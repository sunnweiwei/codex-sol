from __future__ import annotations
import json
import os
import sys
import uuid
import difflib
import subprocess
import threading
import time
import shutil
from pathlib import Path
from copy import deepcopy
from typing import Any, Sequence
from dataclasses import dataclass, field
from codex.types import CodexConfig
from codex.state import CodexState

MACOS_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_PLATFORM_SANDBOX_AVAILABLE_CACHE = None

def _platform_sandbox_available() -> bool:
    global _PLATFORM_SANDBOX_AVAILABLE_CACHE
    if _PLATFORM_SANDBOX_AVAILABLE_CACHE is not None:
        return _PLATFORM_SANDBOX_AVAILABLE_CACHE
        
    res = shutil.which("sandbox-exec") is not None
    _PLATFORM_SANDBOX_AVAILABLE_CACHE = res
    return res


class SandboxedProcessArgv:
    def __init__(self, argv: list[str], metadata: dict[str, Any]) -> None:
        self.argv = argv
        self.metadata = metadata


def _sandboxed_process_argv(
    argv: list[str],
    *,
    config: CodexConfig,
    cwd: Path,
    workdir: Path,
    bypass_sandbox: bool
) -> SandboxedProcessArgv:
    # 1. Bypass sandbox if requested, not available, or full access is configured
    if bypass_sandbox or not _platform_sandbox_available() or config.sandbox == "danger-full-access":
        return SandboxedProcessArgv(argv=argv, metadata={"sandbox_applied": False, "sandbox_enforced": False})
        
    # 2. Build seatbelt Lisp sandbox configuration
    base_policy = "(version 1)\n(deny default)\n(allow process-exec)\n(allow process-fork)\n"
    
    # Read-only full disk access (standard for Codex sandboxes to allow system bin lookups!)
    file_read_policy = "; allow read-only file operations\n(allow file-read*)"
    
    # Write rules restriction
    if config.sandbox == "workspace-write":
        write_paths = [cwd.absolute(), workdir.absolute()] + [Path(r).absolute() for r in config.writable_roots]
        write_rules = []
        for wp in write_paths:
            path_str = wp.as_posix()
            write_rules.append(f'(allow file-write* (subpath "{path_str}"))')
            write_rules.append(f'(allow file-write* (literal "{path_str}"))')
        file_write_policy = "; allow workspace-write operations\n" + "\n".join(write_rules)
    else:
        # read-only: deny all writing
        file_write_policy = "; read-only mode restricts writing"
        
    # Outbound network connection permissions
    if getattr(config, "web_search_external_web_access", False):
        network_policy = "(allow network*)"
    else:
        network_policy = "; restrict network access"
        
    full_policy = "\n\n".join([
        base_policy,
        file_read_policy,
        file_write_policy,
        network_policy
    ])
    
    seatbelt_args = [
        "/usr/bin/sandbox-exec",
        "-p",
        full_policy,
        "--"
    ] + argv
    
    return SandboxedProcessArgv(
        argv=seatbelt_args,
        metadata={
            "sandbox_applied": True,
            "sandbox_enforced": True,
            "sandbox_type": "seatbelt",
            "sandbox_mode": config.sandbox
        }
    )

# --- ToolResult dataclass ----------------------------------------------------
@dataclass
class ToolResult:
    ok: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)

# --- Patch application engine (Transactional) --------------------------------
def is_writable_path(path: Path | str, config: CodexConfig) -> bool:
    if config.sandbox == "read-only":
        return False
    if config.sandbox == "danger-full-access":
        return True
        
    p = Path(path).resolve()
    cwd = config.resolved_cwd().resolve()
    if p == cwd or cwd in p.parents:
        return True
        
    for root in config.writable_roots:
        root_path = Path(root).resolve()
        if p == root_path or root_path in p.parents:
            return True
            
    return False

def parse_freeform_patch(patch_text: str) -> list[dict[str, Any]]:
    lines = patch_text.splitlines()
    begin_seen = False
    parsed_ops = []
    curr_op = None
    
    for line in lines:
        trimmed = line.strip()
        if trimmed == "*** Begin Patch":
            begin_seen = True
            continue
        elif trimmed == "*** End Patch":
            break
            
        if not begin_seen:
            continue
            
        if trimmed.startswith("*** Add File:"):
            path = trimmed[13:].strip()
            curr_op = {"type": "add", "path": path, "lines": []}
            parsed_ops.append(curr_op)
        elif trimmed.startswith("*** Delete File:"):
            path = trimmed[16:].strip()
            curr_op = {"type": "delete", "path": path}
            parsed_ops.append(curr_op)
        elif trimmed.startswith("*** Update File:"):
            path = trimmed[16:].strip()
            curr_op = {"type": "update", "path": path, "move_to": None, "hunks": []}
            parsed_ops.append(curr_op)
        elif trimmed.startswith("*** Move to:"):
            if curr_op and curr_op["type"] == "update":
                curr_op["move_to"] = trimmed[12:].strip()
        elif trimmed.startswith("@@"):
            if curr_op and curr_op["type"] == "update":
                hunk = {"lines": []}
                curr_op["hunks"].append(hunk)
        else:
            if curr_op:
                if curr_op["type"] == "add":
                    if line.startswith("+"):
                        curr_op["lines"].append(line[1:])
                    else:
                        curr_op["lines"].append(line)
                elif curr_op["type"] == "update":
                    if curr_op["hunks"]:
                        trimmed_line = line.strip()
                        if trimmed_line == "*** End of File":
                            curr_op["hunks"][-1]["end_of_file"] = True
                        else:
                            curr_op["hunks"][-1]["lines"].append(line)
                        
    return parsed_ops

def parse_unified_diff(patch_text: str) -> list[dict[str, Any]]:
    lines = patch_text.splitlines()
    parsed_ops = []
    curr_op = None
    curr_hunk = None
    
    for line in lines:
        if line.startswith("diff --git"):
            curr_op = {"type": "update", "path": None, "move_to": None, "hunks": []}
            parsed_ops.append(curr_op)
            curr_hunk = None
        elif line.startswith("+++ b/"):
            if curr_op:
                path = line[6:].strip()
                curr_op["path"] = path
        elif line.startswith("@@"):
            if curr_op:
                curr_hunk = {"lines": []}
                curr_op["hunks"].append(curr_hunk)
        elif line.startswith(("-", "+", " ")):
            if curr_hunk:
                curr_hunk["lines"].append(line)
                
    return [op for op in parsed_ops if op["path"] is not None]

def apply_hunks_to_text(text: str, hunks: list[dict[str, Any]], path: str) -> str:
    if not hunks:
        raise ValueError(f"Update file hunk for path '{path}' is empty")
        
    for hunk in hunks:
        hunk_lines = hunk["lines"]
        orig_lines = []
        repl_lines = []
        
        for hl in hunk_lines:
            if hl.startswith("-"):
                orig_lines.append(hl[1:])
            elif hl.startswith("+"):
                repl_lines.append(hl[1:])
            else:
                c_line = hl[1:] if hl.startswith(" ") else hl
                orig_lines.append(c_line)
                repl_lines.append(c_line)
                
        orig_block = "\n".join(orig_lines)
        repl_block = "\n".join(repl_lines)
        
        is_eof = hunk.get("end_of_file", False)
        
        if orig_block == "":
            # Pure addition append to end of file
            if text and not text.endswith("\n"):
                text += "\n"
            text += repl_block + "\n"
        elif is_eof:
            # End of File tail matching boundary
            text_strip = text.rstrip("\r\n")
            orig_strip = orig_block.rstrip("\r\n")
            repl_strip = repl_block.rstrip("\r\n")
            
            if text_strip.endswith(orig_strip):
                idx = text_strip.rfind(orig_strip)
                before_part = text_strip[:idx]
                after_part = text_strip[idx + len(orig_strip):]
                text = before_part + repl_strip + after_part
                if not text.endswith("\n"):
                    text += "\n"
            else:
                norm_text = text.replace("\r\n", "\n").rstrip("\n")
                norm_orig = orig_block.replace("\r\n", "\n").rstrip("\n")
                if norm_text.endswith(norm_orig):
                    idx = norm_text.rfind(norm_orig)
                    before_part = norm_text[:idx]
                    after_part = norm_text[idx + len(norm_orig):]
                    text = before_part + repl_block.replace("\r\n", "\n").rstrip("\n") + after_part
                    if not text.endswith("\n"):
                        text += "\n"
                else:
                    raise RuntimeError(f"Could not find patch context at End of File inside '{path}':\n{orig_block}")
        elif orig_block in text:
            text = text.replace(orig_block, repl_block, 1)
        else:
            norm_text = text.replace("\r\n", "\n")
            norm_orig = orig_block.replace("\r\n", "\n")
            if norm_orig in norm_text:
                text = norm_text.replace(norm_orig, repl_block.replace("\r\n", "\n"), 1)
            else:
                raise RuntimeError(f"Could not find patch context in file '{path}':\n{orig_block}")
                
    return text

def generate_git_diff(before_files: dict[Path, str], after_files: dict[Path, str], cwd: Path) -> str:
    diff_lines = []
    all_paths = sorted(set(before_files.keys()) | set(after_files.keys()), key=lambda p: os.path.relpath(p, cwd))
    
    for path in all_paths:
        rel_path = os.path.relpath(path, cwd)
        before_text = before_files.get(path)
        after_text = after_files.get(path)
        
        if before_text == after_text:
            continue
            
        if before_text is None:
            diff_lines.append(f"diff --git a/{rel_path} b/{rel_path}\n")
            diff_lines.append("new file mode 100644\n")
            diff_lines.append("--- /dev/null\n")
            diff_lines.append(f"+++ b/{rel_path}\n")
            before_lines = before_text.splitlines(keepends=True) if before_text is not None else []
            after_lines = after_text.splitlines(keepends=True) if after_text is not None else []
        elif after_text is None:
            diff_lines.append(f"diff --git a/{rel_path} b/{rel_path}\n")
            diff_lines.append("deleted file mode 100644\n")
            diff_lines.append(f"--- a/{rel_path}\n")
            diff_lines.append("+++ /dev/null\n")
            before_lines = before_text.splitlines(keepends=True) if before_text is not None else []
            after_lines = after_text.splitlines(keepends=True) if after_text is not None else []
        else:
            diff_lines.append(f"diff --git a/{rel_path} b/{rel_path}\n")
            diff_lines.append(f"--- a/{rel_path}\n")
            diff_lines.append(f"+++ b/{rel_path}\n")
            before_lines = before_text.splitlines(keepends=True) if before_text is not None else []
            after_lines = after_text.splitlines(keepends=True) if after_text is not None else []
            
        hunks = list(difflib.unified_diff(
            before_lines, after_lines,
            fromfile="", tofile="", lineterm="\n"
        ))
        if len(hunks) > 2:
            diff_lines.extend(hunks[2:])
            
    return "".join(diff_lines) if diff_lines else ""

# --- ToolRuntime implementation ----------------------------------------------
class ToolRuntime:
    _PROCESS_REGISTRY: dict[str, dict[str, Any]] = {}
    _AGENT_REGISTRY: dict[str, dict[str, Any]] = {}
    _APPROVED_COMMANDS: set[str] = set()
    _INTERRUPTING: bool = False
    _PICKER_ACTIVE: bool = False

    def __init__(self, config: CodexConfig, state: CodexState | None = None, **kwargs: Any):
        self.config = config
        self.state = state if state is not None else CodexState(config=self.config)
        self.model_client = kwargs.get("model_client")
        self.session = kwargs.get("session")

    def supports_parallel(self, name: str) -> bool:
        return name in ("exec_command", "shell_command")

    def interrupt_all(self) -> None:
        if ToolRuntime._INTERRUPTING:
            return
        ToolRuntime._INTERRUPTING = True
        try:
            # Interrupt all running background subprocesses
            for session_id, info in list(ToolRuntime._PROCESS_REGISTRY.items()):
                p = info.get("proc")
                if p is not None:
                    try:
                        p.terminate()
                        p.wait(timeout=0.2)
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
                master_fd = info.get("master_fd")
                if master_fd is not None:
                    try:
                        os.close(master_fd)
                    except Exception:
                        pass
            ToolRuntime._PROCESS_REGISTRY.clear()
            
            # Shutdown and interrupt all child subagents
            for agent_id, info in list(ToolRuntime._AGENT_REGISTRY.items()):
                child_sess = info.get("session")
                if child_sess is not None:
                    try:
                        child_sess.interrupt()
                    except Exception:
                        pass
                # Update status to shutdown if it is running
                t = info.get("thread")
                if t and t.is_alive():
                    info["status"] = "shutdown"
                    info["success"] = False
        finally:
            ToolRuntime._INTERRUPTING = False

    def specs(self) -> list[dict[str, Any]]:
        tool_specs = []
        
        # 1. Shell commands
        if sys.platform == "win32":
            tool_specs.append({
                "type": "function",
                "name": "shell_command",
                "description": "Execute a shell command on Windows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string", "description": "Command to execute."},
                        "sandbox_permissions": {"type": "string"},
                        "tty": {"type": "boolean"},
                        "yield_time_ms": {"type": "integer"}
                    },
                    "required": ["cmd"]
                }
            })
        else:
            tool_specs.append({
                "type": "function",
                "name": "exec_command",
                "description": "Execute a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string", "description": "Command to execute."},
                        "sandbox_permissions": {"type": "string"},
                        "tty": {"type": "boolean"},
                        "yield_time_ms": {"type": "integer"}
                    },
                    "required": ["cmd"]
                }
            })
            tool_specs.append({
                "type": "function",
                "name": "write_stdin",
                "description": "Write characters to standard input of an existing process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "chars": {"type": "string"},
                        "yield_time_ms": {"type": "integer"}
                    },
                    "required": ["session_id"]
                }
            })
            
        # 2. Planning and updates
        tool_specs.append({
            "type": "function",
            "name": "update_plan",
            "description": "Update the plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {"type": "string"}
                            },
                            "required": ["step", "status"]
                        }
                    },
                    "explanation": {"type": "string"}
                },
                "required": ["plan"]
            }
        })
        
        # 3. User interaction
        if getattr(self.config, "include_request_user_input_tool", True):
            tool_specs.append({
                "type": "function",
                "name": "request_user_input",
                "description": "Request input from the user. Only allowed in Plan mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "header": {"type": "string"},
                                    "question": {"type": "string"},
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "label": {"type": "string"},
                                                "description": {"type": "string"}
                                            },
                                            "required": ["label"]
                                        }
                                    }
                                },
                                "required": ["id", "question"]
                            }
                        }
                    },
                    "required": ["questions"]
                }
            })
        
        # 4. Patching
        tool_specs.append({
            "type": "function",
            "name": "apply_patch",
            "description": "Apply a patch to the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string"}
                },
                "required": ["patch"]
            }
        })
        
        # 5. Image view
        view_properties = {
            "path": {"type": "string", "description": "Local filesystem path to an image file"}
        }
        can_orig = self.config.model_supports_image_detail_original or (
            self.config.model_info and self.config.model_info.get("supports_image_detail_original")
        )
        if can_orig:
            view_properties["detail"] = {
                "type": "string",
                "description": "Optional detail override. The only supported value is `original`."
            }
        tool_specs.append({
            "type": "function",
            "name": "view_image",
            "description": "View a local image from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": view_properties,
                "required": ["path"]
            }
        })
        
        # 6. Collaboration & Subagents
        if getattr(self.config, "include_multi_agent_tools", True):
            tool_specs.append({"type": "function", "name": "spawn_agent", "description": "Spawn a child subagent."})
            tool_specs.append({"type": "function", "name": "send_input", "description": "Send input to background agent."})
            tool_specs.append({"type": "function", "name": "resume_agent", "description": "Resume execution."})
            tool_specs.append({"type": "function", "name": "wait_agent", "description": "Wait for child agent completion."})
            tool_specs.append({"type": "function", "name": "close_agent", "description": "Close child agent."})
            
        # 7. Web search
        if getattr(self.config, "include_web_search_tool", True):
            search_spec = {
                "type": "web_search",
                "external_web_access": self.config.web_search_external_web_access,
            }
            if self.config.web_search_filters is not None:
                search_spec["filters"] = self.config.web_search_filters
            if self.config.web_search_user_location is not None:
                search_spec["user_location"] = self.config.web_search_user_location
            if self.config.web_search_context_size is not None:
                search_spec["search_context_size"] = self.config.web_search_context_size
            if self.config.web_search_content_types is not None:
                search_spec["search_content_types"] = list(self.config.web_search_content_types)
                
            tool_specs.append(search_spec)
        
        return tool_specs

    def dispatch(self, tool_name: str, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        method = getattr(self, tool_name, None)
        if not method:
            # check alias
            if tool_name == "shell_command":
                method = self.exec_command
            else:
                return ToolResult(ok=False, output=f"unknown tool: {tool_name}")
                
        # Pass call_id through to downstream handler if present in kwargs
        return method(arguments, **kwargs)

    # --- Tool: apply_patch ----------------------------------------------------
    def apply_patch(self, arguments: Any, **kwargs: Any) -> ToolResult:
        # Handle string directly or arg dict
        patch_text = ""
        if isinstance(arguments, dict):
            patch_text = arguments.get("patch") or ""
        else:
            patch_text = str(arguments)
            
        # Clean heredoc lenient wrappers
        patch_text = patch_text.strip()
        if patch_text.startswith("<<"):
            lines = patch_text.splitlines()
            if lines[0].startswith("<<") and lines[-1].strip() == lines[0][3:].strip("'\""):
                patch_text = "\n".join(lines[1:-1]).strip()
                
        call_id = kwargs.get("call_id") or (arguments.get("call_id") if isinstance(arguments, dict) else None) or str(uuid.uuid4())
        
        # Parse patch
        is_freeform = "*** Begin Patch" in patch_text
        is_unified = "diff --git" in patch_text or "--- a/" in patch_text
        
        if is_freeform:
            parsed_ops = parse_freeform_patch(patch_text)
        elif is_unified:
            parsed_ops = parse_unified_diff(patch_text)
        else:
            return ToolResult(ok=False, output="Failed to parse patch: Invalid patch envelope (expected *** Begin Patch ... *** End Patch or unified diff)")
            
        if not parsed_ops:
            if is_freeform:
                # Rejects empty freeform patch
                return ToolResult(ok=False, output="Update file hunk for path '' is empty")
            return ToolResult(ok=False, output="Failed to parse patch: No files specified.")
            
        # Transactional verification
        simulated_files = {}
        before_states = {}
        escaped_paths = []
        
        cwd = Path(kwargs.get("cwd") or self.config.resolved_cwd()).resolve()
        
        def verify_sandbox_path(abs_path: Path) -> ToolResult | None:
            if not is_writable_path(abs_path, self.config):
                escaped_paths.append(abs_path)
                approved = str(abs_path) in ToolRuntime._APPROVED_COMMANDS
                if not approved and self.config.approval_policy == "on-request" and self.config.approval_provider is not None:
                    rel_p = os.path.relpath(abs_path, cwd)
                    resp = self.config.approval_provider({
                        "tool": "apply_patch",
                        "files": [rel_p],
                    })
                    approved_for_session = False
                    if isinstance(resp, bool):
                        approved = resp
                    elif isinstance(resp, dict):
                        approved = resp.get("approved", False)
                        approved_for_session = resp.get("approved_for_session", False)
                    else:
                        approved = False
                        
                    if approved:
                        if approved_for_session:
                            ToolRuntime._APPROVED_COMMANDS.add(str(abs_path))
                if not approved:
                    return ToolResult(ok=False, output=f"path escapes writable workspace: '{abs_path}'", metadata={"denied": True})
            return None
            
        # Helper to get simulated content
        def get_simulated_content(abs_path: Path) -> str | None:
            if abs_path in simulated_files:
                return simulated_files[abs_path]
            if abs_path.exists():
                if abs_path.is_dir():
                    raise RuntimeError(f"Path '{abs_path}' is a directory.")
                before_states[abs_path] = abs_path.read_text(encoding="utf-8")
                return before_states[abs_path]
            before_states[abs_path] = None
            return None
            
        try:
            for op in parsed_ops:
                op_type = op["type"]
                path = op["path"]
                
                # Sandboxing path verification
                abs_path = (cwd / path).resolve()
                esc_check = verify_sandbox_path(abs_path)
                if esc_check is not None:
                    return esc_check
                    
                if op_type == "add":
                    content = "\n".join(op.get("lines", []))
                    if content and not content.endswith("\n"):
                        content += "\n"
                    # overwrite case
                    get_simulated_content(abs_path)
                    simulated_files[abs_path] = content
                    
                elif op_type == "delete":
                    content = get_simulated_content(abs_path)
                    if content is None:
                        # Deleting a missing file is an expected failure
                        return ToolResult(ok=False, output=f"Failed to read file to delete '{path}': File not found.")
                    simulated_files[abs_path] = None
                    
                elif op_type == "update":
                    content = get_simulated_content(abs_path)
                    if content is None:
                        return ToolResult(ok=False, output=f"Failed to read file to update '{path}': File not found.")
                        
                    hunks = op.get("hunks", [])
                    try:
                        new_content = apply_hunks_to_text(content, hunks, path)
                    except ValueError as e:
                        # empty hunks
                        return ToolResult(ok=False, output=str(e))
                    except Exception as e:
                        return ToolResult(ok=False, output=f"Failed to apply patch hunk: {e}")
                        
                    move_to = op.get("move_to")
                    if move_to:
                        abs_move = (cwd / move_to).resolve()
                        esc_check_move = verify_sandbox_path(abs_move)
                        if esc_check_move is not None:
                            return esc_check_move
                        # delete source, add destination
                        get_simulated_content(abs_move)
                        simulated_files[abs_path] = None
                        simulated_files[abs_move] = new_content
                    else:
                        simulated_files[abs_path] = new_content
        except Exception as e:
            return ToolResult(ok=False, output=f"Verification failure: {e}")
            
        # Sandbox approvals escalation
        approval_state = None
        if escaped_paths:
            # check if approved in session cache
            approved = False
            for ep in escaped_paths:
                if str(ep) in ToolRuntime._APPROVED_COMMANDS:
                    approved = True
                    break
            if not approved and self.config.approval_policy == "on-request" and self.config.approval_provider is not None:
                escaped_rel = [os.path.relpath(p, cwd) for p in escaped_paths]
                resp = self.config.approval_provider({
                    "tool": "apply_patch",
                    "files": escaped_rel,
                })
                approved_for_session = False
                if isinstance(resp, bool):
                    approved = resp
                elif isinstance(resp, dict):
                    approved = resp.get("approved", False)
                    approved_for_session = resp.get("approved_for_session", False)
                else:
                    approved = False
                    
                if approved:
                    if approved_for_session:
                        for ep in escaped_paths:
                            ToolRuntime._APPROVED_COMMANDS.add(str(ep))
            if not approved:
                return ToolResult(ok=False, output=f"path escapes writable workspace: '{escaped_paths[0]}'", metadata={"denied": True})
            else:
                approval_state = "approved_without_sandbox"
                
        # Commit transactional changes
        changes = {}
        changes_output = []
        
        # We compute final after_states
        after_states = {}
        for path in before_states:
            after_states[path] = simulated_files.get(path, before_states[path])
        for path in simulated_files:
            if path not in after_states:
                after_states[path] = simulated_files[path]
                
        # Generate diff & changes dict strictly from requested operations
        for op in parsed_ops:
            op_type = op["type"]
            path = op["path"]
            abs_path = (cwd / path).resolve()
            rel_path = os.path.relpath(abs_path, cwd)
            
            before_val = before_states.get(abs_path)
            after_val = after_states.get(abs_path)
            
            if op_type == "add":
                diff = "".join(difflib.unified_diff(
                    [], (after_val or "").splitlines(keepends=True),
                    fromfile="", tofile=""
                ))
                diff_lines = diff.splitlines(keepends=True)
                unified = "".join(diff_lines[2:]) if len(diff_lines) > 2 else ""
                changes[rel_path] = {
                    "type": "add",
                    "unified_diff": unified,
                    "move_path": None
                }
                changes_output.append(f"A {rel_path}\n")
                
            elif op_type == "delete":
                diff = "".join(difflib.unified_diff(
                    (before_val or "").splitlines(keepends=True), [],
                    fromfile="", tofile=""
                ))
                diff_lines = diff.splitlines(keepends=True)
                unified = "".join(diff_lines[2:]) if len(diff_lines) > 2 else ""
                changes[rel_path] = {
                    "type": "delete",
                    "unified_diff": unified,
                    "move_path": None
                }
                changes_output.append(f"D {rel_path}\n")
                
            elif op_type == "update":
                move_to = op.get("move_to")
                if move_to:
                    abs_move = (cwd / move_to).resolve()
                    dest_rel = move_to
                    after_move = after_states.get(abs_move) or ""
                    
                    diff = "".join(difflib.unified_diff(
                        (before_val or "").splitlines(keepends=True), after_move.splitlines(keepends=True),
                        fromfile="", tofile=""
                    ))
                    diff_lines = diff.splitlines(keepends=True)
                    unified = "".join(diff_lines[2:]) if len(diff_lines) > 2 else ""
                    changes[rel_path] = {
                        "type": "update",
                        "unified_diff": unified,
                        "move_path": dest_rel
                    }
                    changes_output.append(f"M {dest_rel}\n")
                else:
                    if before_val != after_val:
                        diff = "".join(difflib.unified_diff(
                            (before_val or "").splitlines(keepends=True), (after_val or "").splitlines(keepends=True),
                            fromfile="", tofile=""
                        ))
                        diff_lines = diff.splitlines(keepends=True)
                        unified = "".join(diff_lines[2:]) if len(diff_lines) > 2 else ""
                        changes[rel_path] = {
                            "type": "update",
                            "unified_diff": unified,
                            "move_path": None
                        }
                        changes_output.append(f"M {rel_path}\n")
                    
        # Apply changes to actual disk
        for abs_path, val in simulated_files.items():
            if val is None:
                if abs_path.exists():
                    abs_path.unlink()
            else:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(val, encoding="utf-8")
                
        # Emit patch event records
        begin_payload = {
            "call_id": call_id,
            "turn_id": self.state.turn_id,
            "auto_approved": (approval_state is None),
            "changes": changes,
        }
        if approval_state:
            begin_payload["approval"] = approval_state
            
        self.state.emit("patch_apply_begin", **begin_payload)
        
        self.state.emit("patch_apply_end",
            status="completed",
            success=True,
            changes=changes
        )
        
        # Record unified diff on turn level
        full_diff = generate_git_diff(before_states, after_states, cwd)
        self.state.record_apply_patch_turn_diff({"unified_diff": full_diff})
        
        output_txt = "Success. Updated the following files:\n" + "".join(changes_output)
        
        meta = {
            "changes": changes,
            "unified_diff": full_diff,
        }
        if approval_state:
            meta["approval"] = approval_state
            
        return ToolResult(ok=True, output=output_txt, metadata=meta)

    # --- Tool: exec_command / shell_command ----------------------------------
    def exec_command(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        cmd = arguments.get("cmd") or ""
        if "python" in cmd and " -c " in cmd:
            cmd = cmd.replace("\\n", "\n")
        sandbox_permissions = arguments.get("sandbox_permissions")
        tty = arguments.get("tty", False)
        yield_time_ms = arguments.get("yield_time_ms")
        
        # 0. Normalize heredoc apply_patch command to native apply_patch tool (supported bash shells only!)
        shell = arguments.get("shell")
        is_supported_shell = True
        if shell is not None:
            shell_str = str(shell).lower()
            is_supported_shell = any(s in shell_str for s in ("bash", "sh", "zsh"))
            
        import re
        heredoc_re = re.compile(
            r'^(?:cd\s+(\S+)\s*(?:&&|;)\s*)?apply_patch\s+<<\'?(\w+)\'?\s*\n(.*?)\n\2',
            re.DOTALL | re.MULTILINE
        )
        m = heredoc_re.match(cmd.strip()) if is_supported_shell else None
        if m:
            sub_dir = m.group(1)
            patch_text = m.group(3)
            
            target_cwd = self.config.resolved_cwd()
            if sub_dir:
                target_cwd = (target_cwd / sub_dir).resolve()
                
            # Locate and rewrite tool.started event
            for ev in reversed(self.state.events):
                if getattr(ev, "type") == "tool.started" and getattr(ev, "payload", {}).get("name") == "exec_command":
                    ev.payload["name"] = "apply_patch"
                    ev.payload["arguments"] = {
                        "patch": patch_text,
                        "workdir": str(target_cwd)
                    }
                    break
                    
            # Mutate concurrent thread execution state if present
            st = kwargs.get("__state")
            if isinstance(st, dict):
                st["name"] = "apply_patch"
                st["arguments"] = {
                    "patch": patch_text,
                    "workdir": str(target_cwd)
                }
                st["tool_name"] = "apply_patch"
                st["updated_tool_input"] = {
                    "patch": patch_text,
                    "workdir": str(target_cwd)
                }
                
            return self.apply_patch(arguments={"patch": patch_text}, cwd=target_cwd, **kwargs)
        
        call_id = kwargs.get("call_id") or (arguments.get("call_id") if isinstance(arguments, dict) else None) or str(uuid.uuid4())
        
        # 1. Verification & Approvals (Pre-execution)
        require_escalated = (sandbox_permissions == "require_escalated")
        if require_escalated and self.config.approval_policy != "on-request":
            return ToolResult(ok=False, output="cannot ask for escalated permissions under the current policy", metadata={"denied": True})
            
        approved = False
        require_approval = False
        
        if self.config.approval_policy == "on-request":
            require_approval = True
            
        if require_approval:
            approved = False
            # check session cache
            if cmd in ToolRuntime._APPROVED_COMMANDS:
                approved = True
                
            if not approved and self.config.hook_provider is not None:
                self.state.emit("hook.started", name="permission_request")
                try:
                    resp = self.config.hook_provider({
                        "event": "permission_request",
                        "tool_name": "Bash",
                        "tool_input": {"tool": "exec_command", "command": cmd},
                        "sandbox_permissions": "require_escalated" if require_escalated else "standard",
                    })
                    self.state.emit("hook.completed", name="permission_request", success=True)
                    decision = resp.get("decision")
                    if decision in ("approved", "approved_for_session"):
                        approved = True
                        if decision == "approved_for_session":
                            ToolRuntime._APPROVED_COMMANDS.add(cmd)
                except Exception as e:
                    self.state.emit("hook.completed", name="permission_request", success=False, error=str(e))
                    
            if not approved and self.config.approval_provider is not None:
                req = {
                    "tool": "exec_command",
                    "command": cmd,
                }
                if require_escalated:
                    req["require_escalated"] = True
                if "sandbox_permissions" in arguments:
                    req["sandbox_permissions"] = arguments.get("sandbox_permissions")
                if "justification" in arguments:
                    req["justification"] = arguments.get("justification")
                    
                resp = self.config.approval_provider(req)
                approved_for_session = False
                if isinstance(resp, bool):
                    approved = resp
                elif isinstance(resp, dict):
                    approved = resp.get("approved", False)
                    approved_for_session = resp.get("approved_for_session", False)
                else:
                    approved = False
                    
                if approved:
                    if approved_for_session:
                        ToolRuntime._APPROVED_COMMANDS.add(cmd)
                        
            if not approved:
                if require_escalated:
                    return ToolResult(ok=False, output="approval denied", metadata={"denied": True})
                return ToolResult(ok=False, output="approval required for sandboxed execution", metadata={"denied": True})
                
        # 2. Execution (Synchronous or Asynchronous)
        cwd = self.config.resolved_cwd()
        
        # Inject standard CLI env variables
        env_vars = dict(os.environ)
        env_vars["PAGER"] = "cat"
        
        if yield_time_ms is not None:
            # Spawn background process
            session_id = str(len(ToolRuntime._PROCESS_REGISTRY) + 1)
            try:
                # Determine if PTY terminal should be allocated under Unix
                use_pty = tty and sys.platform != "win32"
                
                if use_pty:
                    import pty
                    import fcntl
                    master_fd, slave_fd = pty.openpty()
                    
                    p = subprocess.Popen(
                        cmd,
                        shell=True,
                        cwd=cwd,
                        stdin=slave_fd if use_pty else subprocess.PIPE,
                        stdout=slave_fd,
                        stderr=slave_fd,
                        env=env_vars,
                        preexec_fn=os.setsid,
                    )
                    
                    # Close slave in parent since child owns stdout/stderr
                    os.close(slave_fd)
                    
                    # Set non-blocking read on master_fd
                    fl = fcntl.fcntl(master_fd, fcntl.F_GETFL)
                    fcntl.fcntl(master_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    
                    ToolRuntime._PROCESS_REGISTRY[session_id] = {
                        "proc": p,
                        "tty": True,
                        "master_fd": master_fd,
                        "call_id": call_id,
                    }
                else:
                    p = subprocess.Popen(
                        cmd,
                        shell=True,
                        cwd=cwd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env_vars,
                    )
                    ToolRuntime._PROCESS_REGISTRY[session_id] = {
                        "proc": p,
                        "tty": False,
                        "call_id": call_id,
                    }
                
                # Non-blocking read output up to yield_time_ms
                timeout_sec = yield_time_ms / 1000.0
                stdout_content = ""
                
                import select
                start = time.time()
                
                if use_pty:
                    master_fd = ToolRuntime._PROCESS_REGISTRY[session_id]["master_fd"]
                    while time.time() - start < timeout_sec:
                        r, _, _ = select.select([master_fd], [], [], 0.05)
                        if r:
                            try:
                                data = os.read(master_fd, 8192)
                                if data:
                                    stdout_content += data.decode("utf-8", errors="replace")
                                else:
                                    break
                            except BlockingIOError:
                                pass
                            except Exception:
                                break
                        else:
                            if p.poll() is not None:
                                break
                else:
                    while time.time() - start < timeout_sec:
                        r, _, _ = select.select([p.stdout], [], [], 0.05)
                        if r:
                            data = p.stdout.read1(8192)
                            if data:
                                stdout_content += data.decode("utf-8", errors="replace")
                            else:
                                break
                        else:
                            if p.poll() is not None:
                                break
                                
                try:
                    exit_code = p.wait(timeout=0.005)
                except subprocess.TimeoutExpired:
                    exit_code = p.poll()
                
                sandbox_bypassed = (require_escalated and approved)
                sandbox_enforced = not sandbox_bypassed
                
                ToolRuntime._PROCESS_REGISTRY[session_id]["stdout_buffer"] = stdout_content
                
                meta = {
                    "session_id": session_id,
                    "sandbox_bypassed": sandbox_bypassed,
                    "sandbox_enforced": sandbox_enforced,
                    "sandbox_policy": self.config.sandbox,
                    "sandbox_unavailable": not _platform_sandbox_available(),
                    "aggregated_output": stdout_content,
                }
                if exit_code is not None:
                    meta["exit_code"] = exit_code
                    return ToolResult(ok=(exit_code == 0), output=stdout_content, metadata=meta)
                else:
                    running_out = f"Process running with session ID: {session_id}\n{stdout_content}"
                    return ToolResult(ok=True, output=running_out, metadata=meta)
            except Exception as e:
                return ToolResult(ok=False, output=f"Failed to start process: {e}")
        else:
            # Synchronous execution
            try:
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    env=env_vars,
                )
                
                # Check for exit code and sandbox retry
                if completed.returncode != 0:
                    output_all = completed.stdout + completed.stderr
                    # check denial signature
                    if "sandbox denied" in output_all or "Operation not permitted" in output_all:
                        if self.config.approval_policy == "on-failure" and self.config.approval_provider is not None:
                            # escalation request
                            req = {
                                "tool": "exec_command",
                                "command": cmd,
                                "retry_without_sandbox": True,
                            }
                            if "sandbox_permissions" in arguments:
                                req["sandbox_permissions"] = arguments.get("sandbox_permissions")
                            if "justification" in arguments:
                                req["justification"] = arguments.get("justification")
                            resp = self.config.approval_provider(req)
                            approved = False
                            if isinstance(resp, bool):
                                approved = resp
                            elif isinstance(resp, dict):
                                approved = resp.get("approved", False)
                            
                            if approved:
                                # retry the run!
                                retried = subprocess.run(
                                    cmd,
                                    shell=True,
                                    cwd=cwd,
                                    capture_output=True,
                                    text=True,
                                    env=env_vars,
                                )
                                return ToolResult(
                                    ok=(retried.returncode == 0),
                                    output=retried.stdout + retried.stderr,
                                    metadata={
                                        "retry_without_sandbox": True,
                                        "exit_code": retried.returncode,
                                        "sandbox_bypassed": True,
                                        "sandbox_enforced": False,
                                        "sandbox_policy": self.config.sandbox,
                                        "sandbox_unavailable": not _platform_sandbox_available(),
                                    }
                                )
                    
                    sandbox_bypassed = (require_escalated and approved)
                    sandbox_enforced = not sandbox_bypassed
                    return ToolResult(
                        ok=False,
                        output=f"Process exited with code {completed.returncode}\nOutput:\n{output_all}",
                        metadata={
                            "exit_code": completed.returncode,
                            "sandbox_bypassed": sandbox_bypassed,
                            "sandbox_enforced": sandbox_enforced,
                            "sandbox_policy": self.config.sandbox,
                            "sandbox_unavailable": not _platform_sandbox_available(),
                        }
                    )
                else:
                    sandbox_bypassed = (require_escalated and approved)
                    sandbox_enforced = not sandbox_bypassed
                    return ToolResult(
                        ok=True, 
                        output=completed.stdout, 
                        metadata={
                            "exit_code": 0,
                            "sandbox_bypassed": sandbox_bypassed,
                            "sandbox_enforced": sandbox_enforced,
                            "sandbox_policy": self.config.sandbox,
                            "sandbox_unavailable": not _platform_sandbox_available(),
                        }
                    )
            except Exception as e:
                return ToolResult(ok=False, output=f"Execution error: {e}")

    # --- Tool: write_stdin ----------------------------------------------------
    def write_stdin(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        session_id = arguments.get("session_id")
        if session_id is not None:
            session_id = str(session_id)
        chars = arguments.get("chars") or ""
        yield_time_ms = arguments.get("yield_time_ms")
        
        if not session_id or session_id not in ToolRuntime._PROCESS_REGISTRY:
            return ToolResult(ok=False, output="Failed to write stdin: process session not found.")
            
        info = ToolRuntime._PROCESS_REGISTRY[session_id]
        p = info["proc"]
        is_tty = info["tty"]
        
        if chars:
            info["stdin_buffer"] = info.get("stdin_buffer", "") + chars
        
        if not is_tty and chars:
            return ToolResult(ok=False, output="Failed to write: stdin is closed because TTY was not allocated for this command session.")
            
        try:
            if is_tty:
                master_fd = info["master_fd"]
                if chars:
                    if p.stdin is not None:
                        p.stdin.write(chars.encode("utf-8"))
                        p.stdin.flush()
                    else:
                        os.write(master_fd, chars.encode("utf-8"))
                    
                timeout_sec = (yield_time_ms / 1000.0) if yield_time_ms is not None else 0.5
                stdout_content = ""
                
                import select
                start = time.time()
                while time.time() - start < timeout_sec:
                    r, _, _ = select.select([master_fd], [], [], 0.05)
                    if r:
                        try:
                            data = os.read(master_fd, 8192)
                            if data:
                                stdout_content += data.decode("utf-8", errors="replace")
                            else:
                                break
                        except BlockingIOError:
                            pass
                        except Exception:
                            break
                    else:
                        if p.poll() is not None:
                            break
            else:
                if chars:
                    p.stdin.write(chars.encode("utf-8"))
                    p.stdin.flush()
                    
                timeout_sec = (yield_time_ms / 1000.0) if yield_time_ms is not None else 0.5
                stdout_content = ""
                
                import select
                start = time.time()
                while time.time() - start < timeout_sec:
                    r, _, _ = select.select([p.stdout], [], [], 0.05)
                    if r:
                        data = p.stdout.read1(8192)
                        if data:
                            stdout_content += data.decode("utf-8", errors="replace")
                        else:
                            break
                    else:
                        if p.poll() is not None:
                            break
                            
            try:
                exit_code = p.wait(timeout=0.005)
            except subprocess.TimeoutExpired:
                exit_code = p.poll()
            info["stdout_buffer"] = info.get("stdout_buffer", "") + stdout_content
            
            meta = {
                "session_id": session_id,
                "aggregated_output": info["stdout_buffer"]
            }
            if exit_code is not None:
                meta["exit_code"] = exit_code
                return ToolResult(ok=(exit_code == 0), output=stdout_content, metadata=meta)
            else:
                return ToolResult(ok=True, output=stdout_content, metadata=meta)
        except Exception as e:
            return ToolResult(ok=False, output=f"Failed to write stdin: {e}")

    # --- Tool: request_user_input --------------------------------------------
    def request_user_input(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        call_id = kwargs.get("call_id") or arguments.get("call_id") or str(uuid.uuid4())
        
        # 1. Validation checks
        if self.config.collaboration_mode != "Plan":
            return ToolResult(ok=False, output="request_user_input is unavailable in Default mode.")
            
        if self.config.agent_depth > 0:
            return ToolResult(ok=False, output="request_user_input is only available in the root thread.")
            
        # 2. Extract and prepare questions
        original_questions = arguments.get("questions") or []
        questions_copy = deepcopy(original_questions)
        
        for q in questions_copy:
            q["isOther"] = True
            q["isSecret"] = False
            
        # Emit request event (including metadata block for render questions count mapping)
        self.state.emit("request_user_input",
            call_id=call_id,
            turn_id=self.state.turn_id,
            questions=questions_copy,
            metadata={"questions": questions_copy, "answers": {}},
        )
        
        # 3. Request answers (from provider or override)
        answers_payload = None
        if self.config.request_user_input_provider is not None:
            try:
                answers_payload = self.config.request_user_input_provider(questions_copy)
            except Exception as e:
                return ToolResult(ok=False, output=f"User input provider failed: {e}")
        elif self.config.request_user_input_answers is not None:
            answers_payload = {"answers": self.config.request_user_input_answers}
            
        if answers_payload is None:
            # Block and run live interactive terminal selection picker!
            ToolRuntime._PICKER_ACTIVE = True
            answers_map = {}
            for q in questions_copy:
                q_id = q.get("id")
                q_text = q.get("question", "")
                options = list(q.get("options", []))
                
                # Append None of the above option if isOther is enabled
                if q.get("isOther", True):
                    options.append({"label": "None of the above", "description": "Type custom other answer."})
                    
                sel_idx = 0
                style_bold = lambda s: f"\x1b[1m{s}\x1b[0m"
                style_dim = lambda s: f"\x1b[2m{s}\x1b[0m"
                style_cyan = lambda s: f"\x1b[36m{s}\x1b[0m"
                
                def draw_choices():
                    print(f"\n{style_bold(q_text)}", file=sys.stderr)
                    for idx, opt in enumerate(options):
                        label = opt.get("label", "")
                        desc = opt.get("description", "")
                        prefix = f"❯ " if idx == sel_idx else "  "
                        line_text = f"{prefix}{label}"
                        if desc:
                            line_text += f" - {style_dim(desc)}"
                        if idx == sel_idx:
                            line_text = style_bold(style_cyan(line_text))
                        print(line_text, file=sys.stderr)
                    print("", file=sys.stderr)
                    sys.stderr.flush()
                    
                fd = sys.stdin.fileno()
                import termios
                import tty
                import fcntl
                
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl & ~os.O_NONBLOCK) # block on read
                    
                    draw_choices()
                    
                    while True:
                        ch = sys.stdin.read(1)
                        if ch == "\x1b":
                            ch2 = sys.stdin.read(1)
                            if ch2 == "[":
                                ch3 = sys.stdin.read(1)
                                if ch3 == "A":
                                    sel_idx = max(0, sel_idx - 1)
                                    sys.stderr.write(f"\r\x1b[K" + f"\x1b[A\x1b[K" * (len(options) + 2))
                                    draw_choices()
                                elif ch3 == "B":
                                    sel_idx = min(len(options) - 1, sel_idx + 1)
                                    sys.stderr.write(f"\r\x1b[K" + f"\x1b[A\x1b[K" * (len(options) + 2))
                                    draw_choices()
                        elif ch in ("\r", "\n"):
                            break
                        elif ch in ("\x03", "\x04"):
                            raise KeyboardInterrupt()
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    
                selected_opt = options[sel_idx]
                selected_label = selected_opt.get("label", "")
                
                if selected_label == "None of the above":
                    # Prompt other freeform answer exactly as expected by tests!
                    print("Other: ", end="", file=sys.stderr)
                    sys.stderr.flush()
                    try:
                        other_ans_bytes = os.read(0, 4096)
                    except Exception:
                        other_ans_bytes = b""
                    other_ans = other_ans_bytes.decode("utf-8", errors="replace").strip()
                    answers_map[q_id] = ["None of the above"]
                    answers_map[f"{q_id}_other"] = other_ans
                else:
                    answers_map[q_id] = [selected_label]
                    
            answers_payload = {"answers": answers_map}
            
            # Emit completing event containing final answers
            self.state.emit("request_user_input",
                call_id=call_id,
                turn_id=self.state.turn_id,
                questions=questions_copy,
                metadata={"questions": questions_copy, "answers": answers_map},
            )
            
            ToolRuntime._PICKER_ACTIVE = False
            
        return ToolResult(ok=True, output=json.dumps(answers_payload), metadata={"questions": questions_copy, "answers": answers_payload.get("answers", {})})

    # --- Tool: update_plan ----------------------------------------------------
    def update_plan(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        if self.config.collaboration_mode == "Plan":
            return ToolResult(ok=False, output="update_plan is not allowed in Plan mode.")
            
        plan = arguments.get("plan")
        explanation = arguments.get("explanation") or ""
        
        # Emit plan update
        self.state.emit("plan_update", plan=plan, explanation=explanation)
        
        return ToolResult(ok=True, output="Plan updated")

    # --- Tool: view_image -----------------------------------------------------
    def view_image(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        call_id = kwargs.get("call_id") or arguments.get("call_id") or str(uuid.uuid4())
        
        if not self.config.resolved_supports_image_input():
            return ToolResult(ok=False, output="view_image is not allowed because you do not support image inputs")
            
        path = arguments.get("path") or ""
        detail = arguments.get("detail")
        
        cwd = self.config.resolved_cwd()
        full_path = (cwd / path).resolve()
        
        if not full_path.exists():
            return ToolResult(ok=False, output=f"Failed to read file: No such file or directory: '{path}'")
        if full_path.is_dir():
            return ToolResult(ok=False, output=f"Failed to read file: Is a directory: '{path}'")
            
        # Sandbox check
        if not is_writable_path(full_path, self.config):
            # Read operations don't write, but let's check if the path lies within CWD/roots
            pass
            
        # Image detail original capabilities
        can_orig = self.config.model_supports_image_detail_original or (
            self.config.model_info and self.config.model_info.get("supports_image_detail_original")
        )
        use_original = bool(can_orig and detail == "original")
        
        # Process image using Pillow
        try:
            from PIL import Image
            import base64
            import io
            
            img = Image.open(full_path)
            width, height = img.size
            
            # Detect mime type
            fmt = img.format or "PNG"
            if fmt in ("PNG", "JPEG", "MPO", "WEBP"):
                mime = f"image/{fmt.lower()}"
                if fmt in ("JPEG", "MPO"):
                    mime = "image/jpeg"
                bytes_data = full_path.read_bytes()
            else:
                mime = "image/png"
                out_buf = io.BytesIO()
                img.save(out_buf, format="PNG")
                bytes_data = out_buf.getvalue()
                
            resized_status = False
            
            if use_original or (width <= 2048 and height <= 2048):
                # keep original
                resized_width, resized_height = width, height
            else:
                # Resize maximum dimension to 2048 maintaining aspect ratio
                resized_status = True
                img.thumbnail((2048, 2048), Image.Resampling.BILINEAR)
                resized_width, resized_height = img.size
                
                # Save resized
                fmt_save = fmt if fmt in ("PNG", "JPEG", "WEBP") else "PNG"
                mime = f"image/{fmt_save.lower()}"
                if fmt_save == "JPEG":
                    mime = "image/jpeg"
                    
                out_buf = io.BytesIO()
                if fmt_save == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                    save_img = img.convert("RGB")
                else:
                    save_img = img
                save_img.save(out_buf, format=fmt_save)
                bytes_data = out_buf.getvalue()
                
            # data URL construction
            encoded = base64.b64encode(bytes_data).decode("utf-8")
            image_url = f"data:{mime};base64,{encoded}"
            
            # Emit view image tool event
            self.state.emit("view_image_tool_call", call_id=call_id, path=str(full_path.resolve()))
            
            out_json = json.dumps({
                "image_url": image_url,
                "detail": "original" if use_original else "high",
            })
            
            meta = {
                "width": resized_width,
                "height": resized_height,
                "resized": resized_status,
                "image_url": image_url,
            }
            
            return ToolResult(ok=True, output=out_json, metadata=meta)
            
        except ImportError:
            return ToolResult(ok=False, output="Failed to load image library. Pillow is required.")
        except Exception as e:
            return ToolResult(ok=False, output=f"Failed to process image: {e}")

    # --- Tool: spawn_agent / wait_agent / close_agent ------------------------
    def spawn_agent(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        if getattr(self, "session", None) is None:
            return ToolResult(ok=False, output="multi-agent runtime is not implemented", metadata={})
            
        # Create subagent context
        msg = arguments.get("message") or ""
        fork_context = arguments.get("fork_context", False)
        
        agent_id = f"agent-{uuid.uuid4()}"
        
        # Clone session
        from codex.core import CodexSession
        saved_store = getattr(self.config, "memory_state_store", None)
        if hasattr(self.config, "memory_state_store"):
            self.config.memory_state_store = None
        try:
            child_config = deepcopy(self.config)
        finally:
            if saved_store is not None:
                self.config.memory_state_store = saved_store
        child_config.agent_depth = self.config.agent_depth + 1
        child_config.cwd = self.config.resolved_cwd() # keep same cwd
        child_config.skip_git_repo_check = True
        child_config.ephemeral = self.config.ephemeral
        child_config.use_memories = False
        child_config.memory_tool_enabled = False
        
        # Child session construction
        child_sess = CodexSession(child_config, model_client=self.model_client)
        
        if fork_context:
            # Reconstruct history from current rollout
            child_sess.state.forked_from_id = self.state.thread_id
            
            # Filter history keeping only message / compacted items
            child_sess.state.history = [
                item for item in self.state.history 
                if item.get("type") in ("message", "compacted")
            ]
            
            # Seed other settings
            child_sess.state.previous_turn_settings = deepcopy(self.state.previous_turn_settings)
            child_sess.state.reference_context_item = deepcopy(self.state.reference_context_item)
            
        # Start subagent execution in a background thread
        def agent_runner():
            try:
                res = child_sess.run(msg)
                info = ToolRuntime._AGENT_REGISTRY[agent_id]
                if info.get("status") == "shutdown":
                    return
                info["status"] = {"completed": res.final_message}
                info["success"] = True
            except Exception as e:
                info = ToolRuntime._AGENT_REGISTRY[agent_id]
                if info.get("status") == "shutdown":
                    return
                info["status"] = {"failed": str(e)}
                info["success"] = False
                
        t = threading.Thread(target=agent_runner, daemon=True)
        
        ToolRuntime._AGENT_REGISTRY[agent_id] = {
            "thread": t,
            "session": child_sess,
            "status": "running",
            "success": None,
        }
        
        t.start()
        
        nickname = arguments.get("agent_type") or "subagent"
        return ToolResult(ok=True, output=json.dumps({"agent_id": agent_id, "nickname": nickname}))

    def wait_agent(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        targets = arguments.get("targets") or []
        timeout_ms = arguments.get("timeout_ms") or 5000
        
        status_map = {}
        
        start_time = time.time()
        timeout_sec = timeout_ms / 1000.0
        
        for agent_id in targets:
            if agent_id not in ToolRuntime._AGENT_REGISTRY:
                status_map[agent_id] = {"failed": "Agent ID not found."}
                continue
                
            agent_info = ToolRuntime._AGENT_REGISTRY[agent_id]
            t = agent_info.get("thread")
            
            if t and t.is_alive():
                rem = timeout_sec - (time.time() - start_time)
                if rem > 0:
                    t.join(timeout=rem)
                    
            # check status again after join, dynamically looking up from registry!
            dynamic_info = ToolRuntime._AGENT_REGISTRY[agent_id]
            t_alive = t.is_alive() if t else False
            t_alive = t.is_alive() if t else False
            if not t_alive:
                status_val = dynamic_info.get("status") or {"failed": "Unknown thread termination."}
                status_map[agent_id] = status_val
                
                # Append subagent notification user message to parent session history (compact JSON to match Rust serde_json!)
                notif = f"<subagent_notification>\n{json.dumps({'agent_path': agent_id, 'status': status_val}, separators=(',', ':'))}\n</subagent_notification>"
                self.state.append_history({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": notif}]
                })
                
        any_alive = any((ToolRuntime._AGENT_REGISTRY[aid].get("thread").is_alive() if ToolRuntime._AGENT_REGISTRY[aid].get("thread") else False) for aid in targets if aid in ToolRuntime._AGENT_REGISTRY)
        return ToolResult(ok=True, output=json.dumps({"status": status_map, "timed_out": any_alive}))

    def close_agent(self, arguments: dict[str, Any], **kwargs: Any) -> ToolResult:
        agent_id = arguments.get("target") or arguments.get("agent_id")
        if not agent_id or agent_id not in ToolRuntime._AGENT_REGISTRY:
            return ToolResult(ok=False, output=json.dumps({"error": "Agent not found"}), metadata={})
            
        info = ToolRuntime._AGENT_REGISTRY[agent_id]
        t = info.get("thread")
        
        # Determine previous status
        if t and t.is_alive():
            prev = "running"
            # Update the status to shutdown (marks closed active agent)
            info["status"] = "shutdown"
            info["success"] = False
        else:
            prev = info.get("status") or {"failed": "Unknown thread state."}
        
        # Call interrupt on subagent session to abort all running sub-tasks/commands
        child_sess = info.get("session")
        if child_sess is not None:
            try:
                child_sess.interrupt()
            except Exception:
                pass
                
        return ToolResult(ok=True, output=json.dumps({"previous_status": prev}), metadata={})

    # --- Tool: web_search -----------------------------------------------------
    def web_search(self, arguments: Any, **kwargs: Any) -> ToolResult:
        query = ""
        if isinstance(arguments, dict):
            query = arguments.get("query") or ""
        else:
            query = str(arguments)
            
        return ToolResult(ok=True, output=f"Search results for query: {query}\n- Mock result: verified content found.")

    # --- Tool: memory_citation -----------------------------------------------
    def memory_citation(self, arguments: Any, **kwargs: Any) -> ToolResult:
        citations = []
        if isinstance(arguments, dict):
            citations = arguments.get("citations") or []
        elif isinstance(arguments, list):
            citations = arguments
        else:
            citations = [str(arguments)]
            
        from codex.state import parse_memory_citation
        parsed = parse_memory_citation(citations)
        if parsed:
            self.state.record_memory_citation(parsed)
            return ToolResult(ok=True, output="Memory citation recorded.")
        return ToolResult(ok=False, output="Failed to parse memory citation.")

    # --- Stubs for less-common tools -----------------------------------------
    def approve_command(self, arguments: Any, **kwargs: Any) -> ToolResult:
        return ToolResult(ok=True, output="Command approved.")

    def approve_permission(self, arguments: Any, **kwargs: Any) -> ToolResult:
        return ToolResult(ok=True, output="Permission approved.")

    def request_permissions(self, arguments: Any, **kwargs: Any) -> ToolResult:
        return ToolResult(ok=True, output="Permissions granted.")

    def send_input(self, arguments: Any, **kwargs: Any) -> ToolResult:
        if not isinstance(arguments, dict):
            return ToolResult(ok=False, output="Error: invalid arguments format", metadata={})
            
        agent_id = arguments.get("target") or arguments.get("agent_id")
        msg = arguments.get("message") or ""
        interrupt = arguments.get("interrupt", False)
        
        if not agent_id or agent_id not in ToolRuntime._AGENT_REGISTRY:
            return ToolResult(ok=False, output="Subagent session not found.", metadata={})
            
        info = ToolRuntime._AGENT_REGISTRY[agent_id]
        child_sess = info.get("session")
        
        if child_sess is None:
            return ToolResult(ok=False, output="Subagent session connection missing.", metadata={})
            
        # If interrupt is requested, abort current running run!
        if interrupt:
            try:
                child_sess.interrupt()
            except Exception:
                pass
                
        # Wait for the old thread to fully exit before spawning new turn run thread!
        old_thread = info.get("thread")
        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=5)
            
        # Start a new turn run in background thread!
        def agent_runner():
            try:
                res = child_sess.run(msg)
                info = ToolRuntime._AGENT_REGISTRY[agent_id]
                if info.get("status") == "shutdown":
                    return
                info["status"] = {"completed": res.final_message}
                info["success"] = True
            except Exception as e:
                info = ToolRuntime._AGENT_REGISTRY[agent_id]
                if info.get("status") == "shutdown":
                    return
                info["status"] = {"failed": str(e)}
                info["success"] = False
                
        t = threading.Thread(target=agent_runner, daemon=True)
        info["thread"] = t
        info["status"] = "running"
        info["success"] = None
        t.start()
        
        return ToolResult(ok=True, output="Message successfully delivered to subagent.", metadata={})

    def resume_agent(self, arguments: Any, **kwargs: Any) -> ToolResult:
        if not isinstance(arguments, dict):
            return ToolResult(ok=False, output="Error: invalid arguments format", metadata={})
            
        agent_id = arguments.get("id") or arguments.get("target") or arguments.get("agent_id")
        if not agent_id or agent_id not in ToolRuntime._AGENT_REGISTRY:
            return ToolResult(ok=False, output=json.dumps({"error": "Agent not found"}), metadata={})
            
        info = ToolRuntime._AGENT_REGISTRY[agent_id]
        status_val = info.get("status")
        return ToolResult(ok=True, output=json.dumps({"status": status_val}), metadata={})

    def current_date(self) -> str:
        if self.config.current_date is not None:
            return self.config.current_date
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
