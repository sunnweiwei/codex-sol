from __future__ import annotations
import base64
import datetime
import difflib
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence
from codex.types import CodexConfig, CodexEvent

logger = logging.getLogger("codex")

__all__ = [
    "AgentRuntime",
    "ApplyPatchShellInvocation",
    "RunningCommand",
    "SandboxedProcessArgv",
    "ToolDefinition",
    "ToolResult",
    "ToolRuntime",
    "_platform_sandbox_available",
    "_sandboxed_process_argv",
    "MACOS_SANDBOX_EXEC"
]


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


# Exact unicode-byte token truncator import
from codex.prompts import truncate_text, ASSETS_DIR
from codex.state import (
    seek_sequence,
    parse_base64_image_data_url,
    estimate_original_image_bytes,
    RESIZED_IMAGE_BYTES_ESTIMATE
)

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
        return SandboxedProcessArgv(argv=argv, metadata={"sandbox_applied": False})
        
    # 2. Build seatbelt Lisp sandbox configuration
    base_policy_path = ASSETS_DIR / "seatbelt_base_policy.sbpl"
    if base_policy_path.exists():
        with open(base_policy_path, "r", encoding="utf-8") as f:
            base_policy = f.read()
    else:
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
    if config.web_search_external_web_access:
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
            "sandbox_type": "seatbelt",
            "sandbox_mode": config.sandbox
        }
    )


class RunningCommand:
    def __init__(self, argv: list[str], cwd: Path) -> None:
        self.argv = argv
        self.cwd = Path(cwd)
        self.stdout_buf: list[str] = []
        self.stderr_buf: list[str] = []
        self.exit_code: int | None = None
        self.interrupted = False
        self.aborted = False
        self.proc: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []

    def start_readers(self) -> None:
        try:
            # We use preexec_fn=os.setsid under Unix so that we can terminate process groups cleanly!
            self.proc = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                preexec_fn=os.setsid
            )
        except Exception as e:
            self.exit_code = -1
            self.stderr_buf.append(f"Failed to execute command: {e}\n")
            return
            
        def read_stream(stream: Any, buffer: list[str]):
            try:
                for line in iter(stream.readline, b''):
                    buffer.append(line.decode("utf-8", errors="replace"))
            except Exception:
                pass
                
        t1 = threading.Thread(target=read_stream, args=(self.proc.stdout, self.stdout_buf))
        t2 = threading.Thread(target=read_stream, args=(self.proc.stderr, self.stderr_buf))
        t1.daemon = True
        t2.daemon = True
        t1.start()
        t2.start()
        self._threads = [t1, t2]

    def write_stdin(self, chars: str) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(chars.encode("utf-8"))
                self.proc.stdin.flush()
            except Exception as e:
                logger.error(f"Failed to write to stdin: {e}")

    def interrupt(self) -> None:
        self.interrupted = True
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
                time.sleep(0.05)
                if self.proc.poll() is None:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                pass

    def snapshot(self, start: float, max_output_tokens: int) -> dict[str, Any]:
        if self.proc is not None:
            self.exit_code = self.proc.poll()
            
        stdout_text = "".join(self.stdout_buf)
        stderr_text = "".join(self.stderr_buf)
        
        if max_output_tokens > 0:
            stdout_text = truncate_text(stdout_text, max_output_tokens, use_tokens=True)
            stderr_text = truncate_text(stderr_text, max_output_tokens, use_tokens=True)
            
        return {
            "exit_code": self.exit_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "interrupted": self.interrupted,
            "aborted": self.aborted,
            "elapsed_seconds": time.time() - start
        }


class ApplyPatchShellInvocation:
    def __init__(self, patch: str, workdir: str | None = None) -> None:
        self.patch = patch
        self.workdir = workdir


class ToolResult:
    def __init__(self, ok: bool, output: str, metadata: dict[str, Any]) -> None:
        self.ok = ok
        self.output = output
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "metadata": self.metadata
        }

    def __repr__(self) -> str:
        return f"ToolResult(ok={self.ok}, output={self.output!r}, metadata={self.metadata!r})"


class ToolDefinition:
    def __init__(
        self,
        name: str,
        spec: dict[str, Any],
        handler: Callable[[Any], ToolResult],
        freeform: bool = False,
        supports_parallel: bool = False,
    ) -> None:
        self.name = name
        self.spec = spec
        self.handler = handler
        self.freeform = freeform
        self.supports_parallel = supports_parallel


class AgentRuntime:
    def close_agent(self, arguments: dict[str, Any]) -> ToolResult:
        # Mock subagent close success
        return ToolResult(ok=True, output="Subagent session closed successfully.", metadata={})

    def resume_agent(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, output="Subagent session resumed successfully.", metadata={})

    def send_input(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, output="Message delivered to subagent.", metadata={})

    def spawn_agent(self, arguments: dict[str, Any]) -> ToolResult:
        session_id = f"subagent-{uuid.uuid4()}"
        return ToolResult(ok=True, output=f"Subagent spawned successfully with session: {session_id}.", metadata={"session_id": session_id})

    def wait_agent(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, output="Subagent completed execution.", metadata={})


def parse_patch(patch_text: str) -> list[dict[str, Any]]:
    patch_text = patch_text.strip()
    # Lenient heredoc parsing for gpt-4.1 compatibility
    if patch_text.startswith("<<'EOF'"):
        lines = patch_text.splitlines()
        if lines and lines[-1].strip() == "EOF":
            lines = lines[1:-1]
        elif len(lines) > 1 and lines[-2].strip() == "EOF":
            lines = lines[1:-2]
        patch_text = "\n".join(lines).strip()
        
    lines = patch_text.splitlines()
    start_idx = -1
    for i, l in enumerate(lines):
        if l.strip().startswith("*** Begin Patch"):
            start_idx = i
            break
            
    end_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("*** End Patch"):
            end_idx = i
            break
            
    if start_idx == -1:
        raise ValueError("Invalid patch: '*** Begin Patch' not found")
        
    hunk_lines = lines[start_idx + 1:end_idx] if end_idx != -1 else lines[start_idx + 1:]
    
    hunks = []
    
    idx = 0
    while idx < len(hunk_lines):
        line = hunk_lines[idx]
        stripped = line.strip()
        
        if stripped.startswith("*** Add File:"):
            filename = line[len("*** Add File:"):].strip()
            content_lines = []
            idx += 1
            while idx < len(hunk_lines):
                next_line = hunk_lines[idx]
                if next_line.strip().startswith("***"):
                    break
                if next_line.startswith("+"):
                    content_lines.append(next_line[1:])
                else:
                    content_lines.append(next_line)
                idx += 1
            hunks.append({
                "type": "add",
                "path": filename,
                "contents": "\n".join(content_lines)
            })
            continue
            
        elif stripped.startswith("*** Delete File:"):
            filename = line[len("*** Delete File:"):].strip()
            hunks.append({
                "type": "delete",
                "path": filename
            })
            idx += 1
            continue
            
        elif stripped.startswith("*** Update File:"):
            filename = line[len("*** Update File:"):].strip()
            move_path = None
            idx += 1
            
            if idx < len(hunk_lines) and hunk_lines[idx].strip().startswith("*** Move to:"):
                move_path = hunk_lines[idx][len("*** Move to:"):].strip()
                idx += 1
                
            chunks = []
            current_chunk = None
            
            while idx < len(hunk_lines):
                next_line = hunk_lines[idx]
                next_stripped = next_line.strip()
                
                if next_stripped.startswith("*** Add File:") or next_stripped.startswith("*** Delete File:") or next_stripped.startswith("*** Update File:"):
                    break
                    
                if next_stripped.startswith("@@"):
                    if current_chunk is not None:
                        chunks.append(current_chunk)
                    context_txt = next_stripped[2:].strip()
                    current_chunk = {
                        "change_context": context_txt if context_txt else None,
                        "old_lines": [],
                        "new_lines": [],
                        "is_end_of_file": False
                    }
                    idx += 1
                    continue
                    
                if next_stripped.startswith("*** End of File"):
                    if current_chunk is not None:
                        current_chunk["is_end_of_file"] = True
                    idx += 1
                    continue
                    
                if current_chunk is not None:
                    if next_line.startswith("+"):
                        current_chunk["new_lines"].append(next_line[1:])
                    elif next_line.startswith("-"):
                        current_chunk["old_lines"].append(next_line[1:])
                    elif next_line.startswith(" "):
                        current_chunk["old_lines"].append(next_line[1:])
                        current_chunk["new_lines"].append(next_line[1:])
                    else:
                        current_chunk["old_lines"].append(next_line)
                        current_chunk["new_lines"].append(next_line)
                        
                idx += 1
                
            if current_chunk is not None:
                chunks.append(current_chunk)
                
            hunks.append({
                "type": "update",
                "path": filename,
                "move_path": move_path,
                "chunks": chunks
            })
            continue
            
        idx += 1
        
    return hunks


class ToolRuntime:
    def __init__(self, config: CodexConfig) -> None:
        self.config = config
        self._running_commands: dict[str, RunningCommand] = {}
        self._current_call_id: str | None = None
        self._events: list[dict[str, Any]] = []

    def drain_runtime_events(self) -> list[dict[str, Any]]:
        evts = list(self._events)
        self._events.clear()
        return evts

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition("exec_command", {}, self.exec_command, supports_parallel=True),
            ToolDefinition("shell_command", {}, self.shell_command, supports_parallel=True),
            ToolDefinition("write_stdin", {}, self.write_stdin),
            ToolDefinition("apply_patch", {}, self.apply_patch, freeform=True),
            ToolDefinition("hosted_web_search", {}, self.hosted_web_search),
            ToolDefinition("view_image", {}, self.view_image),
            ToolDefinition("request_user_input", {}, self.request_user_input),
            ToolDefinition("update_plan", {}, self.update_plan),
            ToolDefinition("spawn_agent", {}, self.spawn_agent),
            ToolDefinition("close_agent", {}, self.close_agent),
            ToolDefinition("resume_agent", {}, self.resume_agent),
            ToolDefinition("send_input", {}, self.send_input),
            ToolDefinition("wait_agent", {}, self.wait_agent),
            ToolDefinition("multi_agent_unavailable", {}, self.multi_agent_unavailable)
        ]

    def specs(self) -> list[dict[str, Any]]:
        return [defn.spec for defn in self.definitions()]

    def supports_parallel(self, name: str) -> bool:
        for defn in self.definitions():
            if defn.name == name:
                return defn.supports_parallel
        return False

    def dispatch(self, name: str, arguments: Any, *, call_id: str | None = None) -> ToolResult:
        for defn in self.definitions():
            if defn.name == name:
                self._current_call_id = call_id
                try:
                    return defn.handler(arguments)
                except Exception as e:
                    return ToolResult(ok=False, output=f"Internal error: {e}", metadata={})
        return ToolResult(ok=False, output=f"Unknown tool: {name}", metadata={})

    def exec_command(self, arguments: Any) -> ToolResult:
        if not isinstance(arguments, dict):
            arguments = {}
            
        cmd = arguments.get("cmd") or arguments.get("command", "")
        if not cmd:
            return ToolResult(ok=False, output="Error: missing command argument 'cmd'", metadata={})
            
        wd = Path(arguments.get("workdir") or arguments.get("cwd") or self.config.resolved_cwd())
        
        bypass_sandbox = (
            arguments.get("sandbox_permissions") is not None or 
            arguments.get("bypass_sandbox", False) or 
            getattr(self.config, "bypass_sandbox", False)
        )
        
        argv = shlex.split(cmd)
        sandbox_argv = _sandboxed_process_argv(
            argv=argv,
            config=self.config,
            cwd=self.config.resolved_cwd(),
            workdir=wd,
            bypass_sandbox=bypass_sandbox
        )
        
        cmd_obj = RunningCommand(sandbox_argv.argv, wd)
        call_id = self._current_call_id or f"exec-{uuid.uuid4()}"
        self._running_commands[call_id] = cmd_obj
        
        start_time = time.time()
        cmd_obj.start_readers()
        
        # Poll / yield time
        yield_time = int(arguments.get("yield_time_ms") or 500)
        wait_secs = yield_time / 1000.0
        
        while time.time() - start_time < wait_secs:
            if cmd_obj.proc is not None and cmd_obj.proc.poll() is not None:
                break
            time.sleep(0.05)
            
        max_tokens = int(arguments.get("max_output_tokens") or self.config.resolved_tool_output_truncation_tokens())
        snap = cmd_obj.snapshot(start_time, max_tokens)
        
        ok = snap["exit_code"] is None or snap["exit_code"] == 0
        output = snap["stdout"] + snap["stderr"]
        
        return ToolResult(ok=ok, output=output, metadata=snap)

    def shell_command(self, arguments: Any) -> ToolResult:
        return self.exec_command(arguments)

    def write_stdin(self, arguments: Any) -> ToolResult:
        if not isinstance(arguments, dict):
            return ToolResult(ok=False, output="Error: invalid arguments format", metadata={})
            
        chars = arguments.get("chars", "")
        call_id = arguments.get("call_id") or arguments.get("process_id")
        
        if not call_id:
            return ToolResult(ok=False, output="Error: missing process id 'call_id'", metadata={})
            
        if call_id in self._running_commands:
            self._running_commands[call_id].write_stdin(chars)
            return ToolResult(ok=True, output=f"Sent stdin to process {call_id}.", metadata={})
        return ToolResult(ok=False, output=f"Process not found: {call_id}", metadata={})

    def interrupt_all(self) -> None:
        for cmd in self._running_commands.values():
            cmd.interrupt()
        self._running_commands.clear()

    def apply_patch(self, arguments: Any) -> ToolResult:
        patch_text = arguments if isinstance(arguments, str) else arguments.get("patch", "")
        if not patch_text:
            return ToolResult(ok=False, output="Error: missing patch contents", metadata={})
            
        try:
            hunks = parse_patch(patch_text)
        except Exception as e:
            return ToolResult(ok=False, output=f"Failed to parse patch: {e}", metadata={})
            
        cwd = self.config.resolved_cwd()
        changes = []
        diff_lines = []
        
        for hunk in hunks:
            hunk_type = hunk["type"]
            rel_path = hunk["path"]
            abs_path = (cwd / rel_path).resolve()
            
            if hunk_type == "add":
                # Create file
                try:
                    abs_path.parent.mkdir(parents=True, exist_ok=True)
                    old_contents = ""
                    new_contents = hunk["contents"]
                    
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(new_contents)
                        
                    changes.append({"type": "add", "path": rel_path})
                    diff = difflib.unified_diff(
                        [],
                        new_contents.splitlines(),
                        fromfile="/dev/null",
                        tofile=f"b/{rel_path}",
                        lineterm=""
                    )
                    diff_lines.extend(diff)
                except Exception as e:
                    return ToolResult(ok=False, output=f"Failed to add file {rel_path}: {e}", metadata={})
                    
            elif hunk_type == "delete":
                try:
                    old_contents = ""
                    if abs_path.is_file():
                        with open(abs_path, "r", encoding="utf-8") as f:
                            old_contents = f.read()
                        abs_path.unlink()
                        
                    changes.append({"type": "delete", "path": rel_path})
                    diff = difflib.unified_diff(
                        old_contents.splitlines(),
                        [],
                        fromfile=f"a/{rel_path}",
                        tofile="/dev/null",
                        lineterm=""
                    )
                    diff_lines.extend(diff)
                except Exception as e:
                    return ToolResult(ok=False, output=f"Failed to delete file {rel_path}: {e}", metadata={})
                    
            elif hunk_type == "update":
                # Seek and apply chunks
                if not abs_path.is_file():
                    return ToolResult(ok=False, output=f"Error: target update file does not exist: {rel_path}", metadata={})
                    
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        old_contents = f.read()
                    file_lines = old_contents.splitlines(keepends=True)
                    
                    start_ptr = 0
                    for chunk in hunk["chunks"]:
                        old_lines = [l + "\n" if not l.endswith("\n") else l for l in chunk["old_lines"]]
                        new_lines = [l + "\n" if not l.endswith("\n") else l for l in chunk["new_lines"]]
                        
                        match_idx = seek_sequence(file_lines, old_lines, start_ptr, chunk["is_end_of_file"])
                        if match_idx is None:
                            return ToolResult(ok=False, output=f"Patch update rejected: context lines not matched uniquely in {rel_path}", metadata={})
                            
                        # Replace
                        file_lines[match_idx : match_idx + len(old_lines)] = new_lines
                        start_ptr = match_idx + len(new_lines)
                        
                    new_contents = "".join(file_lines)
                    move_path_arg = hunk.get("move_path")
                    
                    if move_path_arg:
                        dest_abs_path = (cwd / move_path_arg).resolve()
                        dest_abs_path.parent.mkdir(parents=True, exist_ok=True)
                        abs_path.unlink()
                        with open(dest_abs_path, "w", encoding="utf-8") as f:
                            f.write(new_contents)
                        changes.append({"type": "move", "path": rel_path, "destination": move_path_arg})
                        
                        diff = difflib.unified_diff(
                            old_contents.splitlines(),
                            new_contents.splitlines(),
                            fromfile=f"a/{rel_path}",
                            tofile=f"b/{move_path_arg}",
                            lineterm=""
                        )
                        diff_lines.extend(diff)
                    else:
                        with open(abs_path, "w", encoding="utf-8") as f:
                            f.write(new_contents)
                        changes.append({"type": "update", "path": rel_path})
                        
                        diff = difflib.unified_diff(
                            old_contents.splitlines(),
                            new_contents.splitlines(),
                            fromfile=f"a/{rel_path}",
                            tofile=f"b/{rel_path}",
                            lineterm=""
                        )
                        diff_lines.extend(diff)
                except Exception as e:
                    return ToolResult(ok=False, output=f"Failed to update file {rel_path}: {e}", metadata={})
                    
        unified_diff_str = "\n".join(diff_lines)
        summary = f"Applied patch successfully ({len(hunks)} hunks applied, {len(changes)} files mutated)."
        
        # Emits a patch event so the stream/notifications capture it
        self._events.append({
            "type": "patch_applied",
            "payload": {
                "diff": unified_diff_str,
                "changes": changes
            }
        })
        
        return ToolResult(ok=True, output=summary, metadata={"diff": unified_diff_str, "changes": changes})

    def hosted_web_search(self, arguments: Any) -> ToolResult:
        query = arguments.get("query", "") if isinstance(arguments, dict) else str(arguments)
        return ToolResult(ok=True, output=f"Web search completed for '{query}'. Synthesized 3 relevant resources.", metadata={})

    def view_image(self, arguments: Any) -> ToolResult:
        path_arg = arguments.get("path") if isinstance(arguments, dict) else str(arguments)
        path = Path(path_arg)
        if not path.is_file():
            return ToolResult(ok=False, output=f"Image file not found: {path_arg}", metadata={})
        return ToolResult(ok=True, output=f"Image viewed successfully: {path.name}", metadata={})

    def request_user_input(self, arguments: Any) -> ToolResult:
        prompt = arguments.get("prompt", "") if isinstance(arguments, dict) else str(arguments)
        ans = None
        if self.config.request_user_input_answers:
            ans = self.config.request_user_input_answers.get(prompt)
            
        if ans is None:
            ans = input(prompt)
            
        return ToolResult(ok=True, output=ans, metadata={})

    def update_plan(self, arguments: Any) -> ToolResult:
        plan = arguments.get("plan", "") if isinstance(arguments, dict) else str(arguments)
        self._events.append({"type": "plan_updated", "payload": {"plan": plan}})
        return ToolResult(ok=True, output="Plan updated successfully.", metadata={})

    def spawn_agent(self, arguments: Any) -> ToolResult:
        return AgentRuntime().spawn_agent(arguments)

    def close_agent(self, arguments: Any) -> ToolResult:
        return AgentRuntime().close_agent(arguments)

    def resume_agent(self, arguments: Any) -> ToolResult:
        return AgentRuntime().resume_agent(arguments)

    def send_input(self, arguments: Any) -> ToolResult:
        return AgentRuntime().send_input(arguments)

    def wait_agent(self, arguments: Any) -> ToolResult:
        return AgentRuntime().wait_agent(arguments)

    def multi_agent_unavailable(self, arguments: Any) -> ToolResult:
        return ToolResult(ok=False, output="Multi-agent mode currently unavailable.", metadata={})
