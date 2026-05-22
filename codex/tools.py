# -*- coding: utf-8 -*-
"""
Real tools module with integrated Lark-parity apply_patch parser, mutator,
and routing of exec_command under the macOS Seatbelt Sandboxing.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, Iterable, Literal

from .config import CodexConfig
from .patch_parser import (
    parse_patch,
    apply_patch_execution,
    InvalidPatchError,
    InvalidHunkError
)


class ToolResult:
    """The result of executing a tool call, containing success state and stdout/stderr representation."""
    def __init__(self, ok: bool, output: str, metadata: dict[str, Any]) -> None:
        self.ok = ok
        self.output = output
        self.metadata = metadata


class ToolRuntime:
    """The execution runtime managing Codex tool invocations."""
    def __init__(self, config: CodexConfig) -> None:
        self.config = config

    def apply_patch(self, arguments: Any) -> ToolResult:
        """Executes patch parsing and files mutation under lenient herdodoc extraction rules."""
        # 1. Gracefully decode target arguments block
        patch_text = ""
        if isinstance(arguments, str):
            patch_text = arguments
        elif isinstance(arguments, dict):
            # Probe standard fields
            for key in ("patch", "patch_text", "input", "command"):
                if key in arguments and isinstance(arguments[key], str):
                    patch_text = arguments[key]
                    break
            else:
                # Fallback probe: extract first string value
                for val in arguments.values():
                    if isinstance(val, str):
                        patch_text = val
                        break
                else:
                    return ToolResult(
                        ok=False,
                        output="Invalid patch: No patch text field found in arguments payload\n",
                        metadata={}
                    )
        else:
            patch_text = str(arguments)

        # 2. Run parsing and disk mutations, mapping exceptions verbatim
        try:
            cwd = Path(self.config.cwd) if (self.config and self.config.cwd) else Path.cwd()
            
            # Execute patch application
            success_report = apply_patch_execution(patch_text, cwd)
            return ToolResult(ok=True, output=success_report, metadata={})
            
        except InvalidPatchError as e:
            err_msg = f"Invalid patch: {str(e)}\n"
            return ToolResult(ok=False, output=err_msg, metadata={})
            
        except InvalidHunkError as e:
            err_msg = f"Invalid patch hunk on line {e.line_number}: {e.message}\n"
            return ToolResult(ok=False, output=err_msg, metadata={})
            
        except Exception as e:
            err_msg = f"{str(e)}\n"
            return ToolResult(ok=False, output=err_msg, metadata={})

    def dispatch(self, name: str, arguments: Any, *, call_id: str | None = None) -> ToolResult:
        """Dispatches a tool call by name to its corresponding handler method."""
        if name == "apply_patch":
            return self.apply_patch(arguments)
        elif name == "exec_command":
            return self.exec_command(arguments)
        elif name == "request_user_input":
            return self.request_user_input(arguments)
        elif name == "update_plan":
            return self.update_plan(arguments)
        elif name == "view_image":
            return self.view_image(arguments)
        elif name == "write_stdin":
            return self.write_stdin(arguments)
        return ToolResult(ok=False, output=f"Unknown tool '{name}'", metadata={})

    def exec_command(self, arguments: Any) -> ToolResult:
        """Executes system command shell wrapped under the macOS Seatbelt sandbox."""
        command_args: list[str] = []
        
        # Decode target arguments block
        if isinstance(arguments, str):
            import shlex
            try:
                command_args = shlex.split(arguments)
            except Exception:
                command_args = [arguments]
        elif isinstance(arguments, list):
            command_args = [str(a) for a in arguments]
        elif isinstance(arguments, dict):
            for key in ("command", "args", "arguments", "input"):
                if key in arguments:
                    val = arguments[key]
                    if isinstance(val, list):
                        command_args = [str(v) for v in val]
                        break
                    elif isinstance(val, str):
                        import shlex
                        try:
                            command_args = shlex.split(val)
                        except Exception:
                            command_args = [val]
                        break
            else:
                for val in arguments.values():
                    if isinstance(val, list):
                        command_args = [str(v) for v in val]
                        break
                    elif isinstance(val, str):
                        import shlex
                        try:
                            command_args = shlex.split(val)
                        except Exception:
                            command_args = [val]
                        break
                else:
                    return ToolResult(
                        ok=False,
                        output="Error: No command argument found in payload\n",
                        metadata={}
                    )
        else:
            return ToolResult(
                ok=False,
                output=f"Error: Unsupported command format type {type(arguments).__name__}\n",
                metadata={}
            )
            
        if not command_args:
            return ToolResult(ok=False, output="Error: Command target is empty\n", metadata={})
            
        # Route through the compiled macOS Seatbelt Sandboxing Engine
        try:
            from .sandbox import run_sandboxed_command
            return run_sandboxed_command(command_args, self.config)
        except Exception as e:
            return ToolResult(
                ok=False,
                output=f"Execution failed due to internal sandbox error: {str(e)}\n",
                metadata={}
            )

    def request_user_input(self, arguments: Any) -> ToolResult:
        return ToolResult(ok=False, output="request_user_input stub", metadata={})

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "apply_patch",
                "description": "Applies a unified git diff patch to the workspace files under lenient herdoc extract extraction rules.",
                "parameters": {"type": "object", "properties": {"patch": {"type": "string"}}}
            },
            {
                "name": "exec_command",
                "description": "Runs a command under macOS Seatbelt sandboxing constraints with dynamically resolved whitelists.",
                "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}}}
            }
        ]

    def update_plan(self, arguments: Any) -> ToolResult:
        return ToolResult(ok=False, output="update_plan stub", metadata={})

    def view_image(self, arguments: Any) -> ToolResult:
        return ToolResult(ok=False, output="view_image stub", metadata={})

    def write_stdin(self, arguments: Any) -> ToolResult:
        return ToolResult(ok=False, output="write_stdin stub", metadata={})
