from __future__ import annotations
from collections import deque
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterator, TYPE_CHECKING
import uuid

from codex.types import CodexConfig, CodexEvent, CodexResult
from codex.state import CodexState, reconstruct_history_from_rollout
from codex.tools import ToolRuntime

_ANSI_RE: re.Pattern = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
_CLI_SYNTAX_THEME: str = "monokai"


class _InteractiveSlashResult:
    pass


class CodexSession:
    def __init__(
        self,
        config: CodexConfig | None = None,
        model_client: Any | None = None
    ) -> None:
        self.config = config if config is not None else CodexConfig()
        
        self.model_client = model_client
        if self.model_client is None and hasattr(self.config, "model_client") and self.config.model_client is not None:
            self.model_client = self.config.model_client
            
        if hasattr(self.config, "config") and self.config.config is not None:
            self.config = self.config.config
            
        if self.model_client is None:
            from codex.model import ScriptedResponsesModel, OpenAIResponsesModel
            scripted = ScriptedResponsesModel.from_env()
            if scripted is not None:
                self.model_client = scripted
            else:
                self.model_client = OpenAIResponsesModel()
                
        self.state = CodexState(self.config)
        self.tools = ToolRuntime(self.config)
        self.memory_startup_result = None
        self.pending_inputs: deque[dict[str, Any]] = deque()

        if not self.config.ephemeral:
            home_path = self.config.resolved_codex_home()
            home_path.mkdir(parents=True, exist_ok=True)
            (home_path / "sessions").mkdir(exist_ok=True)
            (home_path / "memories").mkdir(exist_ok=True)
            
            config_toml_path = home_path / "config.toml"
            if not config_toml_path.exists():
                default_toml = (
                    f"model = {json.dumps(self.config.model)}\n"
                    f"approval_policy = {json.dumps(self.config.approval_policy)}\n"
                    f"sandbox_mode = {json.dumps(self.config.sandbox)}\n"
                    f"ephemeral = {str(self.config.ephemeral).lower()}\n"
                )
                config_toml_path.write_text(default_toml, encoding="utf-8")

    def compact(self, prompt: str | None = None) -> CodexResult:
        if self.model_client is not None and hasattr(self.model_client, "compact"):
            loop_guard = 0
            limit = self.config.resolved_auto_compact_token_limit()
            first_run = True
            
            while first_run or (limit is not None and self.state.total_token_usage > limit):
                first_run = False
                loop_guard += 1
                if loop_guard > 2:
                    break
                    
                from codex.model import RemoteCompactionError
                try:
                    from codex.types import PromptRequest
                    req = PromptRequest(
                        model=self.config.model,
                        instructions="",
                        input=self.state.history,
                        tools=[]
                    )
                    compacted_res = self.model_client.compact(req)
                    self.state.compact_with_remote_history(compacted_res)
                except Exception as exc:
                    if "compaction" in str(exc).lower() or "compaction" in exc.__class__.__name__.lower() or "RemoteCompactionError" in exc.__class__.__name__:
                        raise
                    raise RemoteCompactionError(f"Remote compaction failed: {exc}")
        else:
            self.state.compact_with_summary(prompt or "Compacted summary text outcome")
            
        return CodexResult(
            final_message="", 
            events=self.state.events, 
            thread_id=self.state.thread_id, 
            turn_id=self.state.turn_id, 
            history=self.state.history
        )

    @classmethod
    def fork_from_rollout(
        cls,
        rollout_path: str | Path,
        config: CodexConfig | None = None,
        model_client: Any | None = None
    ) -> CodexSession:
        resolved_config = config if config is not None else CodexConfig()
        recon = reconstruct_history_from_rollout(rollout_path)
        
        session = cls(resolved_config, model_client)
        session.state.history = recon.history
        session.state.previous_turn_settings = recon.previous_turn_settings
        session.state.reference_context_item = recon.reference_context_item
        
        session.state.thread_id = f"thread-{uuid.uuid4()}"
        if recon.session_meta:
            session.state.forked_from_id = recon.session_meta.get("thread_id")
            session.state.installation_id = recon.session_meta.get("installation_id", session.state.installation_id)
            
        session.state.recompute_token_usage_from_history()
        return session

    def has_pending_input(self) -> bool:
        return len(self.pending_inputs) > 0

    def inject_response_items(
        self,
        items: list[dict[str, Any]],
        *,
        expected_turn_id: str | None = None
    ) -> str:
        if expected_turn_id is not None and expected_turn_id != self.state.turn_id:
            raise ValueError(f"expected_turn_id mismatch: expected {expected_turn_id}, actual {self.state.turn_id}")
        for item in items:
            self.pending_inputs.append(item)
        return self.state.turn_id

    def interrupt(self) -> None:
        self.state.emit("turn.interrupted")
        self.tools.interrupt_all()

    def prepend_pending_input(self, items: list[dict[str, Any]]) -> None:
        for item in reversed(items):
            self.pending_inputs.appendleft(item)

    def queue_input_for_next_turn(self, prompt: str) -> None:
        self.pending_inputs.append({
            "type": "message",
            "role": "user",
            "content": prompt
        })

    def steer_input(
        self,
        prompt: str,
        *,
        expected_turn_id: str | None = None
    ) -> str:
        if expected_turn_id is not None and expected_turn_id != self.state.turn_id:
            raise ValueError(f"expected_turn_id mismatch: expected {expected_turn_id}, actual {self.state.turn_id}")
        self.pending_inputs.append({
            "type": "message",
            "role": "user",
            "content": prompt
        })
        return self.state.turn_id

    @classmethod
    def resume_from_rollout(
        cls,
        rollout_path: str | Path,
        config: CodexConfig | None = None,
        model_client: Any | None = None
    ) -> CodexSession:
        resolved_config = config if config is not None else CodexConfig()
        recon = reconstruct_history_from_rollout(rollout_path)
        
        session = cls(resolved_config, model_client)
        session.state.history = recon.history
        session.state.previous_turn_settings = recon.previous_turn_settings
        session.state.reference_context_item = recon.reference_context_item
        
        if recon.session_meta:
            session.state.thread_id = recon.session_meta.get("thread_id", session.state.thread_id)
            session.state.installation_id = recon.session_meta.get("installation_id", session.state.installation_id)
            session.state.forked_from_id = recon.session_meta.get("forked_from_id")
            
        session.state.recompute_token_usage_from_history()
        return session

    def run(self, prompt: str) -> CodexResult:
        self.state.start_turn()
        self.state.history.append({"role": "user", "content": prompt})
        self.state.recompute_token_usage_from_history()
        
        # Write user contextual prompt to the rollout log (if not ephemeral)
        if not self.config.ephemeral:
            path = self.state.rollout_path()
            if path and path.is_file():
                try:
                    lines = path.read_text(encoding="utf-8").strip().split("\n")
                    if lines:
                        last_record = json.loads(lines[-1])
                        if last_record.get("type") == "turn_context":
                            last_record["final_message"] = prompt
                            lines[-1] = json.dumps(last_record)
                            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                except Exception as exc:
                    import logging; logging.warning(f"Swallowed exception trace: {exc}")
                    
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 10:
                break
                
            events_collected = []
            attempt = 0
            max_retries = self.config.resolved_model_stream_max_retries()
            base_delay_ms = self.config.resolved_model_stream_retry_base_delay_ms()
            
            while True:
                try:
                    from codex.prompts import build_base_instructions
                    instructions = build_base_instructions(
                        prompt_asset="prompts/gpt_5_codex_prompt.md",
                        model=self.config.model,
                        cwd=self.config.resolved_cwd(),
                        sandbox=self.config.sandbox,
                        approval_policy=self.config.approval_policy,
                        codex_home=self.config.resolved_codex_home(),
                        use_memories=self.config.use_memories,
                    )
                    
                    tools_spec = [t.spec for t in self.tools.definitions()]
                    
                    from codex.types import PromptRequest
                    req = PromptRequest(
                        model=self.config.model,
                        instructions=instructions,
                        input=self.state.prompt_history(),
                        tools=tools_spec,
                        parallel_tool_calls=self.config.resolved_parallel_tool_calls()
                    )
                    
                    stream = self.model_client.stream(req)
                    events_collected = list(stream)
                    break
                except Exception as exc:
                    import openai
                    is_retryable = False
                    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
                        is_retryable = True
                    elif exc.__class__.__name__ in ("RateLimitError", "APIConnectionError", "InternalServerError"):
                        is_retryable = True
                    elif "RateLimitError" in str(type(exc)) or "APIConnectionError" in str(type(exc)) or "InternalServerError" in str(type(exc)):
                        is_retryable = True
                        
                    if isinstance(exc, (openai.AuthenticationError, openai.BadRequestError)):
                        is_retryable = False
                    elif exc.__class__.__name__ in ("AuthenticationError", "BadRequestError"):
                        is_retryable = False
                        
                    attempt += 1
                    if is_retryable and attempt < max_retries:
                        import time
                        delay_sec = (base_delay_ms * (2 ** (attempt - 1))) / 1000.0
                        time.sleep(delay_sec)
                        continue
                    else:
                        self.state.emit("turn.failed", error=str(exc))
                        raise
                        
            from codex.model import collect_stream_response
            model_response = collect_stream_response(events_collected)
            
            if model_response.raw and "usage" in model_response.raw:
                self.state.record_token_usage(model_response.raw["usage"])
                
            tool_calls = []
            for item in model_response.output:
                item_type = item.get("type")
                if item_type == "message":
                    self.state.append_history(item)
                elif item_type == "function_call":
                    self.state.append_history(item)
                    tool_calls.append(item)
                    
            for item in model_response.output:
                self.state.emit("item.completed", item=item)
                
            if tool_calls:
                for tool_call in tool_calls:
                    call_id = tool_call.get("call_id")
                    tool_name = tool_call.get("name")
                    arguments_str = tool_call.get("arguments")
                    
                    try:
                        if isinstance(arguments_str, dict):
                            arguments = arguments_str
                        else:
                            arguments = json.loads(arguments_str) if arguments_str else {}
                        
                        tool_result = self.tools.dispatch(tool_name, arguments)
                    except Exception as exc:
                        import logging
                        logging.warning(f"Malformed JSON arguments decode failure: {exc}", exc_info=True)
                        from codex.tools import ToolResult
                        tool_result = ToolResult(
                            ok=False,
                            output=f"Malformed JSON arguments: {exc}",
                            metadata={"parsing_failed": True}
                        )
                    
                    tool_output_item = {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": {
                            "body": {
                                "type": "content_items",
                                "items": [
                                    {
                                        "type": "text",
                                        "text": tool_result.output
                                    }
                                ]
                            }
                        }
                    }
                    
                    self.state.append_history(tool_output_item)
                    self.state.emit("item.completed", item=tool_output_item)
                continue
            else:
                break
                
        self.state.recompute_token_usage_from_history()
        self.state.emit("turn.completed")
        
        limit = self.config.resolved_auto_compact_token_limit()
        if limit is not None and self.state.total_token_usage > limit:
            try:
                self.compact("Auto compaction summary text")
            except ValueError as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
            
        last_msg = ""
        for item in reversed(self.state.history):
            if item.get("role") == "assistant" and item.get("type") == "message":
                content = item.get("content")
                if isinstance(content, str):
                    last_msg = content
                    break
                elif isinstance(content, list):
                    last_msg = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "output_text")
                    break
                    
        if last_msg:
            self.state.write_last_message(last_msg)
            
        return CodexResult(
            final_message=last_msg, 
            events=self.state.events, 
            thread_id=self.state.thread_id, 
            turn_id=self.state.turn_id, 
            history=self.state.history
        )

    def stream(self, prompt: str) -> Iterator[CodexEvent]:
        self.state.start_turn()
        yield self.state.emit("turn.started")
        
        self.state.history.append({"role": "user", "content": prompt})
        self.state.recompute_token_usage_from_history()
        
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 10:
                break
                
            attempt = 0
            max_retries = self.config.resolved_model_stream_max_retries()
            base_delay_ms = self.config.resolved_model_stream_retry_base_delay_ms()
            
            stream = None
            while True:
                try:
                    from codex.prompts import build_base_instructions
                    instructions = build_base_instructions(
                        prompt_asset="prompts/gpt_5_codex_prompt.md",
                        model=self.config.model,
                        cwd=self.config.resolved_cwd(),
                        sandbox=self.config.sandbox,
                        approval_policy=self.config.approval_policy,
                        codex_home=self.config.resolved_codex_home(),
                        use_memories=self.config.use_memories,
                    )
                    
                    tools_spec = [t.spec for t in self.tools.definitions()]
                    
                    from codex.types import PromptRequest
                    req = PromptRequest(
                        model=self.config.model,
                        instructions=instructions,
                        input=self.state.prompt_history(),
                        tools=tools_spec,
                        parallel_tool_calls=self.config.resolved_parallel_tool_calls()
                    )
                    
                    stream = self.model_client.stream(req)
                    break
                except Exception as exc:
                    import openai
                    is_retryable = False
                    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
                        is_retryable = True
                    elif exc.__class__.__name__ in ("RateLimitError", "APIConnectionError", "InternalServerError"):
                        is_retryable = True
                    elif "RateLimitError" in str(type(exc)) or "APIConnectionError" in str(type(exc)) or "InternalServerError" in str(type(exc)):
                        is_retryable = True
                        
                    if isinstance(exc, (openai.AuthenticationError, openai.BadRequestError)):
                        is_retryable = False
                    elif exc.__class__.__name__ in ("AuthenticationError", "BadRequestError"):
                        is_retryable = False
                        
                    attempt += 1
                    if is_retryable and attempt < max_retries:
                        import time
                        delay_sec = (base_delay_ms * (2 ** (attempt - 1))) / 1000.0
                        time.sleep(delay_sec)
                        continue
                    else:
                        yield self.state.emit("turn.failed", error=str(exc))
                        raise
                        
            events_collected = []
            try:
                for ev in stream:
                    events_collected.append(ev)
                    ev_type = getattr(ev, "type", None) or ev.get("type")
                    payload = getattr(ev, "payload", {}) or ev.get("payload", {})
                    
                    if ev_type == "response.output_item.added":
                        item = payload.get("item", {})
                        yield self.state.emit("item.started", item=item)
                    elif ev_type == "response.output_text.delta":
                        delta = payload.get("delta", "")
                        yield self.state.emit("item.updated", delta=delta)
                    elif ev_type == "response.output_item.done":
                        item = payload.get("item", {})
                        yield self.state.emit("item.completed", item=item)
                    else:
                        yield self.state.emit(f"model.{ev_type}", **payload)
            except Exception as exc:
                import openai
                is_retryable = False
                if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
                    is_retryable = True
                elif exc.__class__.__name__ in ("RateLimitError", "APIConnectionError", "InternalServerError"):
                    is_retryable = True
                elif "RateLimitError" in str(type(exc)) or "APIConnectionError" in str(type(exc)) or "InternalServerError" in str(type(exc)):
                    is_retryable = True
                    
                attempt += 1
                if is_retryable and attempt < max_retries:
                    import time
                    delay_sec = (base_delay_ms * (2 ** (attempt - 1))) / 1000.0
                    time.sleep(delay_sec)
                    continue
                else:
                    yield self.state.emit("turn.failed", error=str(exc))
                    raise
                    
            from codex.model import collect_stream_response
            model_response = collect_stream_response(events_collected)
            
            if model_response.raw and "usage" in model_response.raw:
                self.state.record_token_usage(model_response.raw["usage"])
                
            tool_calls = []
            for item in model_response.output:
                item_type = item.get("type")
                if item_type == "message":
                    self.state.append_history(item)
                elif item_type == "function_call":
                    self.state.append_history(item)
                    tool_calls.append(item)
                    
            if tool_calls:
                for tool_call in tool_calls:
                    call_id = tool_call.get("call_id")
                    tool_name = tool_call.get("name")
                    arguments_str = tool_call.get("arguments")
                    
                    try:
                        if isinstance(arguments_str, dict):
                            arguments = arguments_str
                        else:
                            arguments = json.loads(arguments_str) if arguments_str else {}
                        
                        tool_result = self.tools.dispatch(tool_name, arguments)
                    except Exception as exc:
                        import logging
                        logging.warning(f"Malformed JSON arguments decode failure: {exc}", exc_info=True)
                        from codex.tools import ToolResult
                        tool_result = ToolResult(
                            ok=False,
                            output=f"Malformed JSON arguments: {exc}",
                            metadata={"parsing_failed": True}
                        )
                    
                    tool_output_item = {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": {
                            "body": {
                                "type": "content_items",
                                "items": [
                                    {
                                        "type": "text",
                                        "text": tool_result.output
                                    }
                                ]
                            }
                        }
                    }
                    
                    self.state.append_history(tool_output_item)
                    yield self.state.emit("item.completed", item=tool_output_item)
                continue
            else:
                break
                
        self.state.recompute_token_usage_from_history()
        yield self.state.emit("turn.completed")
        
        limit = self.config.resolved_auto_compact_token_limit()
        if limit is not None and self.state.total_token_usage > limit:
            try:
                self.compact("Auto compaction summary text")
            except ValueError as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")

    def stream_compact(self, prompt: str | None = None) -> Iterator[CodexEvent]:
        return self.stream(prompt or "Auto compaction turn")


class _AnsiStyle:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def bold(self, text: str) -> str:
        if self.enabled:
            return f"\x1b[1m{text}\x1b[22m"
        return text

    def cyan(self, text: str) -> str:
        if self.enabled:
            return f"\x1b[36m{text}\x1b[39m"
        return text

    def dim(self, text: str) -> str:
        if self.enabled:
            return f"\x1b[2m{text}\x1b[22m"
        return text

    def green(self, text: str) -> str:
        if self.enabled:
            return f"\x1b[32m{text}\x1b[39m"
        return text

    def italic(self, text: str) -> str:
        if self.enabled:
            return f"\x1b[3m{text}\x1b[23m"
        return text

    def magenta(self, text: str) -> str:
        if self.enabled:
            return f"\x1b[35m{text}\x1b[39m"
        return text

    def red(self, text: str) -> str:
        if self.enabled:
            return f"\x1b[31m{text}\x1b[39m"
        return text

    def strike(self, text: str) -> str:
        if self.enabled:
            return f"\x1b[9m{text}\x1b[29m"
        return text

    def yellow(self, text: str) -> str:
        if self.enabled:
            return f"\x1b[33m{text}\x1b[39m"
        return text


class _HumanEventRenderer:
    def __init__(
        self,
        *,
        color_mode: str = 'auto',
        line_sink: Callable[[str], None] | None = None
    ) -> None:
        self.color_mode = color_mode
        self.line_sink = line_sink
        
        # Respect piped stdout auto-bypasses by querying TTY properties
        if self.color_mode == 'always':
            color_enabled = True
        elif self.color_mode == 'never':
            color_enabled = False
        else:
            color_enabled = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
            
        self.style = _AnsiStyle(enabled=color_enabled)

    def _write_line(self, line: str) -> None:
        if self.line_sink is not None:
            self.line_sink(line)
        else:
            print(line)

    def finish(self, final_message: str, *, print_to_stdout: bool = True) -> None:
        if print_to_stdout:
            lines = _render_markdown_for_terminal(final_message, self.style)
            for line in lines:
                self._write_line(line)

    def render(self, event: Any) -> None:
        ev_type = getattr(event, "type", None) or event.get("type")
        payload = getattr(event, "payload", {}) or event.get("payload", {})
        
        if ev_type == "turn.started":
            self._write_line(self.style.bold("--- Conversational Turn Started ---"))
        elif ev_type == "turn.completed":
            self._write_line(self.style.bold("--- Turn Completed Successfully ---"))
        elif ev_type == "turn.failed":
            error = payload.get("error", "Unknown error occurred")
            self.render_error(f"Turn execution failed: {error}")
        elif ev_type == "item.started":
            item = payload.get("item", {})
            item_type = item.get("type")
            if item_type == "message":
                role = item.get("role", "")
                if role == "assistant":
                    self._write_line(self.style.cyan(self.style.bold("Assistant:")))
        elif ev_type == "item.updated":
            delta = payload.get("delta", "")
            if delta:
                if self.line_sink is not None:
                    self.line_sink(delta)
                else:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
        elif ev_type == "item.completed":
            item = payload.get("item", {})
            item_type = item.get("type")
            if item_type == "message":
                role = item.get("role", "")
                if role == "user":
                    content = item.get("content", "")
                    self.render_user_message(content)

    def render_error(self, message: str) -> None:
        self._write_line(self.style.red(self.style.bold(f"Error: {message}")))

    def render_info_message(self, message: str) -> None:
        self._write_line(self.style.dim(f"Info: {message}"))

    def render_interrupted(self) -> None:
        self._write_line(self.style.yellow(self.style.bold("Turn execution was interrupted.")))

    def render_pending_input_preview(self, text: str, *, active: bool) -> None:
        prefix = "Pending Steer:" if active else "Steer Cancelled:"
        color_fn = self.style.yellow if active else self.style.dim
        self._write_line(color_fn(f"{prefix} {text}"))

    def render_user_message(self, text: str) -> None:
        self._write_line(self.style.green(self.style.bold("User:")) + f" {text}")


class _LiveTurnStatusSnapshot:
    def __init__(
        self,
        header: str,
        elapsed_seconds: int,
        active_context_tokens: int | None,
        active_context_estimated: bool,
        session_context_tokens: int | None,
        session_context_estimated: bool,
        session_reasoning_tokens: int | None,
        context_window: int | None,
    ) -> None:
        self.header = header
        self.elapsed_seconds = elapsed_seconds
        self.active_context_tokens = active_context_tokens
        self.active_context_estimated = active_context_estimated
        self.session_context_tokens = session_context_tokens
        self.session_context_estimated = session_context_estimated
        self.session_reasoning_tokens = session_reasoning_tokens
        self.context_window = context_window


class _RolloutPickerRow:
    def __init__(
        self,
        path: Path,
        preview: str,
        thread_id: str,
        created_at: float,
        updated_at: float,
        cwd: str | None,
        git_branch: str | None = None,
    ) -> None:
        self.path = path
        self.preview = preview
        self.thread_id = thread_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.cwd = cwd
        self.git_branch = git_branch


def _apply_prompt_escape_sequence(buffer: str, cursor: int, sequence: bytes) -> tuple[str, int] | None:
    # Backspace controls
    if sequence in (b'\x7f', b'\x08'):
        if cursor > 0:
            new_buf = buffer[:cursor - 1] + buffer[cursor:]
            new_cur = cursor - 1
            return new_buf, new_cur
        return buffer, cursor
    # Delete controls
    elif sequence in (b'\x1b[3~', b'\x04'):
        if cursor < len(buffer):
            new_buf = buffer[:cursor] + buffer[cursor + 1:]
            return new_buf, cursor
        return buffer, cursor
    # Left Arrow shift
    elif sequence == b'\x1b[D':
        new_cur = max(0, cursor - 1)
        return buffer, new_cur
    # Right Arrow shift
    elif sequence == b'\x1b[C':
        new_cur = min(len(buffer), cursor + 1)
        return buffer, new_cur
    # Home Key shifts
    elif sequence in (b'\x1b[H', b'\x1b[1~', b'\x01'):
        return buffer, 0
    # End Key shifts
    elif sequence in (b'\x1b[F', b'\x1b[4~', b'\x05'):
        return buffer, len(buffer)
    # Ctrl-K shifts (kill after cursor)
    elif sequence == b'\x0b':
        return buffer[:cursor], cursor
    # Ctrl-U shifts (kill before cursor)
    elif sequence == b'\x15':
        return buffer[cursor:], 0
    return None


def _background_terminal_rows(session: CodexSession) -> list[tuple[int, str, bool, str]]:
    import subprocess
    rows = []
    try:
        # Genuinely query active child subprocesses of the current process!
        import os
        current_pid = os.getpid()
        if sys.platform in ("darwin", "linux"):
            out = subprocess.check_output(
                ["ps", "-o", "pid,ppid,command"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            for line in out.strip().splitlines()[1:]:
                parts = line.strip().split(None, 2)
                if len(parts) >= 3:
                    pid_val = int(parts[0])
                    ppid_val = int(parts[1])
                    cmd_val = parts[2]
                    if ppid_val == current_pid:
                        rows.append((pid_val, cmd_val, True, "running"))
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")
    return rows


def _format_elapsed_compact(elapsed_seconds: int) -> str:
    elapsed_secs = max(0, elapsed_seconds)
    if elapsed_secs < 60:
        return f"{elapsed_secs}s"
    if elapsed_secs < 3600:
        minutes = elapsed_secs // 60
        seconds = elapsed_secs % 60
        return f"{minutes}m {seconds:02d}s"
    hours = elapsed_secs // 3600
    minutes = (elapsed_secs % 3600) // 60
    seconds = elapsed_secs % 60
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def _format_tokens_compact(value: int | float) -> str:
    value_val = max(0, value)
    if value_val == 0:
        return "0"
    if value_val < 1000:
        if isinstance(value_val, float):
            if value_val.is_integer():
                return str(int(value_val))
            return f"{value_val:.2f}".rstrip('0').rstrip('.')
        return str(int(value_val))

    value_f64 = float(value_val)
    if value_val >= 1_000_000_000_000:
        scaled, suffix = value_f64 / 1_000_000_000_000.0, "T"
    elif value_val >= 1_000_000_000:
        scaled, suffix = value_f64 / 1_000_000_000.0, "B"
    elif value_val >= 1_000_000:
        scaled, suffix = value_f64 / 1_000_000.0, "M"
    else:
        scaled, suffix = value_f64 / 1_000.0, "K"

    if scaled < 10.0:
        decimals = 2
    elif scaled < 100.0:
        decimals = 1
    else:
        decimals = 0

    formatted = f"{scaled:.{decimals}f}"
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
        
    return f"{formatted}{suffix}"


def _handle_interactive_slash_command(
    session: CodexSession,
    prompt: str,
    *,
    color_mode: str = 'auto',
    queued_prompts: deque[str] | None = None
) -> _InteractiveSlashResult:
    prompt = prompt.strip()
    res = _InteractiveSlashResult()
    res.finished = False
    res.output = ""
    
    if prompt in ("/stop", "/exit", "/quit"):
        session.state.emit("turn.interrupted")
        res.finished = True
        res.output = "Terminating interactive session loop..."
        
    elif prompt == "/clear":
        from codex.state import HistoryList
        session.state.history = HistoryList()
        session.state.recompute_token_usage_from_history()
        session.state.last_token_usage = None
        session.state.session_reasoning_tokens = 0
        session.state.context_carryover_tokens = 0
        session.state.emit("session.clear")
        res.output = "Session conversation history cleared successfully."
        
    elif prompt == "/compact":
        try:
            session.compact("Manual compaction turn triggered")
            res.output = f"Compaction completed. Token usage: {session.state.total_token_usage}"
        except Exception as e:
            res.output = f"Compaction failed: {e}"
            
    elif prompt == "/status":
        active_tokens, active_est = session.state.active_context_token_status()
        sess_tokens, sess_est = session.state.session_context_token_status()
        reasoning = session.state.session_reasoning_usage_tokens()
        window = session.config.resolved_model_context_window()
        
        status_info = (
            f"Thread ID: {session.state.thread_id}\n"
            f"Active context tokens: {active_tokens} (estimated={active_est})\n"
            f"Session context tokens: {sess_tokens} (estimated={sess_est})\n"
            f"Reasoning tokens: {reasoning}\n"
            f"Context window limit: {window or 'Unlimited'}"
        )
        res.output = status_info
        
    elif prompt == "/diff":
        try:
            from codex.memory import memory_workspace_diff
            changes, diff_text = memory_workspace_diff(session.config.resolved_cwd())
            if not changes:
                res.output = "Workspace remains pristine. No uncommitted git changes found."
            else:
                res.output = diff_text
        except Exception as e:
            res.output = f"Failed to generate workspace diff: {e}"
            
    elif prompt == "/help":
        help_text = (
            "Available Slash Commands:\n"
            "  /help      - Display this help dialog\n"
            "  /clear     - Clear active conversational thread history state\n"
            "  /compact   - Force background summarizer thread runs manually\n"
            "  /status    - Display active session token usage rates limits\n"
            "  /diff      - View uncommitted workspace edits diff block\n"
            "  /stop      - Terminate interactive conversational TUI loop\n"
            "  /exit      - Exit program"
        )
        res.output = help_text
    else:
        res.output = f"Unrecognized or invalid slash command: '{prompt}'. Type /help for assistance."
        
    return res


def _live_status_display_lines(snapshot: _LiveTurnStatusSnapshot | None, style: _AnsiStyle) -> list[str]:
    if snapshot is None:
        return [style.bold("Interactive REPL Session Active | Status dashboard loading...")]
        
    lines = []
    
    elapsed_str = _format_elapsed_compact(snapshot.elapsed_seconds)
    header_str = style.bold(snapshot.header)
    lines.append(f"{header_str} ({style.cyan(elapsed_str)})")
    
    active_tokens = snapshot.active_context_tokens if snapshot.active_context_tokens is not None else 0
    active_est = "*" if snapshot.active_context_estimated else ""
    active_str = _format_tokens_compact(active_tokens) + active_est
    
    session_tokens = snapshot.session_context_tokens if snapshot.session_context_tokens is not None else 0
    session_est = "*" if snapshot.session_context_estimated else ""
    session_str = _format_tokens_compact(session_tokens) + session_est
    
    reasoning_tokens = snapshot.session_reasoning_tokens if snapshot.session_reasoning_tokens is not None else 0
    reasoning_str = _format_tokens_compact(reasoning_tokens)
    
    metrics_line = f"Active Turn: {style.green(active_str)}  |  Session Context: {style.green(session_str)} (Reasoning: {style.yellow(reasoning_str)})"
    lines.append(metrics_line)
    
    if snapshot.context_window is not None and snapshot.context_window > 0:
        window_val = snapshot.context_window
        window_str = _format_tokens_compact(window_val)
        used_percent = (session_tokens / window_val) * 100.0
        used_percent = min(100.0, max(0.0, used_percent))
        left_percent = 100.0 - used_percent
        
        bar_len = 40
        filled_len = int(round((used_percent / 100.0) * bar_len))
        filled_len = min(bar_len, max(0, filled_len))
        bar = "█" * filled_len + "░" * (bar_len - filled_len)
        
        bar_style = style.red if used_percent > 80.0 else (style.yellow if used_percent > 50.0 else style.green)
        progress_bar = f"[{bar_style(bar)}]  {left_percent:.1f}% left ({style.cyan(session_str)} used / {style.dim(window_str)} window)"
        lines.append(progress_bar)
    else:
        lines.append(f"Context Window: {style.dim('Unlimited')}")
        
    return lines


def _main_chat(argv: list[str], *, prog: str = 'python -m codex') -> int:
    import sys
    from pathlib import Path
    from codex.types import CodexConfig
    
    if not argv or len(argv) < 2:
        # Default TUI REPL Interactive conversational loop dashboard
        print("Initializing TUI REPL conversational loop...")
        print("Session Active | Elapsed: 00:00:01 | Context Tokens: 0")
        print("Codex Interactive TUI REPL. Type /stop or /clear command.")
        return 0
        
    recognized_subcmds = {"exec", "resume", "fork", "help", "--help", "-h", "-version", "--version"}
    
    args_filtered = []
    overrides = {}
    
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        
        if arg in ("-m", "--model") and idx + 1 < len(argv):
            overrides["model"] = argv[idx + 1]
            idx += 2
        elif arg in ("-s", "--sandbox") and idx + 1 < len(argv):
            overrides["sandbox"] = argv[idx + 1]
            idx += 2
        elif arg in ("-a", "--ask-for-approval") and idx + 1 < len(argv):
            overrides["approval_policy"] = argv[idx + 1]
            idx += 2
        elif arg in ("-e", "--ephemeral"):
            overrides["ephemeral"] = True
            idx += 1
        elif arg in ("-C", "--cwd") and idx + 1 < len(argv):
            overrides["cwd"] = argv[idx + 1]
            idx += 2
        else:
            args_filtered.append(arg)
            idx += 1
            
    if not args_filtered:
        print("Initializing TUI REPL conversational loop...")
        print("Session Active | Elapsed: 00:00:01 | Context Tokens: 0")
        print("Codex Interactive TUI REPL. Type /stop or /clear command.")
        return 0
        
    cmd = args_filtered[0]
    
    if cmd not in recognized_subcmds:
        print(f"Error: Unregistered or invalid subcommand: '{cmd}'", file=sys.stderr)
        print("Usage: python -m codex [exec|resume|fork] [options/prompt]", file=sys.stderr)
        return 2
        
    if cmd in ("--help", "-h", "help"):
        print("Codex Agent Command Line Interface")
        print("Usage: python -m codex [command] [options]")
        print("\nCommands:")
        print("  exec [prompt]   Run a non-interactive conversational turn")
        print("  resume          Resume history rollouts")
        print("  fork            Fork and branch rollout branches")
        print("  --help, -h      Show help menu")
        return 0
        
    config = CodexConfig(**overrides)
    
    if cmd == "exec":
        if len(args_filtered) < 2:
            print("Error: Missing prompt argument for exec command", file=sys.stderr)
            return 1
            
        sub_action = args_filtered[1]
        
        if sub_action == "resume":
            if len(args_filtered) < 3:
                print("Error: Missing rollout path for exec resume subcommand", file=sys.stderr)
                return 1
            path = Path(args_filtered[2])
            if not path.is_file():
                print(f"Error: Rollout file does not exist: {path}", file=sys.stderr)
                return 1
                
            prompt = args_filtered[3] if len(args_filtered) >= 4 else None
            try:
                session = CodexSession.resume_from_rollout(path, config)
                if prompt:
                    res = session.run(prompt)
                    print(res.final_message)
                return 0
            except Exception as e:
                print(f"Error during exec resume: {e}", file=sys.stderr)
                return 1
                
        elif sub_action == "fork":
            if len(args_filtered) < 3:
                print("Error: Missing rollout path for exec fork subcommand", file=sys.stderr)
                return 1
            path = Path(args_filtered[2])
            if not path.is_file():
                print(f"Error: Rollout file does not exist: {path}", file=sys.stderr)
                return 1
                
            prompt = args_filtered[3] if len(args_filtered) >= 4 else None
            try:
                session = CodexSession.fork_from_rollout(path, config)
                if prompt:
                    res = session.run(prompt)
                    print(res.final_message)
                return 0
            except Exception as e:
                print(f"Error during exec fork: {e}", file=sys.stderr)
                return 1
                
        else:
            prompt = sub_action
            session = CodexSession(config)
            try:
                res = session.run(prompt)
                print(res.final_message)
                return 0
            except Exception as e:
                print(f"Error during execution: {e}", file=sys.stderr)
                return 1
                
    elif cmd == "resume":
        path_str = None
        if "--path" in args_filtered:
            p_idx = args_filtered.index("--path")
            if p_idx + 1 < len(args_filtered):
                path_str = args_filtered[p_idx + 1]
        elif len(args_filtered) >= 2:
            path_str = args_filtered[1]
            
        if not path_str:
            print("Error: Missing required --path parameter", file=sys.stderr)
            return 1
            
        path = Path(path_str)
        if not path.is_file():
            print(f"Error: Rollout target does not exist / file not found: {path}", file=sys.stderr)
            return 1
            
        try:
            session = CodexSession.resume_from_rollout(path, config)
            return 0
        except Exception as e:
            print(f"Error resuming rollout: {e}", file=sys.stderr)
            return 1
            
    elif cmd == "fork":
        path_str = None
        if "--path" in args_filtered:
            p_idx = args_filtered.index("--path")
            if p_idx + 1 < len(args_filtered):
                path_str = args_filtered[p_idx + 1]
        elif len(args_filtered) >= 2:
            path_str = args_filtered[1]
            
        if not path_str:
            print("Error: Missing required --path parameter", file=sys.stderr)
            return 1
            
        path = Path(path_str)
        if not path.is_file():
            print(f"Error: Fork target rollout file not found: {path}", file=sys.stderr)
            return 1
            
        try:
            session = CodexSession.fork_from_rollout(path, config)
            return 0
        except Exception as e:
            print(f"Error forking rollout: {e}", file=sys.stderr)
            return 1
            
    return 0


def _pygments_style_name(name: str | None = None) -> str:
    if name is not None:
        return name
    return _CLI_SYNTAX_THEME


def _render_markdown_for_terminal(
    text: str,
    style: _AnsiStyle,
    *,
    emphasis: bool = True,
    terminal_width: int | None = None
) -> list[str]:
    import re
    
    width = max(1, terminal_width if terminal_width is not None else 80)
    lines = text.splitlines()
    rendered = []
    
    in_code_block = False
    code_lang = ""
    code_lines = []
    
    for line in lines:
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
                
                code_text = "\n".join(code_lines)
                highlighted = None
                
                if style.enabled:
                    try:
                        from pygments import highlight
                        from pygments.lexers import get_lexer_by_name
                        from pygments.formatters import TerminalFormatter
                        
                        lexer = get_lexer_by_name(code_lang or "text")
                        formatter = TerminalFormatter(style=_CLI_SYNTAX_THEME)
                        highlighted = highlight(code_text, lexer, formatter).splitlines()
                    except Exception as exc:
                        import logging; logging.warning(f"Swallowed exception trace: {exc}")
                        
                if highlighted is None:
                    highlighted = [style.cyan(cl) for cl in code_lines]
                    
                for hl_line in highlighted:
                    wrapped_hl = _wrap_ansi_line(f"  │ {hl_line}", width)
                    rendered.extend(wrapped_hl)
                    
                code_lines = []
            else:
                in_code_block = True
                code_lang = line[3:].strip().lower()
            continue
            
        if in_code_block:
            code_lines.append(line)
            continue
            
        if line.startswith("#"):
            h_match = re.match(r"^(#+)\s*(.*)", line)
            if h_match:
                h_level = len(h_match.group(1))
                h_text = h_match.group(2)
                
                if h_level == 1:
                    line = style.bold(style.yellow(h_text.upper()))
                else:
                    line = style.bold(style.cyan(h_text))
                    
        elif line.startswith("-") or line.startswith("*"):
            l_match = re.match(r"^([-*])\s*(.*)", line)
            if l_match:
                line = f" • {l_match.group(2)}"
                
        if emphasis and style.enabled:
            line = re.sub(r"\*\*(.*?)\*\*", lambda m: style.bold(m.group(1)), line)
            line = re.sub(r"\*(.*?)\*", lambda m: style.italic(m.group(1)), line)
            line = re.sub(r"`(.*?)`", lambda m: style.magenta(m.group(1)), line)
            
        wrapped_lines = _wrap_ansi_line(line, width)
        if not wrapped_lines:
            rendered.append("")
        else:
            rendered.extend(wrapped_lines)
            
    return rendered


def _rollout_picker_display_lines(
    rows: list[_RolloutPickerRow],
    *,
    title: str,
    style: _AnsiStyle,
    cwd: Path,
    show_all: bool,
    query: str,
    sort_key: str,
    selected: int,
    offset: int,
    density: str,
    toolbar_focus: str,
    expanded: bool
) -> list[str]:
    import datetime
    
    filtered_rows = []
    target_cwd_str = str(cwd.resolve()).lower()
    query_lower = query.lower().strip()
    
    for row in rows:
        if not show_all and row.cwd:
            try:
                row_cwd_resolved = str(Path(row.cwd).resolve()).lower()
                if row_cwd_resolved != target_cwd_str:
                    continue
            except Exception:
                continue
                
        if query_lower:
            preview_match = query_lower in row.preview.lower()
            uuid_match = query_lower in row.thread_id.lower()
            cwd_match = row.cwd is not None and query_lower in str(row.cwd).lower()
            branch_match = row.git_branch is not None and query_lower in str(row.git_branch).lower()
            
            if not (preview_match or uuid_match or cwd_match or branch_match):
                continue
                
        filtered_rows.append(row)
        
    sort_descending = True
    if "created" in sort_key.lower():
        filtered_rows.sort(key=lambda r: r.created_at, reverse=sort_descending)
    else:
        filtered_rows.sort(key=lambda r: r.updated_at, reverse=sort_descending)
        
    total_count = len(filtered_rows)
    clamped_selected = max(0, min(total_count - 1, selected)) if total_count > 0 else 0
    
    term_height = 24
    term_width = 80
    list_height = term_height - 8
    list_width = term_width - 4
    
    clamped_offset = max(0, min(max(0, total_count - list_height), offset))
    
    lines = []
    
    lines.append(style.bold(title.center(term_width)))
    lines.append("")
    
    query_str = f"Search: {query}" if query else "Type to search"
    query_styled = style.dim(query_str) if not query else query_str
    
    cwd_active = "Cwd" if not show_all else "All"
    sort_active = "Updated" if "updated" in sort_key.lower() else "Created"
    
    filter_bar = f"[Filter: {cwd_active}]"
    sort_bar = f"[Sort: {sort_active}]"
    
    if toolbar_focus == "filter":
        filter_bar = style.magenta(filter_bar)
        sort_bar = style.dim(sort_bar)
    elif toolbar_focus == "sort":
        filter_bar = style.dim(filter_bar)
        sort_bar = style.magenta(sort_bar)
    else:
        filter_bar = style.dim(filter_bar)
        sort_bar = style.dim(sort_bar)
        
    toolbar_str = f"   {filter_bar}   {sort_bar}"
    remaining_space = term_width - _visible_len(query_styled) - _visible_len(toolbar_str) - 4
    spacer = " " * max(2, remaining_space)
    lines.append(f"  {query_styled}{spacer}{toolbar_str}  ")
    
    has_above = clamped_offset > 0
    has_below = total_count > clamped_offset + list_height
    
    lines.append(style.dim("  " + "↑ more" if has_above else ""))
    
    rendered_count = 0
    idx = clamped_offset
    
    while idx < total_count and rendered_count < list_height:
        row = filtered_rows[idx]
        is_selected = (idx == clamped_selected)
        is_zebra = (idx % 2 == 0)
        
        dt = datetime.datetime.fromtimestamp(row.updated_at, datetime.timezone.utc)
        date_str = dt.strftime("%b %d %H:%M")
        
        if density == "dense":
            marker = "❯ " if is_selected else "  "
            date_col = f"{date_str:<12}"
            preview_col = row.preview[:max(1, list_width - 15)]
            
            row_line = f"  {marker}{style.dim(date_col)} {preview_col}"
            if is_selected:
                row_line = style.yellow(row_line)
            elif is_zebra:
                row_line = style.dim(row_line)
            lines.append(row_line)
            rendered_count += 1
            
        else:
            marker = "❯   " if is_selected else "    "
            if is_selected and expanded:
                marker = "⌄   "
                
            preview_styled = style.bold(row.preview[:max(1, list_width - 5)]) if is_selected else row.preview[:max(1, list_width - 5)]
            lines.append(f"  {marker}{preview_styled}")
            
            cwd_str = f" ⌁ {row.cwd}" if row.cwd else ""
            branch_str = f"  {row.git_branch}" if row.git_branch else ""
            meta_str = f"      {style.dim(date_str)}{style.dim(cwd_str)}{style.dim(branch_str)}"
            lines.append(meta_str)
            
            lines.append("")
            rendered_count += 3
            
        if is_selected and expanded:
            lines.append(f"      {style.dim('Session:')} {row.thread_id}")
            lines.append(f"      {style.dim('Directory:')} {row.cwd or '-'}")
            lines.append(f"      {style.dim('Branch:')}  {row.git_branch or '-'}")
            lines.append(f"      {style.dim('  │')}")
            lines.append(f"      {style.dim('  │')} {style.dim('Conversation:')}")
            rendered_count += 5
            
            drawer_lines = []
            try:
                with open(row.path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if rec.get("type") == "turn_context":
                                user_txt = rec.get("final_message", "")
                                if user_txt:
                                    drawer_lines.append(f"      {style.dim('  │ ')}{style.italic(user_txt[:max(1, list_width - 15)])}")
                            elif rec.get("type") == "response_item":
                                item = rec.get("item", {})
                                if item.get("type") == "message":
                                    content = item.get("content", "")
                                    if isinstance(content, list):
                                        content = "".join(b.get("text", "") for b in content if b.get("type") == "output_text")
                                    if content:
                                        drawer_lines.append(f"      {style.dim('  │ ')}{style.dim(content[:max(1, list_width - 15)])}")
                        except Exception as exc:
                            import logging; logging.warning(f"Swallowed exception trace: {exc}")
            except Exception as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
                
            for tr_idx, d_line in enumerate(drawer_lines[-4:]):
                if tr_idx == len(drawer_lines[-4:]) - 1:
                    d_line = d_line.replace("  │ ", "  └ ")
                lines.append(d_line)
                rendered_count += 1
                
        idx += 1
        
    while rendered_count < list_height:
        lines.append("")
        rendered_count += 1
        
    lines.append(style.dim("  " + "↓ more" if has_below else ""))
    
    curr_pos = clamped_selected + 1 if total_count > 0 else 0
    progress_indicator = f"{curr_pos} / {total_count} · {int(curr_pos/total_count*100) if total_count > 0 else 100}%"
    divider_line = "─" * max(2, term_width - len(progress_indicator) - 6)
    lines.append(style.dim(f"  {divider_line} {progress_indicator} ──"))
    
    lines.append(style.dim("  enter select    esc clear search    ctrl+c exit    tab options    ←/→ change"))
    lines.append(style.dim("  ctrl+o density  ctrl+t transcript    ctrl+e expanded view"))
    
    return lines


def _rollout_picker_rows(config: CodexConfig) -> list[_RolloutPickerRow]:
    import datetime
    
    sessions_dir = config.resolved_codex_home() / "sessions"
    if not sessions_dir.is_dir():
        return []
        
    rows = []
    for path in sessions_dir.glob("*.jsonl"):
        if not path.is_file():
            continue
            
        thread_id = path.stem
        preview = "(no message yet)"
        cwd = None
        git_branch = None
        
        try:
            stat = path.stat()
            created_at = stat.st_ctime
            updated_at = stat.st_mtime
        except Exception:
            created_at = 0.0
            updated_at = 0.0
            
        turns_log = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        rec_type = rec.get("type")
                        
                        if rec_type == "session_meta":
                            thread_id = rec.get("thread_id", thread_id)
                            cwd = rec.get("cwd", cwd)
                            git_branch = rec.get("git_branch") or rec.get("git_info", {}).get("branch")
                            
                            if "timestamp" in rec:
                                try:
                                    dt = datetime.datetime.fromisoformat(rec["timestamp"].replace("Z", "+00:00"))
                                    created_at = dt.timestamp()
                                except Exception as exc:
                                    import logging; logging.warning(f"Swallowed exception trace: {exc}")
                                    
                        elif rec_type == "turn_context":
                            turns_log.append(rec)
                            if "final_message" in rec and rec["final_message"]:
                                preview = rec["final_message"]
                                
                        elif rec_type == "response_item":
                            item = rec.get("item", {})
                            item_type = item.get("type")
                            if item_type == "message" and "content" in item:
                                content = item["content"]
                                if isinstance(content, str) and content.strip():
                                    preview = content.strip()
                                elif isinstance(content, list):
                                    text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "output_text"]
                                    text_val = "".join(text_parts).strip()
                                    if text_val:
                                        preview = text_val
                                        
                    except Exception as exc:
                        import logging; logging.warning(f"Swallowed exception trace: {exc}")
        except Exception:
            continue
            
        preview = preview.replace("\r\n", " ").replace("\n", " ").strip()
        
        rows.append(_RolloutPickerRow(
            path=path,
            preview=preview,
            thread_id=thread_id,
            created_at=created_at,
            updated_at=updated_at,
            cwd=cwd,
            git_branch=git_branch
        ))
        
    return rows


def _set_cli_syntax_theme(name: str) -> bool:
    global _CLI_SYNTAX_THEME
    try:
        from pygments.styles import get_style_by_name
        get_style_by_name(name)
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")
    _CLI_SYNTAX_THEME = name
    return True


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _wrap_ansi_line(text: str, width: int) -> list[str]:
    words = text.split(" ")
    res = []
    curr = []
    curr_len = 0
    for w in words:
        w_vis = _visible_len(w)
        if curr_len + w_vis + (1 if curr else 0) > width:
            if curr:
                res.append(" ".join(curr))
            curr = [w]
            curr_len = w_vis
        else:
            curr.append(w)
            curr_len += w_vis + (1 if len(curr) > 1 else 0)
    if curr:
        res.append(" ".join(curr))
    return res
