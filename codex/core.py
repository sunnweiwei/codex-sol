from __future__ import annotations
import datetime
import json
import logging
import uuid
import time
from pathlib import Path
from typing import Iterator, Any

from codex.types import CodexConfig, CodexEvent, CodexResult, ModelResponse, PromptRequest
from codex.model import ModelClient, collect_stream_response
from codex.prompts import (
    build_base_instructions,
    build_initial_context_items,
    build_permissions_instructions
)
from codex.state import (
    CodexState,
    reconstruct_history_from_rollout,
    is_user_turn_boundary,
    strip_memory_citations,
    strip_proposed_plan_blocks,
    build_compaction_summary_text
)
from codex.tools import ToolRuntime, ToolResult
from codex.memory import start_memory_startup_task, MemoryBackgroundTask

logger = logging.getLogger("codex")

__all__ = [
    "CodexSession",
    "_MAX_AGENT_WAIT_TIMEOUT_MS",
    "_MIN_AGENT_WAIT_TIMEOUT_MS"
]


_MAX_AGENT_WAIT_TIMEOUT_MS = 60000
_MIN_AGENT_WAIT_TIMEOUT_MS = 1000


class CodexSession:
    def __init__(
        self,
        config: CodexConfig | None = None,
        state: CodexState | None = None,
        model_client: ModelClient | None = None,
        *args: Any,
        **kwargs: Any
    ):
        self.config = config if config is not None else CodexConfig()
        self.state = state if state is not None else CodexState(self.config)
        self.model_client = model_client if model_client is not None else ModelClient()
        self.tools = ToolRuntime(self.config)
        
        self._pending_inputs: list[dict[str, Any]] = []
        self.memory_startup_result: Any | None = None
        
        # Start background memory task if requested
        if self.config.use_memories and self.config.memory_startup_background:
            try:
                self._mem_task = start_memory_startup_task(
                    codex_home=self.config.resolved_codex_home(),
                    model_client=self.model_client,
                    base_config=self.config,
                    run_phase2=self.config.memory_run_phase2_on_startup
                )
            except Exception as e:
                logger.error(f"Failed to launch memory background task: {e}")

    @classmethod
    def resume_from_rollout(
        cls,
        rollout_path: str | Path,
        config: CodexConfig | None = None,
        model_client: ModelClient | None = None
    ) -> CodexSession:
        recon = reconstruct_history_from_rollout(rollout_path)
        cfg = config if config is not None else CodexConfig()
        state = CodexState(
            config=cfg,
            thread_id=recon.session_meta.get("id") if recon.session_meta else str(uuid.uuid4()),
            history=recon.history,
            previous_turn_settings=recon.previous_turn_settings,
            reference_context_item=recon.reference_context_item
        )
        state._rollout_path = Path(rollout_path)
        return cls(config=cfg, state=state, model_client=model_client)

    @classmethod
    def fork_from_rollout(
        cls,
        rollout_path: str | Path,
        config: CodexConfig | None = None,
        model_client: ModelClient | None = None
    ) -> CodexSession:
        recon = reconstruct_history_from_rollout(rollout_path)
        cfg = config if config is not None else CodexConfig()
        # Create a brand new unique thread and separate rollout path for the fork
        state = CodexState(
            config=cfg,
            thread_id=str(uuid.uuid4()),
            forked_from_id=recon.session_meta.get("id") if recon.session_meta else None,
            history=recon.history,
            previous_turn_settings=recon.previous_turn_settings,
            reference_context_item=recon.reference_context_item
        )
        return cls(config=cfg, state=state, model_client=model_client)

    def history(self) -> list[dict[str, Any]]:
        return self.state.history

    def is_compacted(self) -> bool:
        # returns whether the active epoch carried over from a previous compaction
        return self.state.context_carryover_tokens > 0

    def last_message(self) -> str | None:
        for item in reversed(self.state.history):
            if item.get("type") == "message" and item.get("role") == "assistant":
                content = item.get("content", [])
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("input_text", "output_text"):
                        parts.append(c.get("text", ""))
                return "".join(parts)
        return None

    def prepend_pending_input(self, items: list[dict[str, Any]]) -> None:
        self._pending_inputs = list(items) + self._pending_inputs

    def queue_input_for_next_turn(self, prompt: str) -> None:
        self._pending_inputs.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}]
        })

    def has_pending_input(self) -> bool:
        return len(self._pending_inputs) > 0

    def interrupt(self) -> None:
        self.tools.interrupt_all()

    def steer_input(self, prompt: str, *, expected_turn_id: str | None = None) -> str:
        # Registers a steer event to context and returns steer block
        self.state.emit("steer_registered", prompt=prompt, expected_turn_id=expected_turn_id)
        return f"<steer_input>{prompt}</steer_input>"

    def inject_response_items(self, items: list[dict[str, Any]], *, expected_turn_id: str | None = None) -> str:
        for item in items:
            self.state.append_history(item)
        return f"Injected {len(items)} items."

    def compact(self, prompt: str | None = None) -> CodexResult:
        events = list(self.stream_compact(prompt))
        last_msg = self.last_message() or ""
        return CodexResult(
            final_message=last_msg,
            events=events,
            thread_id=self.state.thread_id,
            turn_id=self.state.turn_id,
            history=self.state.history
        )

    def stream_compact(self, prompt: str | None = None) -> Iterator[CodexEvent]:
        # Context Compaction turn runs local model call to generate summary
        self.state.start_turn()
        yield self.state.emit("turn_started", turn_id=self.state.turn_id, thread_id=self.state.thread_id)
        
        # Build compaction instructions using standard prompts
        sys_prompt = build_memory_consolidation_prompt(self.config.resolved_codex_home() / "memories")
        
        user_messages = []
        for item in self.state.history:
            if is_user_turn_boundary(item):
                content = item.get("content", [])
                txt = "".join(c.get("text", "") for c in content if c.get("type") == "input_text")
                user_messages.append(txt)
                
        prompt_content = f"Summarize the conversation so far. Prompt details: {prompt or 'None'}"
        req = PromptRequest(
            model=self.config.model,
            instructions=sys_prompt,
            input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt_content}]}],
            tools=[]
        )
        
        # Stream response
        stream_events = self.model_client.stream(req)
        for evt in stream_events:
            if evt.type == "response.output_text.delta" or evt.type == "output_text_delta":
                delta = evt.payload.get("delta", "")
                yield self.state.emit("agent_message_content_delta", text=delta)
                
        collected = collect_stream_response(stream_events)
        out_text = ""
        for item in collected.output:
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        out_text += c.get("text", "")
                        
        summary_text = out_text if out_text else "Conversation compacted."
        
        # Compact state
        initial_context = build_initial_context_items(self.config)
        self.state.compact_with_summary(summary_text, initial_context)
        
        yield self.state.emit("turn_complete", turn_id=self.state.turn_id, thread_id=self.state.thread_id)

    def run(self, prompt: str, *, dry_run: bool = False, inline_review: bool = True) -> CodexResult:
        events = list(self.stream(prompt, dry_run=dry_run, inline_review=inline_review))
        last_msg = self.last_message() or ""
        return CodexResult(
            final_message=last_msg,
            events=events,
            thread_id=self.state.thread_id,
            turn_id=self.state.turn_id,
            history=self.state.history,
            memory_citations=self.state.memory_citations
        )

    def stream(self, prompt: str, *, dry_run: bool = False, inline_review: bool = True) -> Iterator[CodexEvent]:
        # 1. Verification checks & Compaction triage
        auto_compact_limit = self.config.resolved_auto_compact_token_limit()
        if auto_compact_limit is not None and self.state.active_context_tokens() > auto_compact_limit:
            # Trigger compaction
            for evt in self.stream_compact("Auto compaction triggered"):
                yield evt
                
        # 2. Start turn
        self.state.start_turn()
        yield self.state.emit("turn_started", turn_id=self.state.turn_id, thread_id=self.state.thread_id)
        
        # 3. Write metadata header & seed records immediately
        # Absolute first turn started
        records = self.state.read_rollout_records()
        has_meta = any(r.get("type") == "session_meta" for r in records)
        
        if not has_meta:
            meta = {
                "id": self.state.thread_id,
                "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "cwd": self.config.resolved_cwd().as_posix(),
                "originator": "cli",
                "cli_version": "0.124.0",
                "source": "cli",
                "thread_source": "cli",
                "model_provider": "openai",
                "base_instructions": {"text": build_base_instructions(
                    prompt_asset="gpt_5_codex_prompt.md",
                    model=self.config.model,
                    cwd=self.config.resolved_cwd(),
                    sandbox=self.config.sandbox,
                    approval_policy=self.config.approval_policy,
                    codex_home=self.config.resolved_codex_home(),
                    memory_tool_enabled=self.config.memory_tool_enabled,
                    use_memories=self.config.use_memories
                )},
                "memory_mode": "enabled" if self.config.use_memories else "disabled"
            }
            if self.state.forked_from_id:
                meta["forked_from_id"] = self.state.forked_from_id
            self.state.append_rollout_record("session_meta", meta)
            
        ctx_item = {
            "turn_id": self.state.turn_id,
            "cwd": self.config.resolved_cwd().as_posix(),
            "current_date": self.config.current_date or datetime.datetime.utcnow().strftime("%Y-%m-%d"),
            "timezone": self.config.timezone or "America/Los_Angeles",
            "approval_policy": self.config.approval_policy,
            "sandbox_policy": {
                "sandbox_kind": "seatbelt",
                "policy": self.config.sandbox
            },
            "model": self.config.model,
            "personality": self.config.collaboration_mode,
            "collaboration_mode": self.config.collaboration_mode,
            "user_instructions": getattr(self.config, "user_instructions", ""),
            "developer_instructions": build_base_instructions(
                prompt_asset="gpt_5_codex_prompt.md",
                model=self.config.model,
                cwd=self.config.resolved_cwd(),
                sandbox=self.config.sandbox,
                approval_policy=self.config.approval_policy,
                codex_home=self.config.resolved_codex_home(),
                memory_tool_enabled=self.config.memory_tool_enabled,
                use_memories=self.config.use_memories
            ),
            "final_output_json_schema": self.config.output_schema
        }
        if self.state.last_token_usage:
            ctx_item["summary"] = self.state.last_token_usage
        self.state.append_rollout_record("turn_context", ctx_item)
        
        # 4. Inject initial contextual bootstrapping messages
        if not self.state.history:
            initial_context = build_initial_context_items(self.config)
            for item in initial_context:
                self.state.append_history(item)
                
        # 5. Append user prompt message
        user_message = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}]
        }
        self.state.append_history(user_message)
        
        # 6. Inference and Tool Execution loop
        iterations = 0
        max_iter = self.config.max_iterations or 15
        
        while iterations < max_iter:
            iterations += 1
            
            # Recompile prompt request
            base_text = build_base_instructions(
                prompt_asset="gpt_5_codex_prompt.md",
                model=self.config.model,
                cwd=self.config.resolved_cwd(),
                sandbox=self.config.sandbox,
                approval_policy=self.config.approval_policy,
                codex_home=self.config.resolved_codex_home(),
                memory_tool_enabled=self.config.memory_tool_enabled,
                use_memories=self.config.use_memories
            )
            
            history_inputs = self.state.prompt_history()
            
            req = PromptRequest(
                model=self.config.model,
                instructions=base_text,
                input=history_inputs,
                tools=self.tools.specs(),
                parallel_tool_calls=self.config.resolved_parallel_tool_calls(),
                reasoning=self.config.resolved_reasoning(),
                verbosity=self.config.resolved_verbosity(),
                output_schema=self.config.output_schema,
                client_metadata={"x-codex-installation-id": self.state.installation_id}
            )
            
            if dry_run:
                yield self.state.emit("dry_run_halted", request=req.to_responses_kwargs())
                return
                
            # Invoke Live model stream!
            stream_events = self.model_client.stream(req)
            
            for evt in stream_events:
                if evt.type == "response.output_text.delta" or evt.type == "output_text_delta":
                    delta = evt.payload.get("delta", "")
                    yield self.state.emit("agent_message_content_delta", text=delta)
                elif evt.type == "response.completed" or evt.type == "completed":
                    # Save token count / usage event to rollout!
                    usage = evt.payload.get("usage") or evt.payload.get("token_usage")
                    if usage:
                        self.state.record_token_usage(usage)
                        yield self.state.emit("token_count", token_usage=usage)
                        
            collected = collect_stream_response(stream_events)
            
            # Process and append model response items
            # The output list contains ResponseItem message and/or tool call blocks
            if not collected.output:
                break
                
            has_tool_calls = False
            tool_calls_to_run = []
            
            for item in collected.output:
                self.state.append_history(item)
                item_type = item.get("type", "")
                
                if item_type == "message":
                    # Check for memory citations
                    content = item.get("content", [])
                    msg_text = "".join(c.get("text", "") for c in content if c.get("type") in ("input_text", "output_text"))
                    _, citations = strip_memory_citations(msg_text)
                    if citations:
                        parsed_citations = parse_memory_citation(citations)
                        if parsed_citations:
                            self.state.record_memory_citation(parsed_citations)
                            
                    yield self.state.emit("agent_message", item=item)
                    
                elif item_type in ("function_call", "custom_tool_call"):
                    has_tool_calls = True
                    tool_calls_to_run.append(item)
                    yield self.state.emit("item_started", item=item)
                    
            if not has_tool_calls:
                # No tool calls: turn completed!
                break
                
            # Run tool calls
            for call in tool_calls_to_run:
                call_id = call.get("call_id") or call.get("id") or f"call-{uuid.uuid4()}"
                tool_name = call.get("name")
                # Custom Lark patch arguments are passed under "patch" or directly as raw strings!
                args = call.get("arguments")
                
                # Check sandbox permissions & approval
                requires_approval = False
                policy = self.config.approval_policy.lower()
                if policy == "always" or policy == " granular":
                    requires_approval = True
                elif policy == "untrusted" or policy == "unless-trusted":
                    # Untrusted defaults:
                    requires_approval = True
                    
                if requires_approval:
                    # Emits a prompt approval request
                    yield self.state.emit("entered_review_mode", call_id=call_id, tool=tool_name, arguments=args)
                    
                    # Interactively prompt if live, else fetch configured answers
                    prompt_query = f"Approve tool execution '{tool_name}' with arguments {args}? (yes/no): "
                    res = self.tools.dispatch("request_user_input", {"prompt": prompt_query})
                    
                    if res.output.strip().lower() not in ("yes", "y", "approve", "accept"):
                        # User declined: abort tool execution!
                        yield self.state.emit("exited_review_mode", decision="decline")
                        output_item = {
                            "type": "function_call_output" if call.get("type") == "function_call" else "custom_tool_call_output",
                            "call_id": call_id,
                            "name": tool_name,
                            "output": {"type": "text", "text": "aborted"}
                        }
                        self.state.append_history(output_item)
                        yield self.state.emit("item_completed", item=output_item)
                        continue
                        
                    yield self.state.emit("exited_review_mode", decision="approve")
                    
                # Run the actual tool call!
                tool_res = self.tools.dispatch(tool_name, args, call_id=call_id)
                
                # Record patch turn diffs
                if tool_name == "apply_patch" and tool_res.ok:
                    diff_str = tool_res.metadata.get("diff", "")
                    self.state.record_apply_patch_turn_diff(diff_str)
                    
                # Format output item under snake_case
                # If custom_tool_call: custom_tool_call_output
                # Else: function_call_output
                if call.get("type") == "custom_tool_call":
                    output_item = {
                        "type": "custom_tool_call_output",
                        "call_id": call_id,
                        "name": tool_name,
                        "output": {
                            "body": {
                                "type": "content_items",
                                "content": [{"type": "input_text", "text": tool_res.output}]
                            }
                        }
                    }
                else:
                    output_item = {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": {
                            "body": {
                                "type": "content_items",
                                "content": [{"type": "input_text", "text": tool_res.output}]
                            }
                        }
                    }
                    
                self.state.append_history(output_item)
                yield self.state.emit("item_completed", item=output_item)
                
        # 7. Finalize turn
        last_msg = self.last_message() or ""
        self.state.write_last_message(last_msg)
        yield self.state.emit("turn_complete", turn_id=self.state.turn_id, thread_id=self.state.thread_id)
