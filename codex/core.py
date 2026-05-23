from __future__ import annotations
import json
import os
import sys
import uuid
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable
from copy import deepcopy
from codex.types import CodexConfig, CodexEvent, CodexResult, get_default_model_slug, find_model_info
from codex.state import CodexState, build_compaction_summary_text, reconstruct_history_from_rollout
from codex.prompts import (
    approx_token_count, 
    build_local_compaction_request, 
    prepare_prompt_history,
    PromptRequest
)
from codex.model import (
    ModelClient, 
    ModelStreamEvent, 
    collect_stream_response, 
    default_model_client
)
from codex.tools import ToolRuntime, ToolResult
from codex.memory import (
    MemoryStateStore, 
    MemoryThreadRecord, 
    MemoryStartupResult,
    run_memory_startup_once, 
    run_memory_phase2_once
)

_MAX_AGENT_WAIT_TIMEOUT_MS = 60000
_MIN_AGENT_WAIT_TIMEOUT_MS = 1000

# --- CodexSession class implementation ---------------------------------------
class CodexSession:
    def __init__(
        self,
        config: CodexConfig,
        model_client: ModelClient | None = None,
        state: CodexState | None = None,
    ):
        self.config = config
        from codex.model import ModelClient, default_model_client
        if model_client is None or type(model_client) is ModelClient:
            self.model_client = default_model_client()
        else:
            self.model_client = model_client
        self.state = state if state is not None else CodexState(config=self.config)
        
        # Inject standard session bindings into state and tools
        self.tools = ToolRuntime(
            self.config, 
            self.state, 
            model_client=self.model_client, 
            session=self
        )
        self.memory_startup_result: MemoryStartupResult | None = None
        self._pending_steer_inputs: list[str] = []
        self._active_regular_turn = False
        self._session_started = False

    @classmethod
    def resume_from_rollout(
        cls,
        rollout_path: str | Path,
        config: CodexConfig | None = None,
        model_client: ModelClient | None = None
    ) -> CodexSession:
        recon = reconstruct_history_from_rollout(rollout_path)
        cfg = config if config is not None else CodexConfig()
        
        meta = recon.session_meta or {}
        inner = meta.get("meta") if isinstance(meta, dict) else None
        if isinstance(inner, dict):
            r_thread_id = inner.get("id") or inner.get("session_id") or str(uuid.uuid4())
        else:
            r_thread_id = meta.get("id") or meta.get("session_id") or str(uuid.uuid4())
            
        state = CodexState(
            config=cfg,
            thread_id=r_thread_id,
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
        
        meta = recon.session_meta or {}
        inner = meta.get("meta") if isinstance(meta, dict) else None
        if isinstance(inner, dict):
            r_forked_from_id = inner.get("id") or inner.get("session_id")
        else:
            r_forked_from_id = meta.get("id") or meta.get("session_id")
            
        # Create a brand new unique thread and separate rollout path for the fork
        state = CodexState(
            config=cfg,
            thread_id=str(uuid.uuid4()),
            forked_from_id=r_forked_from_id,
            history=recon.history,
            previous_turn_settings=recon.previous_turn_settings,
            reference_context_item=recon.reference_context_item
        )
        return cls(config=cfg, state=state, model_client=model_client)

    def interrupt(self) -> None:
        self._interrupted = True
        self.tools.interrupt_all()

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

    def steer_input(self, text: str, expected_turn_id: str | None = None) -> str:
        if not getattr(self, "_active_regular_turn", False) or self.state.turn_id is None:
            raise RuntimeError("no active turn")
        if expected_turn_id is not None:
            if expected_turn_id != self.state.turn_id:
                raise AssertionError("steer returned the wrong active turn id")
        self._pending_steer_inputs.append(text)
        return self.state.turn_id

    def compact(
        self,
        prompt: str | None = None,
        *,
        model: str | None = None,
        trigger: str = "manual",
        phase: str = "pre_sampling",
        reason: str | None = None,
        initial_context_injected: bool = False,
        parent_turn_id: str | None = None,
    ) -> CodexResult:
        events = list(self.stream_compact(
            prompt,
            model=model,
            trigger=trigger,
            phase=phase,
            reason=reason,
            initial_context_injected=initial_context_injected,
            parent_turn_id=parent_turn_id
        ))
        remote_comp = any(getattr(e, "payload", {}).get("remote_compaction") for e in events if getattr(e, "type") == "context_compaction.completed")
        final_msg = getattr(self.state, "_last_local_compaction_summary", "") if not remote_comp else ""
        return CodexResult(
            events=events,
            final_message=final_msg,
            thread_id=self.state.thread_id,
            turn_id=self.state.turn_id,
            history=self.state.history,
            memory_citations=self.state.memory_citations,
        )

    def stream_compact(
        self,
        prompt: str | None = None,
        *,
        model: str | None = None,
        trigger: str = "manual",
        phase: str = "pre_sampling",
        reason: str | None = None,
        initial_context_injected: bool = False,
        parent_turn_id: str | None = None,
    ) -> Iterator[CodexEvent]:
        p_turn_id = parent_turn_id if parent_turn_id is not None else self.state.turn_id
        old_turn_id = self.state.turn_id
        
        self.state.turn_id = str(uuid.uuid4())
        yield self.state.emit("turn.started", turn_id=self.state.turn_id, model_context_window=150000)
        
        # 1. Pre-compaction hook
        additional_contexts = None
        if self.config.hook_provider is not None:
            hook_req = {
                "event": "pre_compact",
                "parent_turn_id": p_turn_id,
                "trigger": trigger,
                "phase": phase,
                "model": model if model is not None else (self.config.model or get_default_model_slug()),
            }
            if reason is not None:
                hook_req["reason"] = reason
            yield self.state.emit("hook.started", name="pre_compact")
            try:
                resp = self.config.hook_provider(hook_req)
                yield self.state.emit("hook.completed", name="pre_compact", success=True)
                if resp:
                    additional_contexts = resp.get("additional_contexts")
                    if resp.get("should_stop"):
                        stop_reason = resp.get("stop_reason") or "pre_compact hook abort"
                        yield self.state.emit("turn.aborted", turn_id=self.state.turn_id, thread_id=self.state.thread_id, reason=stop_reason)
                        self.state.turn_id = old_turn_id
                        return
            except Exception as he:
                yield self.state.emit("hook.completed", name="pre_compact", success=False, error=str(he))
                
        try:
            initial_context = self.state.build_initial_context() if initial_context_injected else []
            
            use_remote = False
            if hasattr(self.model_client, "compact") and self.config.remote_compaction != "off":
                use_remote = True
                
            remote_succeeded = False
            compacted_message = ""
            implementation = "responses"
            is_remote = False
            summary_text = ""
            
            if use_remote:
                try:
                    # Build request containing prompt history
                    prompt_history = prepare_prompt_history(self.state.history, self.config)
                    if additional_contexts:
                        hook_block = "\n<hook_context>\n" + "\n".join(additional_contexts) + "\n</hook_context>"
                        for item in prompt_history:
                            if item.get("role") == "user":
                                item["content"][-1]["text"] += hook_block
                                
                    compaction_model = model if model is not None else (self.config.model or get_default_model_slug())
                    request = PromptRequest(
                        model=compaction_model,
                        instructions=self.state.get_base_instructions(),
                        input=prompt_history,
                        tools=self.tools.specs(),
                        parallel_tool_calls=self.config.resolved_parallel_tool_calls(),
                        verbosity=self.config.resolved_verbosity(),
                    )
                    
                    compacted_history = self.model_client.compact(
                        request,
                        session_id=self.state.thread_id,
                        thread_id=self.state.thread_id,
                        installation_id=self.state.installation_id,
                    )
                    
                    # Apply remote history compaction
                    self.state.compact_with_remote_history(compacted_history, initial_context)
                    
                    remote_succeeded = True
                    compacted_message = ""
                    implementation = "responses_compact"
                    is_remote = True
                except Exception as e:
                    # Emit stream error upon remote compaction failure, then fallback to local
                    yield self.state.emit("stream_error", error=str(e))
                    
            if not remote_succeeded:
                # Fall back to local compaction summary run
                try:
                    compact_request = build_local_compaction_request(self.state.history, self.config, model=model, additional_contexts=additional_contexts)
                    yield self.state.emit("model.request", model=compact_request.model, tool_names=[])
                    response = self.model_client.create(compact_request)
                    
                    for item in response.output:
                        if item.get("type") == "message" and item.get("role") == "assistant":
                            for part in item.get("content", []):
                                if isinstance(part, dict) and "text" in part:
                                    summary_text += part["text"]
                                    
                    self.state.compact_with_summary(summary_text, initial_context)
                    
                    compacted_message = build_compaction_summary_text(summary_text)
                    implementation = "responses"
                    is_remote = False
                except Exception as e:
                    raise e
                    
            # 2. Post-compaction hook
            if self.config.hook_provider is not None:
                hook_req = {
                    "event": "post_compact",
                    "parent_turn_id": p_turn_id,
                    "trigger": trigger,
                    "phase": phase,
                    "implementation": implementation,
                    "remote_compaction": is_remote,
                    "compacted_message": compacted_message,
                    "model": model if model is not None else (self.config.model or get_default_model_slug()),
                }
                if reason is not None:
                    hook_req["reason"] = reason
                yield self.state.emit("hook.started", name="post_compact")
                try:
                    resp = self.config.hook_provider(hook_req)
                    yield self.state.emit("hook.completed", name="post_compact", success=True)
                    if resp and resp.get("should_stop"):
                        stop_reason = resp.get("stop_reason") or "post_compact hook abort"
                        payload = {
                            "turn_id": p_turn_id,
                            "implementation": implementation,
                            "remote_compaction": is_remote,
                            "compacted_message": compacted_message,
                            "trigger": trigger,
                            "phase": phase,
                            "initial_context_injected": initial_context_injected,
                        }
                        if reason is not None:
                            payload["reason"] = reason
                        yield self.state.emit("context_compaction.completed", **payload)
                        yield self.state.emit("turn.aborted", turn_id=self.state.turn_id, thread_id=self.state.thread_id, reason=stop_reason)
                        return
                except Exception as he:
                    yield self.state.emit("hook.completed", name="post_compact", success=False, error=str(he))
                    
            # Re-inject canonical turn context record to rollout
            m_info = find_model_info(self.config.model or get_default_model_slug())
            effort_val = self.config.model_reasoning_effort
            if effort_val is None and m_info:
                effort_val = m_info.get("default_reasoning_level")
            summary_val = self.config.model_reasoning_summary
            if summary_val is None:
                summary_val = m_info.get("default_reasoning_summary") if m_info else "none"
                
            sandbox_policy_dict = {
                "type": self.config.sandbox,
                "writable_roots": [str(p.absolute()) for p in self.config.writable_roots] if getattr(self.config, "writable_roots", None) is not None else [],
                "network_access": getattr(self.config, "network_access", "restricted") == "full",
            }
            
            current_turn_context = {
                "turn_id": self.state.turn_id,
                "cwd": str(self.config.resolved_cwd()),
                "current_date": self.config.current_date,
                "sandbox_policy": sandbox_policy_dict,
                "permission_profile": {
                    "type": "managed",
                    "network": getattr(self.config, "network_access", "restricted") if getattr(self.config, "network_access", "restricted") != "full" else "full",
                },
                "file_system_sandbox_policy": {
                    "kind": "restricted" if self.config.sandbox in ("workspace-write", "read-only") else "full",
                },
                "truncation_policy": {
                    "mode": "tokens",
                    "limit": self.config.resolved_auto_compact_token_limit() or 100_000,
                },
                "approval_policy": self.config.approval_policy,
                "model": self.config.model or get_default_model_slug(),
                "effort": effort_val,
                "summary": summary_val,
                "realtime_active": False,
            }
            self.state.write_rollout_record("turn_context", current_turn_context)
            
            # Emit compaction completed event
            payload = {
                "turn_id": p_turn_id,
                "implementation": implementation,
                "remote_compaction": is_remote,
                "compacted_message": compacted_message,
                "trigger": trigger,
                "phase": phase,
                "initial_context_injected": initial_context_injected,
            }
            if reason is not None:
                payload["reason"] = reason
            yield self.state.emit("context_compaction.completed", **payload)
            yield self.state.emit("turn.completed", usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_output_tokens": 0})
        finally:
            self.state.turn_id = old_turn_id

    def stream(self, prompt: str = "", *args: Any, **kwargs: Any) -> Iterable[CodexEvent]:
        self._active_regular_turn = True
        self._auto_compacted_this_turn = False
        self._interrupted = False
        # 1. Initialize memory state store on config if enabled
        if self.config.use_memories:
            if getattr(self.config, "memory_state_store", None) is None:
                self.config.memory_state_store = MemoryStateStore(self.config.resolved_codex_home() / "memory-state.sqlite3")
                
        # 1. Sync or async memory startup pipeline
        if self.config.use_memories and self.memory_startup_result is None:
            if not self.config.memory_startup_background:
                store = self.config.memory_state_store
                try:
                    startup = run_memory_startup_once(
                        self.config,
                        state_store=store,
                        current_thread_id=self.state.thread_id,
                        model_client=self.model_client,
                        max_rollouts=1,
                        max_unused_days=36500,
                        max_rollout_age_days=36500,
                        min_rollout_idle_hours=self.config.memory_min_rollout_idle_hours,
                    )
                    self.memory_startup_result = startup
                    if self.config.memory_run_phase2_on_startup:
                        run_memory_phase2_once(
                            self.config,
                            state_store=store,
                            model_client=self.model_client,
                        )
                except Exception:
                    pass
            else:
                def worker():
                    bg_store = MemoryStateStore(self.config.resolved_codex_home() / "memory-state.sqlite3")
                    try:
                        startup = run_memory_startup_once(
                            self.config,
                            state_store=bg_store,
                            current_thread_id=self.state.thread_id,
                            model_client=self.model_client,
                            max_rollouts=1,
                            max_unused_days=36500,
                            max_rollout_age_days=36500,
                            min_rollout_idle_hours=self.config.memory_min_rollout_idle_hours,
                        )
                        self.memory_startup_result = startup
                        if self.config.memory_run_phase2_on_startup:
                            run_memory_phase2_once(
                                self.config,
                                state_store=bg_store,
                                model_client=self.model_client,
                            )
                    except Exception:
                        pass
                    finally:
                        bg_store.close()
                threading.Thread(target=worker, daemon=True).start()
                
        # 2. Record this thread in state DB for tracking
        if self.config.use_memories:
            store = self.config.memory_state_store
            try:
                store.upsert_thread(MemoryThreadRecord(
                    thread_id=self.state.thread_id,
                    rollout_path=self.state.rollout_path(),
                    cwd=self.config.resolved_cwd(),
                    updated_at=datetime.now(timezone.utc),
                    git_branch=self.config.resolved_git_branch(),
                    source=self.config.session_source,
                    memory_mode="enabled",
                ))
            except Exception:
                pass
                
        # 3. Trigger hooks before turn run
        if not getattr(self, "_session_started", False):
            self._session_started = True
            if self.config.hook_provider is not None:
                yield self.state.emit("hook.started", name="session_start")
                try:
                    self.config.hook_provider({
                        "event": "session_start",
                        "thread_id": self.state.thread_id,
                        "installation_id": self.state.installation_id,
                        "model": self.config.model or get_default_model_slug(),
                    })
                    yield self.state.emit("hook.completed", name="session_start", success=True)
                except Exception as e:
                    yield self.state.emit("hook.completed", name="session_start", success=False, error=str(e))
                    
        additional_contexts = None
        if self.config.hook_provider is not None and prompt:
            yield self.state.emit("hook.started", name="user_prompt_submit")
            try:
                resp = self.config.hook_provider({
                    "event": "user_prompt_submit",
                    "prompt": prompt,
                })
                yield self.state.emit("hook.completed", name="user_prompt_submit", success=True)
                additional_contexts = resp.get("additional_contexts")
            except Exception as e:
                yield self.state.emit("hook.completed", name="user_prompt_submit", success=False, error=str(e))
                
        # 4. Bootstrap initial context and prepend to history if empty
        if not self.state.history:
            from codex.prompts import build_initial_context_items
            initial_context = build_initial_context_items(self.config)
            if additional_contexts:
                hook_block = "\n<hook_context>\n" + "\n".join(additional_contexts) + "\n</hook_context>"
                for item in initial_context:
                    if item.get("role") == "user":
                        item["content"][-1]["text"] += hook_block
            for item in initial_context:
                self.state.append_history(item)
                
        # 5. Append user prompt message
        if prompt:
            self.state.append_history({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            })
            yield self.state.emit("user_message", message=prompt)
            
        # Emit "thread.started" if not already present
        has_started = any(ev.type == "thread.started" for ev in self.state.events)
        if not has_started:
            yield self.state.emit("thread.started",
                thread_id=self.state.thread_id,
                model=self.config.model or get_default_model_slug(),
            )
            
        # Emit "turn.started" exactly ONCE per turn run
        self.state.turn_id = str(uuid.uuid4())
        yield self.state.emit("turn.started",
            turn_id=self.state.turn_id,
            model_context_window=150000,
        )
            
        # 5. Iterations loop
        max_iters = self.config.max_iterations
        iterations = 0
        final_message = ""
        
        while True:
            if getattr(self, "_interrupted", False):
                raise RuntimeError("turn aborted")
                
            if max_iters is not None and iterations >= max_iters:
                yield self.state.emit("turn.failed", error="max_iterations exceeded")
                break
                
            iterations += 1
            
            # Write turn context to rollout
            # Resolve effort and summary for turn context recording (defaults to catalog-matched values)
            m_info = find_model_info(self.config.model or get_default_model_slug())
            effort_val = self.config.model_reasoning_effort
            if effort_val is None and m_info:
                effort_val = m_info.get("default_reasoning_level")
            summary_val = self.config.model_reasoning_summary
            if summary_val is None:
                summary_val = m_info.get("default_reasoning_summary") if m_info else "none"
                
            sandbox_policy_dict = {
                "type": self.config.sandbox,
                "writable_roots": [str(p.absolute()) for p in self.config.writable_roots] if getattr(self.config, "writable_roots", None) is not None else [],
                "network_access": getattr(self.config, "network_access", "restricted") == "full",
            }
            
            current_turn_context = {
                "turn_id": self.state.turn_id,
                "cwd": str(self.config.resolved_cwd()),
                "current_date": self.config.current_date,
                "sandbox_policy": sandbox_policy_dict,
                "permission_profile": {
                    "type": "managed",
                    "network": getattr(self.config, "network_access", "restricted") if getattr(self.config, "network_access", "restricted") != "full" else "full",
                },
                "file_system_sandbox_policy": {
                    "kind": "restricted" if self.config.sandbox in ("workspace-write", "read-only") else "full",
                },
                "truncation_policy": {
                    "mode": "tokens",
                    "limit": self.config.resolved_auto_compact_token_limit() or 100_000,
                },
                "approval_policy": self.config.approval_policy,
                "model": self.config.model or get_default_model_slug(),
                "effort": effort_val,
                "summary": summary_val,
                "realtime_active": False,
            }
            self.state.write_rollout_record("turn_context", current_turn_context)
            
            # Model downshift compaction check
            if self.state.previous_turn_settings and not getattr(self, "_auto_compacted_this_turn", False):
                prev_model = self.state.previous_turn_settings.get("model")
                curr_model = self.config.model or get_default_model_slug()
                if prev_model and prev_model != curr_model:
                    prev_info = find_model_info(prev_model)
                    curr_info = find_model_info(curr_model)
                    prev_window = prev_info.get("context_window") if prev_info else None
                    curr_window = curr_info.get("context_window") if curr_info else None
                    
                    if prev_window and curr_window and curr_window < prev_window:
                        active_tokens = self.state.active_context_tokens()
                        if active_tokens > 0.90 * curr_window:
                            try:
                                self._auto_compacted_this_turn = True
                                start_idx = len(self.state.events)
                                self.compact(
                                    model=prev_model,
                                    trigger="auto",
                                    phase="pre_sampling",
                                    reason="model_downshift",
                                    initial_context_injected=False,
                                    parent_turn_id=self.state.turn_id,
                                )
                                for i in range(start_idx, len(self.state.events)):
                                    yield self.state.events[i]
                            except Exception:
                                pass
                                
            # Auto-compaction check
            if self.config.model_auto_compact_token_limit is not None and not getattr(self, "_auto_compacted_this_turn", False):
                trigger_limit = self.config.model_auto_compact_token_limit
                should_compact = False
                if iterations == 1:
                    should_compact = (self.state.total_token_usage > trigger_limit)
                else:
                    should_compact = (self.state.active_context_tokens() > trigger_limit)
                    
                if should_compact:
                    try:
                        self._auto_compacted_this_turn = True
                        start_idx = len(self.state.events)
                        compaction_phase = "pre_sampling" if iterations == 1 else "mid_turn"
                        self.compact(trigger="auto", phase=compaction_phase, parent_turn_id=self.state.turn_id, initial_context_injected=True)
                        for i in range(start_idx, len(self.state.events)):
                            yield self.state.events[i]
                    except Exception:
                        pass
                        
            # Compile prompt
            prompt_history = prepare_prompt_history(self.state.history, self.config)
            instructions = self.state.get_base_instructions()
            
            request = PromptRequest(
                model=self.config.model or get_default_model_slug(),
                instructions=instructions,
                input=prompt_history,
                tools=self.tools.specs(),
                parallel_tool_calls=self.config.resolved_parallel_tool_calls(),
                reasoning=self.config.resolved_reasoning(),
                output_schema=self.config.output_schema,
                client_metadata={"x-codex-installation-id": self.state.installation_id},
                include=["reasoning.encrypted_content"] if self.config.resolved_reasoning() is not None else [],
                verbosity=self.config.resolved_verbosity(),
            )
            
            yield self.state.emit("model.request",
                model=request.model,
                tool_names=[t.get("name", t.get("type")) for t in request.tools]
            )
            
            retry_cnt = 0
            max_retries = self.config.model_stream_max_retries or 0
            response_events = []
            
            while True:
                try:
                    if hasattr(self.model_client, "stream"):
                        stream_source = self.model_client.stream(request)
                    else:
                        # Emulate stream from create()
                        response = self.model_client.create(request)
                        simulated = []
                        for item in response.output:
                            item_id = item.get("id") or item.get("call_id") or str(uuid.uuid4())
                            item["id"] = item_id
                            simulated.append(CodexEvent(type="item.started", payload={"item": item}))
                            simulated.append(CodexEvent(type="item.completed", payload={"item": item}))
                        simulated.append(CodexEvent(type="model.response", payload={
                            "response_id": response.id,
                            "response": {
                                "id": response.id,
                                "output": response.output
                            }
                        }))
                        simulated.append(CodexEvent(type="turn.completed", payload={
                            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "reasoning_output_tokens": 0}
                        }))
                        stream_source = simulated
                        
                    for ev in stream_source:
                        response_events.append(ev)
                        ev_type = getattr(ev, "type", "unknown")
                        payload = deepcopy(getattr(ev, "payload", {}))
                        if ev_type == "item.completed" and self._pending_steer_inputs:
                            payload["pending_input"] = True
                        yield self.state.emit(ev_type, **payload)
                    break
                except Exception as e:
                    if retry_cnt < max_retries:
                        retry_cnt += 1
                        yield self.state.emit("stream_error", error=str(e))
                        delay_ms = self.config.model_stream_retry_base_delay_ms or 0
                        time.sleep((delay_ms * (2 ** (retry_cnt - 1))) / 1000.0)
                    else:
                        yield self.state.emit("model.failed", error=str(e))
                        raise e
                        
            # Process stream results
            model_response = collect_stream_response(response_events)
            yield self.state.emit("model.response", response_id=model_response.id)
            
            # Parse token usage
            usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "reasoning_output_tokens": 0}
            for ev in response_events:
                if getattr(ev, "type") == "turn.completed" and "usage" in getattr(ev, "payload", {}):
                    usage = getattr(ev, "payload")["usage"]
                    
            self.state.record_token_usage(usage)
            yield self.state.emit("token_count", usage=usage)
            
            # Commit draft outputs
            has_tool_calls = False
            tool_calls_to_run = []
            
            for item in model_response.output:
                item_type = item.get("type")
                if item_type in ("function_call", "custom_tool_call"):
                    has_tool_calls = True
                    tool_calls_to_run.append(item)
                    self.state.append_history(item)
                    
                    # Mark memories polluted if memory_disable_on_external_context is enabled and external tool runs
                    name = item.get("name")
                    if name == "hosted_web_search" and self.config.use_memories and self.config.memory_disable_on_external_context:
                        store = getattr(self.config, "memory_state_store", None)
                        if store is not None:
                            try:
                                store.upsert_thread(MemoryThreadRecord(
                                    thread_id=self.state.thread_id,
                                    rollout_path=self.state.rollout_path(),
                                    cwd=self.config.resolved_cwd(),
                                    updated_at=datetime.now(timezone.utc),
                                    git_branch=self.config.resolved_git_branch(),
                                    source=self.config.session_source,
                                    memory_mode="polluted",
                                ))
                            except Exception:
                                pass
                                
                elif item_type == "message" and item.get("role") == "assistant":
                    self.state.append_history(item)
                    content = item.get("content", [])
                    final_message = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                else:
                    self.state.append_history(item)
                    # Mark memories polluted if memory_disable_on_external_context is enabled and external web search runs
                    if item_type == "web_search_call" and self.config.use_memories and self.config.memory_disable_on_external_context:
                        store = getattr(self.config, "memory_state_store", None)
                        if store is not None:
                            try:
                                store.upsert_thread(MemoryThreadRecord(
                                    thread_id=self.state.thread_id,
                                    rollout_path=self.state.rollout_path(),
                                    cwd=self.config.resolved_cwd(),
                                    updated_at=datetime.now(timezone.utc),
                                    git_branch=self.config.resolved_git_branch(),
                                    source=self.config.session_source,
                                    memory_mode="polluted",
                                ))
                            except Exception:
                                pass
                    
            if has_tool_calls:
                parallel_enabled = self.config.resolved_parallel_tool_calls()
                parallel_group = []
                if parallel_enabled:
                    for t_call in tool_calls_to_run:
                        name = t_call.get("name")
                        is_parallel = (hasattr(self.tools, "supports_parallel") and self.tools.supports_parallel(name))
                        if is_parallel:
                            parallel_group.append(t_call)
                            
                if len(parallel_group) > 1:
                    # Execute parallel group concurrently, and drain in order!
                    states = []
                    for t_call in parallel_group:
                        call_id = t_call.get("call_id") or t_call.get("id") or str(uuid.uuid4())
                        name = t_call.get("name")
                        raw_args = t_call.get("arguments") or t_call.get("input") or "{}"
                        if isinstance(raw_args, str):
                            try:
                                arguments = json.loads(raw_args)
                            except Exception:
                                arguments = raw_args
                        else:
                            arguments = raw_args
                            
                        # Hook invocation logic (pre_tool_use)
                        should_run_hook = (self.config.hook_provider is not None) and (name in ("exec_command", "shell_command", "apply_patch"))
                        blocked = False
                        block_reason = ""
                        
                        tool_input = {}
                        tool_name = "Bash" if name in ("exec_command", "shell_command") else name
                        
                        if tool_name == "Bash":
                            tool_input = {"command": arguments.get("cmd") if isinstance(arguments, dict) else arguments}
                        else:  # apply_patch
                            tool_input = {"command": arguments.get("patch") if isinstance(arguments, dict) else arguments}
                            
                        updated_tool_input = deepcopy(tool_input)
                        
                        if should_run_hook:
                            yield self.state.emit("hook.started", name="pre_tool_use")
                            try:
                                req = {
                                    "event": "pre_tool_use",
                                    "tool_name": tool_name,
                                    "tool_input": tool_input,
                                }
                                if tool_name == "apply_patch":
                                    req["matcher_aliases"] = ["Write", "Edit"]
                                    
                                resp = self.config.hook_provider(req)
                                yield self.state.emit("hook.completed", name="pre_tool_use", success=True)
                                
                                if resp.get("should_block"):
                                    blocked = True
                                    block_reason = resp.get("block_reason") or "blocked by test hook"
                                elif resp.get("updated_input"):
                                    updated_val = resp.get("updated_input").get("command")
                                    updated_tool_input = {"command": updated_val}
                                    if tool_name == "Bash":
                                        if isinstance(arguments, dict):
                                            arguments["cmd"] = updated_val
                                        else:
                                            arguments = updated_val
                                    else: # apply_patch
                                        if isinstance(arguments, dict):
                                            arguments["patch"] = updated_val
                                        else:
                                            arguments = updated_val
                                    t_call["arguments"] = json.dumps(arguments) if isinstance(arguments, dict) else arguments
                                    if "input" in t_call:
                                        t_call["input"] = arguments
                            except Exception as e:
                                yield self.state.emit("hook.completed", name="pre_tool_use", success=False, error=str(e))
                                
                        yield self.state.emit("tool.started", call_id=call_id, name=name, arguments=arguments)
                        
                        states.append({
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                            "t_call": t_call,
                            "blocked": blocked,
                            "block_reason": block_reason,
                            "tool_name": tool_name,
                            "updated_tool_input": updated_tool_input,
                            "should_run_hook": should_run_hook,
                            "tool_res": None,
                            "thread": None,
                        })
                        
                    # Launch background threads concurrently for unblocked calls!
                    for state in states:
                        if not state["blocked"]:
                            def run_tool(st):
                                try:
                                    st["tool_res"] = self.tools.dispatch(st["name"], st["arguments"], call_id=st["call_id"])
                                except Exception as ex:
                                    st["tool_res"] = ToolResult(ok=False, output=f"Internal error: {ex}", metadata={})
                            t = threading.Thread(target=run_tool, args=(state,), daemon=True)
                            state["thread"] = t
                            t.start()
                            
                    # Join and drain sequentially in order!
                    for state in states:
                        call_id = state["call_id"]
                        name = state["name"]
                        t_call = state["t_call"]
                        blocked = state["blocked"]
                        block_reason = state["block_reason"]
                        tool_name = state["tool_name"]
                        updated_tool_input = state["updated_tool_input"]
                        should_run_hook = state["should_run_hook"]
                        
                        if state["thread"] is not None:
                            state["thread"].join()
                            
                        if blocked:
                            tool_res = ToolResult(ok=False, output=block_reason, metadata={})
                        else:
                            tool_res = state["tool_res"] or ToolResult(ok=False, output="No result returned", metadata={})
                            
                        yield self.state.emit("tool.completed", call_id=call_id, ok=tool_res.ok, output=tool_res.output, metadata=tool_res.metadata)
                        
                        if should_run_hook and not blocked:
                            yield self.state.emit("hook.started", name="post_tool_use")
                            try:
                                post_req = {
                                    "event": "post_tool_use",
                                    "tool_name": tool_name,
                                    "tool_input": updated_tool_input,
                                    "tool_response": tool_res.output,
                                }
                                if tool_name == "apply_patch":
                                    post_req["matcher_aliases"] = ["Write", "Edit"]
                                self.config.hook_provider(post_req)
                                yield self.state.emit("hook.completed", name="post_tool_use", success=True)
                            except Exception as e:
                                yield self.state.emit("hook.completed", name="post_tool_use", success=False, error=str(e))
                                
                        if t_call.get("type") == "custom_tool_call":
                            output_item = {
                                "type": "custom_tool_call_output",
                                "call_id": call_id,
                                "output": tool_res.output
                            }
                        else:
                            output_item = {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": tool_res.output
                            }
                        self.state.append_history(output_item)
                    continue
                else:
                    # Run sequentially
                    for t_call in tool_calls_to_run:
                        for event in self._execute_tool_call_sequentially(t_call):
                            yield event
                    continue
                
            if self._pending_steer_inputs:
                steers = list(self._pending_steer_inputs)
                self._pending_steer_inputs.clear()
                for steer_text in steers:
                    self.state.append_history({
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": steer_text}]
                    })
                continue
                
            if final_message:
                yield self.state.emit("agent_message", message=final_message)
            yield self.state.emit("turn.completed", usage=self.state.last_token_usage)
            break
            
        self.state.write_last_message(final_message)
        self._active_regular_turn = False
        self._final_message_stored = final_message

    def _execute_tool_call_sequentially(self, t_call: dict[str, Any]) -> Iterable[CodexEvent]:
        call_id = t_call.get("call_id") or t_call.get("id") or str(uuid.uuid4())
        name = t_call.get("name")
        raw_args = t_call.get("arguments") or t_call.get("input") or "{}"
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except Exception:
                arguments = raw_args
        else:
            arguments = raw_args
            
        # Hook invocation logic (pre_tool_use & post_tool_use)
        should_run_hook = (self.config.hook_provider is not None) and (name in ("exec_command", "shell_command", "apply_patch"))
        blocked = False
        block_reason = ""
        
        tool_input = {}
        tool_name = "Bash" if name in ("exec_command", "shell_command") else name
        
        if tool_name == "Bash":
            tool_input = {"command": arguments.get("cmd") if isinstance(arguments, dict) else arguments}
        else:  # apply_patch
            tool_input = {"command": arguments.get("patch") if isinstance(arguments, dict) else arguments}
            
        updated_tool_input = deepcopy(tool_input)
        
        if should_run_hook:
            yield self.state.emit("hook.started", name="pre_tool_use")
            try:
                req = {
                    "event": "pre_tool_use",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                }
                if tool_name == "apply_patch":
                    req["matcher_aliases"] = ["Write", "Edit"]
                    
                resp = self.config.hook_provider(req)
                yield self.state.emit("hook.completed", name="pre_tool_use", success=True)
                
                if resp.get("should_block"):
                    blocked = True
                    block_reason = resp.get("block_reason") or "blocked by test hook"
                elif resp.get("updated_input"):
                    updated_val = resp.get("updated_input").get("command")
                    updated_tool_input = {"command": updated_val}
                    if tool_name == "Bash":
                        if isinstance(arguments, dict):
                            arguments["cmd"] = updated_val
                        else:
                            arguments = updated_val
                    else: # apply_patch
                        if isinstance(arguments, dict):
                            arguments["patch"] = updated_val
                        else:
                            arguments = updated_val
                    t_call["arguments"] = json.dumps(arguments) if isinstance(arguments, dict) else arguments
                    if "input" in t_call:
                        t_call["input"] = arguments
            except Exception as e:
                yield self.state.emit("hook.completed", name="pre_tool_use", success=False, error=str(e))
                
        yield self.state.emit("tool.started", call_id=call_id, name=name, arguments=arguments)
        
        if blocked:
            tool_res = ToolResult(ok=False, output=block_reason, metadata={})
        else:
            tool_res = self.tools.dispatch(name, arguments, call_id=call_id)
            
        yield self.state.emit("tool.completed", call_id=call_id, ok=tool_res.ok, output=tool_res.output, metadata=tool_res.metadata)
        
        if should_run_hook and not blocked:
            yield self.state.emit("hook.started", name="post_tool_use")
            try:
                post_req = {
                    "event": "post_tool_use",
                    "tool_name": tool_name,
                    "tool_input": updated_tool_input,
                    "tool_response": tool_res.output,
                }
                if tool_name == "apply_patch":
                    post_req["matcher_aliases"] = ["Write", "Edit"]
                self.config.hook_provider(post_req)
                yield self.state.emit("hook.completed", name="post_tool_use", success=True)
            except Exception as e:
                yield self.state.emit("hook.completed", name="post_tool_use", success=False, error=str(e))
        
        if t_call.get("type") == "custom_tool_call":
            output_item = {
                "type": "custom_tool_call_output",
                "call_id": call_id,
                "output": tool_res.output
            }
        else:
            output_item = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_res.output
            }
        self.state.append_history(output_item)

    def run(self, prompt: str = "") -> CodexResult:
        list(self.stream(prompt))
        final_msg = getattr(self, "_final_message_stored", "")
        return CodexResult(
            events=self.state.events,
            final_message=final_msg,
            thread_id=self.state.thread_id,
            turn_id=self.state.turn_id,
            history=self.state.history,
            memory_citations=self.state.memory_citations,
        )
