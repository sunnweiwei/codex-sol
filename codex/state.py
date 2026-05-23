from __future__ import annotations
import json
import os
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass, field
from codex.types import CodexConfig, CodexEvent, CodexResult, find_model_info, get_default_model_slug
from codex.prompts import build_base_instructions, build_initial_context_items, approx_token_count, truncate_text, prepare_prompt_history, ASSETS_DIR

# --- Constants ---------------------------------------------------------------
CODEX_ROLLOUT_ITEM_TYPES = frozenset({
    "session_meta",
    "response_item",
    "turn_context",
    "compacted",
    "event_msg",
})

# --- Rollout Reconstruction types --------------------------------------------
@dataclass
class RolloutReconstruction:
    history: list[dict[str, Any]]
    previous_turn_settings: dict[str, Any] | None
    reference_context_item: dict[str, Any] | None
    session_meta: dict[str, Any] | None = None
    legacy_compaction_without_replacement_history: bool = False

# --- Shell command parsing actions -------------------------------------------
def parse_command_actions(command: str) -> list[dict[str, Any]]:
    import shlex
    try:
        tokens = shlex.split(command)
    except Exception:
        tokens = command.split()
        
    if not tokens:
        return [{"type": "unknown", "command": command}]
        
    # Check for "rg --files"
    if "rg" in command and "--files" in command:
        path = None
        for t in tokens:
            if t not in ("rg", "--files", "|", "head", "-n", "50") and not t.isdigit() and not t.startswith("-"):
                path = t
        return [{
            "type": "list_files",
            "command": command,
            "path": path,
        }]
        
    # Check for other search commands (rg, grep, git grep)
    first = tokens[0]
    if first in ("rg", "grep") or (len(tokens) >= 2 and tokens[0] == "git" and tokens[1] == "grep"):
        query = None
        path = None
        cmd_len = 2 if first == "git" else 1
        args = tokens[cmd_len:]
        non_opts = [a for a in args if not a.startswith("-")]
        if len(non_opts) >= 1:
            query = non_opts[0]
        if len(non_opts) >= 2:
            path = non_opts[1]
            
        return [{
            "type": "search",
            "command": command,
            "query": query,
            "path": path,
        }]
        
    # Check for read commands (cat, less, head, tail)
    if first in ("cat", "less", "head", "tail"):
        path = None
        args = tokens[1:]
        non_opts = [a for a in args if not a.startswith("-")]
        if non_opts:
            path = str(Path(non_opts[0]).absolute())
        return [{
            "type": "read",
            "command": command,
            "name": first,
            "path": path,
        }]
        
    return [{
        "type": "unknown",
        "command": command,
    }]

# --- Memory Citation Parsers -------------------------------------------------
def strip_memory_citations(text: str) -> tuple[str, list[str]]:
    match = re.search(r'<oai-mem-citation>(.*?)</oai-mem-citation>', text, re.DOTALL)
    if not match:
        return text, []
        
    visible_text = text[:match.start()] + text[match.end():]
    citation_block = match.group(1)
    citations = []
    
    entries_match = re.search(r'<citation_entries>(.*?)</citation_entries>', citation_block, re.DOTALL)
    if entries_match:
        for line in entries_match.group(1).splitlines():
            trimmed = line.strip()
            if trimmed:
                citations.append(trimmed)
                
    ids_match = re.search(r'<rollout_ids>(.*?)</rollout_ids>', citation_block, re.DOTALL)
    if ids_match:
        for line in ids_match.group(1).splitlines():
            trimmed = line.strip()
            if trimmed:
                citations.append(trimmed)
                
    return visible_text, citations

def parse_memory_citation(citations: list[str]) -> dict[str, Any] | None:
    entries = []
    rollout_ids = []
    
    entry_pat = re.compile(r'^([^:]+):(\d+)-(\d+)\|note=\[(.*)\]$')
    uuid_pat = re.compile(r'^[a-fA-F0-9-]{36}$')
    
    for line in citations:
        line = line.strip()
        if not line:
            continue
        match = entry_pat.match(line)
        if match:
            file, start, end, note = match.groups()
            entries.append({
                "path": file,
                "line_start": int(start),
                "line_end": int(end),
                "note": note,
            })
        elif (uuid_pat.match(line) or '-' in line) and len(line) == 36:
            if line not in rollout_ids:
                rollout_ids.append(line)
            
    return {
        "entries": entries,
        "rollout_ids": rollout_ids,
    }

# --- Compaction Helpers ------------------------------------------------------
def summarization_prompt() -> str:
    assets_dir = Path(__file__).parent / "assets"
    prompt_file = assets_dir / "prompts" / "compact" / "prompt.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    return "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary..."

def build_compaction_summary_text(summary_suffix: str) -> str:
    assets_dir = Path(__file__).parent / "assets"
    prefix_file = assets_dir / "prompts" / "compact" / "summary_prefix.md"
    prefix = ""
    if prefix_file.exists():
        prefix = prefix_file.read_text(encoding="utf-8")
    else:
        prefix = "Another language model started to solve this problem..."
        
    return f"{prefix}\n{summary_suffix}"

def collect_user_messages(history: list[dict[str, Any]]) -> list[str]:
    messages = []
    for item in history:
        if item.get("type") == "message" and item.get("role") == "user":
            if is_real_user_message(item):
                content = item.get("content", [])
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("output_text", "input_text"):
                        parts.append(part.get("text", ""))
                txt = "".join(parts)
                if txt.startswith("Another language model started"):
                    continue
                if txt:
                    messages.append(txt)
    return messages

def build_compacted_history(
    initial_context: list[dict[str, Any]],
    user_messages: list[str],
    summary_text: str,
    max_tokens: int = 20000,
) -> list[dict[str, Any]]:
    history = list(initial_context)
    
    selected_messages = []
    if max_tokens > 0:
        remaining = max_tokens
        for message in reversed(user_messages):
            if remaining == 0:
                break
            tokens = approx_token_count(message)
            if tokens <= remaining:
                selected_messages.append(message)
                remaining = remaining - tokens
            else:
                truncated = truncate_text(message, remaining)
                selected_messages.append(truncated)
                break
        selected_messages.reverse()
        
    for message in selected_messages:
        history.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": message}],
        })
        
    summary_text = summary_text if summary_text else "(no summary available)"
    history.append({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": summary_text}],
    })
    
    return history

def drop_last_n_user_turns(history: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    history = list(history)
    for _ in range(n):
        user_idx = -1
        for idx in range(len(history) - 1, -1, -1):
            item = history[idx]
            if item.get("type") == "message" and item.get("role") == "user":
                user_idx = idx
                break
        if user_idx != -1:
            history = history[:user_idx]
        else:
            history = []
            break
    return history

def insert_initial_context(compacted_history: list[dict[str, Any]], initial_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted_history = list(compacted_history)
    if not initial_context:
        return compacted_history
        
    last_user_idx = -1
    for idx in range(len(compacted_history) - 1, -1, -1):
        item = compacted_history[idx]
        if item.get("type") == "message" and item.get("role") == "user":
            text = "".join(part.get("text", "") for part in item.get("content", []) if isinstance(part, dict))
            if not text.startswith("Another language model started"):
                last_user_idx = idx
                break
                
    if last_user_idx == -1:
        for idx in range(len(compacted_history) - 1, -1, -1):
            item = compacted_history[idx]
            if item.get("type") == "message":
                last_user_idx = idx
                break
                
    if last_user_idx != -1:
        compacted_history[last_user_idx:last_user_idx] = initial_context
    else:
        compacted_history.extend(initial_context)
        
    return compacted_history

def is_real_user_message(item: dict[str, Any]) -> bool:
    if item.get("type") != "message" or item.get("role") != "user":
        return False
    content = item.get("content", [])
    text = "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
    if text.startswith("<environment_context>") or text.startswith("<permissions instructions>") or text.startswith("Filesystem sandboxing defines which files") or text.startswith("# AGENTS.md instructions for"):
        return False
    return True

def should_keep_compacted_item(item: dict[str, Any]) -> bool:
    item_type = item.get("type")
    role = item.get("role")
    
    if item_type == "message":
        if role == "developer":
            return False
        if role == "user":
            return is_real_user_message(item)
        if role == "assistant":
            return True
        return False
        
    if item_type in ("compaction", "compacted"):
        return True
        
    return False


# --- Rollout Reconstruction implementation -----------------------------------
def reconstruct_history_from_rollout(source: Path | str | list[dict[str, Any]]) -> RolloutReconstruction:
    if isinstance(source, list):
        records = source
        session_meta = next((r.get("payload") for r in records if r.get("type") == "session_meta"), None)
    else:
        path = Path(source)
        if not path.exists():
            return RolloutReconstruction([], None, None)
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        session_meta = next((r.get("payload") for r in records if r.get("type") == "session_meta"), None)
        
    if isinstance(session_meta, dict):
        inner = session_meta.get("meta")
        if isinstance(inner, dict):
            session_meta = {**session_meta, **inner}
            
    if not records:
        return RolloutReconstruction([], None, None)
        
    base_replacement_history = None
    previous_turn_settings = None
    reference_context_item = None
    saw_legacy_compaction_without_replacement_history = False
    
    pending_rollback_turns = 0
    active_segment_has_user_message = False
    suffix_start_index = 0
    
    for idx in range(len(records) - 1, -1, -1):
        record = records[idx]
        rec_type = record.get("type")
        payload = record.get("payload", {})
        
        if rec_type == "compacted":
            if "replacement_history" in payload and payload["replacement_history"] is not None:
                if base_replacement_history is None:
                    base_replacement_history = payload["replacement_history"]
                    suffix_start_index = idx + 1
            else:
                saw_legacy_compaction_without_replacement_history = True
        elif rec_type == "event_msg":
            msg_type = payload.get("type")
            if msg_type == "thread_rolled_back":
                pending_rollback_turns += int(payload.get("num_turns", 0))
            elif msg_type == "user_message":
                active_segment_has_user_message = True
        elif rec_type == "turn_context":
            if pending_rollback_turns > 0 and active_segment_has_user_message:
                pending_rollback_turns = max(0, pending_rollback_turns - 1)
                active_segment_has_user_message = False
            else:
                if previous_turn_settings is None:
                    previous_turn_settings = {
                        "model": payload.get("model"),
                        "realtime_active": payload.get("realtime_active", False),
                    }
                if reference_context_item is None:
                    reference_context_item = payload
        elif rec_type == "response_item":
            role = payload.get("role")
            if role == "user":
                active_segment_has_user_message = True
                
        if (base_replacement_history is not None and 
            previous_turn_settings is not None and 
            reference_context_item is not None):
            break
            
    history = []
    if base_replacement_history is not None:
        history = list(base_replacement_history)
        
    rollout_suffix = records[suffix_start_index:]
    for record in rollout_suffix:
        rec_type = record.get("type")
        payload = record.get("payload", {})
        
        if rec_type == "response_item":
            history.append(payload)
        elif rec_type == "compacted":
            if "replacement_history" in payload and payload["replacement_history"] is not None:
                history = list(payload["replacement_history"])
            else:
                saw_legacy_compaction_without_replacement_history = True
                user_messages = collect_user_messages(history)
                rebuilt = build_compacted_history([], user_messages, payload.get("message", ""))
                history = rebuilt
        elif rec_type == "event_msg" and payload.get("type") == "thread_rolled_back":
            history = drop_last_n_user_turns(history, int(payload.get("num_turns", 0)))
            
    if saw_legacy_compaction_without_replacement_history:
        reference_context_item = None
        
    return RolloutReconstruction(
        history=history,
        previous_turn_settings=previous_turn_settings,
        reference_context_item=reference_context_item,
        session_meta=session_meta,
        legacy_compaction_without_replacement_history=saw_legacy_compaction_without_replacement_history,
    )

# --- CodexState class implementation -----------------------------------------
class CodexState:
    def __init__(
        self,
        config: CodexConfig,
        thread_id: str | None = None,
        turn_id: str | None = None,
        installation_id: str | None = None,
        forked_from_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        events: list[CodexEvent] | None = None,
        memory_citations: list[dict[str, Any]] | None = None,
        previous_turn_settings: dict[str, Any] | None = None,
        reference_context_item: dict[str, Any] | None = None,
        last_token_usage: dict[str, Any] | None = None,
        total_token_usage: int = 0,
        session_reasoning_tokens: int = 0,
        context_carryover_tokens: int = 0,
        context_carryover_estimated: bool = False,
    ):
        self.config = config
        self.thread_id = thread_id if thread_id is not None else str(uuid.uuid4())
        self.turn_id = turn_id if turn_id is not None else str(uuid.uuid4())
        self.installation_id = installation_id if installation_id is not None else str(uuid.uuid4())
        self.forked_from_id = forked_from_id
        
        self.history = list(history) if history is not None else []
        self.events = list(events) if events is not None else []
        self.memory_citations = list(memory_citations) if memory_citations is not None else []
        
        self.previous_turn_settings = previous_turn_settings
        self.reference_context_item = reference_context_item
        
        self.total_token_usage = total_token_usage
        self.session_reasoning_tokens = session_reasoning_tokens
        self.context_carryover_tokens = context_carryover_tokens
        self.context_carryover_estimated = context_carryover_estimated
        self.on_event = None
        
        # We model turn_diff tracking
        self.current_turn_diffs: list[str] = []
        
        # We track session creation time
        self.created_at = datetime.now(timezone.utc)
        
        # Initialize token usage
        if last_token_usage is not None:
            self.last_token_usage = dict(last_token_usage)
        else:
            self.last_token_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated": False
            }
            
        pass

    def rollout_path(self) -> Path:
        dt = self.created_at.astimezone(timezone.utc)
        date_path = dt.strftime("%Y/%m/%d")
        time_str = dt.strftime("%Y-%m-%dT%H-%M-%S")
        return self.config.resolved_codex_home() / "sessions" / date_path / f"rollout-{time_str}-{self.thread_id}.jsonl"

    def write_rollout_record(self, rec_type: str, payload: dict[str, Any]) -> None:
        if self.config.ephemeral:
            return
        # Lazily ensure session meta is written before the first turn log
        path = self.rollout_path()
        if not path.exists() and rec_type != "session_meta":
            self.ensure_session_meta_written()
            
        # Filter and map event_msg records to rollout
        if rec_type == "event_msg":
            evt_type = payload.get("type")
            if evt_type == "turn.started":
                mapped_payload = {
                    "type": "task_started",
                    "turn_id": payload.get("turn_id"),
                    "collaboration_mode_kind": "default",
                }
            elif evt_type == "user_message":
                mapped_payload = {
                    "type": "user_message",
                    "message": payload.get("message", ""),
                }
            elif evt_type == "agent_message":
                mapped_payload = {
                    "type": "agent_message",
                    "message": payload.get("message", ""),
                }
            elif evt_type == "turn.completed":
                mapped_payload = {
                    "type": "task_complete",
                    "status": "completed",
                }
            elif evt_type in ("patch_apply_begin", "patch_apply_end"):
                mapped_payload = payload
            else:
                # Omit other telemetry events from persistent rollout
                return
            payload = mapped_payload
            
        path = self.rollout_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        line_dict = {
            "timestamp": ts,
            "type": rec_type,
            "payload": payload,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line_dict) + "\n")

    def ensure_session_meta_written(self) -> None:
        if self.config.ephemeral:
            return
        path = self.rollout_path()
        if path.exists():
            return
            
        base_instr = build_base_instructions(
            prompt_asset="auto",
            model=self.config.model,
            cwd=self.config.resolved_cwd(),
            sandbox=self.config.sandbox,
            approval_policy=self.config.approval_policy,
            codex_home=self.config.resolved_codex_home(),
            memory_tool_enabled=self.config.memory_tool_enabled,
            use_memories=self.config.use_memories,
        )
        
        memory_mode = "enabled" if self.config.use_memories else "disabled"
        payload = {
            "meta": {
                "id": self.thread_id,
                "source": self.config.session_source,
                "memory_mode": memory_mode,
                "base_instructions": {"text": base_instr},
                "cwd": str(self.config.resolved_cwd().absolute()),
            }
        }
        if self.forked_from_id:
            payload["meta"]["forked_from_id"] = self.forked_from_id
        self.write_rollout_record("session_meta", payload)

    def read_rollout_records(self) -> list[dict[str, Any]]:
        path = self.rollout_path()
        if not path.exists():
            return []
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def append_history(self, item: dict[str, Any]) -> None:
        if self.config.collaboration_mode == "Plan" and item.get("type") == "message" and item.get("role") == "assistant":
            content = item.get("content", [])
            full_text = ""
            text_part = None
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    full_text += part["text"]
                    text_part = part
            if full_text:
                plan = extract_proposed_plan_text(full_text)
                clean_text = strip_proposed_plan_blocks(full_text)
                if plan is not None:
                    item["proposed_plan"] = plan
                else:
                    item["proposed_plan"] = None
                if text_part is not None:
                    text_part["text"] = clean_text
                    
        self.history.append(deepcopy(item))
        # Appending history invalidates the last exact usage, making it an estimate!
        self.last_token_usage["estimated"] = True
        self.write_rollout_record("response_item", item)

    def emit(self, event_type: str, **payload: Any) -> CodexEvent:
        event = CodexEvent(type=event_type, payload=payload)
        self.events.append(event)
        if getattr(self, "on_event", None) is not None:
            try:
                self.on_event(event)
            except Exception:
                pass
        # Persistent record
        self.write_rollout_record("event_msg", {"type": event_type, **payload})
        return event

    def record_memory_citation(self, citation: dict[str, Any]) -> None:
        self.memory_citations.append(deepcopy(citation))

    def approx_history_tokens(self) -> int:
        tokens = 0
        for item in self.history:
            item_type = item.get("type")
            if item_type == "message" and "content" in item:
                for part in item["content"]:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        tokens += approx_token_count(part["text"])
            elif "arguments" in item and isinstance(item["arguments"], str):
                tokens += approx_token_count(item["arguments"])
            elif "output" in item and isinstance(item["output"], str):
                tokens += approx_token_count(item["output"])
            elif "input" in item and isinstance(item["input"], str):
                tokens += approx_token_count(item["input"])
        return tokens

    def get_base_instructions(self) -> str:
        return build_base_instructions(
            prompt_asset="auto",
            model=self.config.model,
            cwd=self.config.resolved_cwd(),
            sandbox=self.config.sandbox,
            approval_policy=self.config.approval_policy,
            codex_home=self.config.resolved_codex_home(),
            memory_tool_enabled=self.config.memory_tool_enabled,
            use_memories=self.config.use_memories,
        )

    def build_initial_context(self) -> list[dict[str, Any]]:
        return build_initial_context_items(self.config)

    def estimate_token_count_with_base_instructions(self) -> int:
        return approx_token_count(self.get_base_instructions())

    def active_context_tokens(self) -> int:
        if not self.last_token_usage.get("estimated", True):
            return self.last_token_usage.get("total_tokens", 0)
            
        tokens = self.estimate_token_count_with_base_instructions()
        
        initial_context_items = build_initial_context_items(self.config)
        for item in initial_context_items:
            for part in item.get("content", []):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    tokens += approx_token_count(part["text"])
                    
        prepared_history = prepare_prompt_history(self.history, self.config)
        for item in prepared_history:
            item_type = item.get("type")
            if item_type == "message" and "content" in item:
                for part in item["content"]:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        tokens += approx_token_count(part["text"])
            elif item_type in ("function_call_output", "custom_tool_call_output") and "output" in item:
                val = item["output"]
                if isinstance(val, str):
                    tokens += approx_token_count(val)
                elif isinstance(val, dict) and "content" in val:
                    for part in val["content"]:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            tokens += approx_token_count(part["text"])
            elif "arguments" in item and isinstance(item["arguments"], str):
                tokens += approx_token_count(item["arguments"])
            elif "input" in item and isinstance(item["input"], str):
                tokens += approx_token_count(item["input"])
                
        return tokens

    def active_context_token_status(self) -> tuple[int, bool]:
        is_estimated = self.last_token_usage.get("estimated", True)
        if not is_estimated:
            tokens = self.last_token_usage.get("total_tokens", 0)
        else:
            is_estimated = True
            tokens = self.active_context_tokens()
        return tokens, is_estimated

    def session_context_token_status(self) -> tuple[int, bool]:
        active_tokens, active_estimated = self.active_context_token_status()
        total_session_context = self.context_carryover_tokens + active_tokens
        is_estimated = self.context_carryover_estimated or active_estimated
        return total_session_context, is_estimated

    def start_new_context_epoch(self, carryover_tokens: int | None = None, *, estimated: bool = False) -> None:
        self.context_carryover_tokens = carryover_tokens if carryover_tokens is not None else 0
        self.context_carryover_estimated = estimated

    def start_turn(self) -> None:
        self.turn_id = str(uuid.uuid4())
        self.current_turn_diffs = []
        # Emit turn started
        self.emit("turn.started", turn_id=self.turn_id, model=self.config.model)
        
        # Write turn context record to rollout
        self.write_rollout_record("turn_context", {
            "turn_id": self.turn_id,
            "model": self.config.model,
            "realtime_active": False,
            "approval_policy": self.config.approval_policy,
            "summary": "none",
        })

    def record_token_usage(self, usage: dict[str, Any] | None) -> None:
        if usage is None:
            return
            
        reasoning = usage.get("reasoning_output_tokens", 0)
        if "output_tokens_details" in usage and isinstance(usage["output_tokens_details"], dict):
            reasoning = max(reasoning, usage["output_tokens_details"].get("reasoning_tokens", 0))
            
        total = usage.get("total_tokens", 0)
        self.total_token_usage += total
        self.session_reasoning_tokens += reasoning
        
        self.last_token_usage = {**usage, "estimated": False}

    def recompute_token_usage_from_history(self) -> None:
        estimated_cnt = self.active_context_tokens()
        self.last_token_usage = {
            "input_tokens": estimated_cnt,
            "output_tokens": 0,
            "total_tokens": estimated_cnt,
            "estimated": True,
        }

    def record_apply_patch_turn_diff(self, metadata: Any) -> str | None:
        if isinstance(metadata, dict) and "unified_diff" in metadata:
            diff = metadata["unified_diff"]
            if diff:
                self.current_turn_diffs.append(diff)
                self.emit("turn_diff", unified_diff=diff)
                return "\n".join(self.current_turn_diffs)
        return None

    def session_usage_tokens(self) -> int | None:
        return self.total_token_usage

    def session_reasoning_usage_tokens(self) -> int | None:
        return self.session_reasoning_tokens

    def token_usage_info(self) -> dict[str, Any] | None:
        return {
            "total_token_usage": self.total_token_usage,
            "session_reasoning_tokens": self.session_reasoning_tokens,
            "last_token_usage": self.last_token_usage,
        }

    def write_last_message(self, message: str) -> None:
        path = self.config.resolved_output_last_message()
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(message, encoding="utf-8")
            except Exception:
                pass

    def prompt_history(self) -> list[dict[str, Any]]:
        return prepare_prompt_history(self.history, self.config)

    def compact_with_summary(self, summary_suffix: str, initial_context: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        self._last_local_compaction_summary = summary_suffix
        summary_text = build_compaction_summary_text(summary_suffix)
        user_messages = collect_user_messages(self.history)
        new_history = build_compacted_history(initial_context or [], user_messages, summary_text)
        
        self.history = new_history
        self.last_token_usage["estimated"] = True
        
        # Write compaction record to rollout
        self.write_rollout_record("compacted", {
            "message": summary_text,
            "replacement_history": new_history,
        })
        
        return new_history

    def compact_with_remote_history(
        self,
        compacted_history: list[dict[str, Any]],
        initial_context: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        filtered = [item for item in compacted_history if should_keep_compacted_item(item)]
        new_history = insert_initial_context(filtered, initial_context or [])
        
        self.history = new_history
        self.last_token_usage["estimated"] = True
        
        # Write compaction record to rollout
        self.write_rollout_record("compacted", {
            "message": "",
            "replacement_history": new_history,
        })
        
        return new_history

# --- TUI and Markdown Link Rendering Helpers ---------------------------------
def is_local_path_like_link(url: str) -> bool:
    return url.startswith("file:///")

def render_local_link_target(url: str, cwd: Path | None = None) -> str | None:
    if not url.startswith("file:///"):
        return None
        
    raw_target = url[len("file:///"):].replace("\\", "/")
    if "#" in raw_target:
        path_part, hash_part = raw_target.split("#", 1)
    else:
        path_part, hash_part = raw_target, ""
        
    path_part = Path(path_part).absolute()
    if cwd is not None:
        try:
            rel_path = path_part.relative_to(Path(cwd).absolute())
            res = str(rel_path)
        except ValueError:
            res = str(path_part)
    else:
        res = str(path_part)
        
    if hash_part:
        res += f"#{hash_part}"
    return res

# --- Plan block stripping and extraction helpers ----------------------------
def strip_proposed_plan_blocks(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out_lines = []
    in_plan = False
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "<proposed_plan>":
            in_plan = True
            continue
        if stripped_line == "</proposed_plan>":
            in_plan = False
            continue
            
        if not in_plan:
            out_lines.append(line)
            
    return "".join(out_lines)

def extract_proposed_plan_text(text: str) -> str | None:
    lines = text.splitlines(keepends=True)
    plan_lines = []
    saw_plan = False
    in_plan = False
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "<proposed_plan>":
            in_plan = True
            saw_plan = True
            continue
        if stripped_line == "</proposed_plan>":
            in_plan = False
            continue
        if in_plan:
            plan_lines.append(line)
            
    return "".join(plan_lines) if saw_plan else None

# --- Summarization Prompt Helper ---------------------------------------------
def summarization_prompt() -> str:
    path = ASSETS_DIR / "prompts" / "compact" / "prompt.md"
    return path.read_text(encoding="utf-8")


