from __future__ import annotations
import base64
from dataclasses import dataclass, field
import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import uuid
import threading
from typing import Any, Callable, Iterator, Literal, Optional, Sequence

from codex.types import CodexConfig, CodexEvent

_THREAD_LOCAL = threading.local()

def get_active_state() -> Optional[CodexState]:
    """Retrieve currently active thread-local CodexState context session."""
    return getattr(_THREAD_LOCAL, "active_state", None)

def set_active_state(state: CodexState) -> None:
    """Register active thread-local CodexState context session."""
    _THREAD_LOCAL.active_state = state


RESIZED_IMAGE_BYTES_ESTIMATE = 7373
ORIGINAL_IMAGE_PATCH_SIZE = 32
ORIGINAL_IMAGE_MAX_PATCHES = 10000

class HistoryList(list):
    """Custom list subclass that allows setting dynamic properties like memory_citations."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.memory_citations: list[dict[str, Any]] = []

@dataclass
class CodexState:
    config: CodexConfig
    thread_id: str = field(default_factory=lambda: f"thread-{uuid.uuid4()}")
    turn_id: str = field(default_factory=lambda: f"turn-{uuid.uuid4()}")
    installation_id: str = field(default_factory=lambda: f"inst-{uuid.uuid4()}")
    forked_from_id: str | None = None
    history: list[dict] = field(default_factory=HistoryList)
    events: list[CodexEvent] = field(default_factory=list)
    memory_citations: list[dict] = field(default_factory=list)
    previous_turn_settings: dict[str, Any] | None = None
    reference_context_item: dict[str, Any] | None = None
    last_token_usage: dict[str, Any] | None = None
    total_token_usage: int = 0
    session_reasoning_tokens: int = 0
    context_carryover_tokens: int = 0
    context_carryover_estimated: bool = False

    def __post_init__(self) -> None:
        set_active_state(self)


    def active_context_token_status(self) -> tuple[int | None, bool]:
        if self.last_token_usage is not None and "input_tokens" in self.last_token_usage:
            return (self.last_token_usage["input_tokens"], False)
        return (self.estimate_token_count_with_base_instructions(), True)

    def active_context_tokens(self) -> int:
        tokens, _ = self.active_context_token_status()
        return tokens if tokens is not None else 0

    def append_history(self, item: dict) -> None:
        self.history.append(item)
        self.recompute_token_usage_from_history()
        if not self.config.ephemeral:
            path = self.rollout_path()
            if path:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_meta = not path.exists() or path.stat().st_size == 0
                    with open(path, "a", encoding="utf-8") as f:
                        if write_meta:
                            meta_rec = {
                                "type": "session_meta",
                                "thread_id": self.thread_id,
                                "installation_id": self.installation_id,
                                "forked_from_id": self.forked_from_id,
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
                                "cwd": str(self.config.resolved_cwd()),
                                "model_provider": str(self.config.model)
                            }
                            f.write(json.dumps(meta_rec) + "\n")
                        
                        response_rec = {
                            "type": "response_item",
                            "turn_id": self.turn_id,
                            "item": item
                        }
                        f.write(json.dumps(response_rec) + "\n")
                except Exception as exc:
                    import logging; logging.warning(f"Swallowed exception trace: {exc}")

    def approx_history_tokens(self) -> int:
        return sum((estimate_response_item_model_visible_bytes(item) + 3) // 4 for item in self.history)

    def compact_with_remote_history(
        self,
        compacted_history: list[dict[str, Any]],
        initial_context: list[dict] | None = None
    ) -> list[dict]:
        processed_history = [
            dict(item) for item in compacted_history
            if should_keep_compacted_history_item(item)
        ]
        
        new_history = insert_initial_context_before_last_real_user_or_summary(
            processed_history,
            initial_context or []
        )
        
        hist_list = HistoryList()
        if hasattr(self.history, "memory_citations"):
            hist_list.memory_citations = list(self.history.memory_citations)
        for msg in new_history:
            hist_list.append(msg)
            
        self.history = hist_list
        self.reference_context_item = None
        self.recompute_token_usage_from_history()
        
        self.emit("compacted", message="")
        
        if not self.config.ephemeral:
            path = self.rollout_path()
            if path:
                try:
                    with open(path, "a", encoding="utf-8") as f:
                        rec = {
                            "type": "compacted",
                            "message": "",
                            "replacement_history": list(self.history)
                        }
                        f.write(json.dumps(rec) + "\n")
                except Exception as exc:
                    import logging; logging.warning(f"Swallowed exception trace: {exc}")
        return self.history

    def compact_with_summary(
        self,
        summary_suffix: str,
        initial_context: list[dict] | None = None
    ) -> list[dict]:
        summary_text = build_compaction_summary_text(summary_suffix)
        user_msgs = [
            m.get("content", "") for m in self.history
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        
        limit = self.config.resolved_auto_compact_token_limit()
        if limit is None:
            limit = 20000
            
        new_history = build_compacted_history(
            initial_context or [],
            user_msgs,
            summary_text,
            max_tokens=limit
        )
        
        hist_list = HistoryList()
        if hasattr(self.history, "memory_citations"):
            hist_list.memory_citations = list(self.history.memory_citations)
        for msg in new_history:
            hist_list.append(msg)
            
        self.history = hist_list
        self.reference_context_item = None
        self.recompute_token_usage_from_history()
        
        self.emit("compacted", message=summary_text)
        
        if not self.config.ephemeral:
            path = self.rollout_path()
            if path:
                try:
                    with open(path, "a", encoding="utf-8") as f:
                        rec = {
                            "type": "compacted",
                            "message": summary_text,
                            "replacement_history": list(self.history)
                        }
                        f.write(json.dumps(rec) + "\n")
                except Exception as exc:
                    import logging; logging.warning(f"Swallowed exception trace: {exc}")
        return self.history

    def emit(self, event_type: str, **payload: object) -> CodexEvent:
        evt = CodexEvent(type=event_type, payload=payload)
        self.events.append(evt)
        return evt

    def estimate_token_count_with_base_instructions(self) -> int:
        try:
            from codex.prompts import build_base_instructions
            base_str = build_base_instructions(
                prompt_asset="prompts/gpt_5_codex_prompt.md",
                model=self.config.model,
                cwd=Path(self.config.cwd),
                sandbox=self.config.sandbox,
                approval_policy=self.config.approval_policy,
                codex_home=self.config.resolved_codex_home(),
                use_memories=self.config.use_memories,
            )
            base_tokens = approx_token_count(base_str)
        except Exception as exc:
            import logging
            logging.warning(f"Fatal assets loading exception trace: {exc}")
            from codex.types import ConfigurationError
            raise ConfigurationError(f"Fatal system configuration error: Missing static prompts assets. Details: {exc}")
            
        if self.config.use_memories and self.memory_citations:
            cit_text = "\n\n## Memory Citation Annotations\n"
            cit_text += "You have retrieved context citations from prior session runs:\n"
            for cit in self.memory_citations:
                thread_id = cit.get("thread_id")
                turn_id = cit.get("turn_id")
                summary = cit.get("summary", "")
                cit_text += f"- [memory:{thread_id}:{turn_id}] Note: {summary}\n"
            base_tokens += approx_token_count(cit_text)
            
        return base_tokens + self.approx_history_tokens()

    def prompt_history(self) -> list[dict[str, Any]]:
        return prepare_prompt_history(self.history, self.config)

    def read_rollout_records(self) -> list[dict[str, Any]]:
        path = self.rollout_path()
        if not path or not path.is_file():
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        try:
                            records.append(json.loads(stripped))
                        except Exception as exc:
                            import logging; logging.warning(f"Swallowed exception trace: {exc}")
        except Exception as exc:
            import logging; logging.warning(f"Swallowed exception trace: {exc}")
        return records

    def recompute_token_usage_from_history(self) -> None:
        self.total_token_usage = self.approx_history_tokens()

    def record_apply_patch_turn_diff(self, metadata: Any) -> str | None:
        unified_diff = str(metadata) if metadata is not None else ""
        
        # Emit patch lifecycle turn event
        self.emit("patch", unified_diff=unified_diff)
        
        # Log to dynamic persistent session file if not running in ephemeral sandbox
        if not self.config.ephemeral:
            path = self.rollout_path()
            if path:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "a", encoding="utf-8") as f:
                        rec = {
                            "type": "patch",
                            "turn_id": self.turn_id,
                            "unified_diff": unified_diff,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
                        }
                        f.write(json.dumps(rec) + "\n")
                except Exception as exc:
                    import logging; logging.warning(f"Swallowed exception trace: {exc}")
        return unified_diff


    def record_memory_citation(self, citation: dict) -> None:
        self.memory_citations.append(citation)
        if hasattr(self.history, "memory_citations"):
            self.history.memory_citations.append(citation)

    def record_token_usage(self, usage: dict[str, Any] | None) -> None:
        if usage is not None:
            self.last_token_usage = usage
            total = usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
            self.total_token_usage += total
            
            reasoning = usage.get("reasoning_output_tokens") or usage.get("reasoning_tokens") or 0
            self.session_reasoning_tokens += reasoning

    def rollout_path(self) -> Path | None:
        if self.config.ephemeral:
            return None
        return self.config.resolved_codex_home() / "sessions" / f"{self.thread_id}.jsonl"

    def session_context_token_status(self) -> tuple[int | None, bool]:
        if self.total_token_usage > 0:
            return (self.total_token_usage, False)
        return (self.estimate_token_count_with_base_instructions(), True)

    def session_reasoning_usage_tokens(self) -> int | None:
        return self.session_reasoning_tokens if self.session_reasoning_tokens > 0 else None

    def session_usage_tokens(self) -> int | None:
        return self.total_token_usage if self.total_token_usage > 0 else None

    def start_new_context_epoch(
        self,
        carryover_tokens: int | None = None,
        *,
        estimated: bool = False
    ) -> None:
        self.context_carryover_tokens = carryover_tokens if carryover_tokens is not None else 0
        self.context_carryover_estimated = estimated

    def start_turn(self) -> None:
        self.turn_id = f"turn-{uuid.uuid4()}"
        self.events = []
        if not self.config.ephemeral:
            path = self.rollout_path()
            if path:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_meta = not path.exists() or path.stat().st_size == 0
                    with open(path, "a", encoding="utf-8") as f:
                        if write_meta:
                            meta_rec = {
                                "type": "session_meta",
                                "thread_id": self.thread_id,
                                "installation_id": self.installation_id,
                                "forked_from_id": self.forked_from_id,
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
                                "cwd": str(self.config.resolved_cwd()),
                                "model_provider": str(self.config.model)
                            }
                            f.write(json.dumps(meta_rec) + "\n")
                        
                        turn_rec = {
                            "type": "turn_context",
                            "turn_id": self.turn_id,
                            "thread_id": self.thread_id,
                            "final_message": "",
                            "events": [],
                            "cwd": str(self.config.resolved_cwd()),
                            "model": self.config.model,
                            "sandbox_policy": {
                                "sandbox_mode": self.config.sandbox,
                                "writable_roots": [str(p) for p in self.config.writable_roots]
                            },
                            "approval_policy": self.config.approval_policy,
                            "realtime_active": (self.config.collaboration_mode != "Default")
                        }
                        f.write(json.dumps(turn_rec) + "\n")
                except Exception as exc:
                    import logging; logging.warning(f"Swallowed exception trace: {exc}")

    def token_usage_info(self) -> dict[str, Any] | None:
        return {
            "total_token_usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": self.session_reasoning_tokens,
                "total_tokens": self.total_token_usage
            },
            "last_token_usage": self.last_token_usage or {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
                "total_tokens": 0
            },
            "model_context_window": self.config.resolved_model_context_window()
        }

    def write_last_message(self, message: str) -> None:
        out_path = self.config.resolved_output_last_message()
        if out_path is not None:
            try:
                out_path_p = Path(out_path).expanduser().resolve()
                out_path_p.parent.mkdir(parents=True, exist_ok=True)
                out_path_p.write_text(message, encoding="utf-8")
            except Exception as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")


class RolloutReconstruction:
    def __init__(
        self,
        history: list[dict[str, Any]],
        previous_turn_settings: dict[str, Any] | None,
        reference_context_item: dict[str, Any] | None,
        session_meta: dict[str, Any] | None = None,
        legacy_compaction_without_replacement_history: bool = False,
    ) -> None:
        self.history = history
        self.previous_turn_settings = previous_turn_settings
        self.reference_context_item = reference_context_item
        self.session_meta = session_meta
        self.legacy_compaction_without_replacement_history = legacy_compaction_without_replacement_history


def approx_token_count(text: str) -> int:
    """Calculates approximate token count using 4-bytes/token ceiling heuristic."""
    byte_len = len(text.encode('utf-8'))
    char_len = len(text)
    if byte_len > char_len:
        import warnings
        import logging
        msg = "4-bytes/token ceiling heuristic may result in underestimations on multi-byte text"
        warnings.warn(msg, UserWarning)
        logging.warning(msg)
    return (byte_len + 3) // 4

def approx_bytes_for_tokens(tokens: int) -> int:
    """Converts a token budget back into approximate bytes cost."""
    return tokens * 4

def approx_tokens_from_byte_count(bytes_count: int) -> int:
    """Converts byte count back into tokens using ceiling heuristic."""
    return max(0, (bytes_count + 3) // 4)

def parse_base64_image_data_url(url: str) -> Optional[str]:
    """Helper verifying and extracting base64 payload from inline data URLs."""
    if not url.lower().startswith("data:"):
        return None
    try:
        comma_idx = url.find(',')
        if comma_idx == -1:
            return None
        metadata = url[:comma_idx]
        payload = url[comma_idx + 1:]
        
        metadata_parts = metadata[5:].split(';')
        mime_type = metadata_parts[0]
        if not mime_type.lower().startswith("image/"):
            return None
            
        has_base64 = any(part.lower() == "base64" for part in metadata_parts[1:])
        if not has_base64:
            return None
            
        return payload
    except Exception:
        return None

def estimate_original_image_bytes(image_url: str) -> Optional[int]:
    """Decodes original detail images dynamically calculating 32px grids."""
    payload = parse_base64_image_data_url(image_url)
    if not payload:
        return None
    try:
        payload_clean = "".join(payload.split())
        img_data = base64.b64decode(payload_clean)
    except Exception:
        return None

    try:
        from PIL import Image
    except ImportError:
        import warnings
        import logging
        msg = "PIL/Pillow is missing and original image patch sizing falls back to low-detail"
        warnings.warn(msg, UserWarning)
        logging.warning(msg)
        return None

    try:
        img = Image.open(io.BytesIO(img_data))
        width, height = img.size
        
        patches_wide = math.ceil(width / ORIGINAL_IMAGE_PATCH_SIZE)
        patches_high = math.ceil(height / ORIGINAL_IMAGE_PATCH_SIZE)
        patch_count = min(ORIGINAL_IMAGE_MAX_PATCHES, patches_wide * patches_high)
        
        return patch_count * 4
    except Exception:
        return None

def image_data_url_estimate_adjustment(item: dict[str, Any]) -> tuple[int, int]:
    """Scans response items for discount-eligible image URLs."""
    payload_bytes = 0
    replacement_bytes = 0
    
    def accumulate(image_url: str, detail: Optional[str] = None):
        nonlocal payload_bytes, replacement_bytes
        payload = parse_base64_image_data_url(image_url)
        if payload:
            payload_bytes += len(payload)
            est = None
            if detail == "original":
                est = estimate_original_image_bytes(image_url)
            
            if est is not None:
                replacement_bytes += est
            else:
                replacement_bytes += RESIZED_IMAGE_BYTES_ESTIMATE

    item_type = item.get("type")
    
    if item_type == "message":
        content = item.get("content")
        if isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") in ("input_image", "image_url"):
                    img_url_obj = content_item.get("image_url")
                    if isinstance(img_url_obj, dict):
                        url = img_url_obj.get("url")
                        detail = img_url_obj.get("detail")
                    else:
                        url = img_url_obj
                        detail = content_item.get("detail")
                    if isinstance(url, str):
                        accumulate(url, detail)
                        
    elif item_type in ("function_call_output", "custom_tool_call_output"):
        output = item.get("output", {})
        body = output.get("body", {})
        if isinstance(body, dict) and body.get("type") == "content_items":
            items = body.get("items", [])
            for content_item in items:
                if isinstance(content_item, dict) and content_item.get("type") in ("input_image", "image_url"):
                    img_url_obj = content_item.get("image_url")
                    if isinstance(img_url_obj, dict):
                        url = img_url_obj.get("url")
                        detail = img_url_obj.get("detail")
                    else:
                        url = img_url_obj
                        detail = content_item.get("detail")
                    if isinstance(url, str):
                        accumulate(url, detail)
                        
    return payload_bytes, replacement_bytes

def estimate_response_item_model_visible_bytes(item: dict[str, Any]) -> int:
    """Calculates visible model cost bytes including base64 discounts & reasoning offsets."""
    item_type = item.get("type")
    
    if item_type in ("reasoning", "compaction", "context_compaction"):
        content = item.get("encrypted_content") or item.get("content")
        if not content:
            return 0
        if isinstance(content, list):
            text = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        else:
            text = str(content)
        
        encoded_len = len(text.encode('utf-8'))
        return max(0, (encoded_len * 3) // 4 - 650)
        
    try:
        raw_json = json.dumps(item, separators=(',', ':'))
        raw = len(raw_json.encode('utf-8'))
    except Exception:
        raw = 0
        
    payload_bytes, replacement_bytes = image_data_url_estimate_adjustment(item)
    if payload_bytes == 0 or replacement_bytes == 0:
        return raw
    else:
        return max(0, raw - payload_bytes + replacement_bytes)

def truncate_middle_with_token_budget(text: str, token_budget: int) -> str:
    """
    Truncates a string from the middle to fit within a given token budget.
    Aligns to character boundaries and formats as:
    prefix + "…" + removed_tokens + " tokens truncated…" + suffix
    """
    from codex.memory import truncate_middle_with_token_budget as memory_truncate
    res, _ = memory_truncate(text, token_budget)
    return res


def should_keep_compacted_history_item(item: dict[str, Any]) -> bool:
    role = item.get("role")
    if role == "developer" or role == "system":
        return False
    if role in ("user", "assistant"):
        return True
    return False

def insert_initial_context_before_last_real_user_or_summary(
    processed_history: list[dict[str, Any]],
    initial_context: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not initial_context:
        return processed_history
        
    last_user_idx = -1
    for idx in range(len(processed_history) - 1, -1, -1):
        item = processed_history[idx]
        if item.get("role") == "user":
            last_user_idx = idx
            break
            
    if last_user_idx != -1:
        return processed_history[:last_user_idx] + initial_context + processed_history[last_user_idx:]
    else:
        return initial_context + processed_history

def build_compacted_history(
    initial_context: list[dict],
    user_messages: list[str],
    summary_text: str,
    max_tokens: int = 20000
) -> list[dict]:
    compacted_history = []
    
    for turn in initial_context:
        compacted_history.append(dict(turn))
        
    init_cost = sum(approx_token_count(json.dumps(t)) for t in initial_context)
    sum_cost = approx_token_count(summary_text)
    remaining_budget = max_tokens - (init_cost + sum_cost)
    
    selected_messages = []
    if remaining_budget > 0:
        for msg in reversed(user_messages):
            if remaining_budget <= 0:
                break
            cost = approx_token_count(msg)
            if cost <= remaining_budget:
                selected_messages.append(msg)
                remaining_budget -= cost
            else:
                truncated = truncate_middle_with_token_budget(msg, remaining_budget)
                if truncated:
                    selected_messages.append(truncated)
                break
        selected_messages.reverse()
        
    for msg in selected_messages:
        compacted_history.append({"role": "user", "content": msg})
        
    compacted_history.append({"role": "user", "content": summary_text})
    
    return compacted_history

def build_compaction_summary_text(summary_suffix: str) -> str:
    from codex.prompts import ASSETS_DIR
    prefix_file = ASSETS_DIR / "prompts" / "compact" / "summary_prefix.md"
    prefix = prefix_file.read_text(encoding="utf-8").strip()
    return f"{prefix}\n{summary_suffix}"

def extract_proposed_plan_text(text: str) -> str | None:
    if not text:
        return None
    lines = text.splitlines(keepends=True)
    in_block = False
    block_lines = []
    current_block = []
    saw_block = False
    
    for line in lines:
        stripped = line.strip()
        if not in_block:
            if stripped in ("```proposed_plan", "<proposed_plan>"):
                in_block = True
                saw_block = True
                current_block = []
        else:
            if (stripped == "```" and not line.startswith("<")) or stripped == "</proposed_plan>":
                in_block = False
                block_lines = list(current_block)
            else:
                current_block.append(line)
                
    if in_block:
        block_lines = list(current_block)
        
    if saw_block:
        return "".join(block_lines).strip()
    return None

def strip_proposed_plan_blocks(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines(keepends=True)
    in_block = False
    out_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not in_block:
            if stripped in ("```proposed_plan", "<proposed_plan>"):
                in_block = True
            else:
                out_lines.append(line)
        else:
            if (stripped == "```" and not line.startswith("<")) or stripped == "</proposed_plan>":
                in_block = False
    return "".join(out_lines)

def strip_memory_citations(text: str) -> tuple[str, list[str]]:
    if not text:
        return "", []
        
    markers = []
    
    bracket_re = re.compile(r"\[(memory:[\w\-]+(?::turn-[\w\-]+)?)\]")
    def bracket_sub(m: re.Match) -> str:
        markers.append(m.group(1))
        return ""
    clean_text = bracket_re.sub(bracket_sub, text)
    
    xml_re = re.compile(r"<oai-mem-citation>(.*?)</oai-mem-citation>", re.DOTALL)
    def xml_sub(m: re.Match) -> str:
        markers.append(f"<oai-mem-citation>{m.group(1)}</oai-mem-citation>")
        return ""
    clean_text = xml_re.sub(xml_sub, clean_text)
    
    unclosed_xml_re = re.compile(r"<oai-mem-citation>(.*?)$", re.DOTALL)
    def unclosed_sub(m: re.Match) -> str:
        markers.append(f"<oai-mem-citation>{m.group(1)}</oai-mem-citation>")
        return ""
    clean_text = unclosed_xml_re.sub(unclosed_sub, clean_text)
    
    return clean_text, markers

def parse_memory_citation(citations: list[str]) -> dict | None:
    bracket_re = re.compile(r"^\[?memory:([\w\-]+):([\w\-]+)\]?$")
    for cit in citations:
        m = bracket_re.match(cit.strip())
        if m:
            return {
                "thread_id": m.group(1),
                "turn_id": m.group(2),
                "summary": ""
            }
            
    rollout_ids = []
    entries = []
    seen_rollouts = set()
    
    def extract_block(text: str, open_tag: str, close_tag: str) -> Optional[str]:
        if open_tag in text and close_tag in text:
            parts = text.split(open_tag, 1)
            if len(parts) == 2:
                body = parts[1].split(close_tag, 1)[0]
                return body
        return None
        
    for cit in citations:
        ids_body = extract_block(cit, "<rollout_ids>", "</rollout_ids>") or extract_block(cit, "<thread_ids>", "</thread_ids>")
        if ids_body:
            for line in ids_body.splitlines():
                rid = line.strip()
                if rid and rid not in seen_rollouts:
                    rollout_ids.append(rid)
                    seen_rollouts.add(rid)
                    
        entries_body = extract_block(cit, "<citation_entries>", "</citation_entries>")
        if entries_body:
            for line in entries_body.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "|note=[" in line:
                    loc, note_part = line.rsplit("|note=[", 1)
                    note = note_part.rstrip("]").strip()
                    if ":" in loc:
                        path, lines = loc.rsplit(":", 1)
                        if "-" in lines:
                            start, end = lines.split("-", 1)
                            try:
                                entries.append({
                                    "path": path.strip(),
                                    "line_start": int(start.strip()),
                                    "line_end": int(end.strip()),
                                    "note": note
                                })
                            except ValueError as exc:
                                import logging; logging.warning(f"Swallowed exception trace: {exc}")
                                
    if entries or rollout_ids:
        return {
            "entries": entries,
            "rollout_ids": rollout_ids
        }
        
    for cit in citations:
        clean = cit.strip()
        if clean.startswith("[") and clean.endswith("]"):
            clean = clean[1:-1]
        parts = clean.split(":")
        if len(parts) == 3 and parts[0] == "memory":
            if parts[2].startswith("turn-") and parts[2][5:].isdigit():
                return {
                    "thread_id": parts[1],
                    "turn_id": parts[2],
                    "summary": ""
                }
                
    return None

def cleanse_message_citations(msg: dict[str, Any]) -> dict[str, Any]:
    copied = dict(msg)
    content = copied.get("content")
    if isinstance(content, str):
        if "[memory:" in content:
            clean_text, _ = strip_memory_citations(content)
            copied["content"] = clean_text
    elif isinstance(content, list):
        new_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                text = block["text"]
                if isinstance(text, str) and "[memory:" in text:
                    clean_text, _ = strip_memory_citations(text)
                    new_block = dict(block)
                    new_block["text"] = clean_text
                    new_content.append(new_block)
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        copied["content"] = new_content
    return copied

def prepare_prompt_history(
    history: list[dict[str, Any]],
    config: CodexConfig
) -> list[dict[str, Any]]:
    messages = [dict(m) for m in history]
    
    has_system_msg = len(messages) > 0 and messages[0].get("role") == "system"
    
    from codex.prompts import build_base_instructions
    try:
        base_prompt = build_base_instructions(
            prompt_asset="prompts/gpt_5_codex_prompt.md",
            model=config.model,
            cwd=config.resolved_cwd(),
            sandbox=config.sandbox,
            approval_policy=config.approval_policy,
            codex_home=config.resolved_codex_home(),
            use_memories=config.use_memories
        )
    except Exception as exc:
        import logging
        logging.warning(f"Fatal prompts assets compilation failed: {exc}")
        from codex.types import ConfigurationError
        raise ConfigurationError(f"Fatal system configuration error: Failed to compile system base prompts instructions. Details: {exc}")
        
    citations = getattr(history, "memory_citations", [])
    
    if config.use_memories and citations:
        cit_text = "\n\n## Memory Citation Annotations\n"
        cit_text += "You have retrieved context citations from prior session runs:\n"
        for cit in citations:
            thread_id = cit.get("thread_id")
            turn_id = cit.get("turn_id")
            summary = cit.get("summary", "")
            cit_text += f"- [memory:{thread_id}:{turn_id}] Note: {summary}\n"
        base_prompt += cit_text
        
    if has_system_msg:
        messages[0] = {
            "role": "system",
            "content": f"{base_prompt}\n\n{messages[0]['content']}"
        }
    else:
        messages.insert(0, {
            "role": "system",
            "content": base_prompt
        })
        
    cleaned_messages = []
    for msg in messages:
        cleaned_messages.append(cleanse_message_citations(msg))
        
    return cleaned_messages

def normalize_rollout_record(raw_record: dict) -> dict:
    normalized = {"type": raw_record.get("type")}
    
    if not normalized["type"]:
        payload = raw_record.get("payload")
        if isinstance(payload, dict) and "type" in payload:
            normalized["type"] = payload["type"]
        else:
            return {}
            
    rec_type = normalized["type"]
    payload = raw_record.get("payload")
    item = raw_record.get("item")
    
    source_dict = raw_record
    if isinstance(payload, dict):
        source_dict = payload
    elif isinstance(item, dict):
        source_dict = item
        
    if rec_type == "session_meta":
        normalized["thread_id"] = raw_record.get("thread_id") or source_dict.get("id") or source_dict.get("thread_id")
        normalized["installation_id"] = raw_record.get("installation_id") or source_dict.get("installation_id")
        normalized["forked_from_id"] = raw_record.get("forked_from_id") or source_dict.get("forked_from_id")
        normalized["cwd"] = raw_record.get("cwd") or source_dict.get("cwd")
        normalized["model_provider"] = raw_record.get("model_provider") or source_dict.get("model_provider")
        normalized["memory_mode"] = raw_record.get("memory_mode") or source_dict.get("memory_mode")
        
    elif rec_type == "turn_context":
        normalized["turn_id"] = raw_record.get("turn_id") or source_dict.get("turn_id")
        normalized["thread_id"] = raw_record.get("thread_id") or source_dict.get("thread_id")
        normalized["final_message"] = raw_record.get("final_message") or source_dict.get("final_message") or source_dict.get("user_instructions")
        normalized["events"] = raw_record.get("events") or source_dict.get("events") or []
        normalized["cwd"] = raw_record.get("cwd") or source_dict.get("cwd")
        normalized["model"] = raw_record.get("model") or source_dict.get("model")
        normalized["sandbox_policy"] = raw_record.get("sandbox_policy") or source_dict.get("sandbox_policy")
        normalized["approval_policy"] = raw_record.get("approval_policy") or source_dict.get("approval_policy")
        normalized["realtime_active"] = raw_record.get("realtime_active") or source_dict.get("realtime_active")
        
    elif rec_type == "response_item":
        normalized["turn_id"] = raw_record.get("turn_id")
        resp_item = None
        if "item" in raw_record:
            resp_item = raw_record["item"]
        elif "payload" in raw_record:
            resp_item = raw_record["payload"]
        else:
            resp_item = raw_record
        normalized["item"] = resp_item
        
    elif rec_type == "compacted":
        normalized["message"] = raw_record.get("message") or source_dict.get("message")
        normalized["replacement_history"] = raw_record.get("replacement_history") or source_dict.get("replacement_history")
        
    elif rec_type == "event_msg":
        normalized["payload_type"] = source_dict.get("type") if isinstance(source_dict, dict) else None
        normalized["num_turns"] = source_dict.get("num_turns") if isinstance(source_dict, dict) else None
        normalized["turn_id"] = source_dict.get("turn_id") if isinstance(source_dict, dict) else None
        
    return normalized

def drop_last_n_user_turns(history: list[dict], n: int) -> None:
    for _ in range(n):
        last_user_idx = -1
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx != -1:
            del history[last_user_idx:]
        else:
            history.clear()
            break

def reconstruct_history_from_rollout(
    source: Path | str | list[dict[str, Any]]
) -> RolloutReconstruction:
    history = HistoryList()
    previous_turn_settings = None
    reference_context_item = None
    session_meta = None
    legacy_compaction_without_replacement_history = False
    
    seen_turn_ids = set()
    skipped_duplicate_turn_ids = set()
    metadata_stack = []
    raw_records = []
    
    if isinstance(source, list):
        raw_records = source
    else:
        file_path = Path(source)
        if not file_path.is_file():
            # If not direct file, lookup by ID dynamically in both flat and nested schemas
            parent_home = Path.home() / ".codex"
            # Try to dynamic search home path override
            if "CODEX_HOME" in os.environ:
                parent_home = Path(os.environ["CODEX_HOME"])
            
            # Find candidate paths
            flat_candidate = parent_home / "sessions" / f"{source}.jsonl"
            if flat_candidate.is_file():
                file_path = flat_candidate
            else:
                sessions_dir = parent_home / "sessions"
                resolved = None
                if sessions_dir.is_dir():
                    for path in sessions_dir.glob(f"**/*{source}*.jsonl"):
                        if path.is_file():
                            resolved = path
                            break
                if resolved:
                    file_path = resolved
                else:
                    raise FileNotFoundError(f"Rollout session log file not found: {file_path}")
                    
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        raw_records.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise RuntimeError(f"Failed to read rollout stream: {e}")
            
    for record in raw_records:
        normalized = normalize_rollout_record(record)
        if not normalized or "type" not in normalized:
            continue
            
        rec_type = normalized["type"]
        
        if rec_type == "session_meta":
            if session_meta is None:
                session_meta = normalized
                
        elif rec_type == "turn_context":
            turn_id = normalized.get("turn_id")
            if turn_id is not None:
                if turn_id in seen_turn_ids:
                    skipped_duplicate_turn_ids.add(turn_id)
                    continue
                seen_turn_ids.add(turn_id)
                
            metadata_stack.append((reference_context_item, previous_turn_settings))
            reference_context_item = record
            
            user_prompt = normalized.get("final_message")
            if user_prompt:
                history.append({"role": "user", "content": user_prompt})
                
            previous_turn_settings = {
                "model": normalized.get("model", "gpt-5.5"),
                "realtime_active": normalized.get("realtime_active", False)
            }
            
        elif rec_type == "response_item":
            turn_id = normalized.get("turn_id") or record.get("turn_id")
            if turn_id is not None and turn_id in skipped_duplicate_turn_ids:
                continue
                
            item_data = normalized.get("item")
            if not isinstance(item_data, dict):
                continue
                
            item_type = item_data.get("type")
            if item_type == "ghost_snapshot":
                continue
                
            if item_type == "message":
                role = item_data.get("role", "assistant")
                content_items = item_data.get("content", [])
                msg_text = ""
                if isinstance(content_items, list):
                    for content in content_items:
                        if isinstance(content, dict) and "text" in content:
                            msg_text += content["text"]
                elif isinstance(content_items, str):
                    msg_text = content_items
                history.append({"role": role, "content": msg_text})
                
        elif rec_type == "compacted":
            metadata_stack.clear()
            replacement_history = normalized.get("replacement_history")
            summary_message = normalized.get("message", "")
            
            if replacement_history is not None:
                history.clear()
                for msg in replacement_history:
                    if isinstance(msg, dict):
                        history.append(msg)
                reference_context_item = None
            else:
                legacy_compaction_without_replacement_history = True
                reference_context_item = None
                
                user_messages = [msg["content"] for msg in history if msg.get("role") == "user"]
                rebuilt_context = build_compacted_history(
                    initial_context=[], 
                    user_messages=user_messages, 
                    summary_text=summary_message
                )
                
                history.clear()
                for msg in rebuilt_context:
                    history.append(msg)
                    
        elif rec_type == "event_msg":
            payload_type = normalized.get("payload_type")
            if payload_type == "thread_rolled_back":
                num_turns = normalized.get("num_turns", 0)
                drop_last_n_user_turns(history, num_turns)
                
                for _ in range(num_turns):
                    if metadata_stack:
                        reference_context_item, previous_turn_settings = metadata_stack.pop()
                    else:
                        reference_context_item = None
                        previous_turn_settings = None
                
    return RolloutReconstruction(
        history=history,
        previous_turn_settings=previous_turn_settings,
        reference_context_item=reference_context_item,
        session_meta=session_meta,
        legacy_compaction_without_replacement_history=legacy_compaction_without_replacement_history
    )

def summarization_prompt() -> str:
    from codex.prompts import ASSETS_DIR
    prompt_file = ASSETS_DIR / "prompts" / "compact" / "prompt.md"
    return prompt_file.read_text(encoding="utf-8")

def short_display_path(path: str) -> str:
    normalized = path.replace('\\', '/')
    trimmed = normalized.rstrip('/')
    if not trimmed:
        return normalized
    parts = trimmed.split('/')
    filtered = []
    for p in reversed(parts):
        if p and p not in ("build", "dist", "node_modules", "src"):
            filtered.append(p)
    if filtered:
        return filtered[0]
    return trimmed

def split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()

def normalize_tokens(tokens: list[str]) -> list[str]:
    if len(tokens) >= 2 and tokens[0] in ("yes", "y", "no", "n") and tokens[1] == "|":
        tokens = tokens[2:]
    if len(tokens) == 3:
        shell, flag, script = tokens
        if shell in ("bash", "zsh") and flag in ("-c", "-lc"):
            try:
                split_script = shlex.split(script)
                if split_script:
                    return split_script
            except ValueError as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
    return tokens

def split_on_connectors(tokens: list[str]) -> list[list[str]]:
    parts = []
    current = []
    connectors = {"|", "&&", "||", ";"}
    for tok in tokens:
        if tok in connectors:
            if current:
                parts.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        parts.append(current)
    return parts

def cd_target(args: list[str]) -> Optional[str]:
    if not args:
        return None
    target = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            if i + 1 < len(args):
                return args[i + 1]
            return None
          
        if arg in ("-L", "-P") or arg.startswith("-"):
            i += 1
            continue
        target = arg
        i += 1
    return target

def skip_flag_values(tokens: list[str], flags_with_args: set[str]) -> list[str]:
    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            if tok in flags_with_args:
                i += 2
            else:
                i += 1
        else:
            result.append(tok)
            i += 1
    return result

def sed_read_path(args: list[str]) -> Optional[str]:
    if any(arg == "-i" or arg.startswith("-i") or arg == "--in-place" for arg in args):
        return None
        
    has_script_flag = False
    file_operands = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-e", "--expression", "-f", "--file"):
            if arg in ("-e", "--expression"):
                has_script_flag = True
            i += 2
            continue
        if arg.startswith("-"):
            if "e" in arg or "f" in arg:
                has_script_flag = True
                i += 2
            else:
                i += 1
            continue
        file_operands.append(arg)
        i += 1
        
    if not file_operands:
        return None
        
    if has_script_flag:
        return file_operands[0]
    else:
        return file_operands[1] if len(file_operands) > 1 else None

def awk_data_file_operand(args: list[str]) -> Optional[str]:
    has_program = False
    file_operands = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-F", "-v", "-f", "-e", "--field-separator", "--assign", "--file", "--source"):
            if arg in ("-f", "--file", "-e", "--source"):
                has_program = True
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        file_operands.append(arg)
        i += 1
        
    if not file_operands:
        return None
        
    if has_program:
        return file_operands[0]
    else:
        return file_operands[1] if len(file_operands) > 1 else None

def python_script_is_file_walk(script: str) -> bool:
    indicators = ("os.walk", "os.listdir", "os.scandir", "glob.glob", "glob.iglob", "pathlib.Path")
    return any(ind in script for ind in indicators)

def get_first_operand(tokens: list[str], flags_with_args: set[str]) -> Optional[str]:
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            if tok in flags_with_args:
                i += 2
            elif tok.startswith("-n") or tok.startswith("-c"):
                suffix = tok[2:]
                if suffix and (suffix[0].isdigit() or suffix[0] in ("+", "-")):
                    i += 1
                else:
                    i += 2
            else:
                i += 1
        else:
            return tok
    return None

def parse_grep_like_summary(main_cmd: list[str], args: list[str]) -> dict[str, Any]:
    cmd_str = shlex.join(main_cmd)
    pattern = None
    operands = []
    after_double_dash = False
    
    i = 0
    while i < len(args):
        arg = args[i]
        if after_double_dash:
            operands.append(arg)
            i += 1
            continue
        if arg == "--":
            after_double_dash = True
            i += 1
            continue
        if arg in ("-e", "--regexp", "-f", "--file"):
            if i + 1 < len(args) and pattern is None:
                pattern = args[i+1]
            i += 2
            continue
        if arg in ("-m", "--max-count", "-C", "--context", "-A", "--after-context", "-B", "--before-context"):
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        operands.append(arg)
        i += 1
        
    has_pattern = pattern is not None
    query = pattern if has_pattern else (operands[0] if operands else None)
    path_idx = 0 if has_pattern else 1
    path = short_display_path(operands[path_idx]) if len(operands) > path_idx else None
    
    return {"type": "search", "cmd": cmd_str, "query": query, "path": path}

def is_mutating_xargs_command(tokens: list[str]) -> bool:
    sub = xargs_subcommand(tokens)
    if sub is None:
        return False
    return xargs_is_mutating_subcommand(sub)

def xargs_subcommand(tokens: list[str]) -> Optional[list[str]]:
    if not tokens or tokens[0] != "xargs":
        return None
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            rest = tokens[i+1:]
            return rest if rest else None
        if not token.startswith("-"):
            rest = tokens[i:]
            return rest if rest else None
            
        takes_value = token in ("-E", "-e", "-I", "-L", "-n", "-P", "-s")
        if takes_value and len(token) == 2:
            i += 2
        else:
            i += 1
    return None

def xargs_is_mutating_subcommand(tokens: list[str]) -> bool:
    if not tokens:
        return False
    head = tokens[0]
    tail = tokens[1:]
    if head in ("perl", "ruby"):
        return xargs_has_in_place_flag(tail)
    if head == "sed":
        return xargs_has_in_place_flag(tail) or any(token == "--in-place" for token in tail)
    if head == "rg":
        return any(token == "--replace" for token in tail)
    return False

def xargs_has_in_place_flag(tokens: list[str]) -> bool:
    return any(token == "-i" or token.startswith("-i") or token == "-pi" or token.startswith("-pi") for token in tokens)

def is_small_formatting_command(tokens: list[str]) -> bool:
    if not tokens:
        return False
    cmd = tokens[0]
    if cmd in ("wc", "tr", "cut", "sort", "uniq", "tee", "column", "yes", "printf"):
        return True
    if cmd == "xargs":
        return not is_mutating_xargs_command(tokens)
    if cmd == "awk":
        return awk_data_file_operand(tokens[1:]) is None
    if cmd == "head":
        if len(tokens) == 1:
            return True
        if len(tokens) == 2:
            return tokens[1].startswith("-")
        if len(tokens) == 3:
            flag, count = tokens[1], tokens[2]
            if flag in ("-n", "-c") and count.isdigit():
                return True
        return False
    if cmd == "tail":
        if len(tokens) == 1:
            return True
        if len(tokens) == 2:
            return tokens[1].startswith("-")
        if len(tokens) == 3:
            flag, count = tokens[1], tokens[2]
            if flag == "-n":
                if count.isdigit() or (count.startswith("+") and count[1:].isdigit()):
                    return True
            if flag == "-c":
                if count.isdigit() or (count.startswith("+") and count[1:].isdigit()):
                    return True
        return False
    if cmd == "sed":
        return sed_read_path(tokens[1:]) is None
    return False

def is_echo_command(cmd: str) -> bool:
    tokens = split_command(cmd)
    return len(tokens) > 0 and tokens[0] == "echo"

def is_true_command(cmd: str) -> bool:
    return cmd.strip() == "true"

def is_nl_formatting_command(cmd: str) -> bool:
    tokens = split_command(cmd)
    if len(tokens) > 0 and tokens[0] == "nl":
        return all(tok.startswith("-") for tok in tokens[1:])
    return False

def simplify_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changed = True
    while changed:
        changed = False
        if len(commands) <= 1:
            break
            
        if commands[0].get("type") == "unknown" and is_echo_command(commands[0].get("cmd", "")):
            commands = commands[1:]
            changed = True
            continue
            
        true_idx = None
        for idx, cmd in enumerate(commands):
            if cmd.get("type") == "unknown" and is_true_command(cmd.get("cmd", "")):
                true_idx = idx
                break
        if true_idx is not None:
            commands = commands[:true_idx] + commands[true_idx + 1:]
            changed = True
            continue
            
        nl_idx = None
        for idx, cmd in enumerate(commands):
            if cmd.get("type") == "unknown" and is_nl_formatting_command(cmd.get("cmd", "")):
                nl_idx = idx
                break
        if nl_idx is not None:
            commands = commands[:nl_idx] + commands[nl_idx + 1:]
            changed = True
            continue
            
    return commands

def summarize_segment(tokens: list[str]) -> dict[str, Any]:
    cmd_str = shlex.join(tokens)
    head = tokens[0]
    tail = tokens[1:]
    
    if head in ("ls", "eza", "exa"):
        flags = {"-I", "-w", "--block-size", "--format", "--time-style", "--color", "--quoting-style", "--ignore-glob", "--sort", "--time"}
        path_operand = get_first_operand(tokens, flags)
        path = short_display_path(path_operand) if path_operand else None
        return {"type": "list_files", "cmd": cmd_str, "path": path}
        
    if head == "tree":
        flags = {"-L", "-P", "-I", "--charset", "--filelimit", "--sort"}
        path_operand = get_first_operand(tokens, flags)
        path = short_display_path(path_operand) if path_operand else None
        return {"type": "list_files", "cmd": cmd_str, "path": path}
        
    if head == "du":
        flags = {"-d", "--max-depth", "-B", "--block-size", "--exclude", "--time-style"}
        path_operand = get_first_operand(tokens, flags)
        path = short_display_path(path_operand) if path_operand else None
        return {"type": "list_files", "cmd": cmd_str, "path": path}
        
    if head in ("rg", "rga", "ripgrep-all"):
        if "--files" in tokens:
            flags = {"-g", "--glob", "--iglob", "-t", "--type", "--type-add", "--type-not", "-m", "--max-count", "-A", "-B", "-C", "--context", "--max-depth"}
            path_operand = get_first_operand(tokens, flags)
            path = short_display_path(path_operand) if path_operand else None
            return {"type": "list_files", "cmd": cmd_str, "path": path}
            
    if head in ("python", "python3") and len(tail) >= 2 and tail[0] == "-c":
        script = tail[1]
        if python_script_is_file_walk(script):
            return {"type": "list_files", "cmd": cmd_str, "path": None}
            
    if head in ("rg", "rga", "ripgrep-all"):
        flags = {"-g", "--glob", "--iglob", "-t", "--type", "--type-add", "--type-not", "-m", "--max-count", "-A", "-B", "-C", "--context", "--max-depth"}
        candidates = skip_flag_values(tail, flags)
        non_flags = [tok for tok in candidates if not tok.startswith("-")]
        query = non_flags[0] if non_flags else None
        path = short_display_path(non_flags[1]) if len(non_flags) >= 2 else None
        return {"type": "search", "cmd": cmd_str, "query": query, "path": path}
        
    if head in ("grep", "egrep", "fgrep"):
        return parse_grep_like_summary(tokens, tail)
        
    if head == "git" and tail and tail[0] == "grep":
        return parse_grep_like_summary(tokens, tail[1:])
        
    if head == "git" and tail and tail[0] == "ls-files":
        flags = {"--exclude", "--exclude-from", "--pathspec-from-file"}
        path_operand = get_first_operand(tail, flags)
        path = short_display_path(path_operand) if path_operand else None
        return {"type": "list_files", "cmd": cmd_str, "path": path}
        
    if head in ("ag", "ack", "pt"):
        flags = {"-G", "-g", "--file-search-regex", "--ignore-dir", "--ignore-file", "--path-to-ignore"}
        candidates = skip_flag_values(tail, flags)
        non_flags = [tok for tok in candidates if not tok.startswith("-")]
        query = non_flags[0] if non_flags else None
        path = short_display_path(non_flags[1]) if len(non_flags) >= 2 else None
        return {"type": "search", "cmd": cmd_str, "query": query, "path": path}
        
    if head == "cat":
        path = get_first_operand(tokens, set())
        if path:
            return {"type": "read", "cmd": cmd_str, "name": short_display_path(path), "path": path}
            
    if head in ("bat", "batcat"):
        flags = {"--theme", "--language", "--style", "--terminal-width", "--tabs", "--line-range", "--map-syntax"}
        path = get_first_operand(tokens, flags)
        if path:
            return {"type": "read", "cmd": cmd_str, "name": short_display_path(path), "path": path}
            
    if head == "less":
        flags = {"-p", "-P", "-x", "-y", "-z", "-j", "--pattern", "--prompt", "--tabs", "--shift", "--jump-target"}
        path = get_first_operand(tokens, flags)
        if path:
            return {"type": "read", "cmd": cmd_str, "name": short_display_path(path), "path": path}
            
    if head == "more":
        path = get_first_operand(tokens, set())
        if path:
            return {"type": "read", "cmd": cmd_str, "name": short_display_path(path), "path": path}
            
    if head in ("head", "tail"):
        flags = {"-n", "-c"}
        path = get_first_operand(tokens, flags)
        if path:
            return {"type": "read", "cmd": cmd_str, "name": short_display_path(path), "path": path}
            
    return {"type": "unknown", "cmd": cmd_str}

def parse_command_actions(command: str) -> list[dict[str, Any]]:
    tokens = split_command(command)
    if not tokens:
        return []
        
    normalized = normalize_tokens(tokens)
    parts = split_on_connectors(normalized)
    
    raw_cmd_str = command
    
    commands = []
    cwd = None
    
    for segment in parts:
        if not segment:
            continue
        if segment[0] == "cd":
            target = cd_target(segment[1:])
            if target:
                if cwd:
                    cwd = os.path.normpath(os.path.join(cwd, target))
                else:
                    cwd = os.path.normpath(target)
            continue
            
        parsed = summarize_segment(segment)
        
        if parsed["type"] == "read":
            original_path = parsed["path"]
            if cwd:
                resolved = os.path.normpath(os.path.join(cwd, original_path))
                parsed["path"] = resolved
                parsed["name"] = short_display_path(resolved)
                
        commands.append(parsed)
        
    commands = [cmd for cmd in commands if not is_small_formatting_command(split_command(cmd["cmd"]))]
    
    commands = simplify_commands(commands)
    
    if not commands:
        return []
        
    deduped = []
    for cmd in commands:
        if deduped and deduped[-1] == cmd:
            continue
        deduped.append(cmd)
        
    if any(cmd["type"] == "unknown" for cmd in deduped):
        inner_cmd = raw_cmd_str
        if len(tokens) == 3 and tokens[0] in ("bash", "zsh") and tokens[1] in ("-c", "-lc"):
            inner_cmd = tokens[2]
        return [{"type": "unknown", "cmd": inner_cmd}]
        
    return deduped


CODEX_ROLLOUT_ITEM_TYPES: frozenset[str] = frozenset({
    "session_meta",
    "response_item",
    "compacted",
    "turn_context",
    "event_msg"
})
