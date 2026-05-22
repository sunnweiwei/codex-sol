"""
State and rollouts management, history replaying, and context window compaction.
Implements dynamic prompts compilation state, token budgeting, citation parsing,
and custom command segmentation.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

from .config import CodexConfig
from .types import RolloutReconstruction

# Target constant types in rollout parsing
CODEX_ROLLOUT_ITEM_TYPES = frozenset([
    "Compacted",
    "EventMsg",
    "ResponseItem",
    "TurnContext",
    "SessionMeta"
])

# Warning summary prefix used in compaction
SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its thinking process. "
    "You also have access to the state of the tools that were used by that language model. Use this to build on the "
    "work that has already been done and avoid duplicating work. Here is the summary produced by the other language "
    "model, use the information in this summary to assist with your own analysis:"
)

# Base models matching specifications

class ResponseItem:
    """
    In-memory representation of a history item sent to the LLM context.
    """
    def __init__(
        self,
        id: Optional[str],
        role: str,
        content: str,
        phase: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.id = id
        self.role = role
        self.content = content
        self.phase = phase
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "phase": self.phase,
            "metadata": self.metadata
        }

class SessionHistory:
    """
    Maintains the active sequence of past conversation turns.
    """
    def __init__(self) -> None:
        self.items: List[ResponseItem] = []
        self.history_version: int = 0

    def add_item(self, item: ResponseItem) -> None:
        self.items.append(item)
        self.history_version += 1

    def clear(self) -> None:
        self.items.clear()
        self.history_version = 0

    def for_prompt(self, supports_images: bool = False) -> List[ResponseItem]:
        normalized = []
        for item in self.items:
            clean_content = item.content
            if not supports_images and "image_data" in item.metadata:
                item_copy = ResponseItem(
                    id=item.id,
                    role=item.role,
                    content=clean_content,
                    phase=item.phase,
                    metadata={k: v for k, v in item.metadata.items() if k != "image_data"}
                )
                normalized.append(item_copy)
            else:
                normalized.append(item)
        return normalized

class TransitionKind:
    INSTRUCTION = "instruction"
    TOOL_RESULT = "tool_result"
    COMPACTION = "compaction"

class TransitionEvent:
    def __init__(self, kind: str, data: Dict[str, Any], timestamp: int) -> None:
        self.kind = kind
        self.data = data
        self.timestamp = timestamp

class RolloutReconstructionEngine:
    """
    Replays a transaction trace stream to rebuild complete SessionHistory snapshots.
    """
    @staticmethod
    def parse_jsonl(log_content: str) -> List[TransitionEvent]:
        events = []
        for line in log_content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            events.append(TransitionEvent(
                kind=payload["kind"],
                data=payload["data"],
                timestamp=payload["timestamp"]
            ))
        return events

    def replay(self, events: List[TransitionEvent]) -> SessionHistory:
        history = SessionHistory()
        for event in events:
            self._apply_transition(history, event)
        return history

    def _apply_transition(self, history: SessionHistory, event: TransitionEvent) -> None:
        d = event.data
        if event.kind == TransitionKind.INSTRUCTION:
            history.add_item(ResponseItem(
                id=d.get("id"),
                role="user",
                content=d["text"],
                metadata=d.get("metadata", {})
            ))
        elif event.kind == TransitionKind.TOOL_RESULT:
            history.add_item(ResponseItem(
                id=d.get("id"),
                role="tool",
                content=d["output"],
                metadata={"tool_name": d["name"]}
            ))
        elif event.kind == TransitionKind.COMPACTION:
            history.clear()
            summary_content = f"{d['prefix']}\n{d['summary']}"
            history.add_item(ResponseItem(
                id=d.get("id"),
                role="user",
                content=summary_content,
                metadata={"compacted": True}
            ))

    def rollback_to_turn(
        self,
        events: List[TransitionEvent],
        target_turn_index: int,
        initial_context: List[ResponseItem]
    ) -> SessionHistory:
        instruction_count = 0
        cut_index = 0
        for idx, event in enumerate(events):
            if event.kind == TransitionKind.INSTRUCTION:
                if instruction_count == target_turn_index:
                    break
                instruction_count += 1
            cut_index = idx + 1

        retained_events = events[:cut_index]
        history = self.replay(retained_events)
        return self.recover_initial_context(history, initial_context)

    def recover_initial_context(
        self,
        history: SessionHistory,
        initial_context: List[ResponseItem]
    ) -> SessionHistory:
        last_real_user_idx = None
        last_user_or_summary_idx = None

        for idx, item in enumerate(history.items):
            if item.role == "user":
                last_user_or_summary_idx = idx
                if not item.metadata.get("compacted", False):
                    last_real_user_idx = idx

        insertion_index = last_real_user_idx
        if insertion_index is None:
            insertion_index = last_user_or_summary_idx

        rebuilt_items = []
        if insertion_index is not None:
            rebuilt_items.extend(history.items[:insertion_index])
            rebuilt_items.extend(initial_context)
            rebuilt_items.extend(history.items[insertion_index:])
        else:
            rebuilt_items.extend(initial_context)
            rebuilt_items.extend(history.items)

        history.items = rebuilt_items
        history.history_version += 1
        return history

class MemoryCitation:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.rollout_ids: List[str] = []

class CitationStreamParser:
    """
    Strips XML-like <oai-mem-citation> blocks on the fly during text streaming.
    """
    def __init__(self) -> None:
        self.buffer = ""
        self.inside_citation = False
        self.current_citation_payload = ""
        self.citations_extracted: List[str] = []

    def push_delta(self, delta: str) -> Tuple[str, List[str]]:
        self.buffer += delta
        visible_text = []
        new_citations = []

        while self.buffer:
            if not self.inside_citation:
                tag_idx = self.buffer.find("<oai-mem-citation>")
                if tag_idx == -1:
                    partial_idx = self.buffer.rfind("<")
                    if partial_idx != -1 and "<oai-mem-citation>".startswith(self.buffer[partial_idx:]):
                        visible_text.append(self.buffer[:partial_idx])
                        self.buffer = self.buffer[partial_idx:]
                        break
                    else:
                        visible_text.append(self.buffer)
                        self.buffer = ""
                else:
                    visible_text.append(self.buffer[:tag_idx])
                    self.inside_citation = True
                    self.buffer = self.buffer[tag_idx + len("<oai-mem-citation>"):]
            else:
                end_idx = self.buffer.find("</oai-mem-citation>")
                if end_idx == -1:
                    partial_idx = self.buffer.rfind("<")
                    if partial_idx != -1 and "</oai-mem-citation>".startswith(self.buffer[partial_idx:]):
                        self.current_citation_payload += self.buffer[:partial_idx]
                        self.buffer = self.buffer[partial_idx:]
                        break
                    else:
                        self.current_citation_payload += self.buffer
                        self.buffer = ""
                else:
                    self.current_citation_payload += self.buffer[:end_idx]
                    self.citations_extracted.append(self.current_citation_payload)
                    new_citations.append(self.current_citation_payload)
                    self.current_citation_payload = ""
                    self.inside_citation = False
                    self.buffer = self.buffer[end_idx + len("</oai-mem-citation>"):]

        return "".join(visible_text), new_citations

    def finish(self) -> Tuple[str, List[str]]:
        rem = self.buffer
        self.buffer = ""
        if self.inside_citation:
            return "", [self.current_citation_payload + rem]
        return rem, []

class TruncationPolicy:
    TOKENS = "tokens"
    BYTES = "bytes"

class HistoryCompactor:
    @staticmethod
    def is_summary_message(message: str) -> bool:
        return message.startswith(SUMMARY_PREFIX)

    def collect_user_messages(self, history: SessionHistory) -> List[str]:
        messages = []
        for item in history.items:
            if item.role == "user":
                if not self.is_summary_message(item.content):
                    messages.append(item.content)
        return messages

    def build_compacted_history(
        self,
        initial_context: List[ResponseItem],
        user_messages: List[str],
        summary_text: str
    ) -> List[ResponseItem]:
        initial_context_dicts = [item.to_dict() for item in initial_context]
        compacted_dicts = build_compacted_history(
            initial_context=initial_context_dicts,
            user_messages=user_messages,
            summary_text=summary_text,
            max_tokens=20000
        )
        # Re-wrap back to ResponseItem objects
        compacted_items = []
        for item_dict in compacted_dicts:
            compacted_items.append(ResponseItem(
                id=item_dict.get("id"),
                role=item_dict["role"],
                content=item_dict["content"],
                phase=item_dict.get("phase"),
                metadata=item_dict.get("metadata")
            ))
        return compacted_items

# Core verified standalone functions

def approx_token_count(text: str) -> int:
    """
    Coarse ceiling token approximation proxy (len(text) // 4).
    """
    char_len = len(text)
    if char_len == 0:
        return 0
    return (char_len + 3) // 4

def truncate_middle_with_budget(text: str, max_chars: int) -> str:
    char_len = len(text)
    if char_len <= max_chars:
        return text
        
    notice = "\n\n... [TRUNCATED DUE TO CONTEXT LIMITS] ...\n\n"
    notice_len = len(notice)
    
    if max_chars <= notice_len:
        return text[:max_chars]
        
    budget = max_chars - notice_len
    half_budget = budget // 2
    
    start_chunk = text[:half_budget]
    end_chunk = text[char_len - (budget - half_budget):]
    return f"{start_chunk}{notice}{end_chunk}"

def truncate_text(text: str, budget: int, policy: str = TruncationPolicy.TOKENS) -> str:
    if policy == TruncationPolicy.TOKENS:
        max_chars = budget * 4
    else:
        max_chars = budget
    return truncate_middle_with_budget(text, max_chars)

def is_summary_message(message: str) -> bool:
    return message.startswith(SUMMARY_PREFIX)

def collect_user_messages(history: Union[List[Dict[str, Any]], List[ResponseItem]]) -> List[str]:
    messages = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            role = item.role
            content = item.content
            
        if role == "user":
            if isinstance(content, list):
                text_pieces = []
                for p in content:
                    if isinstance(p, dict):
                        text_pieces.append(p.get("text") or p.get("input_text") or p.get("output_text") or "")
                    else:
                        text_pieces.append(str(p))
                content = "\n".join(text_pieces)
                
            if not is_summary_message(content):
                messages.append(content)
    return messages

def build_compacted_history(
    initial_context: list[dict],
    user_messages: list[str],
    summary_text: str,
    max_tokens: int = 20000
) -> list[dict]:
    selected_messages: List[str] = []
    if max_tokens > 0:
        remaining = max_tokens
        for message in reversed(user_messages):
            if remaining <= 0:
                break
            tokens = approx_token_count(message)
            if tokens <= remaining:
                selected_messages.append(message)
                remaining -= tokens
            else:
                truncated = truncate_text(message, remaining, policy=TruncationPolicy.TOKENS)
                selected_messages.append(truncated)
                break
                
        selected_messages.reverse()
        
    compacted_history = []
    
    # Preserve initial context dictionaries
    for item in initial_context:
        compacted_history.append(dict(item))
        
    # Inject compiled user messages
    for msg in selected_messages:
        compacted_history.append({
            "id": None,
            "role": "user",
            "content": msg,
            "phase": None,
            "metadata": {}
        })
        
    # Format warning-prefixed compaction summary message
    clean_summary, _ = strip_memory_citations(summary_text)
    
    if not clean_summary.strip():
        summary_payload = f"{SUMMARY_PREFIX}\n(no summary available)"
    elif not clean_summary.startswith(SUMMARY_PREFIX):
        summary_payload = f"{SUMMARY_PREFIX}\n{clean_summary}"
    else:
        summary_payload = clean_summary
        
    compacted_history.append({
        "id": None,
        "role": "user",
        "content": summary_payload,
        "phase": None,
        "metadata": {"compacted": True}
    })
    
    return compacted_history

def build_compaction_summary_text(summary_suffix: str) -> str:
    return f"{SUMMARY_PREFIX}\n{summary_suffix}"

def extract_proposed_plan_text(text: str) -> Optional[str]:
    pattern = r"\*\*\* Proposed Plan \*\*\*(.*?)\*\*\* End of Proposed Plan \*\*\*"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def strip_proposed_plan_blocks(text: str) -> str:
    pattern = r"\*\*\* Proposed Plan \*\*\*.*?\*\*\* End of Proposed Plan \*\*\*"
    return re.sub(pattern, "", text, flags=re.DOTALL)

def parse_memory_citation(citations: list[str]) -> Optional[dict]:
    if not citations:
        return None
        
    entries = []
    rollout_ids = []
    found_any = False
    
    for payload in citations:
        entries_match = re.search(r"<citation_entries>(.*?)</citation_entries>", payload, re.DOTALL)
        rollouts_match = re.search(r"<rollout_ids>(.*?)</rollout_ids>", payload, re.DOTALL)
        
        if not entries_match and not rollouts_match:
            continue
            
        found_any = True
        
        if entries_match:
            for line in entries_match.group(1).strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                path_range = parts[0].split(":")
                path = path_range[0]
                lines = path_range[1] if len(path_range) > 1 else ""
                note = parts[1].replace("note=", "") if len(parts) > 1 else ""
                entries.append({
                    "path": path,
                    "lines": lines,
                    "note": note
                })
                
        if rollouts_match:
            for line in rollouts_match.group(1).strip().split("\n"):
                line = line.strip()
                if line:
                    rollout_ids.append(line)
                    
    if not found_any:
        return None
        
    return {
        "entries": entries,
        "rollout_ids": rollout_ids
    }

def strip_memory_citations(text: str) -> tuple[str, list[str]]:
    parser = CitationStreamParser()
    visible_delta, cits_1 = parser.push_delta(text)
    visible_tail, cits_2 = parser.finish()
    return visible_delta + visible_tail, cits_1 + cits_2

def parse_command_actions(command: str) -> list[dict[str, Any]]:
    # 1. Tokenize using shlex with fallback splits
    try:
        tokens = shlex.split(command)
    except Exception:
        tokens = command.split()
        
    if not tokens:
        return []
        
    # Helper to strip bash/zsh wrappers recursively
    def strip_wrappers(toks: list[str]) -> list[str]:
        if len(toks) >= 3 and toks[0] in ("bash", "zsh", "sh", "pwsh", "cmd") and toks[1] in ("-c", "-lc", "/c"):
            nested = toks[2]
            try:
                nested_toks = shlex.split(nested)
                return strip_wrappers(nested_toks)
            except Exception:
                nested_toks = nested.split()
                return strip_wrappers(nested_toks)
        return toks
        
    tokens = strip_wrappers(tokens)
    
    # 2. Segment by logical operators (&&, ||, ;, |)
    segments = []
    current = []
    last_op = "start"
    for tok in tokens:
        if tok in ("&&", "||", ";", "|"):
            if current:
                segments.append((last_op, current))
                current = []
            last_op = tok
        else:
            current.append(tok)
    if current:
        segments.append((last_op, current))
        
    # 3. Replay segments tracking dynamic CWD context
    actions = []
    cwd_state = ""
    
    def clean_path(path_str: str) -> str:
        parts = []
        for p in path_str.split("/"):
            if p == "..":
                if parts and parts[-1] != "..":
                    parts.pop()
                else:
                    parts.append("..")
            elif p == "." or not p:
                continue
            else:
                parts.append(p)
        prefix = "/" if path_str.startswith("/") else ""
        return prefix + "/".join(parts)
        
    for op, cmd_toks in segments:
        if not cmd_toks:
            continue
            
        cmd_name = cmd_toks[0]
        cmd_str = " ".join(cmd_toks)
        
        # Handle 'cd' navigation to update CWD state
        if cmd_name == "cd":
            target = cmd_toks[1] if len(cmd_toks) > 1 else ""
            if target:
                if target.startswith("/"):
                    new_path = target
                else:
                    new_path = cwd_state + "/" + target if cwd_state else target
                cwd_state = clean_path(new_path)
            # Add to action list as unknown command action
            actions.append({
                "type": "unknown",
                "command": cmd_str
            })
            
        # Match 'read' category command (cat, less, etc.)
        elif cmd_name in ("cat", "less", "more", "head", "tail", "nano", "vim", "vi", "view", "bat"):
            target_arg = cmd_toks[1] if len(cmd_toks) > 1 else ""
            name = os.path.basename(target_arg) if target_arg else ""
            if target_arg:
                if target_arg.startswith("/"):
                    resolved_path = target_arg
                else:
                    resolved_path = cwd_state + "/" + target_arg if cwd_state else target_arg
                resolved_path = clean_path(resolved_path)
            else:
                resolved_path = ""
                
            actions.append({
                "type": "read",
                "command": cmd_str,
                "name": name,
                "path": resolved_path
            })
            
        # Match 'listFiles' category command (ls, find, etc.)
        elif cmd_name in ("ls", "find", "tree", "fd"):
            target_path = None
            for tok in cmd_toks[1:]:
                if not tok.startswith("-"):
                    target_path = tok
                    break
            actions.append({
                "type": "listFiles",
                "command": cmd_str,
                "path": target_path
            })
            
        # Match 'search' category command (grep, rg, etc.)
        elif cmd_name in ("grep", "egrep", "fgrep", "ripgrep", "rg", "ag", "ack"):
            non_flags = [t for t in cmd_toks[1:] if not t.startswith("-")]
            query = non_flags[0] if len(non_flags) >= 1 else None
            path = non_flags[1] if len(non_flags) >= 2 else None
            actions.append({
                "type": "search",
                "command": cmd_str,
                "query": query,
                "path": path
            })
            
        # Default category is unknown
        else:
            actions.append({
                "type": "unknown",
                "command": cmd_str
            })
            
    return actions

def reconstruct_history_from_rollout(source: Path | str | list[dict[str, Any]]) -> RolloutReconstruction:
    # 1. Resolve source contents
    events_list = []
    if isinstance(source, list):
        events_list = source
    elif isinstance(source, (Path, str)):
        path_obj = Path(source)
        if path_obj.is_file():
            with open(path_obj, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = str(source)
            
        # Parse content as JSONL stream of event dictionaries
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            events_list.append(json.loads(line))
            
    # 2. Convert raw event dictionaries to TransitionEvent objects
    events = []
    for payload in events_list:
        if "kind" not in payload or "data" not in payload:
            continue
        events.append(TransitionEvent(
            kind=payload["kind"],
            data=payload["data"],
            timestamp=payload.get("timestamp", 0)
        ))
        
    # 3. Replay events chronologically to reconstruct conversation history
    replayer = RolloutReconstructionEngine()
    history = replayer.replay(events)
    
    # 4. Map history elements into standard list of dictionaries
    history_dicts = [item.to_dict() for item in history.items]
    
    # 5. Extract plan
    plan_text = ""
    for item in history.items:
        extracted = extract_proposed_plan_text(item.content)
        if extracted:
            plan_text = extracted
            
    return RolloutReconstruction(
        history=history_dicts,
        plan=plan_text
    )

def prepare_prompt_history(history: list[dict[str, Any]], config: CodexConfig) -> list[dict[str, Any]]:
    prepared = []
    for item in history:
        msg = dict(item)
        for key in ["id", "phase", "metadata"]:
            msg.pop(key, None)
            
        content = msg.get("content")
        if isinstance(content, list):
            pieces = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("input_text") or part.get("output_text") or ""
                    if text:
                        pieces.append(text)
                else:
                    pieces.append(str(part))
            msg["content"] = "\n".join(pieces)
            
        prepared.append(msg)
    return prepared

def summarization_prompt() -> str:
    return (
        "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.\n\n"
        "Include:\n"
        "- Current progress and key decisions made\n"
        "- Important context, constraints, or user preferences\n"
        "- What remains to be done (clear next steps)\n"
        "- Any critical data, examples, or references needed to continue\n\n"
        "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
    )
