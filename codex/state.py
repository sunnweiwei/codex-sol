from __future__ import annotations
import json
import logging
import uuid
import datetime
import os
import re
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence
from codex.types import CodexConfig, CodexEvent, CodexResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger("codex")

CODEX_ROLLOUT_ITEM_TYPES = frozenset({"session_meta", "response_item", "compacted", "turn_context", "event_msg"})

__all__ = [
    "CodexState",
    "RolloutReconstruction",
    "build_compacted_history",
    "build_compaction_summary_text",
    "extract_proposed_plan_text",
    "parse_command_actions",
    "parse_memory_citation",
    "prepare_prompt_history",
    "reconstruct_history_from_rollout",
    "strip_memory_citations",
    "strip_proposed_plan_blocks",
    "summarization_prompt",
    "CODEX_ROLLOUT_ITEM_TYPES",
    "seek_sequence",
    "normalise",
    "is_local_path_like_link",
    "render_local_link_target"
]


# Helper: exact unicode-byte token truncator from prompts.py
from codex.prompts import truncate_text, ASSETS_DIR, build_base_instructions

RESIZED_IMAGE_BYTES_ESTIMATE = 7373

_ORIGINAL_IMAGE_ESTIMATE_CACHE: dict[str, int] = {}

def estimate_original_image_bytes(image_url: str) -> int | None:
    key = hashlib.sha1(image_url.encode('utf-8')).hexdigest()
    if key in _ORIGINAL_IMAGE_ESTIMATE_CACHE:
        return _ORIGINAL_IMAGE_ESTIMATE_CACHE[key]
        
    res = parse_base64_image_data_url(image_url)
    if res is None:
        return None
        
    try:
        import base64
        decoded = base64.b64decode(res)
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(decoded))
            width, height = img.size
            patch_size = 32
            patches_wide = (width + patch_size - 1) // patch_size
            patches_high = (height + patch_size - 1) // patch_size
            patch_count = min(10000, patches_wide * patches_high)
            est_bytes = patch_count * 4
            _ORIGINAL_IMAGE_ESTIMATE_CACHE[key] = est_bytes
            return est_bytes
        except ImportError:
            return RESIZED_IMAGE_BYTES_ESTIMATE
    except Exception:
        return None
        
def parse_base64_image_data_url(url: str) -> str | None:
    if not url.lower().startswith("data:"):
        return None
    if "," not in url:
        return None
    metadata, payload = url.split(",", 1)
    if not metadata.lower().startswith("data:image/"):
        return None
    parts = metadata[5:].split(";")
    if "base64" not in [p.lower() for p in parts]:
        return None
    return payload

def normalise(s: str) -> str:
    s = s.strip()
    replacements = {
        '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-', '\u2015': '-', '\u2212': '-',
        '\u2018': '\'', '\u2019': '\'', '\u201a': '\'', '\u201b': '\'',
        '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
        '\u00a0': ' ', '\u2002': ' ', '\u2003': ' ', '\u2004': ' ', '\u2005': ' ', '\u2006': ' ',
        '\u2007': ' ', '\u2008': ' ', '\u2009': ' ', '\u200a': ' ', '\u202f': ' ', '\u205f': ' ',
        '\u3000': ' '
    }
    res = []
    for c in s:
        res.append(replacements.get(c, c))
    return "".join(res)

def seek_sequence(lines: list[str], pattern: list[str], start: int, eof: bool) -> int | None:
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None
        
    search_start = len(lines) - len(pattern) if (eof and len(lines) >= len(pattern)) else start
    
    # 1. Exact match
    for i in range(search_start, len(lines) - len(pattern) + 1):
        if lines[i : i + len(pattern)] == pattern:
            return i
            
    # 2. Rstrip match
    for i in range(search_start, len(lines) - len(pattern) + 1):
        ok = True
        for p_idx, pat in enumerate(pattern):
            if lines[i + p_idx].rstrip('\r\n\t ') != pat.rstrip('\r\n\t '):
                ok = False
                break
        if ok:
            return i
            
    # 3. Trim match
    for i in range(search_start, len(lines) - len(pattern) + 1):
        ok = True
        for p_idx, pat in enumerate(pattern):
            if lines[i + p_idx].strip() != pat.strip():
                ok = False
                break
        if ok:
            return i
            
    # 4. Normalise match
    for i in range(search_start, len(lines) - len(pattern) + 1):
        ok = True
        for p_idx, pat in enumerate(pattern):
            if normalise(lines[i + p_idx]) != normalise(pat):
                ok = False
                break
        if ok:
            return i
            
    return None

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
        
    path_part = Path(path_part).resolve()
    if cwd is not None:
        try:
            path_part = path_part.relative_to(cwd.resolve())
        except ValueError:
            pass
            
    rel_path = path_part.as_posix()
    loc_str = ""
    if hash_part:
        range_match = re.match(r'^L(\d+)C(\d+)-L(\d+)C(\d+)$', hash_part)
        if range_match:
            l1, c1, l2, c2 = range_match.groups()
            loc_str = f":{l1}:{c1}-{l2}:{c2}"
        else:
            range_match2 = re.match(r'^L(\d+)-L(\d+)$', hash_part)
            if range_match2:
                l1, l2 = range_match2.groups()
                loc_str = f":{l1}-{l2}"
            else:
                line_col_match = re.match(r'^L(\d+)C(\d+)$', hash_part)
                if line_col_match:
                    l, c = line_col_match.groups()
                    loc_str = f":{l}:{c}"
                else:
                    line_match = re.match(r'^L(\d+)$', hash_part)
                    if line_match:
                        l = line_match.group(1)
                        loc_str = f":{l}"
                        
    return f"{rel_path}{loc_str}"

def estimate_item_token_count(item: dict[str, Any]) -> int:
    model_visible_bytes = estimate_response_item_model_visible_bytes(item)
    return (model_visible_bytes + 3) // 4

def estimate_response_item_model_visible_bytes(item: dict[str, Any]) -> int:
    item_type = item.get("type", "")
    
    # 1. Reasoning / Compaction / ContextCompaction base64 length recovery
    if item_type in ("reasoning", "compaction", "context_compaction") and "encrypted_content" in item:
        enc = item.get("encrypted_content")
        if enc is None:
            return 0
        if isinstance(enc, str):
            enc_len = len(enc)
        else:
            enc_len = len(json.dumps(enc))
            
        est_bytes = (enc_len * 3) // 4 - 650
        return max(0, est_bytes)
        
    # 2. Standard serialization
    raw_serialized = json.dumps(item)
    raw_len = len(raw_serialized.encode('utf-8'))
    
    payload_bytes, replacement_bytes = image_data_url_estimate_adjustment(item)
    if payload_bytes == 0 or replacement_bytes == 0:
        return raw_len
    else:
        return max(0, raw_len - payload_bytes + replacement_bytes)

def image_data_url_estimate_adjustment(item: dict[str, Any]) -> tuple[int, int]:
    payload_bytes = 0
    replacement_bytes = 0
    
    def accumulate(url: str, detail: str | None = None):
        nonlocal payload_bytes, replacement_bytes
        res = parse_base64_image_data_url(url)
        if res is not None:
            payload_len = len(res)
            payload_bytes += payload_len
            if detail is not None and detail.lower() == "original":
                est = estimate_original_image_bytes(url)
                replacement_bytes += est if est is not None else RESIZED_IMAGE_BYTES_ESTIMATE
            else:
                replacement_bytes += RESIZED_IMAGE_BYTES_ESTIMATE
                
    item_type = item.get("type", "")
    if item_type == "message":
        content = item.get("content", [])
        for c in content:
            if isinstance(c, dict) and c.get("type") == "input_image":
                accumulate(c.get("image_url", ""), c.get("detail"))
                
    elif item_type in ("function_call_output", "custom_tool_call_output"):
        output = item.get("output", {})
        if isinstance(output, dict) and "body" in output:
            body = output.get("body", {})
            if isinstance(body, dict) and body.get("type") == "content_items":
                items = body.get("content", [])
                for oc in items:
                    if isinstance(oc, dict) and oc.get("type") == "input_image":
                        accumulate(oc.get("image_url", ""), oc.get("detail"))
                        
    return payload_bytes, replacement_bytes


def is_contextual_user_message_content(content: list[dict[str, Any]]) -> bool:
    contextual_prefixes = (
        "# AGENTS.md instructions for",
        "<environment_context>",
        "<skills_instructions>",
        "<user_shell_command>",
        "<turn_aborted>",
        "<subagent_notification>",
        "<goal_context>",
        "<personality_spec>",
        "<permissions instructions>",
        "<model_switch>",
        "<collaboration_mode>",
        "<realtime_conversation>",
    )
    for item in content:
        if isinstance(item, dict) and item.get("type") == "input_text":
            text = item.get("text", "").strip()
            if any(text.startswith(prefix) for prefix in contextual_prefixes):
                return True
    return False


def is_user_turn_boundary(item: dict[str, Any]) -> bool:
    if not isinstance(item, dict) or item.get("type") != "message":
        return False
    role = item.get("role", "")
    content = item.get("content", [])
    
    if role == "user":
        return not is_contextual_user_message_content(content)
        
    if role == "assistant":
        # Multi-agent collaboration spans
        for c in content:
            if isinstance(c, dict) and c.get("type") == "input_text":
                text = c.get("text", "")
                if "<subagent_" in text or "<inter_agent_" in text:
                    return True
    return False


def drop_last_n_user_turns(history_list: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    res = list(history_list)
    for _ in range(n):
        last_idx = -1
        for idx in range(len(res) - 1, -1, -1):
            if is_user_turn_boundary(res[idx]):
                last_idx = idx
                break
        if last_idx != -1:
            res = res[:last_idx]
        else:
            break
    return res


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


def strip_memory_citations(text: str) -> tuple[str, list[str]]:
    visible = []
    citations = []
    
    idx = 0
    length = len(text)
    in_tag = False
    current_citation = []
    
    while idx < length:
        if not in_tag:
            if text.startswith("<oai-mem-citation>", idx):
                in_tag = True
                current_citation = []
                idx += len("<oai-mem-citation>")
            else:
                visible.append(text[idx])
                idx += 1
        else:
            if text.startswith("</oai-mem-citation>", idx):
                in_tag = False
                citations.append("".join(current_citation))
                idx += len("</oai-mem-citation>")
            else:
                current_citation.append(text[idx])
                idx += 1
                
    if in_tag:
        citations.append("".join(current_citation))
        
    return "".join(visible), citations


def parse_memory_citation(citations: list[str]) -> dict[str, Any] | None:
    entries = []
    rollout_ids = []
    seen_rollout_ids = set()
    
    for citation in citations:
        if "<citation_entries>" in citation and "</citation_entries>" in citation:
            _, rest = citation.split("<citation_entries>", 1)
            entries_block, _ = rest.split("</citation_entries>", 1)
            for line in entries_block.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    if "|note=[" in line:
                        location, note_part = line.rsplit("|note=[", 1)
                        if note_part.endswith("]"):
                            note = note_part[:-1].strip()
                            if ":" in location:
                                path, line_range = location.rsplit(":", 1)
                                if "-" in line_range:
                                    line_start_str, line_end_str = line_range.split("-", 1)
                                    entries.append({
                                        "path": path.strip(),
                                        "line_start": int(line_start_str.strip()),
                                        "line_end": int(line_end_str.strip()),
                                        "note": note,
                                    })
                except Exception:
                    pass
                    
        ids_block = None
        if "<rollout_ids>" in citation and "</rollout_ids>" in citation:
            _, rest = citation.split("<rollout_ids>", 1)
            ids_block, _ = rest.split("</rollout_ids>", 1)
        elif "<thread_ids>" in citation and "</thread_ids>" in citation:
            _, rest = citation.split("<thread_ids>", 1)
            ids_block, _ = rest.split("</thread_ids>", 1)
            
        if ids_block is not None:
            for line in ids_block.splitlines():
                line = line.strip()
                if line and line not in seen_rollout_ids:
                    seen_rollout_ids.add(line)
                    rollout_ids.append(line)
                    
    if not entries and not rollout_ids:
        return None
    return {
        "entries": entries,
        "rollout_ids": rollout_ids
    }


def build_compaction_summary_text(summary_suffix: str) -> str:
    prefix_path = ASSETS_DIR / "prompts" / "compact" / "summary_prefix.md"
    if prefix_path.exists():
        with open(prefix_path, "r", encoding="utf-8") as f:
            prefix = f.read().strip()
    else:
        prefix = "Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:"
        
    return prefix + "\n\n" + summary_suffix


def build_compacted_history(
    initial_context: list[dict[str, Any]],
    user_messages: list[str],
    summary_text: str,
    max_tokens: int = 20000
) -> list[dict[str, Any]]:
    history = [dict(item) for item in initial_context]
    
    selected_messages = []
    if max_tokens > 0:
        remaining = max_tokens
        for message in reversed(user_messages):
            if remaining == 0:
                break
            tokens = (len(message) + 3) // 4
            if tokens <= remaining:
                selected_messages.append(message)
                remaining -= tokens
            else:
                truncated = truncate_text(message, remaining, use_tokens=True)
                selected_messages.append(truncated)
                break
        selected_messages.reverse()
        
    for message in selected_messages:
        history.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": message}],
        })
        
    summary_to_use = summary_text if summary_text else "(no summary available)"
    history.append({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": summary_to_use}],
    })
    
    return history


def parse_command_actions(command: str) -> list[dict[str, Any]]:
    subcommands = re.split(r'&&|;|\|\|', command)
    
    actions = []
    for sub in subcommands:
        sub = sub.strip()
        if not sub:
            continue
            
        tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', sub)
        if not tokens:
            continue
            
        binary = tokens[0]
        if binary.startswith(('"', "'")) and binary.endswith(('"', "'")):
            binary = binary[1:-1]
            
        bin_name = Path(binary).name
        
        def clean_token(tok: str) -> str:
            if tok.startswith(('"', "'")) and tok.endswith(('"', "'")):
                return tok[1:-1]
            return tok

        if bin_name in ("cat", "less", "head", "tail", "view", "read_file", "view_file"):
            path_arg = ""
            for tok in tokens[1:]:
                cleaned = clean_token(tok)
                if not cleaned.startswith('-'):
                    path_arg = cleaned
                    break
            if not path_arg:
                path_arg = "unknown"
            
            abs_path = Path(path_arg).absolute()
            actions.append({
                "type": "read",
                "command": sub,
                "name": bin_name,
                "path": abs_path.as_posix()
            })
            
        elif bin_name in ("ls", "find", "list_dir"):
            path_arg = None
            for tok in tokens[1:]:
                cleaned = clean_token(tok)
                if not cleaned.startswith('-'):
                    path_arg = cleaned
                    break
            actions.append({
                "type": "listFiles",
                "command": sub,
                "path": path_arg
            })
            
        elif bin_name in ("grep", "rg", "ripgrep", "grep_search"):
            pattern = None
            path_arg = None
            for tok in tokens[1:]:
                cleaned = clean_token(tok)
                if not cleaned.startswith('-'):
                    if pattern is None:
                        pattern = cleaned
                    else:
                        path_arg = cleaned
                        break
            actions.append({
                "type": "search",
                "command": sub,
                "query": pattern,
                "path": path_arg
            })
            
        else:
            actions.append({
                "type": "unknown",
                "command": sub
            })
            
    return actions


from codex.model import iter_model_stream_events, collect_stream_response
# Normalization helper imported here to keep single dependency
from codex.prompts import build_permissions_instructions


def prepare_prompt_history(history: list[dict[str, Any]], config: CodexConfig) -> list[dict[str, Any]]:
    items = [dict(item) for item in history]
    
    missing_outputs = []
    for idx, item in enumerate(items):
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            has_output = any(
                i.get("type") == "function_call_output" and i.get("call_id") == call_id
                for i in items
            )
            if not has_output:
                missing_outputs.append((idx, {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": {"type": "text", "text": "aborted"}
                }))
        elif item_type == "custom_tool_call":
            call_id = item.get("call_id")
            has_output = any(
                i.get("type") == "custom_tool_call_output" and i.get("call_id") == call_id
                for i in items
            )
            if not has_output:
                missing_outputs.append((idx, {
                    "type": "custom_tool_call_output",
                    "call_id": call_id,
                    "name": item.get("name"),
                    "output": {"type": "text", "text": "aborted"}
                }))
        elif item_type == "local_shell_call":
            call_id = item.get("call_id")
            if call_id:
                has_output = any(
                    i.get("type") == "function_call_output" and i.get("call_id") == call_id
                    for i in items
                )
                if not has_output:
                    missing_outputs.append((idx, {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": {"type": "text", "text": "aborted"}
                    }))
        elif item_type == "tool_search_call":
            call_id = item.get("call_id")
            if call_id:
                has_output = any(
                    i.get("type") == "tool_search_output" and i.get("call_id") == call_id
                    for i in items
                )
                if not has_output:
                    missing_outputs.append((idx, {
                        "type": "tool_search_output",
                        "call_id": call_id,
                        "status": "completed",
                        "execution": "client",
                        "tools": []
                    }))
                    
    for idx, output_item in reversed(missing_outputs):
        items.insert(idx + 1, output_item)
        
    function_call_ids = {i.get("call_id") for i in items if i.get("type") in ("function_call", "local_shell_call") if i.get("call_id")}
    custom_tool_call_ids = {i.get("call_id") for i in items if i.get("type") == "custom_tool_call" if i.get("call_id")}
    tool_search_call_ids = {i.get("call_id") for i in items if i.get("type") == "tool_search_call" if i.get("call_id")}
    
    filtered_items = []
    for item in items:
        item_type = item.get("type")
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if call_id not in function_call_ids:
                continue
        elif item_type == "custom_tool_call_output":
            call_id = item.get("call_id")
            if call_id not in custom_tool_call_ids:
                continue
        elif item_type == "tool_search_output":
            execution = item.get("execution", "client")
            call_id = item.get("call_id")
            if execution == "client" and call_id not in tool_search_call_ids:
                continue
        filtered_items.append(item)
    items = filtered_items
    
    supports_images = config.resolved_supports_image_input()
    if not supports_images:
        placeholder = "image content omitted because you do not support image input"
        for item in items:
            item_type = item.get("type")
            if item_type == "message":
                content = item.get("content", [])
                normalized_content = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "input_image":
                        normalized_content.append({
                            "type": "input_text",
                            "text": placeholder
                        })
                    else:
                        normalized_content.append(c)
                item["content"] = normalized_content
                
            elif item_type in ("function_call_output", "custom_tool_call_output"):
                output = item.get("output", {})
                if isinstance(output, dict) and "body" in output:
                    body = output.get("body", {})
                    if isinstance(body, dict) and body.get("type") == "content_items":
                        normalized_output_content = []
                        for oc in body.get("content", []):
                            if isinstance(oc, dict) and oc.get("type") == "input_image":
                                normalized_output_content.append({
                                    "type": "input_text",
                                    "text": placeholder
                                })
                            else:
                                normalized_output_content.append(oc)
                        body["content"] = normalized_output_content
                    
            elif item_type == "image_generation_call":
                if "result" in item:
                    item["result"] = []
                    
    return items


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


def reconstruct_history_from_rollout(source: Path | str | list[dict[str, Any]]) -> RolloutReconstruction:
    records = []
    if isinstance(source, (Path, str)):
        path = Path(source)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            except Exception as e:
                logger.error(f"Failed to read rollout from {path}: {e}")
    else:
        records = list(source)
        
    history = []
    previous_turn_settings = None
    reference_context_item = None
    session_meta = None
    legacy_compaction_without_replacement = False
    
    for rec in records:
        rec_type = rec.get("type")
        payload = rec.get("payload", {})
        
        if rec_type == "session_meta":
            session_meta = payload
            
        elif rec_type == "turn_context":
            reference_context_item = payload
            previous_turn_settings = {
                "model": payload.get("model", "gpt-5.5"),
                "realtime_active": payload.get("realtime_active", False)
            }
            
        elif rec_type == "response_item":
            history.append(payload)
            
        elif rec_type == "compacted":
            repl = payload.get("replacement_history")
            if repl is not None:
                history = [dict(item) for item in repl]
                reference_context_item = None
            else:
                legacy_compaction_without_replacement = True
                reference_context_item = None
                user_messages = []
                for item in history:
                    if is_user_turn_boundary(item):
                        content = item.get("content", [])
                        txt_parts = []
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "input_text":
                                txt_parts.append(c.get("text", ""))
                        user_messages.append("".join(txt_parts))
                history = build_compacted_history(
                    initial_context=[],
                    user_messages=user_messages,
                    summary_text=payload.get("message", "")
                )
                
        elif rec_type == "event_msg":
            evt_type = payload.get("type")
            evt_payload = payload.get("payload", {})
            
            if evt_type in ("thread_rolled_back", "thread.rolled_back"):
                num_turns = int(evt_payload.get("num_turns", 1))
                history = drop_last_n_user_turns(history, num_turns)
                
    return RolloutReconstruction(
        history=history,
        previous_turn_settings=previous_turn_settings,
        reference_context_item=reference_context_item,
        session_meta=session_meta,
        legacy_compaction_without_replacement_history=legacy_compaction_without_replacement
    )


def summarization_prompt() -> str:
    path = ASSETS_DIR / "prompts" / "compact" / "prompt.md"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CodexState:
    def __init__(
        self,
        config: CodexConfig,
        thread_id: str = None,
        turn_id: str = None,
        installation_id: str = None,
        forked_from_id: str | None = None,
        history: list[dict[str, Any]] = None,
        events: list[CodexEvent] = None,
        memory_citations: list[dict[str, Any]] = None,
        previous_turn_settings: dict[str, Any] | None = None,
        reference_context_item: dict[str, Any] | None = None,
        last_token_usage: dict[str, Any] | None = None,
        total_token_usage: int = 0,
        session_reasoning_tokens: int = 0,
        context_carryover_tokens: int = 0,
        context_carryover_estimated: bool = False,
        *args: Any,
        **kwargs: Any
    ):
        self.config = config
        self.thread_id = thread_id if thread_id is not None else str(uuid.uuid4())
        self.turn_id = turn_id if turn_id is not None else str(uuid.uuid4())
        self.installation_id = installation_id if installation_id is not None else str(uuid.uuid4())
        self.forked_from_id = forked_from_id
        self.history = history if history is not None else []
        self.events = events if events is not None else []
        self.memory_citations = memory_citations if memory_citations is not None else []
        self.previous_turn_settings = previous_turn_settings
        self.reference_context_item = reference_context_item
        self.last_token_usage = last_token_usage
        self.total_token_usage = total_token_usage
        self.session_reasoning_tokens = session_reasoning_tokens
        self.context_carryover_tokens = context_carryover_tokens
        self.context_carryover_estimated = context_carryover_estimated
        self.turn_diff = None
        
        # Save custom properties
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Resolve rollout path
        if hasattr(self, "_rollout_path") and self._rollout_path is not None:
            pass
        else:
            now = datetime.datetime.utcnow()
            year = str(now.year)
            month = f"{now.month:02d}"
            day = f"{now.day:02d}"
            date_str = now.strftime("%Y-%m-%dT%H-%M-%S")
            
            home = self.config.resolved_codex_home()
            dir_path = home / "sessions" / year / month / day
            filename = f"rollout-{date_str}-{self.thread_id}.jsonl"
            self._rollout_path = dir_path / filename

    def rollout_path(self) -> Path:
        return self._rollout_path

    def append_rollout_record(self, item_type: str, payload: dict[str, Any]) -> None:
        path = self.rollout_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        row = {"type": item_type, "payload": payload}
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.error(f"Failed to append rollout record: {e}")

    def read_rollout_records(self) -> list[dict[str, Any]]:
        path = self.rollout_path()
        if not path.exists():
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read rollout records: {e}")
        return records

    def append_history(self, item: dict[str, Any]) -> None:
        self.history.append(item)
        self.append_rollout_record("response_item", item)

    def approx_history_tokens(self) -> int:
        return sum(estimate_item_token_count(item) for item in self.history)

    def active_context_tokens(self) -> int:
        return self.approx_history_tokens() + self.context_carryover_tokens

    def active_context_token_status(self) -> tuple[int | None, bool]:
        return (self.active_context_tokens(), self.context_carryover_estimated)

    def session_context_token_status(self) -> tuple[int | None, bool]:
        # Returns current active context tokens in session, along with carryover estimated status
        return (self.active_context_tokens(), self.context_carryover_estimated)

    def session_usage_tokens(self) -> int | None:
        return self.total_token_usage

    def session_reasoning_usage_tokens(self) -> int | None:
        return self.session_reasoning_tokens

    def record_token_usage(self, usage: dict[str, Any] | None) -> None:
        self.last_token_usage = usage
        if usage is not None:
            input_t = usage.get("input_tokens", 0)
            output_t = usage.get("output_tokens", 0)
            reason_t = usage.get("reasoning_tokens", 0)
            self.total_token_usage = input_t + output_t
            self.session_reasoning_tokens = reason_t

    def recompute_token_usage_from_history(self) -> None:
        records = self.read_rollout_records()
        for rec in records:
            if rec.get("type") == "event_msg":
                payload = rec.get("payload", {})
                evt_type = payload.get("type")
                evt_payload = payload.get("payload", {})
                if evt_type == "token_count":
                    usage = evt_payload.get("token_usage") or evt_payload
                    self.record_token_usage(usage)

    def emit(self, event_type: str, **payload: object) -> CodexEvent:
        event = CodexEvent(type=event_type, payload=payload)
        self.events.append(event)
        # UTC isoformat date string to match offsetdatetime now_utc
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        self.append_rollout_record("event_msg", {
            "type": event_type,
            "payload": payload,
            "timestamp": now_str
        })
        return event

    def start_turn(self) -> None:
        self.turn_id = str(uuid.uuid4())
        self.emit("turn_started", turn_id=self.turn_id, thread_id=self.thread_id)

    def start_new_context_epoch(self, carryover_tokens: int | None = None, *, estimated: bool = False) -> None:
        self.context_carryover_tokens = carryover_tokens if carryover_tokens is not None else 0
        self.context_carryover_estimated = estimated
        self.emit("context_compacted", carryover_tokens=self.context_carryover_tokens, estimated=estimated)

    def compact_with_remote_history(
        self,
        compacted_history: list[dict[str, Any]],
        initial_context: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        self.history = [dict(item) for item in compacted_history]
        self.append_rollout_record("compacted", {
            "message": "Remote context compaction completed.",
            "replacement_history": self.history
        })
        self.reference_context_item = None
        self.start_new_context_epoch(carryover_tokens=self.approx_history_tokens(), estimated=False)
        return self.history

    def compact_with_summary(
        self,
        summary_suffix: str,
        initial_context: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        summary_text = build_compaction_summary_text(summary_suffix)
        
        user_messages = []
        for item in self.history:
            if is_user_turn_boundary(item):
                content = item.get("content", [])
                txt_parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "input_text":
                        txt_parts.append(c.get("text", ""))
                user_messages.append("".join(txt_parts))
                
        self.history = build_compacted_history(
            initial_context=initial_context or [],
            user_messages=user_messages,
            summary_text=summary_text,
            max_tokens=20000
        )
        self.append_rollout_record("compacted", {
            "message": summary_text,
            "replacement_history": self.history
        })
        self.reference_context_item = None
        self.start_new_context_epoch(carryover_tokens=self.approx_history_tokens(), estimated=False)
        return self.history

    def record_apply_patch_turn_diff(self, metadata: Any) -> str | None:
        if isinstance(metadata, str):
            diff = metadata
        else:
            diff = str(metadata)
            
        self.turn_diff = diff
        self.emit("turn_diff", diff=diff)
        return diff

    def record_memory_citation(self, citation: dict[str, Any]) -> None:
        self.memory_citations.append(citation)

    def token_usage_info(self) -> dict[str, Any] | None:
        return self.last_token_usage

    def prompt_history(self) -> list[dict[str, Any]]:
        return prepare_prompt_history(self.history, self.config)

    def estimate_token_count_with_base_instructions(self) -> int:
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
        base_tokens = (len(base_text) + 3) // 4
        items_tokens = sum(estimate_item_token_count(item) for item in self.history)
        return base_tokens + items_tokens

    def write_last_message(self, message: str) -> None:
        path = self.config.resolved_output_last_message()
        if path is not None:
            clean_msg, _ = strip_memory_citations(message)
            clean_msg = strip_proposed_plan_blocks(clean_msg)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(clean_msg)
            except Exception as e:
                logger.error(f"Failed to write last message file at {path}: {e}")
