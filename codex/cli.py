from __future__ import annotations
import datetime
import json
import logging
import os
import re
import sys
import shlex
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

from codex.types import CodexConfig, CodexEvent, CodexResult
from codex.core import CodexSession, _MAX_AGENT_WAIT_TIMEOUT_MS, _MIN_AGENT_WAIT_TIMEOUT_MS
from codex.tools import ToolRuntime
from codex.model import ModelClient

logger = logging.getLogger("codex")

__all__ = [
    "CodexSession",
    "_AnsiStyle",
    "_HumanEventRenderer",
    "_LiveTurnStatusSnapshot",
    "_RolloutPickerRow",
    "_apply_prompt_escape_sequence",
    "_background_terminal_rows",
    "_format_elapsed_compact",
    "_format_tokens_compact",
    "_handle_interactive_slash_command",
    "_live_status_display_lines",
    "_main_chat",
    "_pygments_style_name",
    "_render_markdown_for_terminal",
    "_rollout_picker_display_lines",
    "_rollout_picker_rows",
    "_set_cli_syntax_theme",
    "_visible_len",
    "_wrap_ansi_line",
    "_ANSI_RE",
    "_CLI_SYNTAX_THEME",
    "main"
]


_ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')
_CLI_SYNTAX_THEME = "monokai"


class _InteractiveSlashResult:
    HANDLED: _InteractiveSlashResult
    IGNORED: _InteractiveSlashResult
    EXIT: _InteractiveSlashResult

    def __init__(self, handled: bool, session: CodexSession | None = None, exit: bool = False):
        self.handled = handled
        self.session = session
        self.exit = exit
        
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _InteractiveSlashResult):
            return (self.handled == other.handled and 
                    self.exit == other.exit and 
                    self.session is other.session)
        return False
        
    def __hash__(self) -> int:
        return hash((self.handled, self.exit, id(self.session)))

_InteractiveSlashResult.HANDLED = _InteractiveSlashResult(handled=True)
_InteractiveSlashResult.IGNORED = _InteractiveSlashResult(handled=False)
_InteractiveSlashResult.EXIT = _InteractiveSlashResult(handled=True, exit=True)


class _AnsiStyle:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def bold(self, text: str) -> str:
        return f"\x1b[1m{text}\x1b[22m" if self.enabled else text

    def dim(self, text: str) -> str:
        return f"\x1b[2m{text}\x1b[22m" if self.enabled else text

    def italic(self, text: str) -> str:
        return f"\x1b[3m{text}\x1b[23m" if self.enabled else text

    def strike(self, text: str) -> str:
        return f"\x1b[9m{text}\x1b[29m" if self.enabled else text

    def red(self, text: str) -> str:
        return f"\x1b[31m{text}\x1b[39m" if self.enabled else text

    def green(self, text: str) -> str:
        return f"\x1b[32m{text}\x1b[39m" if self.enabled else text

    def yellow(self, text: str) -> str:
        return f"\x1b[33m{text}\x1b[39m" if self.enabled else text

    def cyan(self, text: str) -> str:
        return f"\x1b[36m{text}\x1b[39m" if self.enabled else text

    def magenta(self, text: str) -> str:
        return f"\x1b[35m{text}\x1b[39m" if self.enabled else text


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub('', text))


def _wrap_ansi_line(text: str, width: int) -> list[str]:
    if _visible_len(text) <= width:
        return [text]
        
    pattern = re.compile(r'(\x1b\[[0-9;?]*[a-zA-Z])')
    parts = pattern.split(text)
    
    lines = []
    current_line_parts = []
    current_visible_len = 0
    active_styles = []
    
    for part in parts:
        if not part:
            continue
        if part.startswith('\x1b'):
            current_line_parts.append(part)
            if part == '\x1b[0m' or part == '\x1b[39m' or part == '\x1b[22m':
                active_styles.clear()
            else:
                active_styles.append(part)
        else:
            words = part.split(' ')
            for idx, word in enumerate(words):
                prefix_space = ' ' if idx > 0 else ''
                word_to_add = prefix_space + word
                word_len = len(word_to_add)
                
                if current_visible_len + word_len <= width:
                    current_line_parts.append(word_to_add)
                    current_visible_len += word_len
                else:
                    if active_styles:
                        current_line_parts.append('\x1b[0m')
                    lines.append(''.join(current_line_parts))
                    
                    current_line_parts = []
                    current_visible_len = 0
                    for style in active_styles:
                        current_line_parts.append(style)
                        
                    word_on_new_line = word
                    word_on_new_line_len = len(word_on_new_line)
                    
                    if word_on_new_line_len > width:
                        for char in word_on_new_line:
                            if current_visible_len + 1 > width:
                                if active_styles:
                                    current_line_parts.append('\x1b[0m')
                                lines.append(''.join(current_line_parts))
                                current_line_parts = []
                                current_visible_len = 0
                                for style in active_styles:
                                    current_line_parts.append(style)
                            current_line_parts.append(char)
                            current_visible_len += 1
                    else:
                        current_line_parts.append(word_on_new_line)
                        current_visible_len += word_on_new_line_len
                        
    if current_line_parts:
        if active_styles:
            current_line_parts.append('\x1b[0m')
        lines.append(''.join(current_line_parts))
        
    return lines


from codex.state import (
    is_local_path_like_link,
    render_local_link_target,
    strip_memory_citations,
    strip_proposed_plan_blocks,
    parse_memory_citation
)

def _render_inline_markdown(text: str, style: _AnsiStyle, cwd: Path | None = None) -> str:
    # 1. Parse markdown links: [label](url)
    def link_replacer(match: re.Match) -> str:
        label = match.group(1)
        url = match.group(2)
        if is_local_path_like_link(url):
            target = render_local_link_target(url, cwd)
            if target is not None:
                return style.cyan(target)
        underlined_url = f"\x1b[4m{url}\x1b[24m"
        return f"{label} ({style.cyan(underlined_url)})"

    text = re.sub(r'\[([^\]]*)\]\(([^)]*)\)', link_replacer, text)
    
    # 2. Parse inline code: `code` -> cyan
    text = re.sub(r'`([^`]+)`', lambda m: style.cyan(m.group(1)), text)
    
    # 3. Parse strong: **strong** -> bold
    text = re.sub(r'\*\*([^*]+)\*\*', lambda m: style.bold(m.group(1)), text)
    text = re.sub(r'__([^_]+)__', lambda m: style.bold(m.group(1)), text)
    
    # 4. Parse emphasis: *emp* -> italic
    text = re.sub(r'\*([^*]+)\*', lambda m: style.italic(m.group(1)), text)
    text = re.sub(r'_([^_]+)_', lambda m: style.italic(m.group(1)), text)
    
    # 5. Parse strikethrough: ~~strike~~ -> strike
    text = re.sub(r'~~([^~]+)~~', lambda m: style.strike(m.group(1)), text)
    
    return text


def _render_markdown_for_terminal(
    text: str,
    style: _AnsiStyle,
    *,
    emphasis: bool = True,
    terminal_width: int | None = None
) -> list[str]:
    lines = text.splitlines()
    rendered_lines = []
    in_code_block = False
    code_block_lines = []
    code_block_lang = None
    
    cwd = Path.cwd()

    for line in lines:
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                lang = line.strip()[3:].strip()
                code_block_lang = lang if lang else None
                code_block_lines = []
            else:
                in_code_block = False
                raw_code = "\n".join(code_block_lines)
                
                # Syntax highlight if optional Pygments is available
                highlighted_lines = []
                try:
                    import pygments
                    from pygments.lexers import get_lexer_by_name
                    from pygments.formatters import TerminalFormatter
                    lexer = None
                    if code_block_lang:
                        try:
                            lexer = get_lexer_by_name(code_block_lang)
                        except Exception:
                            pass
                    if not lexer:
                        lexer = get_lexer_by_name("text")
                    style_name = _pygments_style_name()
                    formatter = TerminalFormatter(style=style_name)
                    highlighted_code = pygments.highlight(raw_code, lexer, formatter)
                    highlighted_lines = highlighted_code.splitlines()
                except ImportError:
                    # Fallback plain cyan style
                    highlighted_lines = [style.cyan(l) for l in code_block_lines]
                    
                for hl in highlighted_lines:
                    rendered_lines.append(hl)
            continue
            
        if in_code_block:
            code_block_lines.append(line)
            continue
            
        # Headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2)
            header_text = _render_inline_markdown(header_text, style, cwd)
            
            if level == 1:
                rendered = style.bold(f"\x1b[4m{header_text}\x1b[24m")
            elif level == 2:
                rendered = style.bold(header_text)
            elif level == 3:
                rendered = style.bold(style.italic(header_text))
            else:
                rendered = style.italic(header_text)
                
            rendered_lines.append(rendered)
            continue
            
        # Blockquotes
        bq_match = re.match(r'^>\s*(.*)$', line)
        if bq_match:
            bq_text = bq_match.group(1)
            bq_text = _render_inline_markdown(bq_text, style, cwd)
            rendered = style.green(f"> {bq_text}")
            rendered_lines.append(rendered)
            continue
            
        # Lists
        ul_match = re.match(r'^([-*+])\s+(.+)$', line)
        if ul_match:
            item_text = ul_match.group(2)
            item_text = _render_inline_markdown(item_text, style, cwd)
            rendered = f"• {item_text}"
            rendered_lines.append(rendered)
            continue
            
        ol_match = re.match(r'^(\d+)\.\s+(.+)$', line)
        if ol_match:
            num = ol_match.group(1)
            item_text = ol_match.group(2)
            item_text = _render_inline_markdown(item_text, style, cwd)
            styled_num = style.cyan(f"{num}.")
            rendered = f"{styled_num} {item_text}"
            rendered_lines.append(rendered)
            continue
            
        # Normal line
        rendered = _render_inline_markdown(line, style, cwd)
        
        if terminal_width is not None:
            wrapped = _wrap_ansi_line(rendered, terminal_width)
            rendered_lines.extend(wrapped)
        else:
            rendered_lines.append(rendered)
            
    return rendered_lines


def _format_elapsed_compact(elapsed_seconds: int) -> str:
    seconds = max(0, elapsed_seconds)
    if seconds < 60:
        return f"{seconds}s"
        
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
        
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if hours >= 24:
        days = hours // 24
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours}h {remaining_minutes}m"
        
    if remaining_minutes == 0:
        return f"{hours}h"
    else:
        return f"{hours}h {remaining_minutes}m"


def _format_tokens_compact(value: int | float) -> str:
    val = max(0.0, float(value))
    if val == 0:
        return "0"
    if val < 1000.0:
        if val.is_integer():
            return str(int(val))
        return f"{val:.2f}".rstrip('0').rstrip('.')
        
    if val >= 1_000_000_000_000.0:
        scaled, suffix = val / 1_000_000_000_000.0, "T"
    elif val >= 1_000_000_000.0:
        scaled, suffix = val / 1_000_000_000.0, "B"
    elif val >= 1_000_000.0:
        scaled, suffix = val / 1_000_000.0, "M"
    else:
        scaled, suffix = val / 1_000.0, "K"
        
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


def _pygments_style_name(name: str | None = None) -> str:
    global _CLI_SYNTAX_THEME
    if name is not None:
        return name
    return _CLI_SYNTAX_THEME


def _set_cli_syntax_theme(name: str) -> bool:
    global _CLI_SYNTAX_THEME
    _CLI_SYNTAX_THEME = name
    return True


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
        self.path = Path(path)
        self.preview = preview
        self.thread_id = thread_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.cwd = cwd
        self.git_branch = git_branch

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "preview": self.preview,
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cwd": self.cwd,
            "git_branch": self.git_branch
        }


class _HumanEventRenderer:
    def __init__(self, *, color_mode: str = "auto", line_sink: Callable[[str], None] | None = None) -> None:
        # Resolve automatic color detection
        enabled = True
        if color_mode == "none":
            enabled = False
        elif color_mode == "auto":
            # standard check: is tty?
            enabled = sys.stdout.isatty()
            
        self.style = _AnsiStyle(enabled=enabled)
        self.line_sink = line_sink

    def _write_line(self, line: str) -> None:
        if self.line_sink is not None:
            self.line_sink(line)
        else:
            print(line, file=sys.stderr)

    def render_user_message(self, text: str) -> None:
        msg = f"{self.style.bold(self.style.green('User Message:'))} {text}"
        self._write_line(msg)

    def render_info_message(self, message: str) -> None:
        msg = f"{self.style.bold(self.style.cyan('Info:'))} {message}"
        print(msg, file=sys.stdout)

    def render_error(self, message: str) -> None:
        msg = f"{self.style.bold(self.style.red('Error:'))} {message}"
        self._write_line(msg)

    def render_interrupted(self) -> None:
        msg = self.style.bold(self.style.yellow("Execution interrupted by user."))
        self._write_line(msg)

    def render_pending_input_preview(self, text: str, *, active: bool) -> None:
        status = "Active" if active else "Queued"
        msg = f"{self.style.dim(f'[{status} Input Preview]:')} {text}"
        self._write_line(msg)

    def render(self, event: Any) -> None:
        # Decodes and renders state transitions
        if hasattr(event, "type") and hasattr(event, "payload"):
            evt_type = event.type
            evt_payload = event.payload
        elif isinstance(event, dict):
            evt_type = event.get("type", "")
            evt_payload = event.get("payload", {})
        else:
            return
            
        if evt_type == "turn_started":
            self.render_info_message(f"Turn started (turn_id: {evt_payload.get('turn_id', '')[:8]})")
            
        elif evt_type == "agent_message_content_delta":
            delta = evt_payload.get("text", "")
            # Print deltas directly to stdout
            sys.stdout.write(delta)
            sys.stdout.flush()
            
        elif evt_type == "item_started":
            item = evt_payload.get("item", {})
            itype = item.get("type", "")
            if itype in ("function_call", "custom_tool_call"):
                name = item.get("name")
                args = item.get("arguments")
                self._write_line(f"\n{self.style.bold(self.style.yellow('Running Tool:'))} {name} with {args}...")
                
        elif evt_type == "item_completed":
            item = evt_payload.get("item", {})
            itype = item.get("type", "")
            if itype in ("function_call_output", "custom_tool_call_output"):
                output = item.get("output", {})
                # Extract output text
                text = ""
                if isinstance(output, dict) and "body" in output:
                    body = output.get("body", {})
                    if isinstance(body, dict) and body.get("type") == "content_items":
                        text = "".join(c.get("text", "") for c in body.get("content", []) if c.get("type") == "input_text")
                # Format output nicely on terminal
                lines = _render_markdown_for_terminal(text, self.style, terminal_width=80)
                self._write_line(self.style.bold(self.style.green("Tool Output:")))
                for l in lines[:20]:  # Limit print size of huge stdout
                    self._write_line(f"  {l}")
                if len(lines) > 20:
                    self._write_line(self.style.dim(f"  ... ({len(lines) - 20} lines of output truncated) ..."))
                    
        elif evt_type == "token_count":
            usage = evt_payload.get("token_usage") or evt_payload
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            tot = inp + out
            self.render_info_message(f"Turn cost: {_format_tokens_compact(tot)} tokens ({_format_tokens_compact(inp)} in, {_format_tokens_compact(out)} out)")

    def finish(self, final_message: str, *, print_to_stdout: bool = True) -> None:
        # Strips plans and citations
        clean, citations = strip_memory_citations(final_message)
        clean = strip_proposed_plan_blocks(clean)
        
        if print_to_stdout:
            columns = int(os.environ.get("COLUMNS") or 90)
            lines = _render_markdown_for_terminal(clean, self.style, terminal_width=columns)
            self._write_line("")
            for idx, l in enumerate(lines):
                prefix = "• " if idx == 0 else "  "
                self._write_line(f"{prefix}{l}")
            self._write_line("")
            
            if citations:
                parsed = parse_memory_citation(citations)
                if parsed:
                    self._write_line(self.style.bold(self.style.green("Memory Citations:")))
                    for ent in parsed.get("entries", []):
                        path = ent.get("path")
                        lines_str = f"{ent.get('line_start')}-{ent.get('line_end')}"
                        note = ent.get("note")
                        self._write_line(f"  - {self.style.bold(path)}:{lines_str}  {self.style.dim(note)}")


def _apply_prompt_escape_sequence(buffer: str, cursor: int, sequence: bytes) -> tuple[str, int] | None:
    if sequence == b'\x01':  # Ctrl-A / Home (Legacy)
        sequence = b'\x1b[H'
    elif sequence == b'\x05':  # Ctrl-E / End (Legacy)
        sequence = b'\x1b[F'

    if sequence in (b'\x1b[D', b'\x1bOD'):  # Left Arrow
        return buffer, max(0, cursor - 1)
    if sequence in (b'\x1b[C', b'\x1bOC'):  # Right Arrow
        return buffer, min(len(buffer), cursor + 1)
    if sequence in (b'\x7f', b'\x08'):  # Backspace
        if cursor > 0:
            new_buffer = buffer[:cursor - 1] + buffer[cursor:]
            return new_buffer, cursor - 1
        return buffer, cursor
    if sequence == b'\x1b[3~':  # Delete
        if cursor < len(buffer):
            new_buffer = buffer[:cursor] + buffer[cursor + 1:]
            return new_buffer, cursor
        return buffer, cursor

    if sequence in (b'\x1b[H', b'\x1bOH'):  # Home / Start of Line
        # Find index of last \n before cursor, or 0
        last_nl = buffer.rfind("\n", 0, cursor)
        target = last_nl + 1 if last_nl != -1 else 0
        return buffer, target

    if sequence in (b'\x1b[F', b'\x1bOF'):  # End / End of Line
        # Find index of next \n after cursor, or end of buffer
        next_nl = buffer.find("\n", cursor)
        target = next_nl if next_nl != -1 else len(buffer)
        return buffer, target

    if sequence in (b'\x1b[A', b'\x1bOA', b'\x1b[B', b'\x1bOB'):  # Up / Down Arrows
        # Split into lines keeping start index of each line
        lines_meta = []
        curr = 0
        for line in buffer.split("\n"):
            lines_meta.append((curr, line))
            curr += len(line) + 1
            
        # Find which line cursor is on
        line_idx = 0
        for idx, (start_idx, line_text) in enumerate(lines_meta):
            if start_idx <= cursor <= start_idx + len(line_text):
                line_idx = idx
                break
                
        start_idx, line_text = lines_meta[line_idx]
        col = cursor - start_idx
        
        if sequence in (b'\x1b[A', b'\x1bOA'):  # Up
            if line_idx == 0:
                return buffer, cursor
            target_line_idx = line_idx - 1
        else:  # Down
            if line_idx == len(lines_meta) - 1:
                return buffer, cursor
            target_line_idx = line_idx + 1
            
        target_start, target_text = lines_meta[target_line_idx]
        target_col = min(col, len(target_text))
        return buffer, target_start + target_col

    return None


def _background_terminal_rows(session: CodexSession) -> list[tuple[int, str, bool, str]]:
    rows = []
    for sess_id, info in list(ToolRuntime._PROCESS_REGISTRY.items()):
        p = info.get("proc")
        pid = p.pid if (p and p.pid) else -1
        cmd = info.get("cmd") or ""
        is_running = (p.poll() is None) if p else False
        exit_code = p.returncode if p else -1
        status_string = "Running" if is_running else f"Exited ({exit_code})"
        rows.append((pid, cmd, is_running, status_string))
    return rows


def _live_status_display_lines(snapshot: _LiveTurnStatusSnapshot | None, style: _AnsiStyle) -> list[str]:
    if snapshot is None:
        return [""]
        
    elapsed_str = _format_elapsed_compact(snapshot.elapsed_seconds)
    header_part = style.bold(style.cyan(f"[{snapshot.header} | {elapsed_str}]"))
    
    context_str = "Context: "
    if snapshot.active_context_tokens is not None:
        c_tokens = _format_tokens_compact(snapshot.active_context_tokens)
        est_flag = " (est)" if snapshot.active_context_estimated else ""
        context_str += f"{c_tokens}{est_flag}"
    else:
        context_str += "unknown"
        
    if snapshot.context_window is not None:
        w_tokens = _format_tokens_compact(snapshot.context_window)
        context_str += f" / {w_tokens} window"
        if snapshot.active_context_tokens is not None:
            pct = (snapshot.active_context_tokens / snapshot.context_window) * 100
            context_str += f" ({pct:.1f}% used)"
            
    session_str = ""
    if snapshot.session_context_tokens is not None:
        s_tokens = _format_tokens_compact(snapshot.session_context_tokens)
        est_flag = " (est)" if snapshot.session_context_estimated else ""
        session_str += f"Total: {s_tokens}{est_flag}"
    if snapshot.session_reasoning_tokens is not None and snapshot.session_reasoning_tokens > 0:
        r_tokens = _format_tokens_compact(snapshot.session_reasoning_tokens)
        session_str += f" ({r_tokens} reasoning)"
        
    lines = [
        f"{header_part} {style.dim(context_str)}",
        f"  {style.dim(session_str)}" if session_str else ""
    ]
    return [l for l in lines if l]


def _rollout_picker_rows(config: CodexConfig) -> list[_RolloutPickerRow]:
    sessions_dir = Path(config.resolved_codex_home()) / "sessions"
    if not sessions_dir.exists():
        return []
        
    rows = []
    for path in sessions_dir.glob("**/*.jsonl"):
        if not path.is_file() or path.name.startswith("session_index"):
            continue
            
        try:
            stat = path.stat()
            created_at = stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_ctime
            updated_at = stat.st_mtime
            
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                continue
            meta = json.loads(first_line)
            if meta.get("type") != "session_meta":
                continue
            payload = meta.get("payload", {})
            inner = payload.get("meta")
            if isinstance(inner, dict):
                thread_id = inner.get("id") or inner.get("session_id") or path.stem
                cwd = inner.get("cwd")
                git = inner.get("git", {})
                branch = git.get("branch") if isinstance(git, dict) else None
            else:
                thread_id = payload.get("thread_id") or payload.get("id") or payload.get("session_id") or path.stem
                cwd = payload.get("cwd")
                git = payload.get("git", {})
                branch = git.get("branch") if isinstance(git, dict) else None
            
            preview = ""
            from codex.state import is_real_user_message
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("type") == "response_item":
                        p = r.get("payload", {})
                        if is_real_user_message(p):
                            content = p.get("content", [])
                            preview = "".join(c.get("text", "") for c in content if c.get("type") == "input_text")
                            preview = preview.strip().replace("\n", " ")[:60]
                            break
                            
            rows.append(_RolloutPickerRow(
                path=path,
                preview=preview if preview else "(no prompt context)",
                thread_id=thread_id,
                created_at=created_at,
                updated_at=updated_at,
                cwd=cwd,
                git_branch=branch
            ))
        except Exception:
            pass
            
    rows.sort(key=lambda r: r.updated_at, reverse=True)
    return rows


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
    filtered = []
    for row in rows:
        if not show_all and row.cwd and Path(row.cwd).resolve() != Path(cwd).resolve():
            continue
            
        match = True
        if query:
            q = query.lower()
            fields = [row.preview, str(row.path), row.thread_id, row.cwd or "", row.git_branch or ""]
            match = any(q in f.lower() for f in fields)
            
        if match:
            filtered.append(row)
            
    if sort_key == "created":
        filtered.sort(key=lambda r: r.created_at, reverse=True)
    else:
        filtered.sort(key=lambda r: r.updated_at, reverse=True)
        
    lines = []
    
    # 1. Header & Search Prompts
    lines.append(style.bold(f"=== {title} ==="))
    search_prompt = f"  Search: {query}" if query else "  Type to search"
    lines.append(style.dim(search_prompt))
    
    # 2. Filter & Sort Status bar
    filter_val = "All" if show_all else "Cwd"
    filter_label = style.green(f"Filter:[{filter_val}]") if not query else style.green(f"Filter:[Search: {query}]")
    sort_val = "Updated" if sort_key == "updated" else "Created"
    sort_label = style.cyan(f"Sort:[{sort_val}]")
    lines.append(f"  {filter_label}  |  {sort_label}")
    
    # 3. Instruction Hints
    action_hint = "fork" if "fork" in title.lower() else "resume"
    lines.append(style.dim(f"  enter {action_hint}  |  ctrl+o dense/comfortable  |  tab focus  |  esc exit"))
    lines.append("")
    
    if not filtered:
        lines.append(style.red("  No rollouts discovered matching the filters."))
        lines.append("")
        return lines
        
    page_size = 8 if density == "compact" else 5
    visible_slice = filtered[offset : offset + page_size]
    
    if offset > 0:
        lines.append(style.dim("  ↑ more"))
        lines.append("")
        
    for idx, row in enumerate(visible_slice):
        global_idx = offset + idx
        is_sel = (global_idx == selected)
        
        dt = datetime.datetime.fromtimestamp(row.updated_at)
        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        cwd_part = f" [{Path(row.cwd).name}]" if row.cwd else ""
        branch_part = f" ({row.git_branch})" if row.git_branch else ""
        
        prefix = style.green("❯ ") if is_sel else "  "
        row_line = f"{prefix}{style.bold(date_str)}{cwd_part}{branch_part}  {style.cyan(row.thread_id[:8])}"
        
        if is_sel:
            lines.append(style.bold(style.yellow(row_line)))
            preview_str = f"❯ {row.preview}"
            lines.append(style.bold(style.magenta(preview_str)))
            lines.append(style.dim(f"     id: {row.thread_id}"))
            if expanded:
                lines.append(style.dim(f"     Path: {row.path.as_posix()}"))
        else:
            lines.append(row_line)
            lines.append(style.dim(f"     Preview: {row.preview}"))
            
        if density != "compact":
            lines.append("")
            
    if offset + page_size < len(filtered):
        lines.append(style.dim("  ↓ more"))
        lines.append("")
        
    lines.append("")
    total = len(filtered)
    lines.append(style.dim(f"  Showing row {selected + 1} of {total} total matching rollouts"))
    
    return lines


def get_char_raw() -> bytes:
    import sys
    import tty
    import termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            import fcntl
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            try:
                ch += sys.stdin.read(2)
            except Exception:
                pass
            fcntl.fcntl(fd, fcntl.F_SETFL, fl)
        return ch.encode('utf-8')
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _handle_interactive_slash_command(
    session: CodexSession,
    prompt: str,
    *,
    color_mode: str = "auto",
    queued_prompts: deque[str] | None = None
) -> _InteractiveSlashResult:
    stripped = prompt.strip()
    if not stripped.startswith("/"):
        return _InteractiveSlashResult.IGNORED
        
    cmd_parts = stripped.split(" ", 1)
    cmd = cmd_parts[0].lower()
    arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
    
    renderer = _HumanEventRenderer(color_mode=color_mode)
    style = _AnsiStyle(enabled=(color_mode != "none"))
    
    if cmd in ("/exit", "/quit"):
        renderer.render_info_message("Exiting Codex REPL. Goodbye!")
        return _InteractiveSlashResult.EXIT
        
    elif cmd == "/help":
        help_text = (
            "Available slash commands:\n"
            "  /help      - Display this help information\n"
            "  /exit, /quit - Terminate the REPL session\n"
            "  /clear     - Clear terminal screen\n"
            "  /compact   - Compact current conversation history manually\n"
            "  /history   - View full history of turns in this session\n"
            "  /goal      - Show or update the session goal\n"
            "  /new       - Start a fresh session\n"
            "  /resume    - Resume a previous session\n"
            "  /fork      - Fork a previous session\n"
            "  /theme     - Set syntax highlighing theme\n"
            "  /ps        - Show active background subprocesses"
        )
        print(style.dim(help_text))
        return _InteractiveSlashResult(handled=True, session=session)
        
    elif cmd == "/new":
        new_sess = CodexSession(session.config, model_client=session.model_client)
        return _InteractiveSlashResult(handled=True, session=new_sess)
    elif cmd == "/resume":
        rollout_path = arg.strip() if arg else ""
        if rollout_path == "--last":
            rows = _rollout_picker_rows(session.config)
            if rows:
                rollout_path = rows[0].path
            else:
                rollout_path = ""
                
        if not rollout_path:
            cur_path = session.state.rollout_path()
            if cur_path.exists() and os.path.getsize(cur_path) > 0:
                rollout_path = cur_path
            else:
                # If non-tty headless CI fallback to latest rollout
                if not sys.stdin.isatty() or not sys.stdout.isatty():
                    rows = _rollout_picker_rows(session.config)
                    if rows:
                        rollout_path = rows[0].path
                else:
                    chosen = run_interactive_tui_picker(session.config, title="Resume a previous session")
                    if not chosen:
                        return _InteractiveSlashResult(handled=True, session=session)
                    rollout_path = chosen
                    
        if not rollout_path:
            print("No sessions found to resume.")
            return _InteractiveSlashResult(handled=True, session=session)
            
        new_sess = CodexSession.resume_from_rollout(rollout_path, session.config, session.model_client)
        return _InteractiveSlashResult(handled=True, session=new_sess)
        
    elif cmd == "/fork":
        rollout_path = arg.strip() if arg else ""
        if rollout_path == "--last":
            rows = _rollout_picker_rows(session.config)
            if rows:
                rollout_path = rows[0].path
            else:
                rollout_path = ""
                
        if not rollout_path:
            cur_path = session.state.rollout_path()
            if cur_path.exists() and os.path.getsize(cur_path) > 0:
                rollout_path = cur_path
            else:
                if not sys.stdin.isatty() or not sys.stdout.isatty():
                    rows = _rollout_picker_rows(session.config)
                    if rows:
                        rollout_path = rows[0].path
                else:
                    chosen = run_interactive_tui_picker(session.config, title="Fork a previous session")
                    if not chosen:
                        return _InteractiveSlashResult(handled=True, session=session)
                    rollout_path = chosen
                    
        if not rollout_path:
            print("No sessions found to fork.")
            return _InteractiveSlashResult(handled=True, session=session)
            
        new_sess = CodexSession.fork_from_rollout(rollout_path, session.config, session.model_client)
        return _InteractiveSlashResult(handled=True, session=new_sess)
        
    elif cmd == "/theme":
        theme_name = arg.strip() if arg else "monokai"
        _set_cli_syntax_theme(theme_name)
        return _InteractiveSlashResult(handled=True, session=session)
        
    elif cmd == "/ps":
        print("Background terminals:", file=sys.stderr)
        for sess_id, info in list(ToolRuntime._PROCESS_REGISTRY.items()):
            p = info.get("proc")
            pid = p.pid if (p and p.pid) else -1
            cmd_str = info.get("cmd") or ""
            is_running = (p.poll() is None) if p else False
            status = "Running" if is_running else "Exited"
            print(f"  - [{sess_id}] PID {pid} | {status} | {cmd_str}", file=sys.stderr)
        return _InteractiveSlashResult(handled=True, session=session)
        
    elif cmd == "/stop":
        num_stopped = len(ToolRuntime._PROCESS_REGISTRY)
        session.tools.interrupt_all()
        s_suffix = "s" if num_stopped != 1 else ""
        print(f"Stopped {num_stopped} background terminal{s_suffix}", file=sys.stderr)
        return _InteractiveSlashResult(handled=True, session=session)
        
    elif cmd == "/clear":
        new_sess = CodexSession(session.config, model_client=session.model_client)
        return _InteractiveSlashResult(handled=True, session=new_sess)
        
    elif cmd == "/compact":
        renderer.render_info_message("Triggering manual context compaction...")
        res = session.compact(arg if arg else None)
        renderer.render_info_message("Context compacted")
        return _InteractiveSlashResult(handled=True, session=session)
        
    elif cmd == "/history":
        renderer.render_info_message(f"=== Session History ({len(session.state.history)} items) ===")
        for idx, item in enumerate(session.state.history):
            role = item.get("role", "system")
            content = item.get("content", [])
            txt = "".join(c.get("text", "") for c in content if c.get("type") in ("input_text", "output_text"))
            print(f"[{idx}] {style.bold(role.upper())}: {txt[:200]}...")
        return _InteractiveSlashResult(handled=True, session=session)
        
    elif cmd == "/goal":
        goal = getattr(session.state, "goal", None) or "No goal configured."
        if arg:
            session.state.goal = arg
            renderer.render_info_message(f"Goal updated to: {arg}")
        else:
            print(style.cyan(f"Current Goal: {goal}"))
        return _InteractiveSlashResult.HANDLED
        
    else:
        print(style.red(f"Unknown slash command: {cmd}. Type /help for assistance."))
        return _InteractiveSlashResult.HANDLED


def run_interactive_tui_picker(config: CodexConfig, title: str = "Resume/Fork Session Rollout Picker") -> Path | None:
    # Launches keyboard-interactive rollout picker.
    # Fallback to simple numeric input if stdout/stdin is not a tty
    rows = _rollout_picker_rows(config)
    if not rows:
        print("No sessions found to resume/fork.")
        return None
        
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # Headless/CI numeric fallback!
        style = _AnsiStyle(enabled=False)
        print(f"=== {title} ===")
        for idx, r in enumerate(rows[:15]):
            dt = datetime.datetime.fromtimestamp(r.updated_at).strftime("%Y-%m-%d %H:%M")
            print(f"[{idx}] {dt} | {r.thread_id[:8]} | {r.preview}")
        try:
            val = input("Select rollout index to resume/fork: ").strip()
            sel_idx = int(val)
            if 0 <= sel_idx < len(rows):
                return rows[sel_idx].path
        except Exception:
            pass
        return None
        
    selected = 0
    offset = 0
    show_all = False
    query = ""
    sort_key = "updated"
    density = "default"
    toolbar_focus = "list"
    expanded = False
    
    style = _AnsiStyle(enabled=True)
    cwd = config.resolved_cwd()
    
    while True:
        # Clear screen
        os.system("clear")
        
        # Filter rows
        filtered = []
        for r in rows:
            if not show_all and r.cwd and Path(r.cwd).resolve() != cwd.resolve():
                continue
            match = True
            if query:
                q = query.lower()
                fields = [r.preview, str(r.path), r.thread_id, r.cwd or "", r.git_branch or ""]
                match = any(q in f.lower() for f in fields)
            if match:
                filtered.append(r)
                
        # Clamping
        selected = max(0, min(selected, len(filtered) - 1))
        page_size = 8 if density == "compact" else 5
        if selected < offset:
            offset = selected
        elif selected >= offset + page_size:
            offset = selected - page_size + 1
            
        lines = _rollout_picker_display_lines(
            rows=rows,
            title=title,
            style=style,
            cwd=cwd,
            show_all=show_all,
            query=query,
            sort_key=sort_key,
            selected=selected,
            offset=offset,
            density=density,
            toolbar_focus=toolbar_focus,
            expanded=expanded
        )
        for l in lines:
            print(l)
            
        # Focused bar highlight visualizer
        if toolbar_focus != "list":
            print(style.bold(style.yellow(f"  [Active Toolbar Focus: {toolbar_focus.upper()}] - Type query text live, press Tab to focus list")))
        else:
            print(style.bold(style.cyan(f"  [Active List Navigation] - Arrow keys to select, press Enter to load, Tab to focus toolbar")))
            
        # Get keypress
        try:
            ch = get_char_raw()
        except Exception:
            # Fallback block
            ch = b'q'
            
        if ch == b'q' or ch == b'\x03':  # q or Ctrl-C
            return None
            
        elif ch == b'\r' or ch == b'\n':  # Enter
            if filtered:
                return filtered[selected].path
            return None
            
        elif ch == b'\t':  # Tab
            # Rotate focus
            if toolbar_focus == "list":
                toolbar_focus = "query"
            elif toolbar_focus == "query":
                toolbar_focus = "sort"
            elif toolbar_focus == "sort":
                toolbar_focus = "show_all"
            else:
                toolbar_focus = "list"
                
        elif ch == b'\x1b[A':  # Up Arrow
            selected = max(0, selected - 1)
        elif ch == b'\x1b[B':  # Down Arrow
            if filtered:
                selected = min(len(filtered) - 1, selected + 1)
                
        elif ch == b'\x1b[D' or ch == b'\x1b[C':  # Arrow Left / Right
            if toolbar_focus == "sort":
                sort_key = "created" if sort_key == "updated" else "updated"
            elif toolbar_focus == "show_all":
                show_all = not show_all
                selected = 0
                offset = 0
                
        elif ch in (b'\x7f', b'\x08'):  # Backspace
            if toolbar_focus == "query" and query:
                query = query[:-1]
                selected = 0
                offset = 0
                
        else:
            # Add plain text keys to search query if query bar has focus
            if toolbar_focus == "query":
                try:
                    char_str = ch.decode('utf-8')
                    if char_str.isprintable():
                        query += char_str
                        selected = 0
                        offset = 0
                except Exception:
                    pass


def _main_chat(argv: list[str], *, prog: str = 'python -m codex') -> int:
    # Parses CLI
    import argparse
    
    # 0. Define parent parser for shared session arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--model", help="OpenAI model to use (default: resolved from catalog)")
    parent_parser.add_argument("--sandbox", choices=["workspace-write", "read-only", "danger-full-access"], default="workspace-write", help="Sandboxing policy")
    parent_parser.add_argument("--approval-policy", choices=["always", "never", "unless-trusted"], default="never", help="Approval policy")
    parent_parser.add_argument("--color", choices=["auto", "always", "none", "never"], default="auto", help="ANSI Color mode")
    parent_parser.add_argument("-C", "--cd", help="Working root directory")
    parent_parser.add_argument("--skip-git-repo-check", action="store_true", help="Skip Git repository checking")
    parent_parser.add_argument("--ephemeral", action="store_true", help="Run without persisting session files")

    parser = argparse.ArgumentParser(prog=prog, description="Codex CLI Python Port Agent", parents=[parent_parser])
    subparsers = parser.add_subparsers(dest="command")
    
    # 1. exec command
    exec_p = subparsers.add_parser("exec", help="Execute a single turn with the prompt", parents=[parent_parser])
    exec_p.add_argument("prompt", help="Prompt to execute")
    exec_p.add_argument("--dry-run", action="store_true", help="Format and print prompt without executing")
    
    # 2. resume command
    resume_p = subparsers.add_parser("resume", help="Resume an existing session", parents=[parent_parser])
    resume_p.add_argument("--path", help="Path to rollout file (jsonl) to resume")
    
    # 3. fork command
    fork_p = subparsers.add_parser("fork", help="Fork an existing session", parents=[parent_parser])
    fork_p.add_argument("--path", help="Path to rollout file (jsonl) to fork from")
    
    args = parser.parse_args(argv)
    
    # Initialize basic config and client
    cwd_path = Path(args.cd).resolve() if args.cd else Path.cwd()
    config = CodexConfig(
        model=args.model,
        sandbox=args.sandbox,
        approval_policy=args.approval_policy,
        cwd=cwd_path,
        skip_git_repo_check=args.skip_git_repo_check,
        ephemeral=args.ephemeral,
    )
    model_client = ModelClient()
    
    color_mode = args.color
    if color_mode == "never":
        color_mode = "none"
        
    renderer = _HumanEventRenderer(color_mode=color_mode)
    style = _AnsiStyle(enabled=(color_mode != "none" and (color_mode == "always" or sys.stdout.isatty())))
    
    # Determine session lifecycle
    session = None
    
    if args.command == "exec":
        # Exec turn run
        session = CodexSession(config, model_client=model_client)
        prompt_txt = args.prompt
        
        renderer.render_info_message("Executing dry-run single turn..." if args.dry_run else "Running single turn exec...")
        
        # Start turn execution stream
        stream_events = session.stream(prompt_txt, dry_run=args.dry_run)
        for evt in stream_events:
            renderer.render(evt)
            
        last_msg = session.last_message()
        if last_msg:
            renderer.finish(last_msg)
        return 0
        
    elif args.command == "resume":
        rollout_path = args.path
        if not rollout_path:
            # TUI picker resume!
            rollout_path = run_interactive_tui_picker(config, title="Resume a previous session")
            if not rollout_path:
                return 0
        session = CodexSession.resume_from_rollout(rollout_path, config, model_client)
        renderer.render_info_message(f"Resumed session successfully (thread_id: {session.state.thread_id[:8]})")
        
    elif args.command == "fork":
        rollout_path = args.path
        if not rollout_path:
            # TUI picker fork!
            rollout_path = run_interactive_tui_picker(config, title="Fork a previous session")
            if not rollout_path:
                return 0
        session = CodexSession.fork_from_rollout(rollout_path, config, model_client)
        renderer.render_info_message(f"Forked new session successfully (thread_id: {session.state.thread_id[:8]}, parent: {session.state.forked_from_id[:8] if session.state.forked_from_id else 'None'})")
        
    else:
        # Standard default interactive REPL starting from scratch!
        session = CodexSession(config, model_client=model_client)
        
    # Start Interactive TUI REPL Loop
    renderer.render_info_message(f"=== Codex CLI Terminal REPL session (model: {session.config.model}) ===")
    print(style.dim("  Type slash commands (e.g. /help) or prompt questions. Press Ctrl-C to exit."))
    print("")
    
    if args.command in ("resume", "fork"):
        last_assistant_msg = ""
        for item in reversed(session.state.history):
            if item.get("type") == "message" and item.get("role") == "assistant":
                content = item.get("content", [])
                last_assistant_msg = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if last_assistant_msg:
                    break
        if last_assistant_msg:
            renderer.finish(last_assistant_msg)
    
    while True:
        try:
            prompt = input(style.bold(style.green("› "))).strip()
            if not prompt:
                continue
                
            # Triage slash commands
            if prompt.startswith("/"):
                res = _handle_interactive_slash_command(session, prompt, color_mode=color_mode)
                if res == _InteractiveSlashResult.EXIT:
                    break
                continue
                
            # Run stream
            start_t = time.time()
            stream_events = session.stream(prompt)
            for evt in stream_events:
                renderer.render(evt)
                
            # Turn finalized
            last_msg = session.last_message()
            if last_msg:
                renderer.finish(last_msg)
                
        except (KeyboardInterrupt, EOFError):
            print("")
            renderer.render_info_message("Exiting Codex REPL. Goodbye!")
            break
            
    return 0


def main() -> int:
    return _main_chat(sys.argv[1:])
