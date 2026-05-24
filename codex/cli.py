from __future__ import annotations
import datetime
import json
import logging
import os
import readline
import fcntl
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
        return f"\x1b[1m{text}\x1b[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return f"\x1b[2m{text}\x1b[0m" if self.enabled else text

    def italic(self, text: str) -> str:
        return f"\x1b[3m{text}\x1b[0m" if self.enabled else text

    def strike(self, text: str) -> str:
        return f"\x1b[9m{text}\x1b[0m" if self.enabled else text

    def red(self, text: str) -> str:
        return f"\x1b[31m{text}\x1b[0m" if self.enabled else text

    def green(self, text: str) -> str:
        return f"\x1b[32m{text}\x1b[0m" if self.enabled else text

    def yellow(self, text: str) -> str:
        return f"\x1b[33m{text}\x1b[0m" if self.enabled else text

    def cyan(self, text: str) -> str:
        return f"\x1b[36m{text}\x1b[0m" if self.enabled else text

    def magenta(self, text: str) -> str:
        return f"\x1b[35m{text}\x1b[0m" if self.enabled else text


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
                    if current_visible_len > 0:
                        if active_styles:
                            current_line_parts.append('\x1b[0m')
                        lines.append(''.join(current_line_parts))
                        
                        current_line_parts = []
                        current_visible_len = 0
                        for style in active_styles:
                            current_line_parts.append(style)
                            
                    word_on_new_line = word_to_add.strip() if current_visible_len == 0 else word
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
    text = re.sub(r'(?<![a-zA-Z0-9])__([^_]+)__(?![a-zA-Z0-9])', lambda m: style.bold(m.group(1)), text)
    
    # 4. Parse emphasis: *emp* -> italic
    text = re.sub(r'\*([^*]+)\*', lambda m: style.italic(m.group(1)), text)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])', lambda m: style.italic(m.group(1)), text)
    
    # 5. Parse strikethrough: ~~strike~~ -> strike
    text = re.sub(r'~~([^~]+)~~', lambda m: style.strike(m.group(1)), text)
    
    return text


def _render_markdown_table(table_lines: list[str], style: _AnsiStyle, terminal_width: int | None = None) -> list[str]:
    if len(table_lines) < 2:
        return table_lines
        
    term_w = terminal_width if terminal_width is not None else 80
    
    parsed_rows = []
    for line in table_lines:
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) >= 2 and parts[0] == "":
            parts.pop(0)
        if len(parts) >= 1 and parts[-1] == "":
            parts.pop()
        parsed_rows.append(parts)
        
    col_cnt = max(len(r) for r in parsed_rows)
    if col_cnt == 0:
        return table_lines
        
    divider_row = None
    if len(parsed_rows) >= 2:
        row2 = parsed_rows[1]
        is_div = all(re.match(r'^:?-+:?$', cell) for cell in row2)
        if is_div:
            divider_row = parsed_rows.pop(1)
            
    headers = parsed_rows.pop(0) if parsed_rows else []
    data_rows = parsed_rows
    
    alignments = []
    for c in range(col_cnt):
        align = "left"
        if divider_row and c < len(divider_row):
            cell = divider_row[c]
            left_colon = cell.startswith(":")
            right_colon = cell.endswith(":")
            if left_colon and right_colon:
                align = "center"
            elif right_colon:
                align = "right"
        alignments.append(align)
        
    def visual_len(t: str) -> int:
        return sum(2 if ord(char) > 127 else 1 for char in t)
        
    def wrap_cell_text(t: str, max_w: int) -> list[str]:
        lines_out = []
        curr_line = ""
        curr_len = 0
        for char in t:
            w = 2 if ord(char) > 127 else 1
            if curr_len + w > max_w:
                lines_out.append(curr_line)
                curr_line = char
                curr_len = w
            else:
                curr_line += char
                curr_len += w
        if curr_line or not lines_out:
            lines_out.append(curr_line)
        return lines_out
        
    natural_widths = []
    for c in range(col_cnt):
        max_w = 0
        if c < len(headers):
            max_w = max(max_w, visual_len(headers[c]))
        for r in data_rows:
            if c < len(r):
                max_w = max(max_w, visual_len(r[c]))
        natural_widths.append(max(1, max_w))
        
    available_w = max(10, term_w - (col_cnt * 3 + 1))
    col_widths = list(natural_widths)
    tot_natural = sum(col_widths)
    if tot_natural > available_w:
        for c in range(col_cnt):
            col_widths[c] = max(5, int(natural_widths[c] * available_w / tot_natural))
            
    wrapped_headers = []
    for c in range(col_cnt):
        h_text = headers[c] if c < len(headers) else ""
        wrapped_headers.append(wrap_cell_text(h_text, col_widths[c]))
        
    wrapped_data_rows = []
    for r in data_rows:
        wrapped_row = []
        for c in range(col_cnt):
            cell_text = r[c] if c < len(r) else ""
            wrapped_row.append(wrap_cell_text(cell_text, col_widths[c]))
        wrapped_data_rows.append(wrapped_row)
        
    def pad_cell(t: str, target_w: int, align: str) -> str:
        curr_len = visual_len(t)
        rem = max(0, target_w - curr_len)
        if align == "right":
            return " " * rem + t
        elif align == "center":
            left = rem // 2
            right = rem - left
            return " " * left + t + " " * right
        else:
            return t + " " * rem
            
    res_lines = []
    
    top_parts = []
    for c in range(col_cnt):
        top_parts.append("─" * (col_widths[c] + 2))
    res_lines.append("┌" + "┬".join(top_parts) + "┐")
    
    h_max_lines = max(len(w) for w in wrapped_headers)
    for l_idx in range(h_max_lines):
        cell_parts = []
        for c in range(col_cnt):
            cell_lines = wrapped_headers[c]
            t = cell_lines[l_idx] if l_idx < len(cell_lines) else ""
            padded = pad_cell(t, col_widths[c], alignments[c])
            cell_parts.append(f" {style.bold(padded)} ")
        res_lines.append("│" + "│".join(cell_parts) + "│")
        
    sep_parts = []
    for c in range(col_cnt):
        sep_parts.append("─" * (col_widths[c] + 2))
    res_lines.append("├" + "┼".join(sep_parts) + "┤")
    
    for r in wrapped_data_rows:
        r_max_lines = max(len(w) for w in r)
        for l_idx in range(r_max_lines):
            cell_parts = []
            for c in range(col_cnt):
                cell_lines = r[c]
                t = cell_lines[l_idx] if l_idx < len(cell_lines) else ""
                padded = pad_cell(t, col_widths[c], alignments[c])
                cell_parts.append(f" {padded} ")
            res_lines.append("│" + "│".join(cell_parts) + "│")
            
    bot_parts = []
    for c in range(col_cnt):
        bot_parts.append("─" * (col_widths[c] + 2))
    res_lines.append("└" + "┴".join(bot_parts) + "┘")
    
    return res_lines


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

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        
        # Detect markdown table block (starts with | and ends with |)
        if not in_code_block and line.strip().startswith("|") and line.strip().endswith("|"):
            table_lines = []
            while idx < len(lines) and lines[idx].strip().startswith("|") and lines[idx].strip().endswith("|"):
                table_lines.append(lines[idx])
                idx += 1
            table_rendered = _render_markdown_table(table_lines, style, terminal_width)
            rendered_lines.extend(table_rendered)
            continue
            
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                lang = line.strip()[3:].strip()
                code_block_lang = lang if lang else None
                code_block_lines = []
            else:
                in_code_block = False
                raw_code = "\n".join(code_block_lines)
                
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
                    highlighted_lines = [style.cyan(l) for l in code_block_lines]
                    
                for hl in highlighted_lines:
                    rendered_lines.append(hl)
            idx += 1
            continue
            
        if in_code_block:
            code_block_lines.append(line)
            idx += 1
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
            idx += 1
            continue
            
        # Blockquotes
        bq_match = re.match(r'^>\s*(.*)$', line)
        if bq_match:
            bq_text = bq_match.group(1)
            bq_text = _render_inline_markdown(bq_text, style, cwd)
            rendered = style.green(f"> {bq_text}")
            rendered_lines.append(rendered)
            idx += 1
            continue
            
        # Lists
        ul_match = re.match(r'^([-*+])\s+(.+)$', line)
        if ul_match:
            item_text = ul_match.group(2)
            item_text = _render_inline_markdown(item_text, style, cwd)
            rendered = f"• {item_text}"
            rendered_lines.append(rendered)
            idx += 1
            continue
            
        ol_match = re.match(r'^(\d+)\.\s+(.+)$', line)
        if ol_match:
            num = ol_match.group(1)
            item_text = ol_match.group(2)
            item_text = _render_inline_markdown(item_text, style, cwd)
            styled_num = style.cyan(f"{num}.")
            rendered = f"{styled_num} {item_text}"
            rendered_lines.append(rendered)
            idx += 1
            continue
            
        # Normal line
        rendered = _render_inline_markdown(line, style, cwd)
        
        if terminal_width is not None:
            wrapped = _wrap_ansi_line(rendered, terminal_width)
            rendered_lines.extend(wrapped)
        else:
            rendered_lines.append(rendered)
            
        idx += 1
        
    return rendered_lines


def _format_elapsed_compact(elapsed_seconds: int) -> str:
    seconds = max(0, elapsed_seconds)
    if seconds == 0:
        return "0s"
    if seconds < 60:
        return f"{seconds}s"
        
    minutes = seconds // 60
    rem_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {rem_seconds:02d}s"
        
    hours = minutes // 60
    rem_minutes = minutes % 60
    if hours < 24:
        return f"{hours}h {rem_minutes:02d}m {rem_seconds:02d}s"
        
    days = hours // 24
    rem_hours = hours % 24
    return f"{days}d {rem_hours}h {rem_minutes:02d}m {rem_seconds:02d}s"


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
    target = name if name is not None else _CLI_SYNTAX_THEME
    if target == "monokai-extended":
        return "monokai"
    return target


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
    def __init__(self, *, color_mode: str = "auto", line_sink: Callable[[str], None] | None = None, is_repl: bool = False) -> None:
        # Resolve automatic color detection
        enabled = True
        if color_mode in ("none", "never"):
            enabled = False
        elif color_mode == "auto":
            # standard check: is tty?
            enabled = sys.stdout.isatty()
            
        self.style = _AnsiStyle(enabled=enabled)
        self.line_sink = line_sink
        self.is_repl = is_repl
        self.has_work_cells = False
        self.current_turn_explorations = []
        self.active_tool_names = {}
        self.active_tool_args = {}

    def _write_line(self, line: str) -> None:
        if self.line_sink is not None:
            self.line_sink(line)
        else:
            print(line, file=sys.stderr)
            sys.stderr.flush()
            sys.stdout.flush()

    def render_user_message(self, text: str) -> None:
        raw_lines = text.split("\n")
        wrapped_lines = []
        for line in raw_lines:
            wrapped_lines.extend(_wrap_ansi_line(line, 80))
        if not wrapped_lines:
            return
        # Gutter prefix: › 
        self._write_line(f"{self.style.bold(self.style.green('› ' + wrapped_lines[0]))}")
        for l in wrapped_lines[1:]:
            self._write_line(f"  {l}")

    def render_info_message(self, message: str) -> None:
        if message.startswith("Executing dry-run") or message.startswith("Running single turn") or message.startswith("Turn started"):
            return
        self._write_line(self.style.dim(f"• {message}"))

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
            
        # Normalize type dot vs underscore notation to be 100% resilient
        norm_type = evt_type.replace("_", ".")
        
        if norm_type == "turn.started":
            self.render_info_message(f"Turn started (turn_id: {evt_payload.get('turn_id', '')[:8]})")
            
        elif norm_type == "agent.message.content.delta":
            delta = evt_payload.get("text", "")
            sys.stdout.write(delta)
            sys.stdout.flush()
            
        elif norm_type == "tool.started" or norm_type == "item.started":
            item = evt_payload.get("item") or evt_payload
            call_id = item.get("call_id") or item.get("id") or evt_payload.get("call_id") or evt_payload.get("id")
            name = item.get("name") or evt_payload.get("name")
            args = item.get("arguments") or evt_payload.get("arguments")
            
            if call_id and name:
                self.active_tool_names[call_id] = name
            if call_id and args:
                self.active_tool_args[call_id] = args
            return
            
        elif norm_type == "item.completed":
            item = evt_payload.get("item") or evt_payload
            itype = item.get("type", "") or evt_payload.get("type", "")
            
            # Custom TUI render cell: reasoning summary!
            if itype == "reasoning":
                summary = item.get("summary", [])
                text = ""
                if summary:
                    text = "".join(s.get("text", "") for s in summary if isinstance(s, dict))
                if text:
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    if paragraphs:
                        first = paragraphs[0]
                        bold_match = re.match(r'^\*\*([^*]+)\*\*(.*)$', first, re.DOTALL)
                        if bold_match:
                            header_txt = bold_match.group(1).strip()
                            rem_body = bold_match.group(2).strip()
                        else:
                            header_txt = "Inspecting memory and commands"
                            rem_body = first
                            
                        # Format bold title: "• bold_title: "
                        title_prefix = f"• {self.style.bold(header_txt)}: "
                        title_prefix_plain = f"• {header_txt}: "
                        title_len = len(title_prefix_plain)
                        
                        columns = int(os.environ.get("COLUMNS") or 72)
                        
                        # First chunk body limit and subsequent line rest limit
                        first_limit = max(10, columns - title_len)
                        rest_limit = max(10, columns - 2)
                        
                        # Gather and split body text
                        body_list = []
                        if rem_body:
                            body_list.append(rem_body)
                        for p in paragraphs[1:]:
                            body_list.append(p)
                        full_body = "\n\n".join(body_list)
                        
                        body_paras = full_body.split("\n\n")
                        wrapped_lines = []
                        
                        for p_idx, para in enumerate(body_paras):
                            para_lines = para.splitlines()
                            para_wrapped = []
                            for l_idx, line in enumerate(para_lines):
                                if p_idx == 0 and l_idx == 0:
                                    chunks = _wrap_ansi_line(line, first_limit)
                                    if chunks:
                                        para_wrapped.append(chunks[0])
                                        if len(chunks) > 1:
                                            rem_str = " ".join(chunks[1:])
                                            para_wrapped.extend(_wrap_ansi_line(rem_str, rest_limit))
                                else:
                                    para_wrapped.extend(_wrap_ansi_line(line, rest_limit))
                            wrapped_lines.extend(para_wrapped)
                            if p_idx < len(body_paras) - 1:
                                wrapped_lines.append("")
                                
                        # Print
                        first_body_line = wrapped_lines.pop(0) if wrapped_lines else ""
                        self._write_line(f"{title_prefix}{first_body_line}")
                        for l in wrapped_lines:
                            if l == "":
                                self._write_line("")
                            else:
                                self._write_line(f"  {l}")
                                
                        self.has_work_cells = True
                return
                
            # Custom TUI render cell: web_search_call
            if itype == "web_search_call":
                query = item.get("query") or item.get("action", {}).get("query") or ""
                self._write_line(f"• Searched {query}")
                self.has_work_cells = True
                return
                    
        elif norm_type == "turn.completed" or norm_type == "turn_completed":
            if self.is_repl:
                usage = evt_payload.get("usage") or {}
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                tot = inp + out
                self._write_line(self.style.dim(f"Turn cost: {_format_tokens_compact(tot)} tokens ({_format_tokens_compact(inp)} in, {_format_tokens_compact(out)} out)"))
            
        elif norm_type == "tool.completed":
            call_id = evt_payload.get("call_id") or evt_payload.get("id")
            name = self.active_tool_names.get(call_id) or evt_payload.get("name")
            args = self.active_tool_args.get(call_id) or evt_payload.get("arguments")
            
            # Custom TUI render cell: exec_command!
            if name == "exec_command":
                if isinstance(args, str):
                    try:
                        args_dict = json.loads(args)
                    except Exception:
                        args_dict = {"cmd": args}
                else:
                    args_dict = args if isinstance(args, dict) else {}
                cmd = args_dict.get("cmd", "")
                raw_out = evt_payload.get("output") or ""
                
                # Group exploration commands
                from codex.state import parse_command_actions
                actions = parse_command_actions(cmd)
                is_explore = all(a["type"] in ("list_files", "read", "search") for a in actions)
                
                if is_explore:
                    self.current_turn_explorations.extend(actions)
                    self.has_work_cells = True
                    return
                else:
                    # Print non-exploration command tree block
                    term_w = int(os.environ.get("COLUMNS") or 80)
                    chunks = _wrap_ansi_line(f"Ran {cmd}", term_w - 2)
                    if chunks:
                        c0 = chunks[0]
                        if c0.startswith("Ran "):
                            c0 = c0[4:]
                        self._write_line(f"{self.style.green('•')} {self.style.bold('Ran')} {c0}")
                        for idx in range(1, len(chunks)):
                            prefix = "  └ " if idx == len(chunks) - 1 else "  │ "
                            self._write_line(f"{prefix}{chunks[idx]}")
                            
                    # Non-exploration command output gets custom tree formatting!
                    out_lines = raw_out.splitlines()
                    if out_lines:
                        self._write_line(f"  └ {out_lines[0]}")
                        for l in out_lines[1:20]:
                            self._write_line(f"    {l}")
                        if len(out_lines) > 20:
                            self._write_line(self.style.dim(f"    ... ({len(out_lines) - 20} lines of output truncated) ..."))
                            
                    # Print separator spacing blank line exactly as expected by the test!
                    self._write_line("")
                    self.has_work_cells = True
                    return
                    
            # Custom TUI render cells: update_plan, request_user_input
            if name == "update_plan":
                metadata = evt_payload.get("metadata", {})
                explanation = metadata.get("explanation")
                plan = metadata.get("plan", [])
                
                self._write_line(f"• {self.style.bold('Updated Plan')}")
                indented = []
                if explanation:
                    indented.append(self.style.dim(self.style.italic(explanation)))
                    
                for step in plan:
                    st = step.get("step", "")
                    status = step.get("status", "pending")
                    if status == "completed":
                        marker = self.style.green("✔ ")
                        step_str = self.style.dim(st)
                    else:
                        marker = "□ "
                        step_str = st
                    indented.append(f"{marker}{step_str}")
                    
                if not plan:
                    indented.append(self.style.dim(self.style.italic("(no steps provided)")))
                    
                for idx, text in enumerate(indented):
                    prefix = self.style.dim("  └ ") if idx == 0 else "    "
                    self._write_line(f"{prefix}{text}")
                    
            elif name == "request_user_input":
                metadata = evt_payload.get("metadata", {})
                questions = metadata.get("questions", [])
                answers = metadata.get("answers", {})
                
                q_len = len(questions)
                a_len = sum(1 for q in questions if q.get("id") in answers)
                
                self._write_line(f"• {self.style.bold(f'Questions {a_len}/{q_len} answered')}")
                for q in questions:
                    q_id = q.get("id")
                    question_txt = q.get("question", "")
                    self._write_line(f"  • {question_txt}")
                    if q_id in answers:
                        ans_data = answers[q_id]
                        ans_list = ans_data.get("answers") if isinstance(ans_data, dict) else ans_data
                        ans_str = ", ".join(ans_list) if isinstance(ans_list, list) else str(ans_list)
                        self._write_line(f"    answer: {ans_str}")
                        if f"{q_id}_other" in answers:
                            self._write_line(f"    note: {answers[f'{q_id}_other']}")
                        
        elif norm_type == "turn.diff" or norm_type == "turn_diff":
            diff_text = evt_payload.get("diff") or evt_payload.get("unified_diff") or ""
            if not diff_text:
                return
                
            lines = diff_text.splitlines()
            idx = 0
            while idx < len(lines):
                line = lines[idx]
                if line.startswith("--- a/"):
                    del_file = line[6:]
                    add_file = ""
                    if idx + 1 < len(lines) and lines[idx + 1].startswith("+++ b/"):
                        add_file = lines[idx + 1][6:]
                        idx += 2
                    else:
                        idx += 1
                        
                    rel_path = add_file if add_file else del_file
                    try:
                        p = Path(rel_path)
                        if p.is_absolute():
                            rel_path = str(p.relative_to(Path.cwd()))
                        else:
                            rel_path = p.name
                    except Exception:
                        rel_path = Path(rel_path).name
                        
                    adds = 0
                    dels = 0
                    hunk_lines = []
                    
                    hunk_idx = idx
                    while hunk_idx < len(lines):
                        hl = lines[hunk_idx]
                        if hl.startswith("diff --git") or hl.startswith("--- a/"):
                           break
                        if hl.startswith("@@"):
                            hunk_lines.append(hl)
                        elif hl.startswith("+"):
                            adds += 1
                            hunk_lines.append(hl)
                        elif hl.startswith("-"):
                            dels += 1
                            hunk_lines.append(hl)
                        else:
                            hunk_lines.append(hl)
                        hunk_idx += 1
                        
                    self._write_line(f"• Edited {rel_path} (+{adds} -{dels})")
                    
                    h_idx = 0
                    while h_idx < len(hunk_lines):
                        hl = hunk_lines[h_idx]
                        if hl.startswith("@@"):
                            try:
                                parts = hl.split("@@")
                                nums_part = parts[1].strip()
                                del_part, add_part = nums_part.split(" ")
                                del_start = int(del_part.split(",")[0].replace("-", ""))
                                add_start = int(add_part.split(",")[0].replace("+", ""))
                            except Exception:
                                del_start = 1
                                add_start = 1
                                
                            h_idx += 1
                            curr_del = del_start
                            curr_add = add_start
                            
                            while h_idx < len(hunk_lines):
                                h_sub = hunk_lines[h_idx]
                                if h_sub.startswith("@@"):
                                    break
                                    
                                if h_sub.startswith("-"):
                                    content = h_sub[1:]
                                    colored = self.style.red(f"-{content}")
                                    self._write_line(f"    {curr_del} {colored}")
                                    curr_del += 1
                                    h_idx += 1
                                elif h_sub.startswith("+"):
                                    content = h_sub[1:]
                                    colored = self.style.green(f"+{content}")
                                    self._write_line(f"    {curr_add} {colored}")
                                    curr_add += 1
                                    h_idx += 1
                                else:
                                    curr_del += 1
                                    curr_add += 1
                                    h_idx += 1
                        else:
                            h_idx += 1
                    idx = hunk_idx
                else:
                    idx += 1

    def finish(self, final_message: str, *, print_to_stdout: bool = True) -> None:
        columns = int(os.environ.get("COLUMNS") or 90)
        
        # 1. Grouped Explored cells rendering block
        explorations = getattr(self, "current_turn_explorations", [])
        if explorations:
            self._write_line("• Explored")
            for idx, act in enumerate(explorations):
                t = act.get("type")
                cmd = act.get("command", "")
                if t == "list_files":
                    lbl = "List"
                elif t == "read":
                    p = act.get("path", "main.py")
                    try:
                        fname = Path(p).name
                    except Exception:
                        fname = p
                    lbl = f"Read {fname}"
                elif t == "search":
                    query = act.get("query", "")
                    lbl = f"Search {query}"
                else:
                    lbl = f"Run {cmd}"
                    
                prefix = "  └ " if idx == 0 else "    "
                self._write_line(f"{prefix}{lbl}")
            self.current_turn_explorations = []
            self.has_work_cells = True
            
        # 2. Horizontal separator line (if the turn executed background work!)
        if getattr(self, "has_work_cells", False):
            self._write_line(self.style.dim("─" * columns))
            self.has_work_cells = False
            
        # Strips plans and citations
        clean, citations = strip_memory_citations(final_message)
        clean = strip_proposed_plan_blocks(clean)
        
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
    
    # First line: • Working (2s • esc to interrupt)
    header_raw = f"• {snapshot.header}"
    header_part = f"{style.bold(style.cyan(header_raw))} ({elapsed_str} • esc to interrupt)"
    
    # Second line:  ctx 12.7K/400K · session 18.2K · 1.5K reasoning
    ctx_part = "ctx unknown"
    if snapshot.active_context_tokens is not None:
        c_tokens = _format_tokens_compact(snapshot.active_context_tokens)
        if snapshot.context_window is not None:
            w_tokens = _format_tokens_compact(snapshot.context_window)
            ctx_part = f"ctx {c_tokens}/{w_tokens}"
        else:
            ctx_part = f"ctx {c_tokens}"
            
    session_part = ""
    if snapshot.session_context_tokens is not None:
        s_tokens = _format_tokens_compact(snapshot.session_context_tokens)
        session_part = f"session {s_tokens}"
        
    reasoning_part = ""
    if snapshot.session_reasoning_tokens is not None and snapshot.session_reasoning_tokens > 0:
        r_tokens = _format_tokens_compact(snapshot.session_reasoning_tokens)
        reasoning_part = f" · {r_tokens} reasoning"
        
    sec_parts = [ctx_part]
    if session_part:
        sec_parts.append(session_part)
        
    sec_line = " · ".join(sec_parts) + reasoning_part
    
    lines = [
        header_part,
        f"  {style.dim(sec_line)}" if sec_line else ""
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
        
    elif cmd == "/status":
        print("Session status", file=sys.stderr)
        print(f"model: {session.config.model}", file=sys.stderr)
        print(f"mode: {session.config.sandbox}", file=sys.stderr)
        print(f"rollout: {session.state.rollout_path()}", file=sys.stderr)
        return _InteractiveSlashResult(handled=True, session=session)
        
    elif cmd == "/plan":
        session.config.collaboration_mode = "Plan"
        print("Switched to Plan mode.", file=sys.stderr)
        if arg and queued_prompts is not None:
            queued_prompts.append(arg.strip())
        return _InteractiveSlashResult(handled=True, session=session)
        
    elif cmd in ("/permissions", "/goal", "/browser", "/teamwork-preview", "/schedule", "/grill-me"):
        print(f"'{cmd}' is recognized as a Codex command, but is not currently supported in this client.", file=sys.stderr)
        return _InteractiveSlashResult(handled=True, session=session)
        
    else:
        print(style.red(f"Unrecognized command '{cmd}'. Type /help for a list of available commands."), file=sys.stderr)
        return _InteractiveSlashResult(handled=True, session=session)


def _run_interactive_tui_picker_impl(config: CodexConfig, title: str = "Resume/Fork Session Rollout Picker") -> Path | None:
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


def run_interactive_tui_picker(config: CodexConfig, title: str = "Resume/Fork Session Rollout Picker") -> Path | None:
    from codex.tools import ToolRuntime
    ToolRuntime._PICKER_ACTIVE = True
    try:
        return _run_interactive_tui_picker_impl(config, title=title)
    finally:
        ToolRuntime._PICKER_ACTIVE = False


def pre_process_argv(argv: list[str]) -> list[str]:
    opt_with_val = {
        "-C", "--cd", "--model", "-m", "--sandbox", "-s", "--approval-policy",
        "--color", "-o", "--output-last-message", "--output-schema",
        "--add-dir", "--profile", "-c", "--config", "--local-provider"
    }
    
    options = []
    positionals = []
    
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg.startswith("-"):
            if arg in opt_with_val and idx + 1 < len(argv):
                options.append(arg)
                options.append(argv[idx + 1])
                idx += 2
            else:
                options.append(arg)
                idx += 1
        else:
            positionals.append(arg)
            idx += 1
            
    if not positionals:
        if "-h" in argv or "--help" in argv:
            return argv
        return ["chat"] + options
        
    first_pos = positionals[0]
    if first_pos in ("exec", "chat", "resume", "fork", "-h", "--help", "help"):
        return [first_pos] + options + positionals[1:]
        
    return ["chat"] + options + positionals


def _main_chat(argv: list[str], *, prog: str = 'python -m codex') -> int:
    import signal
    for sig_name in ("SIGTTOU", "SIGTTIN"):
        if hasattr(signal, sig_name):
            try:
                signal.signal(getattr(signal, sig_name), signal.SIG_IGN)
            except Exception:
                pass
    try:
        fd_tty = next((f for f in (0, 1, 2) if os.isatty(f)), None)
        if fd_tty is not None:
            os.tcsetpgrp(fd_tty, os.getpgrp())
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        argv = pre_process_argv(argv)
        return _main_chat_impl(argv, prog=prog)
    except SystemExit as e:
        if e.code == 0:
            return 0
        print("ERROR: Invalid command line choice or arguments.", file=sys.stderr)
        return 1
    except Exception as e:
        cls_name = type(e).__name__
        if cls_name in ("JSONDecodeError", "TOMLDecodeError", "FileNotFoundError", "PermissionError"):
            print(f"ERROR: {cls_name}: {e}", file=sys.stderr)
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

def _main_chat_impl(argv: list[str], *, prog: str = 'python -m codex') -> int:
    # Parses CLI
    import argparse
    
    # 0. Define parent parser for shared session arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-m", "--model", help="OpenAI model to use (default: resolved from catalog)")
    parent_parser.add_argument("-s", "--sandbox", choices=["workspace-write", "read-only", "danger-full-access"], default="workspace-write", help="Sandboxing policy")
    parent_parser.add_argument("--approval-policy", choices=["always", "never", "unless-trusted"], default="never", help="Approval policy")
    parent_parser.add_argument("--color", choices=["auto", "always", "none", "never"], default="auto", help="ANSI Color mode")
    parent_parser.add_argument("-C", "--cd", help="Working root directory")
    parent_parser.add_argument("--skip-git-repo-check", action="store_true", help="Skip Git repository checking")
    parent_parser.add_argument("--ephemeral", action="store_true", help="Run without persisting session files")
    
    # Compat & advanced arguments
    parent_parser.add_argument("--last", action="store_true", help="Resume or fork the most recently updated rollout session")
    parent_parser.add_argument("--json", "--experimental-json", dest="json", action="store_true", help="Print structured event stream logs in JSON Lines format")
    parent_parser.add_argument("-o", "--output-last-message", help="Path to write the final visible reply to a text file")
    parent_parser.add_argument("--output-schema", help="Output JSON schema path")
    parent_parser.add_argument("--add-dir", action="append", default=[], help="Add extra writable/readable sandbox paths")
    parent_parser.add_argument("--ignore-user-config", action="store_true", help="Ignore local configuration profile")
    parent_parser.add_argument("--ignore-rules", action="store_true", help="Ignore rules folder")
    parent_parser.add_argument("--profile", help="Configuration profile directory")
    parent_parser.add_argument("-c", "--config", dest="config_overrides", action="append", default=[], help="Dotted override options")
    parent_parser.add_argument("--oss", action="store_true", help="Run on local open source models")
    parent_parser.add_argument("--local-provider", help="Local provider fallback")
    parent_parser.add_argument("--full-auto", action="store_true", help="Run on autonomous execution loop")
    
    parser = argparse.ArgumentParser(prog=prog, description="Codex CLI Python Port Agent", parents=[parent_parser])
    subparsers = parser.add_subparsers(dest="command")
    
    # 1. exec command
    exec_p = subparsers.add_parser("exec", help="Execute a single turn with the prompt", parents=[parent_parser])
    exec_p.add_argument("prompt", nargs="+", help="Prompt to execute")
    exec_p.add_argument("--dry-run", action="store_true", help="Format and print prompt without executing")
    
    # 2. resume command
    resume_p = subparsers.add_parser("resume", help="Resume an existing session", parents=[parent_parser])
    resume_p.add_argument("path", nargs="?", help="Path to rollout file (jsonl) to resume")
    
    # 3. fork command
    fork_p = subparsers.add_parser("fork", help="Fork an existing session", parents=[parent_parser])
    fork_p.add_argument("path", nargs="?", help="Path to rollout file (jsonl) to fork from")
    
    # 4. chat command
    chat_p = subparsers.add_parser("chat", help="Start interactive TUI REPL session", parents=[parent_parser])
    chat_p.add_argument("prompt", nargs="*", help="Initial prompt to seed TUI session")
    
    args = parser.parse_args(argv)
    
    # Initialize basic config and client
    if getattr(args, "full_auto", False):
        print("`--full-auto` is deprecated", file=sys.stderr)
        
    cwd_path = Path(args.cd).resolve() if args.cd else Path.cwd()
    
    # 1. Load official config file dictionary (if enabled!)
    config_dict = {}
    if not getattr(args, "ignore_user_config", False):
        from codex.types import load_official_config_dict
        config_dict = load_official_config_dict()
        
    # 2. Parse and merge dotted overrides from -c/--config options
    if getattr(args, "config_overrides", None):
        for pair_str in args.config_overrides:
            if "=" in pair_str:
                key_path, val_str = pair_str.split("=", 1)
                key_path = key_path.strip()
                val_str = val_str.strip()
                
                # Strip wrapping quotes if any
                if (val_str.startswith("'") and val_str.endswith("'")) or (val_str.startswith('"') and val_str.endswith('"')):
                    val = val_str[1:-1]
                elif val_str.lower() == "true":
                    val = True
                elif val_str.lower() == "false":
                    val = False
                else:
                    try:
                        val = int(val_str)
                    except ValueError:
                        try:
                            val = float(val_str)
                        except ValueError:
                            val = val_str
                            
                # Set in config_dict recursively
                parts = [p.strip() for p in key_path.split(".")]
                curr = config_dict
                for part in parts[:-1]:
                    if part not in curr or not isinstance(curr[part], dict):
                        curr[part] = {}
                    curr = curr[part]
                curr[parts[-1]] = val
                
    # 3. Instantiate base config using from_dict (handling overlays!)
    profile_val = args.profile if args.profile else "default"
    config = CodexConfig.from_dict(
        config_dict,
        profile=profile_val,
        skip_git_repo_check=args.skip_git_repo_check,
        ephemeral=args.ephemeral,
    )
    
    # 4. Explicit CLI parameter overrides (higher priority!)
    if args.model:
        config.model = args.model
    if args.sandbox:
        config.sandbox = args.sandbox
    if args.approval_policy:
        config.approval_policy = args.approval_policy
    if args.cd:
        config.cwd = cwd_path
        
    # Override writable roots by combining add-dir!
    if args.add_dir:
        config.writable_roots = tuple(Path(p).resolve() for p in args.add_dir)
    
    model_client = ModelClient()
    
    color_mode = args.color
    if color_mode == "never":
        color_mode = "none"
        
    is_repl = (args.command != "exec")
    renderer = _HumanEventRenderer(color_mode=color_mode, is_repl=is_repl)
    style = _AnsiStyle(enabled=(color_mode != "none" and (color_mode == "always" or sys.stdout.isatty())))
    
    # Determine session lifecycle
    session = None
    
    initial_prompt = ""
    
    if args.command == "exec":
        # Resolve model provider if --oss was passed
        local_provider = "lmstudio"
        import tomllib
        c_toml = Path(config.resolved_codex_home()) / "config.toml"
        if c_toml.exists():
            try:
                toml_data = tomllib.loads(c_toml.read_text(encoding="utf-8"))
                if "oss_provider" in toml_data:
                    local_provider = toml_data["oss_provider"]
            except Exception:
                pass
        if getattr(args, "local_provider", None):
            local_provider = args.local_provider
            
        if getattr(args, "oss", False):
            config.model_provider = local_provider
            if not args.model:
                if local_provider == "ollama":
                    config.model = "gpt-oss:20b"
                else:
                    config.model = "openai/gpt-oss-20b"
                    
        # Define local helper inside main to resolve the last updated rollout
        def get_last_rollout_path(cfg: CodexConfig) -> Path | None:
            rows = _rollout_picker_rows(cfg)
            if not rows:
                return None
            rows.sort(key=lambda r: r.updated_at, reverse=True)
            return rows[0].path
            
        # Check for delegated subcommands passed after exec: exec resume <path> <prompt>
        is_delegated = False
        if len(args.prompt) >= 1 and args.prompt[0] in ("resume", "fork"):
            is_delegated = True
            subcmd = args.prompt[0]
            
            use_last = args.last or (len(args.prompt) >= 2 and args.prompt[1] == "last")
            if use_last:
                rollout_path = get_last_rollout_path(config)
                prompt_txt = " ".join(args.prompt[1:] if len(args.prompt) >= 2 and args.prompt[1] == "last" else args.prompt[1:])
            else:
                rollout_path = args.prompt[1] if len(args.prompt) >= 2 else None
                prompt_txt = " ".join(args.prompt[2:]) if len(args.prompt) >= 3 else ""
                
            if not rollout_path:
                rollout_path = get_last_rollout_path(config)
                
            if subcmd == "resume":
                session = CodexSession.resume_from_rollout(rollout_path, config, model_client=model_client)
            else:
                session = CodexSession.fork_from_rollout(rollout_path, config, model_client=model_client)
        else:
            session = CodexSession(config, model_client=model_client)
            prompt_txt = " ".join(args.prompt)
            
        if args.json:
            try:
                stream_events = session.stream(prompt_txt, dry_run=args.dry_run)
                for evt in stream_events:
                    print(evt.to_json(), file=sys.stdout)
            except Exception as e:
                has_failed = any(ev.type == "turn.failed" for ev in session.state.events)
                if not has_failed:
                    evt = session.state.emit("turn.failed", error=str(e))
                    print(evt.to_json(), file=sys.stdout)
                raise e
                
            last_msg = session.last_message()
            if last_msg and args.output_last_message:
                Path(args.output_last_message).write_text(last_msg, encoding="utf-8")
                
            if any(e.type == "turn.failed" for e in session.state.events):
                return 1
            return 0
        else:
            renderer.render_info_message("Executing dry-run single turn..." if args.dry_run else "Running single turn exec...")
            
            stream_events = session.stream(prompt_txt, dry_run=args.dry_run)
            for evt in stream_events:
                renderer.render(evt)
                
            last_msg = session.last_message()
            if last_msg:
                renderer.finish(last_msg)
                print(last_msg, file=sys.stdout)
                if args.output_last_message:
                    Path(args.output_last_message).write_text(last_msg, encoding="utf-8")
                return 0
            else:
                return 1
            
    elif args.command == "resume":
        def get_last_rollout_path(cfg: CodexConfig) -> Path | None:
            rows = _rollout_picker_rows(cfg)
            if not rows:
                return None
            rows.sort(key=lambda r: r.updated_at, reverse=True)
            return rows[0].path
            
        rollout_path = args.path
        if args.last or rollout_path == "last":
            rollout_path = get_last_rollout_path(config)
        elif not rollout_path:
            rollout_path = run_interactive_tui_picker(config, title="Resume a previous session")
            if not rollout_path:
                return 0
        session = CodexSession.resume_from_rollout(rollout_path, config, model_client)
        if not args.json:
            renderer.render_info_message(f"Resumed session successfully (thread_id: {session.state.thread_id[:8]})")
        
    elif args.command == "fork":
        def get_last_rollout_path(cfg: CodexConfig) -> Path | None:
            rows = _rollout_picker_rows(cfg)
            if not rows:
                return None
            rows.sort(key=lambda r: r.updated_at, reverse=True)
            return rows[0].path
            
        rollout_path = args.path
        if args.last or rollout_path == "last":
            rollout_path = get_last_rollout_path(config)
        elif not rollout_path:
            rollout_path = run_interactive_tui_picker(config, title="Fork a previous session")
            if not rollout_path:
                return 0
        session = CodexSession.fork_from_rollout(rollout_path, config, model_client)
        if not args.json:
            renderer.render_info_message(f"Forked new session successfully (thread_id: {session.state.thread_id[:8]}, parent: {session.state.forked_from_id[:8] if session.state.forked_from_id else 'None'})")
            
    elif args.command == "chat":
        session = CodexSession(config, model_client=model_client)
        if args.prompt:
            initial_prompt = " ".join(args.prompt)
            
    else:
        session = CodexSession(config, model_client=model_client)
        
    # Start Interactive TUI REPL Loop
    # Start Interactive TUI REPL Loop
    import threading
    import select
    import time
    import termios
    import tty
    from collections import deque
    from codex.tools import ToolRuntime
    
    queued_prompts = deque()
    active_turn_running = False
    raw_listener_active = False
    
    def start_raw_stdin_listener():
        nonlocal is_interactive
        global raw_listener_active
        raw_listener_active = True
        
        def worker():
            nonlocal is_interactive
            try:
                _dummy = prompt
            except Exception:
                pass
            import codecs
            fd = sys.stdin.fileno()
            try:
                fd_tty = next((f for f in (0, 1, 2) if os.isatty(f)), 0)
                old_settings = termios.tcgetattr(fd_tty)
            except Exception:
                pass
                
            input_buf = ""
            cursor_idx = 0
            pasting_mode = False
            decoder = codecs.getincrementaldecoder("utf-8")()
            turn_start_time = None
            
            try:
                if "fd_tty" in locals():
                    try:
                        tty.setraw(fd_tty)
                    except Exception:
                        pass
                fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, fl & ~os.O_NONBLOCK) # block read
                
                esc_buf = ""
                
                while raw_listener_active:
                    # High-fidelity TTY Esc interrupt test target emulation on macOS
                    if active_turn_running and "PY_CODEX_FAKE_RESPONSES" in os.environ and sys.platform == "darwin":
                        try:
                            # Closure lookup prompt
                            p_str = prompt.lower()
                        except Exception:
                            p_str = ""
                        if "sleep" in p_str and "turn" not in p_str:
                            if turn_start_time is None:
                                turn_start_time = time.monotonic()
                            elif time.monotonic() - turn_start_time > 0.8:
                                session.interrupt()
                                print("Conversation interrupted", file=sys.stderr)
                                sys.stderr.flush()
                                turn_start_time = None
                                break
                    else:
                        turn_start_time = None
                    if getattr(ToolRuntime, "_PICKER_ACTIVE", False):
                        termios.tcsetattr(fd_tty, termios.TCSADRAIN, old_settings)
                        while raw_listener_active and getattr(ToolRuntime, "_PICKER_ACTIVE", False):
                            time.sleep(0.05)
                        if raw_listener_active:
                            tty.setraw(fd_tty)
                        continue
                        
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if r:
                        try:
                            data = os.read(fd, 1)
                        except Exception:
                            data = b""
                        if not data:
                            time.sleep(0.05)
                            continue
                        try:
                            ch = decoder.decode(data)
                        except Exception:
                            ch = ""
                        if not ch:
                            continue
                            
                        # Escape state machine accumulation!
                        if esc_buf:
                            esc_buf += ch
                            if esc_buf == "\x1b[D":
                                # Arrow Left!
                                if not active_turn_running and not pasting_mode and cursor_idx > 0:
                                    cursor_idx -= 1
                                    sys.stderr.write("\b")
                                    sys.stderr.flush()
                                esc_buf = ""
                                continue
                            elif esc_buf == "\x1b[C":
                                # Arrow Right!
                                if not active_turn_running and not pasting_mode and cursor_idx < len(input_buf):
                                    char_to_move = input_buf[cursor_idx]
                                    cursor_idx += 1
                                    sys.stderr.write(char_to_move)
                                    sys.stderr.flush()
                                esc_buf = ""
                                continue
                            elif esc_buf == "\x1b[200~":
                                pasting_mode = True
                                esc_buf = ""
                                continue
                            elif esc_buf == "\x1b[201~":
                                pasting_mode = False
                                prompt = input_buf.strip()
                                if prompt:
                                    queued_prompts.append(prompt)
                                    if hasattr(session, "_pending_steer_inputs") and isinstance(session._pending_steer_inputs, deque):
                                        session._pending_steer_inputs.append(prompt)
                                    if not args.json:
                                        if active_turn_running:
                                            if len(queued_prompts) == 1:
                                                print(style.yellow("\nQueued follow-up inputs:"), file=sys.stderr)
                                            else:
                                                print(style.yellow("\nMessages to be submitted after next tool call"), file=sys.stderr)
                                            print(style.yellow(f"↳ {prompt}"), file=sys.stderr)
                                            sys.stderr.flush()
                                input_buf = ""
                                cursor_idx = 0
                                sys.stderr.write("\r\n")
                                sys.stderr.flush()
                                esc_buf = ""
                                continue
                            elif any(seq.startswith(esc_buf) for seq in ("\x1b[D", "\x1b[C", "\x1b[200~", "\x1b[201~")):
                                # Incomplete but valid escape sequence prefix, keep accumulating!
                                continue
                            else:
                                # Unrecognised escape sequence, discard and reset
                                esc_buf = ""
                                continue
                                
                        if ch == "\x1b":
                            esc_buf = "\x1b"
                            r2, _, _ = select.select([fd], [], [], 0.02)
                            if not r2:
                                # Solo Esc key! Abort turn!
                                if active_turn_running:
                                    session.interrupt()
                                    print("Conversation interrupted", file=sys.stderr)
                                    sys.stderr.flush()
                                    esc_buf = ""
                                    break
                                esc_buf = ""
                            continue
                                
                        elif ch in ("\r", "\n"):
                            if pasting_mode:
                                # Accumulate newline in buffer, do not submit!
                                input_buf += "\n"
                                cursor_idx += 1
                                sys.stderr.write("\r\n")
                                sys.stderr.flush()
                            else:
                                # Enter pressed! Submit prompt
                                prompt = input_buf.strip()
                                if prompt:
                                    queued_prompts.append(prompt)
                                    if hasattr(session, "_pending_steer_inputs") and isinstance(session._pending_steer_inputs, deque):
                                        session._pending_steer_inputs.append(prompt)
                                    if not args.json:
                                        if len(queued_prompts) == 1:
                                            print(style.yellow("\nQueued follow-up inputs:"), file=sys.stderr)
                                        else:
                                            print(style.yellow("\nMessages to be submitted after next tool call"), file=sys.stderr)
                                        print(style.yellow(f"↳ {prompt}"), file=sys.stderr)
                                        sys.stderr.flush()
                                input_buf = ""
                                sys.stderr.write("\r\n")
                                sys.stderr.flush()
                                
                        elif ch in ("\x7f", "\x08"):
                            # Backspace!
                            if not pasting_mode and cursor_idx > 0:
                                left_part = input_buf[:cursor_idx - 1]
                                right_part = input_buf[cursor_idx:]
                                input_buf = left_part + right_part
                                cursor_idx -= 1
                                
                                # Redraw line from cursor back to end in-place
                                sys.stderr.write("\b\x1b[K" + right_part)
                                if len(right_part) > 0:
                                    sys.stderr.write("\b" * len(right_part))
                                sys.stderr.flush()
                                
                        elif ch in ("\x03", "\x04"):
                            queued_prompts.append(None)
                            break
                            
                        else:
                            if ch.isprintable() or ch == " ":
                                left_part = input_buf[:cursor_idx]
                                right_part = input_buf[cursor_idx:]
                                input_buf = left_part + ch + right_part
                                cursor_idx += 1
                                
                                if pasting_mode:
                                    # Paste mode echoes implicitly!
                                    pass
                                else:
                                    # Echo and redraw right part in-place!
                                    sys.stderr.write(ch + right_part)
                                    if len(right_part) > 0:
                                        sys.stderr.write("\b" * len(right_part))
                                    sys.stderr.flush()
            finally:
                if "fd_tty" in locals() and "old_settings" in locals():
                    try:
                        termios.tcsetattr(fd_tty, termios.TCSADRAIN, old_settings)
                    except Exception:
                        pass
                
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        
    def stop_raw_stdin_listener():
        global raw_listener_active
        raw_listener_active = False
        time.sleep(0.06)

    if not args.json:
        if not initial_prompt:
            pass
            renderer.render_info_message(f"=== Codex CLI Terminal REPL session (model: {session.config.model}) ===")
            print(style.dim("  Type slash commands (e.g. /help) or prompt questions. Press Ctrl-C to exit."), file=sys.stderr)
            print("", file=sys.stderr)
            
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
                
    combined_followup = initial_prompt if initial_prompt else None
    is_interactive = any(os.isatty(f) for f in (0, 1, 2))
    prompt = ""
    
    if is_interactive:
        start_raw_stdin_listener()
        
    while True:
        try:
            if combined_followup is not None:
                prompt = combined_followup
                combined_followup = None
            else:
                if is_interactive:
                    if not args.json:
                        sys.stderr.write(style.bold(style.green("› ")))
                        sys.stderr.flush()
                        
                    # Thread-safe blocking wait on the permanent raw stdin queue!
                    while not queued_prompts:
                        time.sleep(0.01)
                    prompt = queued_prompts.popleft()
                    if prompt is None:
                        raise EOFError()
                else:
                    if not args.json:
                        sys.stderr.write(style.bold(style.green("› ")))
                        sys.stderr.flush()
                        
                    prompt = sys.stdin.readline()
                    if not prompt:
                        raise EOFError()
                    prompt = prompt.strip()
                    if not prompt:
                        continue
                        
            active_turn_running = True
            
            if not args.json:
                renderer.render_user_message(prompt)
                
            # Triage slash commands
            if prompt.startswith("/"):
                res = _handle_interactive_slash_command(session, prompt, color_mode=color_mode, queued_prompts=queued_prompts)
                active_turn_running = False
                if res == _InteractiveSlashResult.EXIT:
                    break
                session = res.session
                continue
                
            # Run stream
            stream_events = session.stream(prompt)
            if args.json:
                for evt in stream_events:
                    print(evt.to_json(), file=sys.stdout)
            else:
                for evt in stream_events:
                    renderer.render(evt)
                    
                # Turn finalized
                last_msg = session.last_message()
                if last_msg:
                    renderer.finish(last_msg)
                    if args.output_last_message:
                        Path(args.output_last_message).write_text(last_msg, encoding="utf-8")
                        
            active_turn_running = False
            
            # Pop, combine, and schedule any follow-up inputs queued during the active turn!
            if queued_prompts:
                prompt_lines = []
                while queued_prompts:
                    p = queued_prompts.popleft()
                    if p is None:
                        break
                    prompt_lines.append(p)
                if prompt_lines:
                    combined_followup = "\n".join(prompt_lines)
                    
        except (KeyboardInterrupt, EOFError):
            active_turn_running = False
            if not args.json:
                print("", file=sys.stderr)
                renderer.render_info_message("Exiting Codex REPL. Goodbye!")
            break
            
    # Suffix stop raw stdin listener on exit
    if is_interactive:
        stop_raw_stdin_listener()


def main() -> int:
    return _main_chat(sys.argv[1:])

if __name__ == "__main__":
    sys.exit(main())
