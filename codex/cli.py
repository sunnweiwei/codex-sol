"""
Production-ready python implementation for codex.cli.
Conforms to active Feature specifications under API_SURFACE.md.
Embeds high-fidelity ANSI/VT100 style escape codes, dynamic status footers,
and coordinate-aware visual wrapping engines.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Callable, Pattern


_ANSI_RE: Pattern[str] = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][0-9;]*\x07')


class _AnsiStyle:
    """Wraps global text coloring escape settings for 16-color ANSI/VT100 outputs."""
    
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Indicates whether style escapes are active."""
        return self._enabled

    @property
    def reset(self) -> str:
        """Closes active color styling highlights."""
        return "\x1b[0m" if self._enabled else ""

    @property
    def bold(self) -> str:
        """Selects bold text weight emphasis."""
        return "\x1b[1m" if self._enabled else ""

    @property
    def dim(self) -> str:
        """Selects high-contrast dim text emphasis."""
        return "\x1b[2m" if self._enabled else ""

    @property
    def italic(self) -> str:
        """Selects italic styling emphasis."""
        return "\x1b[3m" if self._enabled else ""

    @property
    def underline(self) -> str:
        """Selects text underline styling."""
        return "\x1b[4m" if self._enabled else ""

    @property
    def inverse(self) -> str:
        """Swaps standard background/foreground color values."""
        return "\x1b[7m" if self._enabled else ""

    @property
    def green(self) -> str:
        """Sets text foreground color to green."""
        return "\x1b[32m" if self._enabled else ""

    @property
    def blue(self) -> str:
        """Sets text foreground color to blue."""
        return "\x1b[34m" if self._enabled else ""

    @property
    def cyan(self) -> str:
        """Sets text foreground color to cyan."""
        return "\x1b[36m" if self._enabled else ""

    @property
    def gray(self) -> str:
        """Sets high-contrast text color to low-contrast gray."""
        return "\x1b[90m" if self._enabled else ""

    @property
    def white(self) -> str:
        """Sets text foreground color to white."""
        return "\x1b[37m" if self._enabled else ""

    @property
    def yellow(self) -> str:
        """Sets text foreground color to yellow."""
        return "\x1b[33m" if self._enabled else ""

    @property
    def magenta(self) -> str:
        """Sets text foreground color to magenta."""
        return "\x1b[35m" if self._enabled else ""

    @property
    def red(self) -> str:
        """Sets text foreground color to red."""
        return "\x1b[31m" if self._enabled else ""

    @property
    def bg_gray(self) -> str:
        """Selects high-contrast gray cell background block."""
        return "\x1b[100m" if self._enabled else ""

    @property
    def bg_blue(self) -> str:
        """Selects blue cell background block."""
        return "\x1b[44m" if self._enabled else ""

    @property
    def bg_green(self) -> str:
        """Selects green cell background block."""
        return "\x1b[42m" if self._enabled else ""

    @property
    def status_bar(self) -> str:
        """Selects the standard low-contrast gray status bar background combo."""
        return "\x1b[37;100m" if self._enabled else ""


class _LiveTurnStatusSnapshot:
    """Carries a metrics data point mapping state coordinates onto the base status-line."""
    
    def __init__(
        self,
        header: str,
        elapsed_seconds: int,
        active_context_tokens: int | None,
        active_context_estimated: bool,
        session_context_tokens: int | None,
        session_context_estimated: bool,
        session_reasoning_tokens: int | None,
        context_window: int | None
    ) -> None:
        self.header = header
        self.elapsed_seconds = elapsed_seconds
        self.active_context_tokens = active_context_tokens
        self.active_context_estimated = active_context_estimated
        self.session_context_tokens = session_context_tokens
        self.session_context_estimated = session_context_estimated
        self.session_reasoning_tokens = session_reasoning_tokens
        self.context_window = context_window


class _HumanEventRenderer:
    """Manages raw interactive stdout logging, framing chat streams inside border layout blocks."""
    
    def __init__(
        self,
        *,
        color_mode: str = 'auto',
        line_sink: Callable[[str], None] | None = None
    ) -> None:
        self.color_mode = color_mode
        self.line_sink = line_sink or print
        
        enabled = True
        if color_mode == 'never':
            enabled = False
        elif color_mode == 'auto':
            enabled = sys.stdout.isatty()
            
        self.style = _AnsiStyle(enabled=enabled)

    def _write(self, line: str) -> None:
        self.line_sink(line)

    def render_user_message(self, text: str) -> None:
        """Renders standard input draft message framed under green square boundary box frames."""
        if not text:
            return
            
        lines = text.split("\n")
        max_width = 72
        
        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(_wrap_ansi_line(line, max_width))
            
        content_width = max(_visible_len(l) for l in wrapped_lines)
        
        border = self.style.gray
        text_col = self.style.green
        reset = self.style.reset
        
        self._write(f"  {border}┌{'─' * (content_width + 2)}┐{reset}")
        for l in wrapped_lines:
            vis_len = _visible_len(l)
            padding = content_width - vis_len
            self._write(f"  {border}│{reset} {text_col}{l}{reset}{' ' * padding} {border}│{reset}")
        self._write(f"  {border}└{'─' * (content_width + 2)}┘{reset}")

    def render(self, event: Any) -> None:
        """Interprets incoming turn events, writing structured bubbles and welcome headers."""
        if event is None:
            return
            
        if isinstance(event, dict):
            ev_type = event.get("type")
            payload = event.get("payload") or event
        else:
            ev_type = getattr(event, "type", None)
            payload = getattr(event, "payload", None) or {}
            
        if not ev_type:
            return
            
        border = self.style.gray
        text_col = self.style.blue
        reset = self.style.reset
        
        # 1. Boot welcome greeting
        if ev_type == "session_start" or ev_type == "boot":
            welcome_text = payload.get("text") or "Welcome to the Python Codex API Engine!"
            lines = welcome_text.split("\n")
            content_width = max(_visible_len(l) for l in lines)
            
            self._write(f"  {border}╭{'─' * (content_width + 2)}╮{reset}")
            for l in lines:
                padding = content_width - _visible_len(l)
                self._write(f"  {border}│{reset} {text_col}{l}{reset}{' ' * padding} {border}│{reset}")
            self._write(f"  {border}╰{'─' * (content_width + 2)}╯{reset}")
            
        # 2. Assistant messages stream replies
        elif ev_type == "agent_message":
            text = payload.get("text", "")
            if text:
                wrapped = []
                for line in text.split("\n"):
                    wrapped.extend(_wrap_ansi_line(line, 72))
                content_width = max(_visible_len(l) for l in wrapped)
                
                self._write(f"  {border}╭{'─' * (content_width + 2)}╮{reset}")
                for l in wrapped:
                    padding = content_width - _visible_len(l)
                    self._write(f"  {border}│{reset} {text_col}{l}{reset}{' ' * padding} {border}│{reset}")
                self._write(f"  {border}╰{'─' * (content_width + 2)}╯{reset}")


def _apply_prompt_escape_sequence(buffer: str, cursor: int, sequence: bytes) -> tuple[str, int] | None:
    """
    Simulates terminal line-editing controls, applying raw key escape bytes onto input prompt buffers.
    Conforms to backspace, arrow directions, home/end, and forward delete bindings.
    """
    if not sequence:
        return buffer, cursor
        
    # Backspace edits
    if sequence in (b"\x7f", b"\x08"):
        if cursor > 0:
            new_buffer = buffer[:cursor-1] + buffer[cursor:]
            return new_buffer, cursor - 1
        return buffer, cursor
        
    # Coordinate motion sequences
    if sequence.startswith(b"\x1b"):
        # Left navigation arrow
        if sequence in (b"\x1b[D", b"\x1bOD"):
            return buffer, max(0, cursor - 1)
        # Right navigation arrow
        elif sequence in (b"\x1b[C", b"\x1bOC"):
            return buffer, min(len(buffer), cursor + 1)
        # Home alignment
        elif sequence in (b"\x1b[H", b"\x1b[1~", b"\x1bOH", b"\x1b[a"):
            return buffer, 0
        # End alignment
        elif sequence in (b"\x1b[F", b"\x1b[4~", b"\x1bOF", b"\x1b[b"):
            return buffer, len(buffer)
        # Forward delete
        elif sequence == b"\x1b[3~":
            if cursor < len(buffer):
                new_buffer = buffer[:cursor] + buffer[cursor+1:]
                return new_buffer, cursor
            return buffer, cursor
            
    return None


def _format_elapsed_compact(elapsed_seconds: int) -> str:
    """Compresses integer seconds into neat compact durations (e.g. 12s, 2m 15s)."""
    if elapsed_seconds < 0:
        return "0s"
    h = elapsed_seconds // 3600
    m = (elapsed_seconds % 3600) // 60
    s = elapsed_seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def _format_tokens_compact(value: int | float) -> str:
    """Formats numeric token sizes into compact human readable tags (e.g. 120, 1.2k, 25k)."""
    if value is None or value < 0:
        return "0"
    val = float(value)
    if val < 1000:
        return str(int(val))
    elif val < 1000000:
        k_val = val / 1000.0
        if k_val >= 100 or abs(k_val - int(k_val)) < 0.05:
            return f"{int(k_val)}k"
        return f"{k_val:.1f}k"
    else:
        m_val = val / 1000000.0
        if m_val >= 100 or abs(m_val - int(m_val)) < 0.05:
            return f"{int(m_val)}M"
        return f"{m_val:.1f}M"


def _live_status_display_lines(snapshot: _LiveTurnStatusSnapshot | None, style: _AnsiStyle) -> list[str]:
    """Generates the coordinate-based status footer printout containing metrics placed at base row 24."""
    if snapshot is None:
        return []
        
    elapsed = _format_elapsed_compact(snapshot.elapsed_seconds)
    
    tokens_block = ""
    if snapshot.active_context_tokens is not None:
        act = _format_tokens_compact(snapshot.active_context_tokens)
        est = "~" if snapshot.active_context_estimated else ""
        tokens_block += f"{est}{act}"
    else:
        tokens_block += "-"
        
    if snapshot.session_context_tokens is not None:
        sess = _format_tokens_compact(snapshot.session_context_tokens)
        est = "~" if snapshot.session_context_estimated else ""
        tokens_block += f" (total: {est}{sess}"
        if snapshot.session_reasoning_tokens is not None:
            reasoning = _format_tokens_compact(snapshot.session_reasoning_tokens)
            tokens_block += f", reasoning: {reasoning}"
        tokens_block += ")"
        
    window = ""
    if snapshot.context_window is not None:
        window = f"/{_format_tokens_compact(snapshot.context_window)}"
        
    content = f" Elapsed: {elapsed} | Tokens: {tokens_block}{window} | Turn: {snapshot.header} | Sandbox: Restricted "
    padded = content.ljust(80)
    
    bg = style.status_bar if style.enabled else ""
    reset = style.reset if style.enabled else ""
    
    # VT100 dynamic coordinates positioning: Save, Hide, Locate bottom (Row 24), print text, Restore, Show
    escapes = f"\x1b[s\x1b[?25l\x1b[24;1H{bg}{padded}{reset}\x1b[u\x1b[?25h"
    return [escapes]


def _render_markdown_for_terminal(
    text: str,
    style: _AnsiStyle,
    *,
    emphasis: bool = True,
    terminal_width: int | None = None
) -> list[str]:
    """Renders basic markdown text into terminal ANSI highlighted lines, wrapped according to column boundaries."""
    if not text:
        return []
        
    width = terminal_width if terminal_width is not None else 80
    lines = text.split("\n")
    rendered = []
    
    in_code = False
    
    for line in lines:
        stripped = line.strip()
        
        # Code block mapping
        if stripped.startswith("```"):
            in_code = not in_code
            continue
            
        if in_code:
            styled = f"{style.dim}{line}{style.reset}" if style.enabled else line
            rendered.extend(_wrap_ansi_line(styled, width))
            continue
            
        # Headers
        if stripped.startswith("# "):
            val = stripped[2:]
            styled = f"{style.bold}{style.blue}# {val}{style.reset}" if style.enabled else line
            rendered.extend(_wrap_ansi_line(styled, width))
            continue
        elif stripped.startswith("## "):
            val = stripped[3:]
            styled = f"{style.bold}{style.cyan}## {val}{style.reset}" if style.enabled else line
            rendered.extend(_wrap_ansi_line(styled, width))
            continue
        elif stripped.startswith("### "):
            val = stripped[4:]
            styled = f"{style.bold}### {val}{style.reset}" if style.enabled else line
            rendered.extend(_wrap_ansi_line(styled, width))
            continue
            
        # Blockquotes
        if stripped.startswith(">"):
            quote = line[1:].strip()
            styled = f"{style.italic}{quote}{style.reset}" if style.enabled else quote
            prefix = f"{style.gray}│ {style.reset}" if style.enabled else "│ "
            wrapped = _wrap_ansi_line(styled, width - 2)
            for w in wrapped:
                rendered.append(f"{prefix}{w}")
            continue
            
        # Bullet list items
        if stripped.startswith(("- ", "* ")) and len(stripped) > 2:
            item = stripped[2:]
            bullet = f"{style.green}• {style.reset}" if style.enabled else "• "
            
            styled = item
            if style.enabled and emphasis:
                styled = re.sub(r'\*\*(.*?)\*\*', f'{style.bold}\\1{style.reset}', styled)
                styled = re.sub(r'\*(.*?)\*', f'{style.italic}\\1{style.reset}', styled)
                styled = re.sub(r'`(.*?)`', f'{style.yellow}\\1{style.reset}', styled)
                
            wrapped = _wrap_ansi_line(styled, width - 2)
            if wrapped:
                rendered.append(f"{bullet}{wrapped[0]}")
                for w in wrapped[1:]:
                    rendered.append(f"  {w}")
            continue
            
        # Paragraph text
        styled = line
        if style.enabled and emphasis:
            styled = re.sub(r'\*\*(.*?)\*\*', f'{style.bold}\\1{style.reset}', styled)
            styled = re.sub(r'\*(.*?)\*', f'{style.italic}\\1{style.reset}', styled)
            styled = re.sub(r'`(.*?)`', f'{style.yellow}\\1{style.reset}', styled)
            
        rendered.extend(_wrap_ansi_line(styled, width))
        
    return rendered


def _visible_len(text: str) -> int:
    """Calculates visible length of text by stripping out all ANSI style escapes."""
    return len(_ANSI_RE.sub('', text))


def _wrap_ansi_line(text: str, width: int) -> list[str]:
    """Slices text blocks wrapping on spaces without splitting ANSI escape sequence blocks."""
    if _visible_len(text) <= width:
        return [text]
        
    segments = []
    current_style = ""
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\x1b":
            match = _ANSI_RE.match(text, i)
            if match:
                seq = match.group(0)
                current_style += seq
                if seq == "\x1b[0m":
                    current_style = ""
                i += len(seq)
                continue
        segments.append((text[i], current_style))
        i += 1
        
    wrapped_lines = []
    current_line = []
    current_len = 0
    
    words = []
    current_word = []
    for char, style in segments:
        if char.isspace() and char != "\xa0":
            if current_word:
                words.append((current_word, False))
                current_word = []
            words.append(([ (char, style) ], True))
        else:
            current_word.append((char, style))
    if current_word:
        words.append((current_word, False))
        
    for word, is_space in words:
        w_len = len(word)
        if current_len + w_len <= width:
            current_line.extend(word)
            current_len += w_len
        else:
            if is_space and current_len == 0:
                continue
            if w_len > width:
                left = width - current_len
                if left > 0:
                    current_line.extend(word[:left])
                    word = word[left:]
                if current_line:
                    wrapped_lines.append(current_line)
                    current_line = []
                    current_len = 0
                while len(word) > width:
                    wrapped_lines.append(word[:width])
                    word = word[width:]
                current_line.extend(word)
                current_len = len(word)
            else:
                if current_line:
                    while current_line and current_line[-1][0].isspace():
                        current_line.pop()
                    if current_line:
                        wrapped_lines.append(current_line)
                current_line = word
                current_len = w_len
                
    if current_line:
        wrapped_lines.append(current_line)
        
    result = []
    last_style = ""
    for line in wrapped_lines:
        line_str = []
        for char, style in line:
            if style != last_style:
                line_str.append(style if style else "\x1b[0m")
                last_style = style
            line_str.append(char)
        if last_style:
            line_str.append("\x1b[0m")
            last_style = ""
        result.append("".join(line_str))
        
    return result
