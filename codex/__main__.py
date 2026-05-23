"""Codex Package Entry Point. Executed when calling python -m codex."""

from __future__ import annotations
import sys
from codex.cli import _main_chat

if __name__ == "__main__":
    sys.exit(_main_chat(sys.argv))
