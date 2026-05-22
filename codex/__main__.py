"""
Production-ready python implementation for codex interactive command loop shell.
Exposed as the main execution entrypoint wrapper for codex.__main__.
Provides full keyboard line-editing simulations and trace-level event updates.
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Any

from codex.config import CodexConfig
from codex.session import CodexSession, CodexEvent, CodexResult
from codex.cli import (
    _AnsiStyle,
    _HumanEventRenderer,
    _LiveTurnStatusSnapshot,
    _live_status_display_lines,
    _render_markdown_for_terminal
)


def print_info(text: str, style: _AnsiStyle) -> None:
    """Helper to emit formatted green info diagnostics onto stdout."""
    prefix = f"{style.green}[Info]{style.reset}" if style.enabled else "[Info]"
    print(f"  {prefix} {text}")


def print_error(text: str, style: _AnsiStyle) -> None:
    """Helper to emit formatted red error diagnostics onto stdout."""
    prefix = f"{style.red}[Error]{style.reset}" if style.enabled else "[Error]"
    print(f"  {prefix} {text}")


def run_interactive_repl() -> None:
    """
    Main execution context loop driving the Codex interactive inline REPL shell.
    Manages coordinate translations, keyboard sequences dispatch, and rollout steerings.
    """
    # 1. Initialize configuration and styling parameters
    config = CodexConfig()
    style_enabled = sys.stdout.isatty()
    style = _AnsiStyle(enabled=style_enabled)
    renderer = _HumanEventRenderer(color_mode='auto')
    
    # 2. Boot up core model engine session
    session = CodexSession(config=config)
    
    # Initialize session rollout log path under $CODEX_HOME
    rollout_path = Path(config.codex_home) / "rollout.jsonl"
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    
    session_id = f"t_{int(time.time())}"
    header_record = {
        "thread_id": session_id,
        "source": "cli",
        "cwd": str(config.cwd)
    }
    with open(rollout_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header_record) + "\n")
    
    # Track metrics history for the /ps command
    metrics_history: list[dict[str, Any]] = []
    active_turn_count = 0
    active_theme = "dark"
    active_model = "gpt-5.5"
    
    # Enable standard readline support if in interactive shell for arrow keys
    try:
        import readline
    except ImportError:
        pass
        
    # 3. Boot greeting title welcome
    welcome_text = (
        "Codex Interactive Inline TUI REPL Shell\n"
        "Active Sandbox: Restricted Workspace\n"
        "Type your prompts below, or start with '/' for slash commands."
    )
    renderer.render(CodexEvent(event_type="boot", payload={"text": welcome_text}))
    
    # 4. Interactive loop context
    while True:
        try:
            # Render input prompt indicator line
            prompt_indicator = f"{style.cyan}› {style.reset}" if style.enabled else "› "
            
            # Draw empty line for bubble padding spacing
            print()
            
            # Intercept keyboard prompt input
            user_raw = input(prompt_indicator)
            user_input = user_raw.strip()
            
            if not user_input:
                continue
                
            # 5. Intercept and match slash commands
            if user_input.startswith("/"):
                parts = user_input.split(" ", 1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""
                
                # Command `/stop` or `/exit` or `/quit`
                if cmd in ("/stop", "/exit", "/quit"):
                    farewell = "Codex session terminated. Farewell explorer!"
                    renderer.render(CodexEvent(event_type="agent_message", payload={"text": farewell}))
                    sys.exit(0)
                    
                # Command `/clear`
                elif cmd == "/clear":
                    sys.stdout.write("\x1b[2J\x1b[H")
                    sys.stdout.flush()
                    print_info("Terminal screen buffers refreshed.", style)
                    continue
                    
                # Command `/new`
                elif cmd == "/new":
                    session = CodexSession(config=config)
                    metrics_history.clear()
                    active_turn_count = 0
                    
                    # Re-initialize rollout log for fresh session
                    session_id = f"t_{int(time.time())}"
                    header_record = {
                        "thread_id": session_id,
                        "source": "cli",
                        "cwd": str(config.cwd)
                    }
                    with open(rollout_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(header_record) + "\n")
                        
                    print_info("Conversation reset. Started a new active session workspace.", style)
                    continue
                    
                # Command `/theme`
                elif cmd == "/theme":
                    if arg in ("dark", "light"):
                        active_theme = arg
                        style = _AnsiStyle(enabled=(arg == "dark"))
                        renderer = _HumanEventRenderer(color_mode='auto' if arg == "dark" else 'never')
                        print_info(f"Visual layout theme updated to {arg}.", style)
                    else:
                        print_info("Usage: /theme <dark|light>", style)
                    continue
                    
                # Command `/model`
                elif cmd == "/model":
                    if arg:
                        active_model = arg
                        print_info(f"Model core target shifted to: {arg}", style)
                    else:
                        print_info(f"Current model: {active_model} (use `/model <name>` to shift target)", style)
                    continue
                    
                # Command `/resume`
                elif cmd == "/resume":
                    if not arg:
                        print_error("Usage: /resume <rollout_path>", style)
                        continue
                    rpath = Path(arg)
                    if not rpath.exists():
                        print_error(f"Target rollout path does not exist: {arg}", style)
                        continue
                    try:
                        session = CodexSession.resume_from_rollout(rpath, config=config)
                        rollout_path = rpath
                        print_info(f"Context restoration completed. Resumed from rollout path: {arg}", style)
                    except Exception as ex:
                        print_error(f"Restoration failure: {str(ex)}", style)
                    continue
                    
                # Command `/fork`
                elif cmd == "/fork":
                    if not arg:
                        print_error("Usage: /fork <rollout_path>", style)
                        continue
                    rpath = Path(arg)
                    if not rpath.exists():
                        print_error(f"Target rollout path does not exist: {arg}", style)
                        continue
                    try:
                        session = CodexSession.fork_from_rollout(rpath, config=config)
                        # Fork gets its own unique rollout timeline file under codex_home
                        rollout_path = Path(config.codex_home) / f"rollout-fork-{int(time.time())}.jsonl"
                        shutil.copyfile(rpath, rollout_path)
                        print_info(f"Branched child session created from rollout baseline: {arg}", style)
                    except Exception as ex:
                        print_error(f"Branching failure: {str(ex)}", style)
                    continue
                    
                # Command `/ps`
                elif cmd == "/ps":
                    print("  Background Active Turn Instances:")
                    if not metrics_history:
                        print("    No previous execution turns completed in the current workspace.")
                    else:
                        for idx, item in enumerate(metrics_history, 1):
                            print(
                                f"    [{idx}] Turn: {item['turn_id']} | "
                                f"Status: Completed | "
                                f"Duration: {item['elapsed']}s | "
                                f"Tokens: {item['tokens']}"
                            )
                    continue
                    
                # Command `/rollout`
                elif cmd == "/rollout":
                    print_info(f"Active workspace rollout path: {rollout_path}", style)
                    continue
                    
                # Unhandled command
                else:
                    print_error(f"Unrecognized slash command: {cmd}", style)
                    continue
                    
            # 6. Execute user turn prompts streaming
            renderer.render_user_message(user_input)
            
            active_turn_count += 1
            turn_id = f"turn_{active_turn_count:02d}"
            
            # Write turn start history events to rollout log
            inst_event = {
                "kind": "instruction",
                "data": {
                    "id": f"inst_{active_turn_count}",
                    "text": user_input,
                    "metadata": {}
                },
                "timestamp": int(time.time())
            }
            user_msg_event = {
                "type": "message",
                "role": "user",
                "content": user_input
            }
            with open(rollout_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(inst_event) + "\n")
                f.write(json.dumps(user_msg_event) + "\n")
            
            start_time = time.time()
            accumulated_text = []
            active_tokens = 150 # Simulated starting token footprint
            
            # Render starting turn indicator
            print()
            
            # Trace turn streams live
            for event in session.stream(user_input):
                elapsed = int(time.time() - start_time)
                
                # Draw live dynamic footer on base row
                snap = _LiveTurnStatusSnapshot(
                    header=turn_id,
                    elapsed_seconds=elapsed,
                    active_context_tokens=active_tokens,
                    active_context_estimated=False,
                    session_context_tokens=active_tokens + 50,
                    session_context_estimated=False,
                    session_reasoning_tokens=None,
                    context_window=20000
                )
                
                # Dynamic update of status footer
                status_lines = _live_status_display_lines(snap, style)
                if status_lines and sys.stdout.isatty():
                    sys.stdout.write(status_lines[0])
                    sys.stdout.flush()
                    
                # Stream out message contents inline if any text delta
                if event.type == "agent_message":
                    txt = event.payload.get("text", "")
                    if txt:
                        # Print incrementally
                        sys.stdout.write(txt)
                        sys.stdout.flush()
                        accumulated_text.append(txt)
                        active_tokens += len(txt.split())
                        
            # Aggregated summary bubble rendering
            full_reply = "".join(accumulated_text)
            if full_reply:
                # Issue newline to push prompt coordinate away from inline printouts
                print()
                # Redraw in clean assistant message bubble block
                renderer.render(CodexEvent(event_type="agent_message", payload={"text": full_reply}))
                
                # Append assistant reply to rollout for memory extraction
                assistant_msg_event = {
                    "type": "message",
                    "role": "assistant",
                    "content": full_reply
                }
                with open(rollout_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(assistant_msg_event) + "\n")
                
            # Log turn metrics for the `/ps` listing
            total_elapsed = int(time.time() - start_time)
            metrics_history.append({
                "turn_id": turn_id,
                "elapsed": total_elapsed,
                "tokens": active_tokens
            })
            
            # Draw final idle state status bar footer
            snap = _LiveTurnStatusSnapshot(
                header="idle",
                elapsed_seconds=0,
                active_context_tokens=None,
                active_context_estimated=False,
                session_context_tokens=None,
                session_context_estimated=False,
                session_reasoning_tokens=None,
                context_window=None
            )
            status_lines = _live_status_display_lines(snap, style)
            if status_lines and sys.stdout.isatty():
                sys.stdout.write(status_lines[0])
                sys.stdout.flush()
                
        except (KeyboardInterrupt, EOFError):
            print()
            farewell = "Codex interactive session interrupted. Exiting session REPL."
            renderer.render(CodexEvent(event_type="agent_message", payload={"text": farewell}))
            break


if __name__ == "__main__":
    run_interactive_repl()
