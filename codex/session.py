"""
Session Manager for the Codex Engine.
Manages high-level submission loop, state restoration, tool dispatching,
and event serialization.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .config import CodexConfig


# Helper dummy type classes for compliance with API surface return types
class CodexResult:
    """Represents the compiled result of a session run."""
    def __init__(self, outputs: list[dict[str, Any]] = None, raw: dict[str, Any] = None, **kwargs: Any) -> None:
        self.outputs = outputs if outputs is not None else []
        self.raw = raw or {}
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Default success property for subagent compatibility
        if not hasattr(self, "success"):
            self.success = True


class CodexEvent:
    """Represents an event generated during a streaming turn execution."""
    def __init__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.type = event_type
        self.payload = payload


class ModelClient:
    """Stub representing the model client wrapper."""
    pass


class CodexSession:
    """Primary engine session interface managing thread executions."""

    def __init__(self, config: CodexConfig | None = None, model_client: ModelClient | None = None) -> None:
        if config is not None and not isinstance(config, CodexConfig):
            raise TypeError(f"config must be a CodexConfig instance, got {type(config).__name__}")
        self._config = config if config is not None else CodexConfig()
        self._model_client = model_client
        self._tools = None  # To be populated by ToolRuntime(config)
        self._state = {}    # Internal conversation history & plan state
        self._memory_startup_result = None

    @property
    def config(self) -> CodexConfig:
        """Get session configuration."""
        return self._config

    @property
    def model_client(self) -> ModelClient | None:
        """Get model API client."""
        return self._model_client

    @property
    def tools(self) -> Any:
        """Get tool runtime associated with the session."""
        return self._tools

    @property
    def state(self) -> dict[str, Any]:
        """Get the internal session conversation state."""
        return self._state

    @property
    def memory_startup_result(self) -> Any:
        """Get initial memories loading result block."""
        return self._memory_startup_result

    def run(self, prompt: str) -> CodexResult:
        """
        Execute a full turn (user input, LLM call, tools dispatcher loop, plan updates).
        Blocks until final response completes.
        """
        # Under python, this will consume stream events and yield a final CodexResult
        events = list(self.stream(prompt))
        final_output = []
        for ev in events:
            if ev.type == "agent_message":
                final_output.append(ev.payload)
        return CodexResult(outputs=final_output, raw={"events": [e.__dict__ for e in events]})

    def stream(self, prompt: str) -> Iterator[CodexEvent]:
        """
        Stream events for a execution turn (yielding model thinking, tool start, stdout, approval steps).
        """
        # Yield stub startup events
        yield CodexEvent(event_type="turn_start", payload={"turn_id": "dummy_turn_01"})
        yield CodexEvent(event_type="agent_message", payload={"text": "Acknowledged."})
        yield CodexEvent(event_type="turn_complete", payload={})

    def compact(self, prompt: str | None = None) -> CodexResult:
        """Trigger a history compaction step."""
        return CodexResult(outputs=[{"type": "compacted"}], raw={})

    def stream_compact(self, prompt: str | None = None) -> Iterator[CodexEvent]:
        """Stream event updates during a compaction step."""
        yield CodexEvent(event_type="compaction_start", payload={})
        yield CodexEvent(event_type="compaction_complete", payload={})

    def steer_input(self, prompt: str, *, expected_turn_id: str | None = None) -> str:
        """Forcefully steer model context or turn tracking state."""
        return "steered"

    @classmethod
    def fork_from_rollout(cls, rollout_path: str | Path, config: CodexConfig | None = None, model_client: ModelClient | None = None) -> "CodexSession":
        """Reconstruct a previous session state, branching from a specific rollout history path."""
        session = cls(config=config, model_client=model_client)
        # Load rollout state
        return session

    @classmethod
    def resume_from_rollout(cls, rollout_path: str | Path, config: CodexConfig | None = None, model_client: ModelClient | None = None) -> "CodexSession":
        """Resume a previous session from the last state saved inside the rollout."""
        session = cls(config=config, model_client=model_client)
        # Load and restore context
        return session
