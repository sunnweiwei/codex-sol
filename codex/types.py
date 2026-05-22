from typing import Any, Dict, List
from dataclasses import dataclass, field

@dataclass
class ModelResponse:
    id: str
    output: List[Dict[str, Any]]
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptRequest:
    model: str
    instructions: str
    input: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    parallel_tool_calls: bool = True
    prompt_cache_key: str | None = None
    reasoning: Dict[str, Any] | None = None
    include: List[str] = field(default_factory=list)
    output_schema: Dict[str, Any] | None = None
    output_schema_strict: bool = True
    verbosity: str | None = None
    service_tier: str | None = None
    client_metadata: Dict[str, str] | None = None

    def to_compact_payload(self) -> Dict[str, Any]:
        return {}

    def to_responses_kwargs(self) -> Dict[str, Any]:
        return {}

@dataclass
class CodexEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __init__(self, type: str, payload: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> None:
        self.type = type
        self.payload = payload if payload is not None else {}
        for key, val in kwargs.items():
            setattr(self, key, val)

KNOWN_EVENT_TYPES: frozenset[str] = frozenset({
    "turn.started",
    "turn.completed",
    "turn.failed",
    "response.started",
    "response.server_reasoning_included",
    "response.rate_limits",
    "response.completed",
    "response.output_item_done",
    "response.delta",
    "error",
    "cancelled",
    "plan.proposed",
    "plan.updated",
    "tool.call",
    "tool.result",
    "memory.consolidated"
})

TERMINAL_TURN_EVENT_TYPES: frozenset[str] = frozenset({
    "turn.completed",
    "turn.failed",
    "response.completed",
    "error",
    "cancelled"
})
