"""
Type definitions, schemas, and events boundaries for the Codex Engine.
Defines ModelResponse, PromptRequest, and exports essential event constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


@dataclass
class ModelResponse:
    """Represents a structured response from the model api execution."""
    id: str
    output: list[dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRequest:
    """Represents a prompt construction sent down to the model client provider."""
    model: str
    instructions: str
    input: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    parallel_tool_calls: bool = True
    prompt_cache_key: str | None = None
    reasoning: dict[str, Any] | None = None
    include: list[str] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    output_schema_strict: bool = True
    verbosity: str | None = None
    service_tier: str | None = None
    client_metadata: dict[str, str] | None = None

    def to_compact_payload(self) -> dict[str, Any]:
        """Compile a minimal payload representation of the request."""
        payload = {
            "model": self.model,
            "instructions": self.instructions,
            "input": self.input,
            "tools": self.tools,
        }
        if not self.parallel_tool_calls:
            payload["parallel_tool_calls"] = False
        if self.prompt_cache_key:
            payload["prompt_cache_key"] = self.prompt_cache_key
        if self.reasoning:
            payload["reasoning"] = self.reasoning
        if self.include:
            payload["include"] = self.include
        if self.output_schema:
            payload["output_schema"] = self.output_schema
            payload["output_schema_strict"] = self.output_schema_strict
        if self.verbosity:
            payload["verbosity"] = self.verbosity
        if self.service_tier:
            payload["service_tier"] = self.service_tier
        if self.client_metadata:
            payload["client_metadata"] = self.client_metadata
        return payload

    def to_responses_kwargs(self) -> dict[str, Any]:
        """Format parameters for OpenAI responses API compatibility."""
        kwargs = {
            "model": self.model,
            "messages": self.input,
            "tools": self.tools,
        }
        if self.instructions:
            # Inject system instructions into messages list
            kwargs["messages"] = [{"role": "developer", "content": self.instructions}] + self.input
        return kwargs


# Dynamic Type hints referenced across modules
class MemoryJobClaim:
    """Represents a claimed lock on global or stage 1 background memory indexing jobs."""
    def __init__(self, job_key: str, worker_id: str, lease_seconds: int) -> None:
        self.job_key = job_key
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds


class MemoryStageOneOutput:
    """Represents the output structure from memory stage one processing."""
    def __init__(self, raw_memory: str, rollout_summary: str, slug: str | None = None) -> None:
        self.raw_memory = raw_memory
        self.rollout_summary = rollout_summary
        self.slug = slug


class MemoryRollout:
    """Represents rollout metadata parsed from session histories."""
    def __init__(self, path: Path | str, contents: str) -> None:
        self.path = Path(path)
        self.contents = contents


class MemoryPhase2Result:
    """Represents the outcomes of global memory consolidation (phase 2)."""
    def __init__(self, success: bool, consolidated_memories: int = 0) -> None:
        self.success = success
        self.consolidated_memories = consolidated_memories


class MemoryStartupResult:
    """Represents the results of immediate memories lookup during startup."""
    def __init__(self, loaded_records: list[Any], active_citation_context: str = "") -> None:
        self.loaded_records = loaded_records
        self.active_citation_context = active_citation_context


class MemoryBackgroundTask:
    """Represents a background scheduler monitoring and compiling memory indexing logs."""
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


class RolloutReconstruction:
    """Represents reconstructed state from session rollout analysis."""
    def __init__(self, history: list[dict[str, Any]], plan: str = "") -> None:
        self.history = history
        self.plan = plan


# Event Categories definitions for validation and frontend consumption
KNOWN_EVENT_TYPES = frozenset([
    "error",
    "warning",
    "guardian_warning",
    "realtime_conversation_started",
    "realtime_conversation_realtime",
    "realtime_conversation_closed",
    "realtime_conversation_sdp",
    "model_reroute",
    "model_verification",
    "context_compacted",
    "thread_rolled_back",
    "turn_started",
    "turn_complete",
    "token_count",
    "agent_message",
    "user_message",
    "agent_reasoning",
    "agent_reasoning_raw_content",
    "agent_reasoning_section_break",
    "session_configured",
    "thread_goal_updated",
    "mcp_startup_update",
    "mcp_startup_complete",
    "mcp_tool_call_begin",
    "mcp_tool_call_end",
    "web_search_begin",
    "web_search_end",
    "image_generation_begin",
    "image_generation_end",
    "exec_command_begin",
    "exec_command_output_delta",
    "terminal_interaction",
    "exec_command_end",
    "view_image_tool_call",
    "exec_approval_request",
    "request_permissions",
    "request_user_input",
    "dynamic_tool_call_request",
    "dynamic_tool_call_response",
    "elicitation_request",
    "apply_patch_approval_request",
    "guardian_assessment",
    "deprecation_notice",
    "stream_error",
    "patch_apply_begin",
    "patch_apply_updated",
    "patch_apply_end",
    "turn_diff",
])

TERMINAL_TURN_EVENT_TYPES = frozenset([
    "turn_complete",
    "error",
])
