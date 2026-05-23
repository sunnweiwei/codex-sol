"""Codex Core API types: configuration dataclasses, event streams, run outcomes, lifecycle steps, and schemas."""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, Dict, List, Tuple

# Type Aliases and Literals
SandboxMode = Literal['read-only', 'workspace-write', 'danger-full-access']
ApprovalPolicy = Literal['never', 'on-request', 'on-request-rule-request-permission', 'on-failure', 'unless-trusted']
CollaborationMode = Literal['Default', 'Subagent', 'Supervisor']
RemoteCompactionMode = Literal['auto', 'disabled', 'enabled', 'v2']
LifecyclePhase = Literal['start', 'turn', 'tool', 'patch', 'compact', 'consolidate']

# Constants
KNOWN_EVENT_TYPES = frozenset({
    "thread.started",
    "turn.started",
    "turn.completed",
    "turn.failed",
    "item.started",
    "item.updated",
    "item.completed",
    "error"
})

TERMINAL_TURN_EVENT_TYPES = frozenset({
    "turn.completed",
    "turn.failed",
    "error"
})

_MODEL_CATALOG_CACHE: dict[str, dict[str, Any]] | None = None

def _load_model_catalog(codex_home: Path | None = None) -> dict[str, dict[str, Any]]:
    global _MODEL_CATALOG_CACHE
    if _MODEL_CATALOG_CACHE is not None:
        return _MODEL_CATALOG_CACHE

    catalog: dict[str, dict[str, Any]] = {}
    
    # Static fallback for gpt-5.5 frontier model
    catalog["gpt-5.5"] = {
        "slug": "gpt-5.5",
        "context_window": 272000,
        "supports_parallel_tool_calls": True,
        "auto_compact_token_limit": None,
        "supports_reasoning_summaries": True,
        "default_reasoning_level": "medium",
        "default_reasoning_summary": "none",
        "default_verbosity": "low",
        "support_verbosity": True,
        "input_modalities": ["text", "image"],
        "supports_image_detail_original": True,
        "truncation_policy": {"mode": "tokens", "limit": 10000}
    }

    # Attempt to dynamically locate models.json in assets
    search_paths = []
    
    if "CODEX_ASSETS_DIR" in os.environ:
        search_paths.append(Path(os.environ["CODEX_ASSETS_DIR"]) / "models.json")
        
    if codex_home:
        search_paths.append(codex_home / "models.json")
        
    search_paths.append(Path.home() / ".codex" / "models.json")
    
    # Package assets directory lookups
    try:
        package_root = Path(__file__).resolve().parent
        search_paths.append(package_root / "assets" / "models.json")
        search_paths.append(package_root.parent / "codex" / "assets" / "models.json")
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")

    # Deduplicate search paths preserving order
    seen_paths = set()
    unique_paths = []
    for p in search_paths:
        try:
            resolved = p.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                unique_paths.append(p)
        except Exception:
            if p not in seen_paths:
                seen_paths.add(p)
                unique_paths.append(p)

    for path in unique_paths:
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "models" in data:
                        for model in data["models"]:
                            slug = model.get("slug")
                            if slug:
                                catalog[slug] = model
                break
            except Exception as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
                
    _MODEL_CATALOG_CACHE = catalog
    return catalog

@dataclass
class CodexConfig:
    """Consolidated configuration container managing sandbox permissions, limits, reasoning, and dynamic environments."""
    model: str = field(default_factory=lambda: os.environ.get("CODEX_MODEL", "gpt-5.5"))
    cwd: Path | str = field(default_factory=lambda: Path.cwd())
    sandbox: SandboxMode = 'workspace-write'
    approval_policy: ApprovalPolicy = 'never'
    writable_roots: tuple[Path | str, ...] = ()
    codex_home: Path | str | None = None
    skip_git_repo_check: bool = False
    ephemeral: bool = False
    max_iterations: int | None = None
    model_auto_compact_token_limit: int | None = None
    agent_depth: int = 0
    web_search_external_web_access: bool = False
    web_search_filters: dict[str, Any] | None = None
    web_search_user_location: dict[str, Any] | None = None
    web_search_context_size: Literal['low', 'medium', 'high'] | None = None
    web_search_content_types: tuple[str, ...] | None = None
    collaboration_mode: CollaborationMode = 'Default'
    approval_provider: Any | None = None
    hook_provider: Any | None = None
    request_user_input_answers: dict[str, Any] | None = None
    request_user_input_provider: Any | None = None
    model_supports_image_input: bool | None = None
    model_supports_image_detail_original: bool | None = None
    memory_tool_enabled: bool = False
    memory_disable_on_external_context: bool = False
    use_memories: bool = True
    memory_state_store: Any | None = None
    memory_startup_background: bool = True
    memory_run_phase2_on_startup: bool = True
    memory_max_rollout_age_days: int = 10
    memory_min_rollout_idle_hours: int = 6
    model_stream_max_retries: int | None = None
    model_stream_retry_base_delay_ms: int = 200
    output_schema: dict[str, Any] | None = None
    input_images: tuple[Path | str, ...] = ()
    remote_compaction: RemoteCompactionMode = field(default_factory=lambda: "auto")
    current_date: str | None = None
    timezone: str | None = None

    # Extra configuration parameters to support resolved methods
    output_last_message: Path | str | None = None
    model_reasoning_effort: str | None = None
    model_reasoning_summary: str | None = None
    model_client: Any = None
    config: Any = None

    def resolved_cwd(self) -> Path:
        """Resolves active process working directory relative to standard path rules."""
        return Path(self.cwd).expanduser().resolve()

    def resolved_codex_home(self) -> Path:
        """Resolves custom codex runtime folder path, falling back to home subdirectory."""
        home = self.codex_home if self.codex_home is not None else Path.home() / ".codex"
        return Path(home).expanduser().resolve()

    def _get_model_info(self) -> dict[str, Any]:
        catalog = _load_model_catalog(self.resolved_codex_home())
        return catalog.get(self.model) or catalog.get("gpt-5.5") or {}

    def resolved_auto_compact_token_limit(self) -> int | None:
        """Resolves auto compaction limit, looking up default model catalog limits."""
        if self.model_auto_compact_token_limit is not None:
            return self.model_auto_compact_token_limit
        model_info = self._get_model_info()
        return model_info.get("auto_compact_token_limit")

    def resolved_model_context_window(self) -> int | None:
        """Resolves token context capacity for the currently active LLM model."""
        model_info = self._get_model_info()
        return model_info.get("context_window")

    def resolved_model_stream_max_retries(self) -> int:
        """Resolves max reconnection retries allowed for streaming inference."""
        if self.model_stream_max_retries is not None:
            return self.model_stream_max_retries
        return 5  # default reconnect attempts (DEFAULT_STREAM_MAX_RETRIES)

    def resolved_model_stream_retry_base_delay_ms(self) -> int:
        """Resolves retry base delay in milliseconds."""
        return self.model_stream_retry_base_delay_ms

    def resolved_output_last_message(self) -> Path | None:
        """Resolves output file path for the final assistant response."""
        if self.output_last_message is not None:
            return Path(self.output_last_message).expanduser().resolve()
        return None

    def resolved_parallel_tool_calls(self) -> bool:
        """Resolves whether active model supports parallel model tool invocations."""
        model_info = self._get_model_info()
        return bool(model_info.get("supports_parallel_tool_calls", False))

    def resolved_reasoning(self) -> dict[str, str | None] | None:
        """Resolves active reasoning summaries configuration effort level and detail output limits."""
        model_info = self._get_model_info()
        if not model_info.get("supports_reasoning_summaries", False):
            return None
        
        effort = self.model_reasoning_effort or model_info.get("default_reasoning_level") or "medium"
        summary = self.model_reasoning_summary or model_info.get("default_reasoning_summary") or "none"
        
        return {
            "effort": effort,
            "summary": summary
        }

    def resolved_supports_image_input(self) -> bool:
        """Resolves whether active model supports multimodal image inputs."""
        if self.model_supports_image_input is not None:
            return self.model_supports_image_input
        model_info = self._get_model_info()
        modalities = model_info.get("input_modalities", [])
        return "image" in modalities

    def resolved_tool_output_truncation_tokens(self) -> int:
        """Resolves truncation bounds for tool results to fit context constraints safely."""
        model_info = self._get_model_info()
        policy = model_info.get("truncation_policy") or {}
        if policy.get("mode") == "tokens" and policy.get("limit") is not None:
            return int(policy["limit"])
        return 10000  # fallback limit

    def resolved_verbosity(self) -> str | None:
        """Resolves verbosity detail level for reasoning and outputs."""
        model_info = self._get_model_info()
        if not model_info.get("support_verbosity", False):
            return None
        return model_info.get("default_verbosity")

@dataclass
class CodexEvent:
    """Structured event emitted by the Codex execution loop during task progress."""
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts event back into wire-compatible dictionary representations."""
        return {
            "type": self.type,
            "payload": self.payload
        }

    def to_json(self) -> str:
        """Serializes event to standardized JSON strings."""
        return json.dumps(self.to_dict())

@dataclass
class CodexResult:
    """Unified outcome representing execution results after driving or compacting conversation turns."""
    final_message: str
    events: list[CodexEvent]
    thread_id: str
    turn_id: str
    history: list[dict[str, Any]]
    memory_citations: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class LifecycleStep:
    """Formal declaration of individual sequential phases inside executing turns."""
    name: str
    event_type: str
    phase: LifecyclePhase
    terminal: bool = False
    mutates_history: bool = False

@dataclass
class ModelResponse:
    """Inference response mapped back from the OpenAI Responses client structure."""
    id: str
    output: list[dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptRequest:
    """Consolidated input block structured for submission to the ModelClient layer."""
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

    def to_responses_kwargs(self) -> dict[str, Any]:
        """Extracts and maps parameters into exact arguments accepted by the Responses API client."""
        kwargs = {
            "model": self.model,
            "instructions": self.instructions,
            "input": self.input,
            "tools": self.tools,
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        if self.prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = self.prompt_cache_key
        if self.reasoning is not None:
            kwargs["reasoning"] = self.reasoning
        if self.include:
            kwargs["include"] = self.include
        if self.output_schema is not None:
            kwargs["output_schema"] = self.output_schema
            kwargs["output_schema_strict"] = self.output_schema_strict
        if self.verbosity is not None:
            kwargs["verbosity"] = self.verbosity
        if self.service_tier is not None:
            kwargs["service_tier"] = self.service_tier
        if self.client_metadata is not None:
            kwargs["client_metadata"] = self.client_metadata
        return kwargs

    def to_compact_payload(self) -> dict[str, Any]:
        """Truncates massive history payloads into highly compressed formats safe for telemetry logs."""
        compact_input = []
        for item in self.input:
            compact_item = dict(item)
            if "content" in compact_item:
                content = compact_item["content"]
                if isinstance(content, str) and len(content) > 100:
                    compact_item["content"] = content[:100] + "... [truncated]"
                elif isinstance(content, list):
                    compact_item["content"] = [
                        (c[:100] + "... [truncated]" if isinstance(c, str) and len(c) > 100 else c)
                        for c in content
                    ]
            compact_input.append(compact_item)
            
        payload = {
            "model": self.model,
            "instructions_len": len(self.instructions),
            "input_turns_count": len(self.input),
            "input_compact": compact_input,
            "tools_count": len(self.tools),
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if self.output_schema is not None:
            payload["output_schema"] = self.output_schema
        return payload


class ConfigurationError(Exception):
    """Fatal system configuration and static assets loading exceptions."""
    pass

