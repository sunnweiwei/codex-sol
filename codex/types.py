from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("codex")

__all__ = [
    "CodexConfig",
    "CodexEvent",
    "CodexResult",
    "LifecycleStep",
    "ModelResponse",
    "PromptRequest",
    "KNOWN_EVENT_TYPES",
    "TERMINAL_TURN_EVENT_TYPES"
]


# Constants
KNOWN_EVENT_TYPES = frozenset({
    "error", "warning", "guardian_warning", "realtime_conversation_started",
    "realtime_conversation_realtime", "realtime_conversation_closed",
    "realtime_conversation_sdp", "model_reroute", "model_verification",
    "context_compacted", "thread_rolled_back", "turn_started", "turn_complete",
    "token_count", "agent_message", "user_message", "agent_reasoning",
    "agent_reasoning_raw_content", "agent_reasoning_section_break",
    "session_configured", "thread_goal_updated", "mcp_startup_update",
    "mcp_startup_complete", "mcp_tool_call_begin", "mcp_tool_call_end",
    "web_search_begin", "web_search_end", "image_generation_begin",
    "image_generation_end", "exec_command_begin", "exec_command_output_delta",
    "terminal_interaction", "exec_command_end", "view_image_tool_call",
    "exec_approval_request", "request_permissions", "request_user_input",
    "dynamic_tool_call_request", "dynamic_tool_call_response", "elicitation_request",
    "apply_patch_approval_request", "guardian_assessment", "deprecation_notice",
    "stream_error", "patch_apply_begin", "patch_apply_updated", "patch_apply_end",
    "turn_diff", "realtime_conversation_list_voices_response", "plan_update",
    "turn_aborted", "shutdown_complete", "entered_review_mode", "exited_review_mode",
    "raw_response_item", "item_started", "item_completed", "hook_started",
    "hook_completed", "agent_message_content_delta", "plan_delta",
    "reasoning_content_delta", "reasoning_raw_content_delta",
    "collab_agent_spawn_begin", "collab_agent_spawn_end", "collab_agent_interaction_begin",
    "collab_agent_interaction_end", "collab_waiting_begin", "collab_waiting_end",
    "collab_close_begin", "collab_close_end", "collab_resume_begin",
    "collab_resume_end"
})

TERMINAL_TURN_EVENT_TYPES = frozenset({"turn_complete", "turn_aborted", "task_complete"})

_MODEL_CATALOG_CACHE = None

# Catalog utilities
def _get_assets_dir() -> Path:
    local_assets = Path(__file__).parent / "assets"
    if local_assets.exists():
        return local_assets
    fallback = Path("/Users/sunweiwei/Agent/eval/codex-impl/your-solution/codex/assets")
    if fallback.exists():
        return fallback
    return local_assets

def _load_models_catalog() -> list[dict[str, Any]]:
    global _MODEL_CATALOG_CACHE
    if _MODEL_CATALOG_CACHE is not None:
        return _MODEL_CATALOG_CACHE
    
    assets_dir = _get_assets_dir()
    models_file = assets_dir / "models.json"
    if not models_file.exists():
        return []
    try:
        with open(models_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            catalog = data.get("models", [])
            _MODEL_CATALOG_CACHE = catalog
            return catalog
    except Exception as e:
        logger.error(f"Failed to load models catalog: {e}")
        return []

def get_model_preset(model_slug: str) -> dict[str, Any] | None:
    models = _load_models_catalog()
    if not models:
        return None
    
    # Match exactly by slug first
    for model in models:
        if model.get("slug") == model_slug:
            return model
            
    # Fallback to longest prefix match (matches Rust core prefix resolution)
    best_match = None
    for model in models:
        slug = model.get("slug", "")
        if model_slug.startswith(slug):
            if best_match is None or len(slug) > len(best_match.get("slug", "")):
                best_match = model
    return best_match

def get_default_model_from_catalog() -> str:
    models = _load_models_catalog()
    if not models:
        return "gpt-5.5"
    
    # Sort by priority ascending. Mark default picker visible.
    sorted_models = sorted(models, key=lambda m: m.get("priority", 99))
    for m in sorted_models:
        if m.get("visibility") == "list":
            return m.get("slug", "gpt-5.5")
    if sorted_models:
        return sorted_models[0].get("slug", "gpt-5.5")
    return "gpt-5.5"

def find_codex_home() -> Path:
    codex_home_env = os.environ.get("CODEX_HOME")
    if codex_home_env:
        path = Path(codex_home_env)
        if not path.exists():
            raise FileNotFoundError(f"CODEX_HOME points to {codex_home_env!r}, but that path does not exist")
        if not path.is_dir():
            raise ValueError(f"CODEX_HOME points to {codex_home_env!r}, but that path is not a directory")
        return path.resolve()
    else:
        return Path.home() / ".codex"


class CodexConfig:
    def __init__(
        self,
        model: str | None = None,
        cwd: Path | str | None = None,
        sandbox: str = "workspace-write",
        approval_policy: str = "never",
        writable_roots: tuple[Path | str, ...] = (),
        codex_home: Path | str | None = None,
        skip_git_repo_check: bool = False,
        ephemeral: bool = False,
        max_iterations: int | None = None,
        model_auto_compact_token_limit: int | None = None,
        agent_depth: int = 0,
        web_search_external_web_access: bool = False,
        web_search_filters: dict[str, Any] | None = None,
        web_search_user_location: dict[str, Any] | None = None,
        web_search_context_size: Literal["low", "medium", "high"] | None = None,
        web_search_content_types: tuple[str, ...] | None = None,
        collaboration_mode: str = "Default",
        approval_provider: Any | None = None,
        hook_provider: Any | None = None,
        request_user_input_answers: dict[str, Any] | None = None,
        request_user_input_provider: Any | None = None,
        model_supports_image_input: bool | None = None,
        model_supports_image_detail_original: bool | None = None,
        memory_tool_enabled: bool = False,
        memory_disable_on_external_context: bool = False,
        use_memories: bool = True,
        memory_state_store: Any | None = None,
        memory_startup_background: bool = True,
        memory_run_phase2_on_startup: bool = True,
        memory_max_rollout_age_days: int = 10,
        memory_min_rollout_idle_hours: int = 6,
        model_stream_max_retries: int | None = None,
        model_stream_retry_base_delay_ms: int = 200,
        output_schema: dict[str, Any] | None = None,
        input_images: tuple[Path | str, ...] = (),
        remote_compaction: str | None = None,
        current_date: str | None = None,
        timezone: str | None = None,
        *args: Any,
        **kwargs: Any
    ):
        self.model = model if model is not None else get_default_model_from_catalog()
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.sandbox = sandbox
        self.approval_policy = approval_policy
        self.writable_roots = writable_roots
        self.codex_home = Path(codex_home) if codex_home is not None else None
        self.skip_git_repo_check = skip_git_repo_check
        self.ephemeral = ephemeral
        self.max_iterations = max_iterations
        self.model_auto_compact_token_limit = model_auto_compact_token_limit
        self.agent_depth = agent_depth
        self.web_search_external_web_access = web_search_external_web_access
        self.web_search_filters = web_search_filters
        self.web_search_user_location = web_search_user_location
        self.web_search_context_size = web_search_context_size
        self.web_search_content_types = web_search_content_types
        self.collaboration_mode = collaboration_mode
        self.approval_provider = approval_provider
        self.hook_provider = hook_provider
        self.request_user_input_answers = request_user_input_answers
        self.request_user_input_provider = request_user_input_provider
        self.model_supports_image_input = model_supports_image_input
        self.model_supports_image_detail_original = model_supports_image_detail_original
        self.memory_tool_enabled = memory_tool_enabled
        self.memory_disable_on_external_context = memory_disable_on_external_context
        self.use_memories = use_memories
        self.memory_state_store = memory_state_store
        self.memory_startup_background = memory_startup_background
        self.memory_run_phase2_on_startup = memory_run_phase2_on_startup
        self.memory_max_rollout_age_days = memory_max_rollout_age_days
        self.memory_min_rollout_idle_hours = memory_min_rollout_idle_hours
        self.model_stream_max_retries = model_stream_max_retries
        self.model_stream_retry_base_delay_ms = model_stream_retry_base_delay_ms
        self.output_schema = output_schema
        self.input_images = input_images
        self.remote_compaction = remote_compaction if remote_compaction is not None else "never"
        self.current_date = current_date
        self.timezone = timezone
        
        # Capture all trailing metadata kwargs to avoid type crashes on inspect and allow extra fields
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def resolved_auto_compact_token_limit(self) -> int | None:
        if self.model_auto_compact_token_limit is not None:
            return self.model_auto_compact_token_limit
        preset = get_model_preset(self.model)
        if preset is None:
            return None
        context_window = preset.get("context_window") or preset.get("max_context_window")
        if context_window is not None:
            # derived limit is 90% of context window
            context_limit = (context_window * 9) // 10
            config_limit = preset.get("auto_compact_token_limit")
            if config_limit is not None:
                return min(config_limit, context_limit)
            return context_limit
        return preset.get("auto_compact_token_limit")

    def resolved_codex_home(self) -> Path:
        if self.codex_home is not None:
            return Path(self.codex_home).absolute()
        return find_codex_home()

    def resolved_cwd(self) -> Path:
        return Path(self.cwd).absolute()

    def resolved_model_context_window(self) -> int | None:
        preset = get_model_preset(self.model)
        if preset is None:
            return None
        return preset.get("context_window") or preset.get("max_context_window")

    def resolved_model_stream_max_retries(self) -> int:
        if self.model_stream_max_retries is not None:
            return self.model_stream_max_retries
        # Try to read standard model provider setting if stored as kwarg
        prov_retries = getattr(self, "stream_max_retries", None) or getattr(self, "request_max_retries", None)
        if prov_retries is not None:
            return int(prov_retries)
        return 5

    def resolved_model_stream_retry_base_delay_ms(self) -> int:
        # Default is already 200 in the constructor
        return self.model_stream_retry_base_delay_ms

    def resolved_output_last_message(self) -> Path | None:
        path = getattr(self, "last_message_file", None) or getattr(self, "output_last_message", None)
        return Path(path) if path is not None else None

    def resolved_parallel_tool_calls(self) -> bool:
        preset = get_model_preset(self.model)
        if preset is None:
            return True
        return preset.get("supports_parallel_tool_calls", True)

    def resolved_reasoning(self) -> dict[str, str | None] | None:
        preset = get_model_preset(self.model)
        if preset is None:
            return None
        
        # Verify model supports reasoning summaries
        supported = preset.get("supported_reasoning_levels") or preset.get("supported_reasoning_efforts")
        if not supported:
            return None
            
        effort = getattr(self, "model_reasoning_effort", None) or getattr(self, "reasoning_effort", None)
        summary = getattr(self, "model_reasoning_summary", None) or getattr(self, "reasoning_summary", None)
        
        if effort is None:
            effort = preset.get("default_reasoning_level") or preset.get("default_reasoning_effort")
        if summary is None:
            summary = preset.get("default_reasoning_summary")
            
        return {
            "effort": effort,
            "summary": summary
        }

    def resolved_supports_image_input(self) -> bool:
        if self.model_supports_image_input is not None:
            return self.model_supports_image_input
        preset = get_model_preset(self.model)
        if preset is None:
            return False
        modalities = preset.get("input_modalities", [])
        return "image" in modalities

    def resolved_tool_output_truncation_tokens(self) -> int:
        overridden = getattr(self, "tool_output_token_limit", None)
        if overridden is not None:
            return int(overridden)
            
        preset = get_model_preset(self.model)
        if preset is None:
            return 10000
            
        truncation_policy = preset.get("truncation_policy", {})
        limit = truncation_policy.get("limit", 10000)
        mode = truncation_policy.get("mode", "tokens")
        
        if mode == "bytes":
            # 4 bytes = 1 token approx (from approx_tokens_from_byte_count)
            return (limit + 3) // 4
        return limit

    def resolved_verbosity(self) -> str | None:
        verb = getattr(self, "model_verbosity", None) or getattr(self, "verbosity", None)
        if verb is not None:
            return str(verb)
        preset = get_model_preset(self.model)
        if preset is None:
            return "medium"
        return preset.get("default_verbosity", "medium")


class CodexEvent:
    def __init__(self, type: str, payload: dict[str, Any] = None):
        self.type = type
        self.payload = payload if payload is not None else {}

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "payload": self.payload}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class CodexResult:
    def __init__(
        self,
        final_message: str,
        events: list[CodexEvent],
        thread_id: str,
        turn_id: str,
        history: list[dict[str, Any]],
        memory_citations: list[dict[str, Any]] = None
    ):
        self.final_message = final_message
        self.events = events
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.history = history
        self.memory_citations = memory_citations if memory_citations is not None else []


class LifecycleStep:
    def __init__(
        self,
        name: str,
        event_type: str,
        phase: str,  # Maps to LifecyclePhase string Literal
        terminal: bool = False,
        mutates_history: bool = False
    ):
        self.name = name
        self.event_type = event_type
        self.phase = phase
        self.terminal = terminal
        self.mutates_history = mutates_history


class ModelResponse:
    def __init__(
        self,
        id: str,
        output: list[dict[str, Any]],
        raw: dict[str, Any] = None
    ):
        self.id = id
        self.output = output
        self.raw = raw if raw is not None else {}


class PromptRequest:
    def __init__(
        self,
        model: str,
        instructions: str,
        input: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parallel_tool_calls: bool = True,
        prompt_cache_key: str | None = None,
        reasoning: dict[str, Any] | None = None,
        include: list[str] = None,
        output_schema: dict[str, Any] | None = None,
        output_schema_strict: bool = True,
        verbosity: str | None = None,
        service_tier: str | None = None,
        client_metadata: dict[str, str] | None = None
    ):
        self.model = model
        self.instructions = instructions
        self.input = input
        self.tools = tools
        self.parallel_tool_calls = parallel_tool_calls
        self.prompt_cache_key = prompt_cache_key
        self.reasoning = reasoning
        self.include = include if include is not None else []
        self.output_schema = output_schema
        self.output_schema_strict = output_schema_strict
        self.verbosity = verbosity
        self.service_tier = service_tier
        self.client_metadata = client_metadata

    def to_compact_payload(self) -> dict[str, Any]:
        return self.to_responses_kwargs()

    def to_responses_kwargs(self) -> dict[str, Any]:
        text = None
        if self.verbosity is not None or self.output_schema is not None:
            text = {}
            if self.verbosity is not None:
                text["verbosity"] = self.verbosity.lower()
            if self.output_schema is not None:
                text["format"] = {
                    "type": "json_schema",
                    "strict": self.output_schema_strict,
                    "schema": self.output_schema,
                    "name": "codex_output_schema",
                }
                
        kwargs = {
            "model": self.model,
            "instructions": self.instructions,
            "input": self.input,
            "tools": self.tools,
            "tool_choice": "auto",
            "parallel_tool_calls": self.parallel_tool_calls,
            "store": True,
            "stream": True,
            "include": self.include,
        }
        if self.reasoning is not None:
            kwargs["reasoning"] = self.reasoning
        if self.service_tier is not None:
            kwargs["service_tier"] = self.service_tier
        if self.prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = self.prompt_cache_key
        if text is not None:
            kwargs["text"] = text
        if self.client_metadata is not None:
            kwargs["client_metadata"] = self.client_metadata
        return kwargs
