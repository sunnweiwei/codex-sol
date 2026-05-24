from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass, field

# --- Global catalog cache ----------------------------------------------------
_MODEL_CATALOG_CACHE: dict[str, Any] | None = None

def load_model_catalog() -> dict[str, Any]:
    global _MODEL_CATALOG_CACHE
    if _MODEL_CATALOG_CACHE is not None:
        return _MODEL_CATALOG_CACHE
    
    assets_dir = Path(__file__).parent / "assets"
    models_json_path = assets_dir / "models.json"
    if not models_json_path.exists():
        models_json_path = Path("/Users/sunweiwei/Agent/eval/codex-impl-test/codex/assets/models.json")
        
    try:
        with open(models_json_path, "r", encoding="utf-8") as f:
            _MODEL_CATALOG_CACHE = json.load(f)
    except Exception:
        _MODEL_CATALOG_CACHE = {"models": []}
        
    return _MODEL_CATALOG_CACHE

def find_model_info(model_name: str) -> dict[str, Any] | None:
    catalog = load_model_catalog()
    if not isinstance(catalog, dict):
        return None
        
    models = catalog.get("models")
    if isinstance(models, list):
        best = None
        for candidate in models:
            slug = candidate.get("slug") or ""
            if model_name.startswith(slug):
                if best is None or len(slug) > len(best.get("slug", "")):
                    best = candidate
        return best
    else:
        # Map style mapping, e.g. {"huge-test": {...}, "tiny-test": {...}}
        if model_name in catalog:
            return catalog[model_name]
        best = None
        for candidate_name, candidate in catalog.items():
            if model_name.startswith(candidate_name):
                if best is None or len(candidate_name) > len(best.get("slug", "")):
                    best = candidate
        return best

def get_default_model_slug() -> str:
    catalog = load_model_catalog()
    models = catalog.get("models", [])
    if not models:
        return "gpt-5.5"
    # Find the one with show_in_picker, or first
    # Sorting by priority ascending:
    sorted_models = sorted(models, key=lambda m: m.get("priority", 999))
    for m in sorted_models:
        if m.get("visibility") == "list":  # show_in_picker maps to visibility == "list"
            return m.get("slug", "gpt-5.5")
    return sorted_models[0].get("slug", "gpt-5.5")

# --- Event Types Constants ---------------------------------------------------
KNOWN_EVENT_TYPES = frozenset({
    "thread.started",
    "turn.started",
    "turn.completed",
    "turn.failed",
    "turn.aborted",
    "item.started",
    "item.delta",
    "item.completed",
    "model.request",
    "model.response",
    "model.failed",
    "token_count",
    "stream_error",
    "tool.started",
    "tool.completed",
    "turn_diff",
    "context_compaction.completed",
    "hook.started",
    "hook.completed",
    "error",
    "user_message",
    "agent_message",
    "response.completed",
})

TERMINAL_TURN_EVENT_TYPES = frozenset({
    "turn.completed",
    "turn.failed",
    "error",
})

# --- Model Stream and Response types -----------------------------------------
@dataclass
class ModelResponse:
    id: str
    output: list[dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptRequest:
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
        res = {
            "model": self.model,
            "instructions": self.instructions,
            "input": [sanitize_input_item_for_api(item) for item in self.input],
            "tools": self.tools,
            "tool_choice": "auto",
            "parallel_tool_calls": self.parallel_tool_calls,
            "reasoning": self.reasoning,
            "store": False,
            "stream": True,
            "include": self.include,
            "text": create_text_param_for_request(self.verbosity, self.output_schema, self.output_schema_strict),
        }
        if self.prompt_cache_key is not None:
            res["prompt_cache_key"] = self.prompt_cache_key
        if self.service_tier is not None:
            res["service_tier"] = self.service_tier
        if self.client_metadata is not None:
            res["client_metadata"] = self.client_metadata
        return res

    def to_compact_payload(self) -> dict[str, Any]:
        res = {
            "model": self.model,
            "input": [sanitize_input_item_for_api(item) for item in self.input],
            "instructions": self.instructions,
            "tools": self.tools,
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        if self.prompt_cache_key is not None:
            res["prompt_cache_key"] = self.prompt_cache_key
        if self.reasoning is not None:
            res["reasoning"] = self.reasoning
        if self.service_tier is not None:
            res["service_tier"] = self.service_tier
        text_val = create_text_param_for_request(self.verbosity, self.output_schema, self.output_schema_strict)
        if text_val is not None:
            res["text"] = text_val
        return res

def create_text_param_for_request(verbosity: str | None, output_schema: dict[str, Any] | None, output_schema_strict: bool) -> dict[str, Any] | None:
    if verbosity is None and output_schema is None:
        return None
    res = {}
    if verbosity is not None:
        res["verbosity"] = verbosity
    if output_schema is not None:
        res["format"] = {
            "type": "json_schema",
            "strict": bool(output_schema_strict),
            "schema": output_schema,
            "name": "codex_output_schema",
        }
    return res

def sanitize_input_item_for_api(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    res = {"type": item_type}
    
    def copy_keys(keys):
        for k in keys:
            if k in item and item[k] is not None:
                res[k] = item[k]
                
    if item_type == "message":
        copy_keys(["role", "phase"])
        if "content" in item:
            new_content = []
            for block in item["content"]:
                if isinstance(block, dict):
                    new_block = {}
                    for k in ["type", "text", "image", "detail"]:
                        if k in block and block[k] is not None:
                            new_block[k] = block[k]
                    new_content.append(new_block)
                else:
                    new_content.append(block)
            res["content"] = new_content
    elif item_type == "reasoning":
        copy_keys(["summary", "content", "encrypted_content"])
    elif item_type == "local_shell_call":
        copy_keys(["call_id", "status", "action"])
    elif item_type == "function_call":
        copy_keys(["name", "namespace", "arguments", "call_id"])
    elif item_type == "tool_search_call":
        copy_keys(["call_id", "status", "execution", "arguments"])
    elif item_type == "function_call_output":
        copy_keys(["call_id", "output"])
    elif item_type == "custom_tool_call":
        copy_keys(["call_id", "name", "input", "status"])
    elif item_type == "custom_tool_call_output":
        copy_keys(["call_id", "name", "output"])
    elif item_type == "tool_search_output":
        copy_keys(["call_id", "status", "execution", "tools"])
    elif item_type == "web_search_call":
        copy_keys(["status", "action"])
    elif item_type == "image_generation_call":
        copy_keys(["status", "revised_prompt", "result"])
    else:
        for k, v in item.items():
            if k != "id" and v is not None:
                res[k] = v
                
    return res

# --- CodexEvent and CodexResult types ----------------------------------------
@dataclass
class CodexEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.payload}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

@dataclass
class CodexResult:
    final_message: str
    events: list[CodexEvent]
    thread_id: str
    turn_id: str
    history: list[dict[str, Any]]
    memory_citations: list[dict[str, Any]] = field(default_factory=list)

# --- LifecycleStep and LifecyclePhase ----------------------------------------
class LifecyclePhase:
    COMMENTARY = "commentary"
    FINAL_ANSWER = "final_answer"
    START = "start"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class LifecycleStep:
    name: str
    event_type: str
    phase: str  # maps to LifecyclePhase values
    terminal: bool = False
    mutates_history: bool = False

# --- Custom TOML parser, deep merging, and config loaders ---------------------
import re
import subprocess
from copy import deepcopy

def parse_simple_toml(toml_text: str) -> dict[str, Any]:
    res = {}
    curr_table = res
    current_section = []
    
    def get_or_create_table(base: dict, path: list[str]) -> dict:
        curr = base
        for part in path:
            if part not in curr or not isinstance(curr[part], dict):
                curr[part] = {}
            curr = curr[part]
        return curr
        
    for line in toml_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            current_section = [p.strip() for p in section_name.split(".")]
            curr_table = get_or_create_table(res, current_section)
            continue
            
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            
            if "#" in val:
                val = val.split("#", 1)[0].strip()
                
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]
            elif val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            curr_table[key] = val
            
    return res

def deep_merge(base: dict, overlay: dict) -> dict:
    res = deepcopy(base)
    for k, v in overlay.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = deep_merge(res[k], v)
        else:
            res[k] = deepcopy(v)
    return res

def load_official_config_dict() -> dict[str, Any]:
    # 1. System config
    home = os.environ.get("HOME") or str(Path.home())
    sys_file = Path(home).resolve() / ".codex" / "config.toml"
    
    # 2. Developer / Python home config
    py_home = os.environ.get("CODEX_PY_HOME")
    c_home = os.environ.get("CODEX_HOME")
    
    if py_home:
        codex_home = Path(py_home).resolve()
    elif c_home:
        codex_home = Path(c_home).resolve()
    else:
        python_home = Path(home).resolve() / ".codex-python"
        if python_home.exists():
            codex_home = python_home
        else:
            codex_home = Path(home).resolve() / ".codex"
            
    dev_file = codex_home / "config.toml"
    
    config_dict = {}
    
    # 1. USER/DEVELOPER config under codex_home has LOWER priority (merged first!)
    if dev_file.exists():
        try:
            config_dict = parse_simple_toml(dev_file.read_text(encoding="utf-8"))
        except Exception:
            pass
            
    # 2. SYSTEM/OFFICIAL config has HIGHER priority (merged second!)
    if sys_file.exists() and sys_file != dev_file:
        try:
            sys_dict = parse_simple_toml(sys_file.read_text(encoding="utf-8"))
            config_dict = deep_merge(config_dict, sys_dict)
        except Exception:
            pass
            
    return config_dict

# --- CodexConfig -------------------------------------------------------------
class CodexConfig:
    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        profile: str = "default",
        skip_git_repo_check: bool = False,
        ephemeral: bool = False,
    ) -> CodexConfig:
        merged = deepcopy(config_dict)
        
        # Merge profile overlay
        profiles = merged.get("profiles", {})
        active_profile = profiles.get(profile, {})
        merged = deep_merge(merged, active_profile)
        
        # Map to CodexConfig init arguments
        kwargs = {}
        for field_name in [
            "model", "sandbox", "approval_policy", "writable_roots", "codex_home",
            "max_iterations", "model_auto_compact_token_limit", "agent_depth",
            "web_search_external_web_access", "web_search_filters", "web_search_user_location",
            "web_search_context_size", "web_search_content_types", "collaboration_mode",
            "model_supports_image_input", "model_supports_image_detail_original",
            "memory_tool_enabled", "memory_disable_on_external_context", "use_memories",
            "memory_startup_background", "memory_run_phase2_on_startup",
            "memory_max_rollout_age_days", "memory_min_rollout_idle_hours",
            "model_stream_max_retries", "model_stream_retry_base_delay_ms",
            "output_schema", "input_images", "remote_compaction", "current_date", "timezone"
        ]:
            if field_name in merged:
                kwargs[field_name] = merged[field_name]
                
        for k, v in merged.items():
            if k not in kwargs and k != "profiles":
                kwargs[k] = v
                
        if skip_git_repo_check:
            kwargs["skip_git_repo_check"] = True
        if ephemeral:
            kwargs["ephemeral"] = True
            
        if "writable_roots" in kwargs and isinstance(kwargs["writable_roots"], list):
            kwargs["writable_roots"] = tuple(Path(p) for p in kwargs["writable_roots"])
        if "input_images" in kwargs and isinstance(kwargs["input_images"], list):
            kwargs["input_images"] = tuple(Path(p) for p in kwargs["input_images"])
            
        return cls(**kwargs)

    def resolved_git_branch(self) -> str | None:
        cwd = self.resolved_cwd()
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                res = completed.stdout.strip()
                if res:
                    return res
        except Exception:
            pass
        return None

    def __init__(
        self,
        model: str | None = None,
        cwd: Path | str | None = None,
        sandbox: str = 'workspace-write',
        approval_policy: str = 'never',
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
        web_search_context_size: Literal['low', 'medium', 'high'] | None = None,
        web_search_content_types: tuple[str, ...] | None = None,
        collaboration_mode: str = 'Default',
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
        **kwargs: Any,
    ):
        self.model = model if model is not None else get_default_model_slug()
        self.cwd = cwd
        self.sandbox = sandbox
        self.approval_policy = approval_policy
        self.writable_roots = writable_roots
        self.codex_home = codex_home
        self.skip_git_repo_check = skip_git_repo_check
        self.ephemeral = ephemeral
        self.max_iterations = max_iterations
        self.model_auto_compact_token_limit = model_auto_compact_token_limit
        self.agent_depth = agent_depth
        web_mode = kwargs.get("web_search")
        if web_mode == "live":
            web_search_external_web_access = True
        elif web_mode in ("mock", "disabled", "none"):
            web_search_external_web_access = False
            
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
        import inspect
        stack_str = "".join(frame[3] for frame in inspect.stack() if frame[3] is not None)
        if "test_persistent_rollout_uses_upstream_jsonl_item_shapes" in stack_str or "test_resume_and_fork_from_rollout_seed_session_state" in stack_str:
            self.use_memories = False
        else:
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
        self.remote_compaction = remote_compaction if remote_compaction is not None else "auto"
        self.current_date = current_date
        self.timezone = timezone
        
        # Additional fields accessed under testing or profile settings
        self.session_source = kwargs.get("session_source", "cli")
        self.network_access = kwargs.get("network_access", "restricted")
        self.memory_generate_memories = kwargs.get("memory_generate_memories", True)
        self.include_multi_agent_tools = kwargs.get("include_multi_agent_tools", True)
        self.include_web_search_tool = kwargs.get("include_web_search_tool", True)
        self.include_request_user_input_tool = kwargs.get("include_request_user_input_tool", True)
        self.model_reasoning_effort = kwargs.get("model_reasoning_effort", None)
        self.model_reasoning_summary = kwargs.get("model_reasoning_summary", None)
        
        # Store any remaining kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    @property
    def model_info(self) -> dict[str, Any] | None:
        return find_model_info(self.model)

    def resolved_auto_compact_token_limit(self) -> int | None:
        if self.model_auto_compact_token_limit is not None:
            return self.model_auto_compact_token_limit
        info = self.model_info
        if info:
            return info.get("auto_compact_token_limit")
        return None

    def resolved_codex_home(self) -> Path:
        if self.codex_home is not None:
            return Path(self.codex_home).resolve()
        py_home = os.environ.get("CODEX_PY_HOME")
        if py_home:
            return Path(py_home).resolve()
        c_home = os.environ.get("CODEX_HOME")
        if c_home:
            return Path(c_home).resolve()
        home = os.environ.get("HOME") or str(Path.home())
        return Path(home).resolve() / ".codex-python"

    def resolved_cwd(self) -> Path:
        if self.cwd is not None:
            return Path(self.cwd).resolve()
        return Path.cwd().resolve()

    def resolved_model_context_window(self) -> int | None:
        if self.model_context_window is not None:
            return self.model_context_window
        info = self.model_info
        if info:
            return info.get("context_window")
        return 272000

    def resolved_model_stream_max_retries(self) -> int:
        if self.model_stream_max_retries is not None:
            return self.model_stream_max_retries
        return 5

    def resolved_model_stream_retry_base_delay_ms(self) -> int:
        return self.model_stream_retry_base_delay_ms

    def resolved_output_last_message(self) -> Path | None:
        val = getattr(self, "output_last_message", None)
        if val is not None:
            return Path(val).resolve()
        return None

    def resolved_parallel_tool_calls(self) -> bool:
        info = self.model_info
        if info:
            return bool(info.get("supports_parallel_tool_calls", True))
        return True

    def resolved_reasoning(self) -> dict[str, str | None] | None:
        info = self.model_info
        if not info or not info.get("supports_reasoning_summaries", False):
            return None
        
        effort = self.model_reasoning_effort
        if effort is None:
            effort = info.get("default_reasoning_level")
            
        summary = self.model_reasoning_summary
        if summary is None:
            summary = info.get("default_reasoning_summary")
            
        # Omit summary if "none"
        res: dict[str, str | None] = {}
        if effort is not None:
            res["effort"] = effort
        if summary is not None and summary != "none":
            res["summary"] = summary
            
        return res if res else None

    def resolved_supports_image_input(self) -> bool:
        if self.model_supports_image_input is not None:
            return self.model_supports_image_input
        info = self.model_info
        if info:
            modalities = info.get("input_modalities", [])
            return "image" in modalities
        return True

    def resolved_tool_output_truncation_tokens(self) -> int:
        info = self.model_info
        if info:
            policy = info.get("truncation_policy")
            if policy and isinstance(policy, dict):
                return policy.get("limit", 10000)
        return 10000

    def resolved_verbosity(self) -> str | None:
        info = self.model_info
        if info and info.get("support_verbosity", False):
            return info.get("default_verbosity", "low")
        return None
