"""
Configuration Wrappers for the Codex Engine.
Defines CodexConfig, bridging standard python configurations to the Rust config schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import os
from typing import Any, Literal

# Define type aliases for compliance with API surface annotations
SandboxMode = Literal['read-only', 'workspace-write', 'danger-full-access']
ApprovalPolicy = Literal['never', 'unless-trusted', 'on-failure', 'on-request']
CollaborationMode = Any  # Can be a string literal like 'Default' or a helper class instance
RemoteCompactionMode = Literal['disabled', 'auto', 'force']


def _resolve_default_model() -> str:
    """Resolve default model slug from models.json assets."""
    try:
        # Try relative paths first, targeting codex/assets/models.json
        assets_dir = Path(__file__).parent / "assets"
        models_json_path = assets_dir / "models.json"
        if not models_json_path.exists():
            # Try upstream relative search
            upstream_path = Path(__file__).parents[1] / "upstream/openai-codex/codex-rs/models-manager/models.json"
            if upstream_path.exists():
                models_json_path = upstream_path
        
        if models_json_path.exists():
            with open(models_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    # Return slug of highest priority model, default to first item
                    for model_entry in data:
                        if model_entry.get("priority") == 0:
                            return model_entry.get("slug", "gpt-5.5")
                    return data[0].get("slug", "gpt-5.5")
    except Exception:
        pass
    return "gpt-5.5"


def _resolve_default_cwd() -> Path:
    """Resolve default current working directory."""
    return Path.cwd()


def _resolve_default_remote_compaction() -> str:
    """Resolve default remote compaction mode."""
    return "auto"


@dataclass
class CodexConfig:
    """Configuration options for a Codex session."""
    model: str = field(default_factory=_resolve_default_model)
    cwd: Path | str = field(default_factory=_resolve_default_cwd)
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
    remote_compaction: RemoteCompactionMode = field(default_factory=_resolve_default_remote_compaction)
    current_date: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        # Validate sandbox and approval options constraints
        valid_sandboxes = {'read-only', 'workspace-write', 'danger-full-access'}
        if self.sandbox not in valid_sandboxes:
            raise ValueError(f"Unsupported sandbox profile: {self.sandbox}")
            
        valid_policies = {'never', 'unless-trusted', 'on-failure', 'on-request'}
        if self.approval_policy not in valid_policies:
            raise ValueError(f"Unsupported approval policy: {self.approval_policy}")

        # Standardize paths to Path objects
        if isinstance(self.cwd, str):
            self.cwd = Path(self.cwd)
        if isinstance(self.codex_home, str):
            self.codex_home = Path(self.codex_home)
        elif self.codex_home is None:
            self.codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
            
        # Ensure writable_roots are standard Path elements
        self.writable_roots = tuple(Path(p) if isinstance(p, str) else p for p in self.writable_roots)
        
        # Standardize input images
        self.input_images = tuple(Path(p) if isinstance(p, str) else p for p in self.input_images)
