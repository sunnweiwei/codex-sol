import os
from typing import Any, Literal
from pathlib import Path
from enum import Enum

class CollaborationMode(str, Enum):
    DEFAULT = "Default"
    EXPERIMENTAL = "Experimental"

class SandboxMode(str, Enum):
    read_only = "read-only"
    workspace_write = "workspace-write"
    danger_full_access = "danger-full-access"

    # Compatibility aliases
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"

class ApprovalPolicy(str, Enum):
    never = "never"
    unless_trusted = "unless-trusted"
    on_failure = "on-failure"
    on_request = "on-request"
    granular = "granular"

    # Compatibility aliases
    NEVER = "never"
    UNLESS_TRUSTED = "unless-trusted"
    ON_FAILURE = "on-failure"
    ON_REQUEST = "on-request"
    GRANULAR = "granular"

class NetworkAccess(str, Enum):
    restricted = "restricted"
    bypass = "bypass"
    enabled = "enabled"  # Support client SDK integration alias

    # Compatibility aliases
    RESTRICTED = "restricted"
    BYPASS = "bypass"
    ENABLED = "enabled"

class CodexConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        fields = [
            "model", "cwd", "sandbox", "approval_policy", "writable_roots", "codex_home",
            "skip_git_repo_check", "ephemeral", "max_iterations", "model_auto_compact_token_limit",
            "agent_depth", "web_search_external_web_access", "web_search_filters",
            "web_search_user_location", "web_search_context_size", "web_search_content_types",
            "collaboration_mode", "approval_provider", "hook_provider", "request_user_input_answers",
            "request_user_input_provider", "model_supports_image_input",
            "model_supports_image_detail_original", "memory_tool_enabled",
            "memory_disable_on_external_context", "use_memories", "memory_state_store",
            "memory_startup_background", "memory_run_phase2_on_startup", "memory_max_rollout_age_days",
            "memory_min_rollout_idle_hours", "model_stream_max_retries", "model_stream_retry_base_delay_ms",
            "output_schema", "input_images", "remote_compaction", "current_date", "timezone"
        ]
        
        # 1. Defaults / Env initialization
        self.model = os.environ.get("CODEX_MODEL", "gpt-5.2-codex")
        self.cwd = Path.cwd()
        self.sandbox = SandboxMode.workspace_write
        self.approval_policy = ApprovalPolicy.never
        self.writable_roots = ()
        self.codex_home = os.environ.get("CODEX_HOME")
        self.skip_git_repo_check = False
        self.ephemeral = False
        self.max_iterations = None
        self.model_auto_compact_token_limit = None
        self.agent_depth = 0
        self.web_search_external_web_access = False
        self.web_search_filters = None
        self.web_search_user_location = None
        self.web_search_context_size = None
        self.web_search_content_types = None
        self.collaboration_mode = "Default"
        self.approval_provider = None
        self.hook_provider = None
        self.request_user_input_answers = None
        self.request_user_input_provider = None
        self.model_supports_image_input = None
        self.model_supports_image_detail_original = None
        self.memory_tool_enabled = False
        self.memory_disable_on_external_context = False
        self.use_memories = True
        self.memory_state_store = None
        self.memory_startup_background = True
        self.memory_run_phase2_on_startup = True
        self.memory_max_rollout_age_days = 10
        self.memory_min_rollout_idle_hours = 6
        self.model_stream_max_retries = None
        self.model_stream_retry_base_delay_ms = 200
        self.output_schema = None
        self.input_images = ()
        self.remote_compaction = "disabled"
        self.current_date = None
        self.timezone = None
        
        # 2. Position mapping
        for i, val in enumerate(args):
            if i < len(fields):
                setattr(self, fields[i], val)
                
        # 3. Keyword overrides
        for key, val in kwargs.items():
            setattr(self, key, val)
            
        # 4. Parse / Normalise Enum types if passed as strings
        if isinstance(self.sandbox, str):
            for enum_val in SandboxMode:
                if enum_val.value == self.sandbox:
                    self.sandbox = enum_val
                    break
        if isinstance(self.approval_policy, str):
            for enum_val in ApprovalPolicy:
                if enum_val.value == self.approval_policy:
                    self.approval_policy = enum_val
                    break
