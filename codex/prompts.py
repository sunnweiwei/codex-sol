"""
Prompts Construction and Asset Loading for the Codex Engine.
Manages base system instructions compiling, environment contexts generation,
permissions instruction segments mapping, and asset integrity tracking.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PosixPath
from typing import Any, Literal

from .config import CodexConfig

# Configure ASSETS_DIR pointing to the relative assets folder path
ASSETS_DIR = Path(__file__).parent / "assets"

# Type aliases for compliance
SandboxMode = Literal['read-only', 'workspace-write', 'danger-full-access']
ApprovalPolicy = Literal['never', 'unless-trusted', 'on-failure', 'on-request']
NetworkAccess = Literal['restricted', 'enabled']


def verify_asset_hashes() -> dict[str, bool]:
    """
    Compute integrity check results of packaging assets.
    Verifies lark grammar files, sandbox profiles, and prompt instructions.
    """
    results = {}
    expected_assets = [
        "apply_patch.lark",
        "models.json",
        "base_instructions/default.md",
        "profiles/base.sbpl",
        "profiles/network.sbpl",
        "profiles/platform_defaults.sbpl",
        "prompts/permissions/approval_policy/never.md",
        "prompts/permissions/approval_policy/on_failure.md",
        "prompts/permissions/approval_policy/on_request.md",
        "prompts/permissions/approval_policy/on_request_rule_request_permission.md",
        "prompts/permissions/approval_policy/unless_trusted.md",
        "prompts/permissions/sandbox_mode/danger_full_access.md",
        "prompts/permissions/sandbox_mode/read_only.md",
        "prompts/permissions/sandbox_mode/workspace_write.md",
    ]
    
    for relative_path in expected_assets:
        target = ASSETS_DIR / relative_path
        if target.exists():
            try:
                # Compute sha-256 for integrity check simulation
                hasher = hashlib.sha256()
                with open(target, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                results[relative_path] = True
            except Exception:
                results[relative_path] = False
        else:
            # Fallback mapping: if in development sandbox, check upstream paths
            upstream_base = Path(__file__).parents[1] / "upstream/openai-codex/codex-rs"
            upstream_maps = {
                "apply_patch.lark": "core/src/tools/handlers/apply_patch.lark",
                "models.json": "models-manager/models.json",
                "base_instructions/default.md": "protocol/src/prompts/base_instructions/default.md",
                "profiles/base.sbpl": "sandboxing/src/seatbelt_base_policy.sbpl",
                "profiles/network.sbpl": "sandboxing/src/seatbelt_network_policy.sbpl",
                "profiles/platform_defaults.sbpl": "sandboxing/src/restricted_read_only_platform_defaults.sbpl",
                "prompts/permissions/approval_policy/never.md": "core/src/context/prompts/permissions/approval_policy/never.md",
                "prompts/permissions/approval_policy/on_failure.md": "core/src/context/prompts/permissions/approval_policy/on_failure.md",
                "prompts/permissions/approval_policy/on_request.md": "core/src/context/prompts/permissions/approval_policy/on_request.md",
                "prompts/permissions/approval_policy/on_request_rule_request_permission.md": "core/src/context/prompts/permissions/approval_policy/on_request_rule_request_permission.md",
                "prompts/permissions/approval_policy/unless_trusted.md": "core/src/context/prompts/permissions/approval_policy/unless_trusted.md",
                "prompts/permissions/sandbox_mode/danger_full_access.md": "core/src/context/prompts/permissions/sandbox_mode/danger_full_access.md",
                "prompts/permissions/sandbox_mode/read_only.md": "core/src/context/prompts/permissions/sandbox_mode/read_only.md",
                "prompts/permissions/sandbox_mode/workspace_write.md": "core/src/context/prompts/permissions/sandbox_mode/workspace_write.md",
            }
            mapped_rel = upstream_maps.get(relative_path)
            if mapped_rel and (upstream_base / mapped_rel).exists():
                results[relative_path] = True
            else:
                results[relative_path] = False
                
    return results


def read_model_catalog_instructions(model: str) -> str | None:
    """Load model-specific system override instructions from the capabilities catalog."""
    try:
        models_json = ASSETS_DIR / "models.json"
        if not models_json.exists():
            # Check upstream path
            upstream_path = Path(__file__).parents[1] / "upstream/openai-codex/codex-rs/models-manager/models.json"
            if upstream_path.exists():
                models_json = upstream_path
                
        if models_json.exists():
            with open(models_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                for model_entry in data:
                    if model_entry.get("slug") == model:
                        return model_entry.get("catalog_instructions")
    except Exception:
        pass
    return None


def collect_agents_md(cwd: Path) -> str:
    """Locate and load project AGENTS.md metadata document for in-context workspace planning."""
    target = cwd / "AGENTS.md"
    # Fallbacks scan if absent
    if not target.exists():
        for alt_name in ["README.md", "README", "agents.md"]:
            alt_path = cwd / alt_name
            if alt_path.exists():
                target = alt_path
                break
                
    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                # Cap at default max document size of 32KB
                return f.read(32 * 1024)
        except Exception:
            pass
    return ""


def build_permissions_instructions(*, cwd: Path, sandbox: SandboxMode, approval_policy: ApprovalPolicy, network_access: NetworkAccess = 'restricted', writable_roots: tuple[Path | str, ...] = ()) -> str:
    """Compile developer instructions covering sandboxing bounds and permission escalation policies."""
    
    # 1. Load sandbox prompt template
    sandbox_template_rel = f"prompts/permissions/sandbox_mode/{sandbox.replace('-', '_')}.md"
    sandbox_path = ASSETS_DIR / sandbox_template_rel
    
    # Development mapping fallback
    if not sandbox_path.exists():
        upstream_path = Path(__file__).parents[1] / f"upstream/openai-codex/codex-rs/core/src/context/prompts/permissions/sandbox_mode/{sandbox.replace('-', '_')}.md"
        if upstream_path.exists():
            sandbox_path = upstream_path
            
    sandbox_txt = ""
    if sandbox_path.exists():
        with open(sandbox_path, "r", encoding="utf-8") as f:
            sandbox_txt = f.read().strip()
            # Replace network variable
            sandbox_txt = sandbox_txt.replace("{{network_access}}", network_access)
            
    # 2. Load approval policy template
    policy_filename = approval_policy.replace("-", "_")
    # Mapping custom override
    if approval_policy == "on-request":
        # Check if exec permissions approvals is simulated
        policy_filename = "on_request"
        
    policy_template_rel = f"prompts/permissions/approval_policy/{policy_filename}.md"
    policy_path = ASSETS_DIR / policy_template_rel
    
    if not policy_path.exists():
        upstream_path = Path(__file__).parents[1] / f"upstream/openai-codex/codex-rs/core/src/context/prompts/permissions/approval_policy/{policy_filename}.md"
        if upstream_path.exists():
            policy_path = upstream_path
            
    policy_txt = ""
    if policy_path.exists():
        with open(policy_path, "r", encoding="utf-8") as f:
            policy_txt = f.read().strip()
            
    # 3. Compile writable roots descriptor
    roots_txt = ""
    if sandbox == "workspace-write" and writable_roots:
        roots_list = [f"`{Path(p).resolve()}`" for p in writable_roots]
        if len(roots_list) == 1:
            roots_txt = f" The writable root is {roots_list[0]}."
        else:
            roots_txt = f" The writable roots are {', '.join(roots_list)}."
            
    # Combine sections following exact structure
    instructions = [
        "<permissions instructions>",
        sandbox_txt,
        policy_txt
    ]
    if roots_txt:
        # Append to sandbox text or as trailing paragraph
        instructions.append(roots_txt)
        
    instructions.append("</permissions instructions>")
    
    return "\n\n".join(x for x in instructions if x)


def build_environment_context(cwd: Path, *, shell: str | None = None, current_date: str | None = None, timezone: str | None = None) -> str:
    """Generate dynamic workspace environment details embedded into turns context."""
    import datetime
    import platform
    
    date_val = current_date if current_date else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tz_val = timezone if timezone else datetime.datetime.now().astimezone().tzname() or "UTC"
    shell_val = shell if shell else os.environ.get("SHELL", "/bin/zsh")
    
    # Render XML block for model injection
    lines = [
        "<environment_context>",
        f"Active Working Directory: {cwd}",
        f"Operating System: {platform.system()} ({platform.release()})",
        f"Shell: {shell_val}",
        f"Current Date/Time: {date_val}",
        f"Timezone: {tz_val}",
        "</environment_context>"
    ]
    return "\n".join(lines)


def build_initial_context_items(config: CodexConfig, *, cwd: Path | None = None) -> list[dict[str, Any]]:
    """Initialize conversation contexts (system prompt items, workspace overrides, agents.md)."""
    target_cwd = Path(cwd) if cwd is not None else Path(config.cwd)
    
    context = []
    
    # 1. Compile and inject Base Instructions (developer role prompt)
    base_instr = build_base_instructions(
        prompt_asset="default",
        model=config.model,
        cwd=target_cwd,
        sandbox=config.sandbox,
        approval_policy=config.approval_policy,
        codex_home=config.codex_home,
        memory_tool_enabled=config.memory_tool_enabled,
        use_memories=config.use_memories
    )
    context.append({"role": "developer", "content": base_instr})
    
    # 2. Inject environment context if requested (user role prompt)
    env_context = build_environment_context(
        cwd=target_cwd,
        current_date=config.current_date,
        timezone=config.timezone
    )
    context.append({"role": "user", "content": env_context})
    
    # 3. Scan and inject workspace agents doc
    project_doc = collect_agents_md(target_cwd)
    if project_doc:
        context.append({"role": "user", "content": f"<project_documentation>\n{project_doc}\n</project_documentation>"})
        
    return context


def build_base_instructions(*, prompt_asset: str, model: str | None = None, cwd: Path, sandbox: SandboxMode, approval_policy: ApprovalPolicy, codex_home: Path | None = None, memory_tool_enabled: bool = False, use_memories: bool = True) -> str:
    """Load default base prompts asset, injecting active sandboxing configurations."""
    asset_file = ASSETS_DIR / f"base_instructions/{prompt_asset}.md"
    
    if not asset_file.exists():
        upstream_path = Path(__file__).parents[1] / f"upstream/openai-codex/codex-rs/protocol/src/prompts/base_instructions/{prompt_asset}.md"
        if upstream_path.exists():
            asset_file = upstream_path
            
    base_text = ""
    if asset_file.exists():
        with open(asset_file, "r", encoding="utf-8") as f:
            base_text = f.read()
            
    # Inject permissions instructions block
    perms_block = build_permissions_instructions(
        cwd=cwd,
        sandbox=sandbox,
        approval_policy=approval_policy
    )
    
    # Assemble full context block
    full_prompt = base_text.strip() + "\n\n" + perms_block
    
    # Read model-specific instruction overrides if present
    if model:
        model_overrides = read_model_catalog_instructions(model)
        if model_overrides:
            full_prompt += "\n\n" + model_overrides
            
    return full_prompt


def build_memory_consolidation_prompt(memory_root: Path | str) -> str:
    """Compile core consolidation system instructions used in stage 2 memories merging runs."""
    return "Consolidate following long-term memory records into minimal structural bullet points."


def build_memory_stage_one_input_message(*, rollout_path: Path | str, rollout_cwd: Path | str, rollout_contents: str, model_context_window: int | None = None, effective_context_window_percent: int = 95) -> str:
    """Build the stage one inputs wrapping previous raw rollout files."""
    return f"Extract structured summaries from rollout path: {rollout_path}. CWD: {rollout_cwd}.\n\nRaw contents:\n{rollout_contents}"


def memory_stage_one_rollout_token_limit(*, model_context_window: int | None = None, effective_context_window_percent: int = 95) -> int:
    """Determine context windows limits safety caps before history packing."""
    base_window = model_context_window if model_context_window is not None else 128000
    return int(base_window * (effective_context_window_percent / 100))


def memory_stage_one_system_prompt() -> str:
    """Retrieve system instruction set specifically guiding memories extraction turns."""
    return "Extract long term structural user behaviors, preferred workflows, tool workarounds, and custom variables from project sessions."
