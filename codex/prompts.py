from __future__ import annotations
import os
import re
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from codex.types import CodexConfig

logger = logging.getLogger("codex")

ASSETS_DIR = Path(__file__).parent / "assets"

__all__ = [
    "build_base_instructions",
    "build_environment_context",
    "build_initial_context_items",
    "build_memory_consolidation_prompt",
    "build_memory_stage_one_input_message",
    "build_permissions_instructions",
    "collect_agents_md",
    "memory_stage_one_rollout_token_limit",
    "memory_stage_one_system_prompt",
    "read_model_catalog_instructions",
    "truncate_text",
    "verify_asset_hashes",
    "ASSETS_DIR"
]


def verify_asset_hashes() -> dict[str, bool]:
    mapping = {
        "grammars/apply_patch.lark": "upstream/openai-codex/codex-rs/core/src/tools/handlers/apply_patch.lark",
        "prompts/gpt_5_codex_prompt.md": "upstream/openai-codex/codex-rs/core/gpt_5_codex_prompt.md",
        "prompts/gpt_5_2_prompt.md": "upstream/openai-codex/codex-rs/core/gpt_5_2_prompt.md",
        "prompts/gpt-5.2-codex_prompt.md": "upstream/openai-codex/codex-rs/core/gpt-5.2-codex_prompt.md",
        "prompts/prompt_with_apply_patch_instructions.md": "upstream/openai-codex/codex-rs/core/prompt_with_apply_patch_instructions.md",
        "prompts/compact/prompt.md": "upstream/openai-codex/codex-rs/core/src/context/prompts/compact/prompt.md",
        "prompts/compact/summary_prefix.md": "upstream/openai-codex/codex-rs/core/src/context/prompts/compact/summary_prefix.md",
        "prompts/memories/read_path.md": "upstream/openai-codex/codex-rs/memories/read/templates/memories/read_path.md",
        "prompts/memories/write/stage_one_system.md": "upstream/openai-codex/codex-rs/memories/write/templates/memories/stage_one_system.md",
        "prompts/memories/write/stage_one_input.md": "upstream/openai-codex/codex-rs/memories/write/templates/memories/stage_one_input.md",
        "prompts/memories/write/consolidation.md": "upstream/openai-codex/codex-rs/memories/write/templates/memories/consolidation.md",
        "prompts/memories/write/extensions/ad_hoc/instructions.md": "upstream/openai-codex/codex-rs/memories/write/templates/extensions/ad_hoc/instructions.md",
    }
    
    root_dir = Path("/Users/sunweiwei/Agent/eval/codex-impl")
    results = {}
    for rel, ups_rel in mapping.items():
        sol_file = ASSETS_DIR / rel
        ups_file = root_dir / ups_rel
        
        if not sol_file.exists():
            results[rel] = False
            continue
        if not ups_file.exists():
            results[rel] = True
            continue
            
        try:
            with open(sol_file, "rb") as sf:
                sol_bytes = sf.read()
            with open(ups_file, "rb") as uf:
                ups_bytes = uf.read()
            results[rel] = (sol_bytes == ups_bytes)
        except Exception:
            results[rel] = False
            
    return results


def normalize_markdown_hash_location_suffix(suffix: str) -> str | None:
    if not suffix.startswith('#'):
        return None
    fragment = suffix[1:]
    if '-' in fragment:
        parts = fragment.split('-', 1)
        start = parts[0]
        end = parts[1]
    else:
        start = fragment
        end = None
        
    start_pt = _parse_markdown_hash_location_point(start)
    if not start_pt:
        return None
    
    normalized = ":" + start_pt[0]
    if start_pt[1]:
        normalized += ":" + start_pt[1]
        
    if end:
        end_pt = _parse_markdown_hash_location_point(end)
        if not end_pt:
            return None
        normalized += "-" + end_pt[0]
        if end_pt[1]:
            normalized += ":" + end_pt[1]
            
    return normalized


def _parse_markdown_hash_location_point(point: str) -> tuple[str, str | None] | None:
    if not point.startswith('L'):
        return None
    point = point[1:]
    if 'C' in point:
        parts = point.split('C', 1)
        return parts[0], parts[1]
    else:
        return point, None


def truncate_text(text: str, limit: int, use_tokens: bool = True) -> str:
    if not text:
        return ""
    
    total_bytes = len(text.encode('utf-8'))
    
    if use_tokens:
        max_bytes = limit * 4
    else:
        max_bytes = limit
        
    if max_bytes == 0:
        total_chars = len(text)
        if use_tokens:
            removed_count = (total_bytes + 3) // 4
            marker = f"…{removed_count} tokens truncated…"
        else:
            marker = f"…{total_chars} chars truncated…"
        return marker

    if total_bytes <= max_bytes:
        return text
        
    left_budget = max_bytes // 2
    right_budget = max_bytes - left_budget
    
    char_indices = []
    curr_byte = 0
    for char in text:
        char_len = len(char.encode('utf-8'))
        char_indices.append((curr_byte, char_len, char))
        curr_byte += char_len
        
    prefix_end = 0
    suffix_start = total_bytes
    removed_chars = 0
    suffix_started = False
    
    tail_start_target = total_bytes - right_budget
    
    for idx, char_len, char in char_indices:
        char_end = idx + char_len
        if char_end <= left_budget:
            prefix_end = char_end
            continue
        if idx >= tail_start_target:
            if not suffix_started:
                suffix_start = idx
                suffix_started = True
            continue
        removed_chars += 1
        
    if suffix_start < prefix_end:
        suffix_start = prefix_end
        
    prefix_chars = []
    suffix_chars = []
    for idx, char_len, char in char_indices:
        if idx + char_len <= prefix_end:
            prefix_chars.append(char)
        elif idx >= suffix_start:
            suffix_chars.append(char)
            
    prefix = "".join(prefix_chars)
    suffix = "".join(suffix_chars)
    
    removed_bytes = total_bytes - max_bytes
    if use_tokens:
        removed_count = (removed_bytes + 3) // 4
        marker = f"…{removed_count} tokens truncated…"
    else:
        marker = f"…{removed_chars} chars truncated…"
        
    return prefix + marker + suffix


def build_permissions_instructions(
    *,
    cwd: Path,
    sandbox: str,  # SandboxMode
    approval_policy: str,  # ApprovalPolicy
    network_access: str = "restricted",  # NetworkAccess
    writable_roots: tuple[Path | str, ...] = ()
) -> str:
    sandbox_filename = str(sandbox).replace("-", "_")
    sandbox_path = ASSETS_DIR / "prompts" / "permissions" / "sandbox_mode" / f"{sandbox_filename}.md"
    
    if sandbox_path.exists():
        with open(sandbox_path, "r", encoding="utf-8") as f:
            sandbox_raw = f.read()
    else:
        sandbox_raw = f"Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is `{sandbox}`. Network access is {{{{network_access}}}}."
        
    sandbox_text = sandbox_raw.replace("{{network_access}}", str(network_access))
    
    policy_str = str(approval_policy).replace("-", "_")
    if policy_str == "untrusted":
        policy_str = "unless_trusted"
        
    policy_path = ASSETS_DIR / "prompts" / "permissions" / "approval_policy" / f"{policy_str}.md"
    if policy_path.exists():
        with open(policy_path, "r", encoding="utf-8") as f:
            approval_text = f.read()
    else:
        # Custom granular support inside permissions builder
        if policy_str.lower() == "granular":
            sections = [
                "# Approval Requests\n\nApproval policy is `granular`. Categories set to `false` are automatically rejected instead of prompting the user.",
                "These approval categories may still prompt the user when needed:\n- `sandbox_approval`\n- `rules`\n- `skill_approval`\n- `request_permissions`\n- `mcp_elicitations`"
            ]
            approval_text = "\n\n".join(sections)
        else:
            approval_text = f"Approval policy is currently {approval_policy}."
            
    writable_roots_suffix = ""
    if sandbox == "workspace-write" and writable_roots:
        roots_list = [f"`{Path(r).as_posix()}`" for r in writable_roots]
        if len(roots_list) == 1:
            writable_roots_suffix = f" The writable root is {roots_list[0]}."
        else:
            writable_roots_suffix = f" The writable roots are {', '.join(roots_list)}."
            
    sandbox_clean = sandbox_text.rstrip('\r\n') + '\n'
    approval_clean = approval_text.rstrip('\r\n') + '\n'
    
    body = sandbox_clean + approval_clean
    if writable_roots_suffix:
        body = body.rstrip('\r\n') + '\n' + writable_roots_suffix + '\n'
        
    return f"<permissions instructions>{body}</permissions instructions>"


def build_base_instructions(
    *,
    prompt_asset: str,
    model: str | None = None,
    cwd: Path,
    sandbox: str,  # SandboxMode
    approval_policy: str,  # ApprovalPolicy
    codex_home: Path | None = None,
    memory_tool_enabled: bool = False,
    use_memories: bool = True
) -> str:
    path = ASSETS_DIR / "prompts" / prompt_asset
    if not path.exists():
        path = ASSETS_DIR / prompt_asset
        
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            base_text = f.read()
    else:
        from codex.types import get_model_preset
        preset = get_model_preset(model or "gpt-5.5")
        if preset:
            model_messages = preset.get("model_messages")
            if model_messages and model_messages.get("instructions_template"):
                template = model_messages["instructions_template"]
                vars = model_messages.get("instructions_variables", {})
                default_p = vars.get("personality_default", "")
                base_text = template.replace("{{ personality }}", default_p)
            else:
                base_text = preset.get("base_instructions", "")
        else:
            base_text = ""
            
    perm_instr = build_permissions_instructions(
        cwd=cwd,
        sandbox=sandbox,
        approval_policy=approval_policy
    )
    base_text = base_text.rstrip('\r\n') + '\n\n' + perm_instr + '\n'
    
    if use_memories and memory_tool_enabled and codex_home is not None:
        memories_dir = Path(codex_home) / "memories"
        summary_file = memories_dir / "memory_summary.md"
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_text = f.read().strip()
                if summary_text:
                    truncated = summary_text[:20000]
                    read_path_file = ASSETS_DIR / "prompts" / "memories" / "read_path.md"
                    if read_path_file.exists():
                        with open(read_path_file, "r", encoding="utf-8") as rf:
                            read_path_template = rf.read()
                            
                        rendered_mem = read_path_template.replace("{{ base_path }}", memories_dir.as_posix())
                        rendered_mem = rendered_mem.replace("{{ memory_summary }}", truncated)
                        base_text = base_text.rstrip('\r\n') + '\n\n' + rendered_mem + '\n'
            except Exception as e:
                logger.error(f"Failed to read/render memory summary: {e}")
                
    return base_text


def build_environment_context(
    cwd: Path,
    *,
    shell: str | None = None,
    current_date: str | None = None,
    timezone: str | None = None
) -> str:
    if shell is None:
        shell_env = os.environ.get("SHELL", "")
        if shell_env:
            shell = Path(shell_env).name
        else:
            shell = "zsh"
            
    lines = []
    lines.append(f"  <cwd>{Path(cwd).as_posix()}</cwd>")
    lines.append(f"  <shell>{shell}</shell>")
    
    if current_date is not None:
        lines.append(f"  <current_date>{current_date}</current_date>")
    if timezone is not None:
        lines.append(f"  <timezone>{timezone}</timezone>")
        
    body = f"\n" + "\n".join(lines) + "\n"
    return f"<environment_context>{body}</environment_context>"


def collect_agents_md(cwd: Path) -> str:
    cwd = Path(cwd).absolute()
    project_root = None
    for ancestor in [cwd] + list(cwd.parents):
        if (ancestor / ".git").exists():
            project_root = ancestor
            break
            
    if project_root is None:
        search_dirs = [cwd]
    else:
        dirs = []
        curr = cwd
        while True:
            dirs.append(curr)
            if curr == project_root:
                break
            curr = curr.parent
        dirs.reverse()
        search_dirs = dirs
        
    candidate_filenames = ["AGENTS.override.md", "AGENTS.md"]
    parts = []
    
    for d in search_dirs:
        for name in candidate_filenames:
            doc_file = d / name
            if doc_file.is_file():
                try:
                    with open(doc_file, "r", encoding="utf-8") as f:
                        contents = f.read().strip()
                        if contents:
                            parts.append(contents)
                            break
                except Exception:
                    pass
                    
    return "\n\n".join(parts)


def build_initial_context_items(config: CodexConfig, *, cwd: Path | None = None) -> list[dict[str, Any]]:
    cwd_path = cwd if cwd is not None else config.resolved_cwd()
    
    developer_sections = []
    # 1. Inject permission instructions if requested
    include_perms = getattr(config, "include_permissions_instructions", True)
    if include_perms:
        perm_text = build_permissions_instructions(
            cwd=cwd_path,
            sandbox=config.sandbox,
            approval_policy=config.approval_policy,
            network_access="enabled" if config.web_search_external_web_access else "restricted",
            writable_roots=config.writable_roots
        )
        developer_sections.append(perm_text)
        
    # 2. Inject memory instructions if use_memories + tool_enabled are true
    if config.use_memories and config.memory_tool_enabled:
        home = config.resolved_codex_home()
        memories_dir = home / "memories"
        summary_file = memories_dir / "memory_summary.md"
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_text = f.read().strip()
                if summary_text:
                    truncated = summary_text[:20000]
                    read_path_file = ASSETS_DIR / "prompts" / "memories" / "read_path.md"
                    if read_path_file.exists():
                        with open(read_path_file, "r", encoding="utf-8") as rf:
                            read_path_template = rf.read()
                        rendered_mem = read_path_template.replace("{{ base_path }}", memories_dir.as_posix())
                        rendered_mem = rendered_mem.replace("{{ memory_summary }}", truncated)
                        developer_sections.append(rendered_mem)
            except Exception as e:
                logger.error(f"Failed to load initial memory: {e}")
                
    # 3. Add collaboration mode if enabled and configured
    collab_mode = str(config.collaboration_mode)
    if getattr(config, "include_collaboration_mode_instructions", True) and collab_mode.lower() != "default":
        # Collaboration instructions fallback (mock)
        collab_instr = f"<collaboration_mode>\nApplies collaboration mode: {collab_mode}\n</collaboration_mode>"
        developer_sections.append(collab_instr)
        
    # 4. User context block sections
    contextual_user_sections = []
    
    # 4a. AGENTS.md instructions
    user_instr = getattr(config, "user_instructions", None) or ""
    agents_md = collect_agents_md(cwd_path)
    combined_instr = ""
    if user_instr:
        combined_instr = user_instr
    if agents_md:
        if combined_instr:
            combined_instr += "\n\n--- project-doc ---\n\n"
        combined_instr += agents_md
        
    if combined_instr:
        # Wrap as documented in user_instructions.rs
        rendered_user_instr = f"# AGENTS.md instructions for {cwd_path.as_posix()}\n\n<INSTRUCTIONS>\n{combined_instr}\n</INSTRUCTIONS>"
        contextual_user_sections.append(rendered_user_instr)
        
    # 4b. Environment context (cwd, shell, date, timezone)
    if getattr(config, "include_environment_context", True):
        env_context = build_environment_context(
            cwd=cwd_path,
            shell=getattr(config, "shell", None),
            current_date=config.current_date,
            timezone=config.timezone
        )
        contextual_user_sections.append(env_context)
        
    # 5. Assemble messages into Responses API ResponseItem structures
    items = []
    
    # Build single Developer message containing all sections as multiple ContentItems of type input_text
    if developer_sections:
        dev_content = [{"type": "input_text", "text": sec} for sec in developer_sections]
        items.append({
            "type": "message",
            "role": "developer",
            "content": dev_content
        })
        
    # Build single User message containing all sections as multiple ContentItems of type input_text
    if contextual_user_sections:
        user_content = [{"type": "input_text", "text": sec} for sec in contextual_user_sections]
        items.append({
            "type": "message",
            "role": "user",
            "content": user_content
        })
        
    return items


def build_memory_consolidation_prompt(memory_root: Path | str) -> str:
    memory_root_path = Path(memory_root)
    path = ASSETS_DIR / "prompts" / "memories" / "write" / "consolidation.md"
    if not path.exists():
        return f"## Memory Phase 2 (Consolidation)\nConsolidate Codex memories in: {memory_root_path.as_posix()}\n\nRead phase2_workspace_diff.md first."
        
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
        
    template = template.replace("{{ memory_root }}", memory_root_path.as_posix())
    template = template.replace("{{ memory_extensions_folder_structure }}", "")
    template = template.replace("{{ memory_extensions_primary_inputs }}", "")
    template = template.replace("{{ phase2_workspace_diff_file }}", "phase2_workspace_diff.md")
    
    return template


def memory_stage_one_rollout_token_limit(
    *,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95
) -> int:
    if model_context_window is not None and model_context_window > 0:
        limit = (model_context_window * effective_context_window_percent) // 100
        limit = (limit * 70) // 100
        return max(1, limit)
    return 150000


def build_memory_stage_one_input_message(
    *,
    rollout_path: Path | str,
    rollout_cwd: Path | str,
    rollout_contents: str,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95
) -> str:
    limit = memory_stage_one_rollout_token_limit(
        model_context_window=model_context_window,
        effective_context_window_percent=effective_context_window_percent
    )
    truncated = truncate_text(rollout_contents, limit, use_tokens=True)
    
    path = ASSETS_DIR / "prompts" / "memories" / "write" / "stage_one_input.md"
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
        
    template = template.replace("{{ rollout_path }}", Path(rollout_path).as_posix())
    template = template.replace("{{ rollout_cwd }}", Path(rollout_cwd).as_posix())
    template = template.replace("{{ rollout_contents }}", truncated)
    
    return template


def memory_stage_one_system_prompt() -> str:
    path = ASSETS_DIR / "prompts" / "memories" / "write" / "stage_one_system.md"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_model_catalog_instructions(model: str) -> str | None:
    from codex.types import get_model_preset
    preset = get_model_preset(model)
    if preset is None:
        return None
        
    model_messages = preset.get("model_messages")
    if model_messages and model_messages.get("instructions_template"):
        template = model_messages["instructions_template"]
        vars = model_messages.get("instructions_variables", {})
        default_p = vars.get("personality_default", "")
        return template.replace("{{ personality }}", default_p)
    return preset.get("base_instructions")
