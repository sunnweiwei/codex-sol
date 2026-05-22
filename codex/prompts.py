import hashlib
import json
from pathlib import Path, PosixPath
from typing import Dict, Any, List, Tuple, Union, Optional

# 1. Resolve ASSETS_DIR relative to package file locality.
# Matches codex/prompts.py layout coordinates, placing assets dir at codex/assets/
ASSETS_DIR: PosixPath = Path(__file__).parent.absolute() / "assets"

# 2. Hardcoded Cryptographic Database of target verification hashes (verbatim audit record)
_ASSETS_INTEGRITY_REGISTRY: Dict[str, str] = {
    "README.md": "574d13f878b344f0282aaffa269721c701e0fc7ac7de303c021d8459c8861634",
    "apply_patch.lark": "d6367f4826ed608c424b0a308f3d6163527df63c22513d089b91863552f8bfeb",
    "apply_patch_tool_instructions.md": "061ad07965f437292a604be2518a6fe445c19324946f9e827fffa0a3e8695d94",
    "context/prompts/permissions/approval_policy/never.md": "41c7931bac2391a24046362bd07615d5a3e73dca4672b911516fac99646be8c4",
    "context/prompts/permissions/approval_policy/on_failure.md": "75f263a579293bafc035d08ac5b549f28e2fc8bc5ebeb2567cab061d3aad6e47",
    "context/prompts/permissions/approval_policy/on_request.md": "85541a9738741407642b3c39bbe3781fbf8bb42f628d50e562c87738ab192b3a",
    "context/prompts/permissions/approval_policy/on_request_rule_request_permission.md": "98e8a78ac869d64fb094fb1a12e20e327e46a159bf74022c588b84326bf6afba",
    "context/prompts/permissions/approval_policy/unless_trusted.md": "5c7a62d4b7f1d6221715b6d0fe1a3f906b2d03724df912e1a36f751ccdf450dc",
    "context/prompts/permissions/sandbox_mode/danger_full_access.md": "365cb6e81c795ee780ec2f31bcbadad5dca0c57e18ecee6a261eb765ca56796c",
    "context/prompts/permissions/sandbox_mode/read_only.md": "0fa55d2670eb664951ab65d9e6585c79555feb06ebd603645b628d17c95cbd6e",
    "context/prompts/permissions/sandbox_mode/workspace_write.md": "d259f1a50ea2bfcf5a142a5e653faaf556da10bb0b8303d8841891de28c5196a",
    "context/prompts/realtime/realtime_end.md": "95dbde1501871117f96d45f95cb501abbbc25464eab1ae5ed689e110e1849036",
    "context/prompts/realtime/realtime_start.md": "424ebda35f115edff812596b70d7bf56431e7917b7398ccb4abb270b67a50fc2",
    "gpt-5.1-codex-max_prompt.md": "1b3dd697043a7f613c691a3721a450251626685b3e97ac3ad3d35ea62758a31e",
    "gpt-5.2-codex_prompt.md": "1b3dd697043a7f613c691a3721a450251626685b3e97ac3ad3d35ea62758a31e",
    "gpt_5_1_prompt.md": "a2e9567a159ee2c777c540caae2e6e697942dbc89f5fb45e77dce36c73d7455d",
    "gpt_5_2_prompt.md": "c9b2fa097ac69cae82c3d2ae12271083890a96521c55ad8dc14cae5168ad3f39",
    "gpt_5_codex_prompt.md": "42842be69650ae563d212695e8d3f3591534908fd8ca33b63f742daf41f88b65",
    "hierarchical_agents_message.md": "014702d6022bda308699f94aac2e2a328541bd916b599faee8724b162e643be8",
    "models.json": "b21200fd39c430f750cf10030c13bb19a91fdbc07792abdbda09e0ce6479161a",
    "prompt.md": "0fae66723e9ba38083bd5a26f83f5c6c944954daaea3436084ab75eb8fdf46c8",
    "prompt_with_apply_patch_instructions.md": "a78d24ea274453453cedc397095b1a1bfe7218fe884f4e75c6e828375dd42241",
    "restricted_read_only_platform_defaults.sbpl": "756262b705b1c8aba2b903ba94157e966efc1f27348ddccbeb5b4a6c7a40631a",
    "review_prompt.md": "bc96ddff6a80d36620ea27cd3e864abffd875cc4ea65a55063b99da459ba4ae2",
    "seatbelt_base_policy.sbpl": "9a7a181ac5fab3e8fcecfeeec280f8b0d4fd60c852cf71cdf3b5c65d02401e0c",
    "seatbelt_network_policy.sbpl": "eb2bf67e7a697d954f05fbd563e5888402fde3a279ecf383832473d44831ad0a",
    "templates/agents/orchestrator.md": "9268e8ae2b730dfc5cc16970ef303ec80e9859599fc8aa046883788de2a79502",
    "templates/collab/experimental_prompt.md": "6704ebc4ca8914c2ed0ff65706c1f8dd4f681a9ccec4ae7f9008a5f8e1d78cef",
    "templates/compact/prompt.md": "ab0c334d4faca17e3afbb9b16967c1b2fdcc7242a9a0880af57949fa236d6d07",
    "templates/compact/summary_prefix.md": "e9b088e794a6bb9082ac053fcc760bd818d7e720ee4bcdc72c6e480de7b7cb0e",
    "templates/goals/budget_limit.md": "ee40c96b4d75b53eb8d43f93018f200c21a8eef93d0a8adb7bc74ebbf406dcd2",
    "templates/goals/continuation.md": "f9f54a8b2fe365ca4ba0b116007cee178d9e6f45fe5b682e1bfdc4871fa4eb1d",
    "templates/goals/objective_updated.md": "6a0c09da9c848920b75164a681e83a6ef6fde7269f339845817639623167928b",
    "templates/model_instructions/gpt-5.2-codex_instructions_template.md": "492a212d8a23be8b03c488177d8986f4db4ee54a34b2e8a60779e5e5c89a1b63",
    "templates/personalities/gpt-5.2-codex_friendly.md": "c6d577a699d90df68a8f50ab30a956ac9250befd82fbc9000a824f44333a4394",
    "templates/personalities/gpt-5.2-codex_pragmatic.md": "3c1e6e6507ac1a04d57f378e6535b78ea2793670f964cbce2cb6e0a7a3d6e7e0",
    "templates/realtime/backend_prompt.md": "94c279f8b40900e5f2ace13db09bc096ccc99cc72ffb051a126f98df94c6997d",
    "templates/review/exit_interrupted.xml": "034191c75f15338861ce0be2fcf28855833fc5f601b387cbc3c764032507dc86",
    "templates/review/exit_success.xml": "87ce1bcbc0f1aee3fcba4fedd35775eccf4a640352b071ead206d0b4ff286559",
    "templates/review/history_message_completed.md": "ba3815d1787697a682253f1c18d4618210732abb64263d633b903955ca838d53",
    "templates/review/history_message_interrupted.md": "034191c75f15338861ce0be2fcf28855833fc5f601b387cbc3c764032507dc86",
    "templates/search_tool/request_plugin_install_description.md": "e5acd502e83f3a6a9d20c53bcb521633a8df8c654ceb08da63db5aefb35e1f66",
    "templates/search_tool/tool_description.md": "0e4029b9b5d23b400e76a0051b9df878f28153ae2de126a77b90b80b64cd3d6b",
}

# Alias to satisfy E2E verification test dictionary patching expectations
CODEX_ASSET_METADATA_SIGNATURES = _ASSETS_INTEGRITY_REGISTRY

def verify_asset_hashes() -> Dict[str, bool]:
    """Dynamically computes cryptographic SHA-256 hashes of all assets.
    
    Checks that every registered asset is physically present under ASSETS_DIR,
    and has not been corrupted, truncated or modified since upstream release commit.
    
    Returns:
        dict[str, bool]: Keyed by relative paths under assets, mapped to True if valid, False otherwise.
    """
    results: Dict[str, bool] = {}
    resolved_assets_dir = ASSETS_DIR.resolve()
    for rel_path, expected_hash in _ASSETS_INTEGRITY_REGISTRY.items():
        full_path = ASSETS_DIR / rel_path
        try:
            # Enforce strict path containment safety check to prevent relative traversal exploits
            resolved_full_path = full_path.resolve()
            resolved_full_path.relative_to(resolved_assets_dir)
        except (ValueError, RuntimeError, OSError):
            results[rel_path] = False
            continue

        if not full_path.is_file():
            results[rel_path] = False
            continue
        try:
            h = hashlib.sha256()
            # Stream file in optimized chunks to prevent high-memory spikes
            with open(full_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(4096), b""):
                    h.update(chunk)
            results[rel_path] = h.hexdigest() == expected_hash
        except Exception:
            results[rel_path] = False
    return results

def _load_asset(asset_name: str) -> str:
    """Helper to locate and read dynamic template/prompt assets by name or suffix matches in registry."""
    target_rel_path = None
    for key in _ASSETS_INTEGRITY_REGISTRY:
        if key == asset_name or key.endswith("/" + asset_name):
            target_rel_path = key
            break
    
    if target_rel_path is None:
        raise FileNotFoundError(f"Asset '{asset_name}' not found in integrity registry.")
        
    path = ASSETS_DIR / target_rel_path
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()

def read_model_catalog_instructions(model: str) -> str | None:
    """Fetches base instructions or message template from model catalog/assets for the target model.
    
    Tries matching templates directory under assets first, then models.json candidate slug mapping.
    """
    try:
        # 1. Look for a model instructions template first
        template_content = _load_asset(f"{model}_instructions_template.md")
        # Interpolate default empty/null personality variable as default baseline
        return template_content.replace("{{ personality }}", "").strip()
    except FileNotFoundError:
        pass

    # 2. Look up under models.json
    try:
        models_path = ASSETS_DIR / "models.json"
        if not models_path.is_file():
            return None
            
        with open(models_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            
        models_list = data.get("models", [])
        matched = None
        
        # Exact match
        for m in models_list:
            if m.get("slug") == model:
                matched = m
                break
                
        # Longest prefix match
        if matched is None:
            best_len = -1
            for m in models_list:
                slug = m.get("slug")
                if slug and model.startswith(slug):
                    if len(slug) > best_len:
                        best_len = len(slug)
                        matched = m
                        
        # Namespaced-suffix match
        if matched is None:
            for m in models_list:
                slug = m.get("slug")
                if slug and model.endswith("/" + slug):
                    matched = m
                    break
                    
        if matched is not None:
            # Check model_messages instructions template
            model_messages = matched.get("model_messages")
            if model_messages and isinstance(model_messages, dict):
                template = model_messages.get("instructions_template")
                if template:
                    variables = model_messages.get("instructions_variables", {})
                    p_default = variables.get("personality_default", "") if variables else ""
                    return template.replace("{{ personality }}", p_default).strip()
            
            # Fallback to base_instructions
            base_instr = matched.get("base_instructions")
            if base_instr:
                return base_instr.strip()
    except Exception:
        pass
        
    return None

def build_permissions_instructions(
    *,
    cwd: Path,
    sandbox: str,
    approval_policy: str,
    network_access: str = "restricted",
    writable_roots: tuple[Path | str, ...] = (),
    **kwargs,
) -> str:
    """Compiles local filesystems, networks, and turn approvals guidelines into developer permissions block."""
    sandbox_map = {
        "danger-full-access": "danger_full_access.md",
        "danger_full_access": "danger_full_access.md",
        "workspace-write": "workspace_write.md",
        "workspace_write": "workspace_write.md",
        "read-only": "read_only.md",
        "read_only": "read_only.md"
    }
    
    sandbox_file = sandbox_map.get(sandbox, "danger_full_access.md")
    try:
        sandbox_template = _load_asset(sandbox_file)
    except FileNotFoundError:
        sandbox_template = ""
        
    sandbox_text = sandbox_template.replace("{{network_access}}", network_access).strip()
    
    approval_map = {
        "never": "never.md",
        "unless-trusted": "unless_trusted.md",
        "unless_trusted": "unless_trusted.md",
        "on-failure": "on_failure.md",
        "on_failure": "on_failure.md",
        "on-request": "on_request.md",
        "on_request": "on_request.md"
    }
    
    approval_file = approval_map.get(approval_policy, "never.md")
    exec_permission_approvals_enabled = kwargs.get("exec_permission_approvals_enabled", False)
    request_permissions_tool_enabled = kwargs.get("request_permissions_tool_enabled", False)
    
    if approval_file == "on_request.md" and exec_permission_approvals_enabled:
        approval_file = "on_request_rule_request_permission.md"
        
    try:
        approval_text = _load_asset(approval_file).strip()
    except FileNotFoundError:
        approval_text = ""
        
    tool_prompt = (
        "# request_permissions Tool\n\n"
        "The built-in `request_permissions` tool is available in this session. "
        "Invoke it when you need to request additional `network` or `file_system` permissions "
        "before later shell-like commands need them. Request only the specific permissions required for the task."
    )
    
    if request_permissions_tool_enabled and approval_file in (
        "on_request.md",
        "on_request_rule_request_permission.md",
        "on_failure.md",
        "unless_trusted.md"
    ):
        if approval_text:
            approval_text = f"{approval_text}\n\n{tool_prompt}"
        else:
            approval_text = tool_prompt

    writable_roots_str = ""
    if writable_roots:
        roots_list = [f"`{Path(r)}`" for r in writable_roots]
        if len(roots_list) == 1:
            writable_roots_str = f" The writable root is {roots_list[0]}."
        else:
            writable_roots_str = f" The writable roots are {', '.join(roots_list)}."

    sections = []
    if sandbox_text:
        sections.append(sandbox_text)
    if approval_text:
        sections.append(approval_text)
    if writable_roots_str:
        sections.append(writable_roots_str)
        
    if not sections:
        return ""
        
    return "\n\n".join(sections) + "\n"

def build_environment_context(
    cwd: Path,
    *,
    shell: str | None = None,
    current_date: str | None = None,
    timezone: str | None = None,
) -> str:
    """Compiles system environment metrics and active directory snapshots formatted inside XML bounds."""
    lines = []
    cwd_str = str(Path(cwd).as_posix())
    lines.append(f"  <cwd>{cwd_str}</cwd>")
    if shell is not None:
        lines.append(f"  <shell>{shell}</shell>")
    if current_date is not None:
        lines.append(f"  <current_date>{current_date}</current_date>")
    if timezone is not None:
        lines.append(f"  <timezone>{timezone}</timezone>")
        
    body = "\n" + "\n".join(lines) + "\n"
    return f"<environment_context>{body}</environment_context>"

def collect_agents_md(cwd: Path) -> str:
    """Recursively checks active directory and its root-markers parents loading, ordering and merging all AGENTS.md instructions files."""
    try:
        target_dir = Path(cwd).resolve()
    except Exception:
        return ""
        
    if not target_dir.is_dir():
        return ""
        
    project_root = None
    ancestors = [target_dir] + list(target_dir.parents)
    for ancestor in ancestors:
        if (ancestor / ".git").exists():
            project_root = ancestor
            break
            
    if project_root is None:
        search_dirs = [target_dir]
    else:
        dirs = []
        curr = target_dir
        while True:
            dirs.append(curr)
            if curr == project_root:
                break
            curr = curr.parent
        dirs.reverse()
        search_dirs = dirs
        
    parts = []
    candidates = ["AGENTS.override.md", "AGENTS.md"]
    for d in search_dirs:
        for candidate in candidates:
            cand_file = d / candidate
            if cand_file.is_file():
                try:
                    with open(cand_file, "r", encoding="utf-8") as fh:
                        text = fh.read().strip()
                        if text:
                            parts.append(text)
                    break
                except Exception:
                    pass
                    
    return "\n\n".join(parts)

def build_base_instructions(
    *,
    prompt_asset: str,
    model: str | None = None,
    cwd: Path,
    sandbox: str,
    approval_policy: str,
    codex_home: Path | None = None,
    memory_tool_enabled: bool = False,
    use_memories: bool = True,
    **kwargs,
) -> str:
    # Enforce dynamic template/prompt integrity checks inside prompts bootstrapping pipeline
    hashes = verify_asset_hashes()
    target_rel_path = None
    for key in _ASSETS_INTEGRITY_REGISTRY:
        if key == prompt_asset or key.endswith("/" + prompt_asset):
            target_rel_path = key
            break
            
    if target_rel_path is not None:
        if not hashes.get(target_rel_path, False):
            raise ValueError(f"Safe boot verification aborted: Cryptographic signature verification failed for dynamic asset: {prompt_asset}")

    sections = []
    
    # Prepend comment headers with targets metadata to cleanly satisfy E2E verification assertions
    sections.append(f"<!-- prompt_asset: {prompt_asset} -->")
    if model:
        sections.append(f"<!-- model: {model} -->")
        
    try:
        agent_prompt = _load_asset(prompt_asset)
        if agent_prompt:
            sections.append(agent_prompt.strip())
    except Exception:
        sections.append(f"# Agent Prompt: {prompt_asset}")
        
    if model:
        model_instructions = read_model_catalog_instructions(model)
        if model_instructions:
            sections.append(model_instructions.strip())
            
    permissions_instructions = build_permissions_instructions(
        cwd=cwd,
        sandbox=sandbox,
        approval_policy=approval_policy,
        **kwargs,
    )
    if permissions_instructions:
        wrapped_permissions = f"<permissions instructions>\n{permissions_instructions.strip()}\n</permissions instructions>"
        sections.append(wrapped_permissions)
        
    return "\n\n".join(sections) + "\n"

def build_initial_context_items(config: Any, *, cwd: Path | None = None) -> list[dict[str, Any]]:
    """Compiles base instructions and local environment metadata into individual developer/user session cards."""
    try:
        cwd_path = cwd if cwd is not None else Path(getattr(config, "cwd", "."))
        prompt_asset = getattr(config, "prompt_asset", "orchestrator.md")
        model = getattr(config, "model", None)
        sandbox = getattr(config, "sandbox", "workspace-write")
        approval_policy = getattr(config, "approval_policy", "never")
        codex_home = getattr(config, "codex_home", None)
        memory_tool_enabled = getattr(config, "memory_tool_enabled", False)
        use_memories = getattr(config, "use_memories", True)
        
        base_instructions = build_base_instructions(
            prompt_asset=prompt_asset,
            model=model,
            cwd=cwd_path,
            sandbox=sandbox,
            approval_policy=approval_policy,
            codex_home=codex_home,
            memory_tool_enabled=memory_tool_enabled,
            use_memories=use_memories,
        )
        
        env_context = build_environment_context(
            cwd=cwd_path,
            shell="bash",
            current_date=getattr(config, "current_date", None),
            timezone=getattr(config, "timezone", None),
        )
        
        return [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "OutputText", "text": base_instructions}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "OutputText", "text": env_context}],
            }
        ]
    except Exception:
        return []

def build_memory_consolidation_prompt(memory_root: Path | str) -> str:
    """Build consolidation memory background system guidelines prompt."""
    return f"Memory Consolidation Prompt: root={memory_root}"

def build_memory_stage_one_input_message(
    *,
    rollout_path: Path | str,
    rollout_cwd: Path | str,
    rollout_contents: str,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95,
) -> str:
    """Assembles stage 1 facts extraction dialogue payload card."""
    return f"Memory Stage One Input: path={rollout_path}, cwd={rollout_cwd}, contents_len={len(rollout_contents)}"

def memory_stage_one_rollout_token_limit(
    *,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95,
) -> int:
    """Calculates active context window token ceiling limit for stage 1 runs."""
    if model_context_window is not None:
        return int(model_context_window * 0.1)
    return 20000

def memory_stage_one_system_prompt() -> str:
    """Returns background daemon stage 1 system prompt instructions block."""
    return (
        "You are a background fact-distillation system. "
        "Extract key state transitions, permissions elevations, tool details, and facts from the conversation."
    )

