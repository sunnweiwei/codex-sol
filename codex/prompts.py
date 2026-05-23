from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Literal
from codex.types import CodexConfig, PromptRequest, load_model_catalog, find_model_info, get_default_model_slug

# Re-exports or constants
ASSETS_DIR = Path(__file__).parent.resolve() / "assets"

# --- Asset Hashing (verify_asset_hashes) ------------------------------------
# We precomputed these hashes for verification
ASSET_SHA256_HASHES = {
    'models.json': 'b21200fd39c430f750cf10030c13bb19a91fdbc07792abdbda09e0ce6479161a',
    'seatbelt_base_policy.sbpl': '9a7a181ac5fab3e8fcecfeeec280f8b0d4fd60c852cf71cdf3b5c65d02401e0c',
    'grammars/apply_patch.lark': 'd6367f4826ed608c424b0a308f3d6163527df63c22513d089b91863552f8bfeb',
    'prompts/gpt_5_codex_prompt.md': '42842be69650ae563d212695e8d3f3591534908fd8ca33b63f742daf41f88b65',
    'prompts/gpt_5_2_prompt.md': 'c9b2fa097ac69cae82c3d2ae12271083890a96521c55ad8dc14cae5168ad3f39',
    'prompts/prompt_with_apply_patch_instructions.md': 'a78d24ea274453453cedc397095b1a1bfe7218fe884f4e75c6e828375dd42241',
    'prompts/gpt-5.2-codex_prompt.md': '1b3dd697043a7f613c691a3721a450251626685b3e97ac3ad3d35ea62758a31e',
    'prompts/compact/summary_prefix.md': 'e9b088e794a6bb9082ac053fcc760bd818d7e720ee4bcdc72c6e480de7b7cb0e',
    'prompts/compact/prompt.md': 'ab0c334d4faca17e3afbb9b16967c1b2fdcc7242a9a0880af57949fa236d6d07',
    'prompts/memories/read_path.md': '71ece7a2ab1c986caf93733320bedd518fefa57cda92bf4fa862df3f229ba46c',
    'prompts/memories/write/stage_one_system.md': 'cf795e8a2f5f52d333af2613bf1ff79178112f5fd2161cc181a8ddf52e59da33',
    'prompts/memories/write/stage_one_input.md': '2e54c74909238022305c269c862910bb29509fda8b58ce671ef011f8d6453047',
    'prompts/memories/write/consolidation.md': 'af0df49d83c5ccc08ad0cadcd2856b05ac33439533daec8ecd00d291a8ad3358',
    'prompts/memories/write/extensions/ad_hoc/instructions.md': 'd36a36083d92f9d44efbd95e0e4b6e81d7d149e812f2bca2009b6dd4b8aa93e7',
    'prompts/permissions/approval_policy/on_request.md': '85541a9738741407642b3c39bbe3781fbf8bb42f628d50e562c87738ab192b3a',
    'prompts/permissions/approval_policy/unless_trusted.md': '5c7a62d4b7f1d6221715b6d0fe1a3f906b2d03724df912e1a36f751ccdf450dc',
    'prompts/permissions/approval_policy/on_failure.md': '75f263a579293bafc035d08ac5b549f28e2fc8bc5ebeb2567cab061d3aad6e47',
    'prompts/permissions/approval_policy/on_request_rule_request_permission.md': '98e8a78ac869d64fb094fb1a12e20e327e46a159bf74022c588b84326bf6afba',
    'prompts/permissions/approval_policy/never.md': '41c7931bac2391a24046362bd07615d5a3e73dca4672b911516fac99646be8c4',
    'prompts/permissions/sandbox_mode/danger_full_access.md': '365cb6e81c795ee780ec2f31bcbadad5dca0c57e18ecee6a261eb765ca56796c',
    'prompts/permissions/sandbox_mode/workspace_write.md': 'd259f1a50ea2bfcf5a142a5e653faaf556da10bb0b8303d8841891de28c5196a',
    'prompts/permissions/sandbox_mode/read_only.md': '0fa55d2670eb664951ab65d9e6585c79555feb06ebd603645b628d17c95cbd6e'
}

def verify_asset_hashes() -> dict[str, bool]:
    import hashlib
    res = {}
    for rel_path, expected_hash in ASSET_SHA256_HASHES.items():
        full_path = ASSETS_DIR / rel_path
        if not full_path.exists():
            res[rel_path] = False
            continue
        try:
            h = hashlib.sha256(full_path.read_bytes()).hexdigest()
            res[rel_path] = (h == expected_hash)
        except Exception:
            res[rel_path] = False
    return res

# --- Model Catalog Instructions ---------------------------------------------
def read_model_catalog_instructions(model: str) -> str | None:
    info = find_model_info(model)
    if not info:
        return None
        
    messages = info.get("model_messages")
    if messages and isinstance(messages, dict):
        template = messages.get("instructions_template")
        if template:
            variables = messages.get("instructions_variables") or {}
            default_personality = variables.get("personality_default", "")
            return template.replace("{{ personality }}", default_personality)
            
    return info.get("base_instructions")

# --- Base Instructions builder ----------------------------------------------
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
) -> str:
    if prompt_asset == "auto":
        resolved_model = model if model is not None else get_default_model_slug()
        instructions = read_model_catalog_instructions(resolved_model)
        if instructions is not None:
            return instructions
        prompt_asset = "gpt_5_codex_prompt.md"
        
    # Otherwise, read prompt asset from assets
    asset_path = ASSETS_DIR / "prompts" / prompt_asset
    if not asset_path.exists():
        asset_path = ASSETS_DIR / "prompts" / "gpt_5_codex_prompt.md"
        
    try:
        return asset_path.read_text(encoding="utf-8")
    except Exception:
        return ""

# --- Helper: build_environment_context ---------------------------------------
def build_environment_context(
    cwd: Path,
    *,
    shell: str | None = None,
    current_date: str | None = None,
    timezone: str | None = None,
) -> str:
    cwd = Path(cwd).absolute()
    lines = ["<environment_context>"]
    lines.append(f"  <cwd>{cwd}</cwd>")
    sh = shell if shell is not None else "bash"
    lines.append(f"  <shell>{sh}</shell>")
    if current_date is not None:
        lines.append(f"  <current_date>{current_date}</current_date>")
    if timezone is not None:
        lines.append(f"  <timezone>{timezone}</timezone>")
    lines.append("</environment_context>")
    return "\n".join(lines)

# --- Helper: build_permissions_instructions ---------------------------------
def build_permissions_instructions(
    *,
    cwd: Path,
    sandbox: str,
    approval_policy: str,
    network_access: str = "restricted",
    writable_roots: tuple[Path | str, ...] = (),
    **kwargs: Any,
) -> str:
    sb_filename = sandbox.replace("-", "_") + ".md"
    sb_path = ASSETS_DIR / "prompts" / "permissions" / "sandbox_mode" / sb_filename
    if sb_path.exists():
        sb_text = sb_path.read_text(encoding="utf-8").strip()
        sb_text = sb_text.replace("{{network_access}}", network_access)
    else:
        sb_text = f"Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is `{sandbox}`. Network access is {network_access}."
        
    roots_text = ""
    if sandbox != "read-only" and writable_roots:
        roots_list = [f"`{Path(p).absolute()}`" for p in writable_roots]
        if len(roots_list) == 1:
            roots_text = f" The writable root is {roots_list[0]}."
        else:
            roots_text = f" The writable roots are {', '.join(roots_list)}."
            
    if roots_text:
        sb_text = sb_text.rstrip(".") + "." + roots_text
        
    ap_filename = approval_policy.replace("-", "_") + ".md"
    ap_path = ASSETS_DIR / "prompts" / "permissions" / "approval_policy" / ap_filename
    if ap_path.exists():
        ap_text = ap_path.read_text(encoding="utf-8").strip()
    else:
        ap_text = f"Approval policy is `{approval_policy}`."
        
    res = sb_text
    if not res.endswith("\n"):
        res += "\n"
    res += ap_text
    if not res.endswith("\n"):
        res += "\n"
        
    reviewer = kwargs.get("approvals_reviewer", "auto_review")
    if reviewer == "auto_review" and approval_policy != "never":
        suffix = "\n\nAuto-approval is enabled for prefix rules. Safe commands matching approved rules do not require prompt checks."
        res = res.strip() + suffix
        
    if not res.endswith("\n"):
        res += "\n"
    return res

# --- Helper: build_initial_context_items -------------------------------------
def build_initial_context_items(config: CodexConfig, *, cwd: Path | None = None) -> list[dict[str, Any]]:
    cwd = cwd if cwd is not None else config.resolved_cwd()
    
    # 1. Developer message: permissions + memory instructions
    permissions = build_permissions_instructions(
        cwd=cwd,
        sandbox=config.sandbox,
        approval_policy=config.approval_policy,
        network_access=config.network_access,
        writable_roots=config.writable_roots,
        approvals_reviewer=getattr(config, "approvals_reviewer", "auto_review"),
    )
    
    dev_content = [{"type": "input_text", "text": f"<permissions instructions>\n{permissions}\n</permissions instructions>"}]
    
    if config.use_memories:
        # Load memory summary and instructions
        home_dir = config.resolved_codex_home()
        sum_file = home_dir / "memories" / "memory_summary.md"
        memory_summary = ""
        if sum_file.exists():
            try:
                memory_summary = sum_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass
                
        read_file = ASSETS_DIR / "prompts" / "memories" / "read_path.md"
        if read_file.exists():
            try:
                prompt_text = read_file.read_text(encoding="utf-8")
                rendered = prompt_text.replace("{{ base_path }}", str(home_dir / "memories"))
                rendered = rendered.replace("{{ memory_summary }}", memory_summary or "No memory baseline.")
                dev_content.append({"type": "input_text", "text": rendered})
            except Exception:
                pass
                
    developer_msg = {
        "type": "message",
        "role": "developer",
        "content": dev_content,
    }
    
    # 2. User message: AGENTS.md + Environment Context
    user_content = []
    
    # Scanned AGENTS.md instructions
    agents_md_content = collect_agents_md(cwd)
    if agents_md_content:
        agents_text = f"# AGENTS.md instructions for {cwd.absolute()}\n\n<INSTRUCTIONS>\n{agents_md_content}\n</INSTRUCTIONS>"
        user_content.append({"type": "input_text", "text": agents_text})
        
    # Environment context
    env_text = build_environment_context(
        cwd=cwd,
        shell="bash",
        current_date=config.current_date,
        timezone=config.timezone,
    )
    user_content.append({"type": "input_text", "text": env_text})
    
    user_msg = {
        "type": "message",
        "role": "user",
        "content": user_content,
    }
    
    return [developer_msg, user_msg]

# --- Scanned AGENTS.md instructions -----------------------------------------
def collect_agents_md(cwd: Path) -> str:
    cwd = cwd.absolute()
    project_root = None
    cursor = cwd
    while True:
        if (cursor / ".git").exists() and (cursor / ".git").is_dir():
            project_root = cursor
            break
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
        
    if project_root is not None:
        search_dirs = []
        cursor = cwd
        while True:
            search_dirs.append(cursor)
            if cursor == project_root:
                break
            parent = cursor.parent
            if parent == cursor:
                break
            cursor = parent
        search_dirs.reverse()
    else:
        search_dirs = [cwd]
        
    found_contents = []
    candidates = ["AGENTS.override.md", "AGENTS.md"]
    for d in search_dirs:
        for name in candidates:
            candidate_path = d / name
            if candidate_path.exists() and candidate_path.is_file():
                try:
                    content = candidate_path.read_text(encoding="utf-8").strip()
                    if content:
                        found_contents.append(content)
                        break
                except Exception:
                    pass
    return "\n\n".join(found_contents)

# --- Truncation Helpers -----------------------------------------------------
def approx_token_count(text: str) -> int:
    return (len(text.encode('utf-8')) + 3) // 4

def approx_tokens_from_byte_count(bytes_cnt: int) -> int:
    return (bytes_cnt + 3) // 4

def truncate_text(s: str, max_tokens: int) -> str:
    if not s:
        return ""
    
    b = s.encode('utf-8')
    max_bytes = max_tokens * 4
    if len(b) <= max_bytes:
        return s
        
    left_budget = max_bytes // 2
    right_budget = max_bytes - left_budget
    
    decoded_chars = list(s)
    byte_offsets = []
    curr = 0
    for c in decoded_chars:
        byte_offsets.append(curr)
        curr += len(c.encode('utf-8'))
    byte_offsets.append(curr)
    
    prefix_end_idx = 0
    for idx, offset in enumerate(byte_offsets):
        if offset <= left_budget:
            prefix_end_idx = idx
            
    prefix_end_byte = byte_offsets[prefix_end_idx]
    
    tail_start_target = len(b) - right_budget
    suffix_start_idx = len(decoded_chars)
    for idx, offset in enumerate(byte_offsets):
        if offset >= tail_start_target:
            suffix_start_idx = idx
            break
            
    if suffix_start_idx < prefix_end_idx:
        suffix_start_idx = prefix_end_idx
        
    suffix_start_byte = byte_offsets[suffix_start_idx]
    
    removed_bytes = len(b) - prefix_end_byte - (len(b) - suffix_start_byte)
    removed_tokens = approx_tokens_from_byte_count(removed_bytes)
    
    marker = f"…{removed_tokens} tokens truncated…"
    prefix = "".join(decoded_chars[:prefix_end_idx])
    suffix = "".join(decoded_chars[suffix_start_idx:])
    return prefix + marker + suffix

# --- Memory Consolidation Prompt --------------------------------------------
EXTENSIONS_FOLDER_STRUCTURE = r"""
Memory extensions (under {{ memory_extensions_root }}/):

- <extension_name>/instructions.md
  - Source-specific guidance for interpreting additional memory signals. If an
    extension folder exists, you must read its instructions.md to determine how to use this memory
    source.

If the user has any memory extensions, you MUST read the instructions for each extension to
determine how to use the memory source. If the workspace diff shows deleted extension resource files,
remove stale memories derived only from those resources. If it has no extension folders, continue
with the standard memory inputs only.
"""

EXTENSIONS_PRIMARY_INPUTS = r"""
Optional source-specific inputs:
Under `{{ memory_extensions_root }}/`:

- `<extension_name>/instructions.md`
  - If extension folders exist, read each instructions.md first and follow it when interpreting
    that extension's memory source.

If the workspace diff shows deleted memory extension resources, use that extension-specific deletion
signal to remove stale memories derived only from those resources.
"""

def build_memory_consolidation_prompt(memory_root: Path | str) -> str:
    memory_root = Path(memory_root).absolute()
    memory_extensions_root = memory_root / "extensions"
    memory_extensions_exist = memory_extensions_root.is_dir()
    
    if memory_extensions_exist:
        folder_structure = EXTENSIONS_FOLDER_STRUCTURE.replace("{{ memory_extensions_root }}", str(memory_extensions_root.absolute()))
        primary_inputs = EXTENSIONS_PRIMARY_INPUTS.replace("{{ memory_extensions_root }}", str(memory_extensions_root.absolute()))
    else:
        folder_structure = ""
        primary_inputs = ""
        
    template_file = ASSETS_DIR / "prompts" / "memories" / "write" / "consolidation.md"
    if not template_file.exists():
        return f"## Memory Phase 2 (Consolidation)\nConsolidate Codex memories in: {memory_root}\n\nRead phase2_workspace_diff.md first."
        
    try:
        content = template_file.read_text(encoding="utf-8")
        rendered = content.replace("{{ memory_root }}", str(memory_root.absolute()))
        rendered = rendered.replace("{{ memory_extensions_folder_structure }}", folder_structure)
        rendered = rendered.replace("{{ memory_extensions_primary_inputs }}", primary_inputs)
        rendered = rendered.replace("{{ phase2_workspace_diff_file }}", "phase2_workspace_diff.md")
        return rendered
    except Exception:
        return f"## Memory Phase 2 (Consolidation)\nConsolidate Codex memories in: {memory_root}\n\nRead phase2_workspace_diff.md first."

# --- Memory Stage One Prompt Builders ----------------------------------------
def memory_stage_one_system_prompt() -> str:
    template_file = ASSETS_DIR / "prompts" / "memories" / "write" / "stage_one_system.md"
    if template_file.exists():
        return template_file.read_text(encoding="utf-8")
    return "Memory Writing Agent: Phase 1"

def memory_stage_one_rollout_token_limit(
    *,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95,
) -> int:
    # 70% of the active model's effective input window
    if model_context_window is None:
        return 150_000
    effective_limit = (model_context_window * effective_context_window_percent) // 100
    limit = (effective_limit * 70) // 100
    return max(1, limit)

def build_memory_stage_one_input_message(
    *,
    rollout_path: Path | str,
    rollout_cwd: Path | str,
    rollout_contents: str,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95,
) -> str:
    rollout_path = Path(rollout_path)
    rollout_cwd = Path(rollout_cwd)
    
    limit = memory_stage_one_rollout_token_limit(
        model_context_window=model_context_window,
        effective_context_window_percent=effective_context_window_percent,
    )
    
    truncated_contents = truncate_text(rollout_contents, limit)
    
    template_file = ASSETS_DIR / "prompts" / "memories" / "write" / "stage_one_input.md"
    if not template_file.exists():
        return f"Analyze rollout {rollout_path}\n{truncated_contents}"
        
    try:
        content = template_file.read_text(encoding="utf-8")
        rendered = content.replace("{{ rollout_path }}", str(rollout_path))
        rendered = rendered.replace("{{ rollout_cwd }}", str(rollout_cwd))
        rendered = rendered.replace("{{ rollout_contents }}", truncated_contents)
        return rendered
    except Exception:
        return f"Analyze rollout {rollout_path}\n{truncated_contents}"

# --- Context checkpoint compaction prompt ------------------------------------
SUMMARIZATION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work.\n"
)

def prepare_prompt_history(history: list[dict[str, Any]], config: CodexConfig) -> list[dict[str, Any]]:
    from copy import deepcopy
    
    # 1. Normalise call-output pairs
    has_call = set()
    has_output = set()
    for item in history:
        item_type = item.get("type")
        if item_type in ("function_call", "custom_tool_call"):
            call_id = item.get("call_id") or item.get("id")
            if call_id:
                has_call.add(call_id)
        elif item_type in ("function_call_output", "custom_tool_call_output"):
            call_id = item.get("call_id")
            if call_id:
                has_output.add(call_id)
                
    placed_calls = set()
    normalized = []
    for item in history:
        item_type = item.get("type")
        if item_type in ("function_call_output", "custom_tool_call_output"):
            call_id = item.get("call_id")
            if call_id not in placed_calls:
                # Strip orphan output
                continue
            normalized.append(deepcopy(item))
        elif item_type in ("function_call", "custom_tool_call"):
            call_id = item.get("call_id") or item.get("id")
            if call_id:
                placed_calls.add(call_id)
            normalized.append(deepcopy(item))
            if call_id and call_id not in has_output:
                # Append synthetic aborted output
                synth = {
                    "type": "function_call_output" if item_type == "function_call" else "custom_tool_call_output",
                    "call_id": call_id,
                    "output": "aborted",
                }
                normalized.append(synth)
        else:
            normalized.append(deepcopy(item))
            
    prompt_history = normalized
    supports_images = config.resolved_supports_image_input()
    truncation_limit = config.resolved_tool_output_truncation_tokens()
    
    placeholder = "image content omitted because you do not support image input"
    
    for item in prompt_history:
        item_type = item.get("type")
        
        if not supports_images:
            if item_type == "message" and "content" in item:
                for idx, part in enumerate(item["content"]):
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        item["content"][idx] = {"type": "input_text", "text": placeholder}
                        
            if item_type in ("function_call_output", "custom_tool_call_output") and isinstance(item.get("output"), dict):
                output_dict = item["output"]
                if "content" in output_dict:
                    for idx, part in enumerate(output_dict["content"]):
                        if isinstance(part, dict) and part.get("type") == "input_image":
                            output_dict["content"][idx] = {"type": "input_text", "text": placeholder}
                            
        if item_type in ("function_call_output", "custom_tool_call_output") and isinstance(item.get("output"), str):
            output_str = item["output"]
            if approx_token_count(output_str) > truncation_limit:
                item["output"] = truncate_text(output_str, truncation_limit)
                
    return prompt_history

def build_compaction_summary_prompt(history: list[dict[str, Any]]) -> str:
    lines = ["Here is the conversation history to compact:\n"]
    for item in history:
        role = item.get("role", "")
        item_type = item.get("type", "")
        if item_type == "message":
            content = item.get("content", [])
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            lines.append(f"{role.upper()}: {text}")
        elif item_type in ("function_call", "custom_tool_call"):
            name = item.get("name", "")
            args = item.get("arguments") or item.get("input") or ""
            lines.append(f"TOOL CALL: {name}({args})")
        elif item_type in ("function_call_output", "custom_tool_call_output"):
            out = item.get("output", "")
            lines.append(f"TOOL OUTPUT: {out}")
    return "\n".join(lines)

def build_local_compaction_request(
    history: list[dict[str, Any]],
    config: CodexConfig,
    model: str | None = None,
    additional_contexts: list[str] | None = None,
) -> PromptRequest:
    prompt = build_compaction_summary_prompt(history)
    comp_model = model if model is not None else "gpt-5.4-mini"
    
    prompt_text = prompt + "\n\n" + SUMMARIZATION_PROMPT
    if additional_contexts:
        hook_block = "\n<hook_context>\n" + "\n".join(additional_contexts) + "\n</hook_context>"
        prompt_text += hook_block
        
    return PromptRequest(
        model=comp_model,
        instructions="You are Codex, a highly skilled software engineering assistant.",
        input=prepare_prompt_history(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt_text}]
                }
            ],
            config
        ),
        tools=[],
        parallel_tool_calls=False,
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )

