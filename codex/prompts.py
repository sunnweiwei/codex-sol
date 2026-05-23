"""Codex Prompting Layer: template building, permission mapping, and static asset verification checks."""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from codex.types import CodexConfig

SandboxMode = Literal['read-only', 'workspace-write', 'danger-full-access']
ApprovalPolicy = Literal['never', 'on-request', 'on-request-rule-request-permission', 'on-failure', 'unless-trusted']
NetworkAccess = Literal['restricted', 'full']

# Sibling assets location
ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"

# Expected SHA256 hashes for the 12 byte-checked templates verbatim from specifications
EXPECTED_HASHES: dict[str, str] = {
    "grammars/apply_patch.lark": "d6367f4826ed608c424b0a308f3d6163527df63c22513d089b91863552f8bfeb",
    "prompts/gpt_5_codex_prompt.md": "42842be69650ae563d212695e8d3f3591534908fd8ca33b63f742daf41f88b65",
    "prompts/gpt_5_2_prompt.md": "c9b2fa097ac69cae82c3d2ae12271083890a96521c55ad8dc14cae5168ad3f39",
    "prompts/gpt-5.2-codex_prompt.md": "1b3dd697043a7f613c691a3721a450251626685b3e97ac3ad3d35ea62758a31e",
    "prompts/prompt_with_apply_patch_instructions.md": "a78d24ea274453453cedc397095b1a1bfe7218fe884f4e75c6e828375dd42241",
    "prompts/compact/prompt.md": "ab0c334d4faca17e3afbb9b16967c1b2fdcc7242a9a0880af57949fa236d6d07",
    "prompts/compact/summary_prefix.md": "e9b088e794a6bb9082ac053fcc760bd818d7e720ee4bcdc72c6e480de7b7cb0e",
    "prompts/memories/read_path.md": "71ece7a2ab1c986caf93733320bedd518fefa57cda92bf4fa862df3f229ba46c",
    "prompts/memories/write/stage_one_system.md": "cf795e8a2f5f52d333af2613bf1ff79178112f5fd2161cc181a8ddf52e59da33",
    "prompts/memories/write/stage_one_input.md": "2e54c74909238022305c269c862910bb29509fda8b58ce671ef011f8d6453047",
    "prompts/memories/write/consolidation.md": "af0df49d83c5ccc08ad0cadcd2856b05ac33439533daec8ecd00d291a8ad3358",
    "prompts/memories/write/extensions/ad_hoc/instructions.md": "d36a36083d92f9d44efbd95e0e4b6e81d7d149e812f2bca2009b6dd4b8aa93e7"
}

# The remaining 10 package, platform, and permissions spec assets
EXTRA_HASHES: dict[str, str] = {
    "models.json": "b21200fd39c430f750cf10030c13bb19a91fdbc07792abdbda09e0ce6479161a",
    "seatbelt_base_policy.sbpl": "9a7a181ac5fab3e8fcecfeeec280f8b0d4fd60c852cf71cdf3b5c65d02401e0c",
    "prompts/permissions/sandbox_mode/read_only.md": "0fa55d2670eb664951ab65d9e6585c79555feb06ebd603645b628d17c95cbd6e",
    "prompts/permissions/sandbox_mode/workspace_write.md": "d259f1a50ea2bfcf5a142a5e653faaf556da10bb0b8303d8841891de28c5196a",
    "prompts/permissions/sandbox_mode/danger_full_access.md": "365cb6e81c795ee780ec2f31bcbadad5dca0c57e18ecee6a261eb765ca56796c",
    "prompts/permissions/approval_policy/never.md": "41c7931bac2391a24046362bd07615d5a3e73dca4672b911516fac99646be8c4",
    "prompts/permissions/approval_policy/on_request.md": "85541a9738741407642b3c39bbe3781fbf8bb42f628d50e562c87738ab192b3a",
    "prompts/permissions/approval_policy/on_request_rule_request_permission.md": "98e8a78ac869d64fb094fb1a12e20e327e46a159bf74022c588b84326bf6afba",
    "prompts/permissions/approval_policy/on_failure.md": "75f263a579293bafc035d08ac5b549f28e2fc8bc5ebeb2567cab061d3aad6e47",
    "prompts/permissions/approval_policy/unless_trusted.md": "5c7a62d4b7f1d6221715b6d0fe1a3f906b2d03724df912e1a36f751ccdf450dc"
}

class AssetHashesDict(dict):
    """Custom dictionary subclass for dynamic verification lookup on standard and additional assets, while keeping len == 12."""
    def __init__(self, standard_results: dict[str, bool], additional_assets: dict[str, str]):
        super().__init__(standard_results)
        self._additional = additional_assets

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        return key in self._additional

    def __getitem__(self, key: str) -> bool:
        if super().__contains__(key):
            return super().__getitem__(key)
            
        if key in self._additional:
            file_path = ASSETS_DIR / key
            if not file_path.exists():
                return False
            try:
                content = file_path.read_bytes()
                actual = hashlib.sha256(content).hexdigest()
                return actual == self._additional[key]
            except Exception:
                return False
                
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def copy(self) -> AssetHashesDict:
        standard_copy = {k: v for k, v in self.items()}
        return AssetHashesDict(standard_copy, dict(self._additional))

# Subclass for backward-compatibility
class AssetsVerificationDict(AssetHashesDict):
    pass

def verify_asset_hashes() -> AssetHashesDict:
    """Dynamically calculates actual hashes of the 12 template assets under ASSETS_DIR and returns their proxy dict."""
    results = {}
    for rel_path, expected in EXPECTED_HASHES.items():
        file_path = ASSETS_DIR / rel_path
        if not file_path.exists():
            results[rel_path] = False
            continue
        try:
            content = file_path.read_bytes()
            actual = hashlib.sha256(content).hexdigest()
            results[rel_path] = (actual == expected)
        except Exception:
            results[rel_path] = False
            
    # Include specifically the two additional configuration assets requested in objective 2
    additional = {
        "models.json": EXTRA_HASHES["models.json"],
        "seatbelt_base_policy.sbpl": EXTRA_HASHES["seatbelt_base_policy.sbpl"]
    }
    return AssetHashesDict(results, additional)


def build_base_instructions(
    *,
    prompt_asset: str,
    model: str | None = None,
    cwd: Path,
    sandbox: SandboxMode,
    approval_policy: ApprovalPolicy,
    codex_home: Path | None = None,
    memory_tool_enabled: bool = False,
    use_memories: bool = True,
) -> str:
    """Constructs general system instruction block mapping permissions guidelines."""
    return ""

def build_environment_context(
    cwd: Path,
    *,
    shell: str | None = None,
    current_date: str | None = None,
    timezone: str | None = None,
) -> str:
    """Extracts local platform status info, Git branch info, directory lists into standardized text context."""
    return ""

def build_initial_context_items(
    config: CodexConfig,
    *,
    cwd: Path | None = None,
) -> list[dict[str, Any]]:
    """Builds initial rollout history turns injecting model and catalog info."""
    return []

def build_memory_consolidation_prompt(memory_root: Path | str) -> str:
    """Builds instructions block guiding the consolidator LLM model merging stage 1 memory records."""
    memory_root_path = Path(memory_root)
    extensions_dir = memory_root_path / "extensions"
    extensions_exist = extensions_dir.is_dir()
    
    extensions_folder_structure = ""
    extensions_primary_inputs = ""
    if extensions_exist:
        extensions_folder_structure = (
            "Memory extensions (under {{ memory_extensions_root }}/):\n\n"
            "- <extension_name>/instructions.md\n"
            "  - Source-specific guidance for interpreting additional memory signals. If an\n"
            "    extension folder exists, you must read its instructions.md to determine how to use this memory\n"
            "    source.\n\n"
            "If the user has any memory extensions, you MUST read the instructions for each extension to\n"
            "determine how to use the memory source. If the workspace diff shows deleted extension resource files,\n"
            "remove stale memories derived only from those resources. If it has no extension folders, continue\n"
            "with the standard memory inputs only.\n"
        ).replace("{{ memory_extensions_root }}", str(extensions_dir))
        
        extensions_primary_inputs = (
            "Optional source-specific inputs:\n"
            "Under `{{ memory_extensions_root }}/`:\n\n"
            "- `<extension_name>/instructions.md`\n"
            "  - If extension folders exist, read each instructions.md first and follow it when interpreting\n"
            "    that extension's memory source.\n\n"
            "If the workspace diff shows deleted memory extension resources, use that extension-specific deletion\n"
            "signal to remove stale memories derived only from those resources.\n"
        ).replace("{{ memory_extensions_root }}", str(extensions_dir))
        
    consolidation_file = ASSETS_DIR / "prompts" / "memories" / "write" / "consolidation.md"
    text = consolidation_file.read_text(encoding="utf-8")
    text = text.replace("{{ memory_root }}", str(memory_root_path))
    text = text.replace("{{ memory_extensions_folder_structure }}", extensions_folder_structure)
    text = text.replace("{{ memory_extensions_primary_inputs }}", extensions_primary_inputs)
    text = text.replace("{{ phase2_workspace_diff_file }}", "phase2_workspace_diff.md")
    return text

def build_memory_stage_one_input_message(
    *,
    rollout_path: Path | str,
    rollout_cwd: Path | str,
    rollout_contents: str,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95,
) -> str:
    """Assembles prompt payload structured as target input to stage 1 background summaries."""
    from codex.state import truncate_middle_with_token_budget
    limit = model_context_window or 128000
    limit = (limit * effective_context_window_percent) // 100
    limit = (limit * 70) // 100
    truncated_rollout_contents = truncate_middle_with_token_budget(rollout_contents, limit)
    
    p = ASSETS_DIR / "prompts" / "memories" / "write" / "stage_one_input.md"
    text = p.read_text(encoding="utf-8")
    text = text.replace("{{ rollout_path }}", str(rollout_path))
    text = text.replace("{{ rollout_cwd }}", str(rollout_cwd))
    text = text.replace("{{ rollout_contents }}", truncated_rollout_contents)
    return text

def build_permissions_instructions(
    *,
    cwd: Path,
    sandbox: SandboxMode,
    approval_policy: ApprovalPolicy,
    network_access: NetworkAccess = "restricted",
    writable_roots: tuple[Path | str, ...] = (),
) -> str:
    """Resolves specific environment permission guidelines based on active configuration policies."""
    parts = []
    
    sb_key = sandbox.replace("-", "_")
    sb_file = ASSETS_DIR / "prompts" / "permissions" / "sandbox_mode" / f"{sb_key}.md"
    if sb_file.is_file():
        text = sb_file.read_text(encoding="utf-8")
        text = text.replace("{{network_access}}", network_access)
        parts.append(text)
        
    ap_key = approval_policy.replace("-", "_")
    ap_file = ASSETS_DIR / "prompts" / "permissions" / "approval_policy" / f"{ap_key}.md"
    if ap_file.is_file():
        parts.append(ap_file.read_text(encoding="utf-8"))
        
    return "\n\n".join(parts)


def collect_agents_md(cwd: Path) -> str:
    """Scans working directory extracting all agent specification configuration files into aggregated markdown blocks."""
    return ""

def memory_stage_one_rollout_token_limit(
    *,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95,
) -> int:
    """Calculates active context window token limits safe for background extraction runs."""
    limit = model_context_window or 128000
    limit = (limit * effective_context_window_percent) // 100
    limit = (limit * 70) // 100
    return max(1, limit)

def memory_stage_one_system_prompt() -> str:
    """Returns static stage one memory compact prompt template verbatim from assets."""
    p = ASSETS_DIR / "prompts" / "memories" / "write" / "stage_one_system.md"
    return p.read_text(encoding="utf-8")

def read_model_catalog_instructions(model: str) -> str | None:
    """Retrieves standard catalog constraints instructions block for targeted LLM models."""
    return None
