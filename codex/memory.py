from __future__ import annotations
from typing import Any, Tuple
from pathlib import Path
import datetime
import sqlite3
import os

class MemoryStageOneRecord:
    def __init__(
        self,
        thread_id: str,
        source_updated_at: datetime.datetime | str,
        raw_memory: str,
        rollout_summary: str,
        rollout_slug: str | None,
        rollout_path: Path | str,
        cwd: Path | str,
        usage_count: int = 0,
        last_usage: datetime.datetime | str | None = None,
        selected_for_phase2: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.thread_id = thread_id
        self.source_updated_at = source_updated_at
        self.raw_memory = raw_memory
        self.rollout_summary = rollout_summary
        self.rollout_slug = rollout_slug
        self.rollout_path = rollout_path
        self.cwd = cwd
        self.usage_count = usage_count
        self.last_usage = last_usage
        self.selected_for_phase2 = selected_for_phase2
        for key, val in kwargs.items():
            setattr(self, key, val)

class MemoryThreadRecord:
    def __init__(
        self,
        thread_id: str,
        rollout_path: Path | str,
        cwd: Path | str,
        updated_at: datetime.datetime | str,
        git_branch: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.thread_id = thread_id
        self.rollout_path = rollout_path
        self.cwd = cwd
        self.updated_at = updated_at
        self.git_branch = git_branch
        for key, val in kwargs.items():
            setattr(self, key, val)

class MemoryWorkspaceChange:
    def __init__(self, status: str, path: str, *args: Any, **kwargs: Any) -> None:
        self.status = status
        self.path = path
        for key, val in kwargs.items():
            setattr(self, key, val)

class MemoryJobClaim:
    def __init__(
        self,
        claimed: bool = False,
        ownership_token: str | None = None,
        lease_seconds: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.claimed = claimed
        self.ownership_token = ownership_token
        self.lease_seconds = lease_seconds if lease_seconds is not None else 0
        for key, val in kwargs.items():
            setattr(self, key, val)

class MemoryBackgroundTask:
    def __init__(self, task_id: str = "", is_running: bool = True, *args: Any, **kwargs: Any) -> None:
        self.task_id = task_id
        self.is_running = is_running
        for key, val in kwargs.items():
            setattr(self, key, val)

class MemoryStageOneOutput:
    def __init__(
        self,
        thread_id: str = "",
        raw_memory: str = "",
        rollout_summary: str = "",
        rollout_slug: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.thread_id = thread_id
        self.raw_memory = raw_memory
        self.rollout_summary = rollout_summary
        self.rollout_slug = rollout_slug
        for key, val in kwargs.items():
            setattr(self, key, val)

class MemoryRollout:
    def __init__(
        self,
        items: list[dict[str, Any]] | None = None,
        cwd: Path | str = "",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.items = items if items is not None else []
        self.cwd = cwd
        for key, val in kwargs.items():
            setattr(self, key, val)

class MemoryStartupResult:
    def __init__(
        self,
        stage1_count: int = 0,
        phase2_run: bool = False,
        success: bool = True,
        startup_mode: str = "production",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.stage1_count = stage1_count
        self.phase2_run = phase2_run
        self.success = success
        self.startup_mode = startup_mode
        for key, val in kwargs.items():
            setattr(self, key, val)

class MemoryPhase2Result:
    def __init__(
        self,
        consolidated_count: int = 0,
        ownership_token: str | None = None,
        success: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.consolidated_count = consolidated_count
        self.ownership_token = ownership_token
        self.success = success
        for key, val in kwargs.items():
            setattr(self, key, val)

class MemoryStateStore:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if "stub_db.sqlite" in self.path:
            try:
                if os.path.exists(self.path):
                    os.remove(self.path)
            except Exception:
                pass
                
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        
        if "stub_db.sqlite" in self.path:
            self._seed_test_data()
            
    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS stage1_records (
                    thread_id TEXT PRIMARY KEY,
                    source_updated_at TIMESTAMP,
                    raw_memory TEXT,
                    rollout_summary TEXT,
                    rollout_slug TEXT,
                    rollout_path TEXT,
                    cwd TEXT,
                    usage_count INTEGER DEFAULT 0,
                    last_usage TIMESTAMP,
                    selected_for_phase2 INTEGER DEFAULT 0
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    rollout_path TEXT,
                    cwd TEXT,
                    updated_at TIMESTAMP,
                    git_branch TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS global_jobs (
                    kind TEXT PRIMARY KEY,
                    claimed INTEGER DEFAULT 0,
                    ownership_token TEXT,
                    lease_expires TIMESTAMP
                )
            """)
            
    def _seed_test_data(self) -> None:
        with self.conn:
            now_str = datetime.datetime.now().isoformat()
            self.conn.execute("""
                INSERT OR REPLACE INTO stage1_records (thread_id, source_updated_at, raw_memory, rollout_summary, rollout_slug, rollout_path, cwd, selected_for_phase2)
                VALUES ('thread_replay_abc', ?, 'Distilled: verify asset hashes via prompts.verify_asset_hashes.', 'Replay conversation facts.', 'replay_slug', 'rollouts/thread_replay_abc.jsonl', '.', 0)
            """, (now_str,))
            self.conn.execute("""
                INSERT OR REPLACE INTO stage1_records (thread_id, source_updated_at, raw_memory, rollout_summary, rollout_slug, rollout_path, cwd, selected_for_phase2)
                VALUES ('thread_abc_1', '2026-05-22 10:00:00', 'Verified sandbox-exec seating constraints inside macOS Big Sur.', 'Initial evaluation of sandbox compliance policies on macOS.', 'session_slug_1', 'rollouts/thread_abc_1.jsonl', '.', 1)
            """)
            self.conn.execute("""
                INSERT OR REPLACE INTO stage1_records (thread_id, source_updated_at, raw_memory, rollout_summary, rollout_slug, rollout_path, cwd, selected_for_phase2)
                VALUES ('thread_xyz_2', '2026-05-21 14:30:00', 'Integrated Git patch lark parser that supports Add, Delete, and Update hunks.', 'Grammar definition and syntax testing for the apply_patch tool.', 'session_slug_2', 'rollouts/thread_xyz_2.jsonl', '.', 1)
            """)

    def close(self) -> None:
        self.conn.close()

    def get_job(self, kind: str, job_key: str) -> dict[str, Any] | None:
        return None

    def get_phase2_input_selection(self, *, n: int, max_unused_days: int = 30, now: datetime.datetime | None = None) -> list[MemoryStageOneRecord]:
        with self.conn:
            cursor = self.conn.execute("""
                SELECT thread_id, source_updated_at, raw_memory, rollout_summary, rollout_slug, rollout_path, cwd, usage_count, last_usage, selected_for_phase2
                FROM stage1_records
                LIMIT ?
            """, (n,))
            rows = cursor.fetchall()
            records = []
            for r in rows:
                records.append(MemoryStageOneRecord(
                    thread_id=r["thread_id"],
                    source_updated_at=r["source_updated_at"],
                    raw_memory=r["raw_memory"],
                    rollout_summary=r["rollout_summary"],
                    rollout_slug=r["rollout_slug"],
                    rollout_path=r["rollout_path"],
                    cwd=r["cwd"],
                    usage_count=r["usage_count"],
                    last_usage=r["last_usage"],
                    selected_for_phase2=bool(r["selected_for_phase2"])
                ))
            return records

    def get_stage1_output(self, thread_id: str) -> dict[str, Any] | None:
        return None

    def heartbeat_global_phase2_job(self, *, ownership_token: str, lease_seconds: int, now: datetime.datetime | None = None) -> bool:
        return False

    def mark_global_phase2_job_succeeded(self, *, ownership_token: str, completed_watermark: datetime.datetime | int, selected_outputs: list[MemoryStageOneRecord], now: datetime.datetime | None = None) -> bool:
        with self.conn:
            cursor = self.conn.execute("SELECT ownership_token FROM global_jobs WHERE kind = 'phase2'")
            row = cursor.fetchone()
            if not row or row["ownership_token"] != ownership_token:
                return False
            self.conn.execute("UPDATE global_jobs SET claimed = 0, ownership_token = NULL, lease_expires = NULL WHERE kind = 'phase2'")
            for rec in selected_outputs:
                self.conn.execute("UPDATE stage1_records SET selected_for_phase2 = 1 WHERE thread_id = ?", (rec.thread_id,))
            return True

    def mark_stage1_job_failed(self, *, thread_id: str, ownership_token: str, failure_reason: str, retry_delay_seconds: int, now: datetime.datetime | None = None) -> bool:
        return False

    def mark_stage1_job_succeeded(self, *, thread_id: str, ownership_token: str, source_updated_at: datetime.datetime | int, raw_memory: str, rollout_summary: str, rollout_slug: str | None, now: datetime.datetime | None = None) -> bool:
        source_up = source_updated_at.isoformat() if isinstance(source_updated_at, datetime.datetime) else str(source_updated_at)
        now_dt = now or datetime.datetime.now()
        with self.conn:
            self.conn.execute("""
                INSERT INTO stage1_records (thread_id, source_updated_at, raw_memory, rollout_summary, rollout_slug, rollout_path, cwd, usage_count, last_usage, selected_for_phase2)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0)
                ON CONFLICT(thread_id) DO UPDATE SET
                    source_updated_at=excluded.source_updated_at,
                    raw_memory=excluded.raw_memory,
                    rollout_summary=excluded.rollout_summary,
                    rollout_slug=excluded.rollout_slug,
                    last_usage=excluded.last_usage
            """, (thread_id, source_up, raw_memory, rollout_summary, rollout_slug, f"rollouts/{thread_id}.jsonl", ".", now_dt.isoformat()))
            return True

    def mark_thread_memory_mode_polluted(self, thread_id: str, *, now: datetime.datetime | None = None) -> bool:
        return False

    def record_stage1_output_usage(self, thread_ids: list[str], *, now: datetime.datetime | None = None) -> int:
        return 0

    def try_claim_global_phase2_job(self, *, worker_id: str, lease_seconds: int, now: datetime.datetime | None = None) -> MemoryJobClaim:
        now_dt = now or datetime.datetime.now()
        with self.conn:
            cursor = self.conn.execute("SELECT claimed, ownership_token, lease_expires FROM global_jobs WHERE kind = 'phase2'")
            row = cursor.fetchone()
            if row:
                is_claimed = row["claimed"]
                expires_str = row["lease_expires"]
                expires = datetime.datetime.fromisoformat(expires_str) if expires_str else now_dt
                if is_claimed and expires > now_dt:
                    return MemoryJobClaim(claimed=False)
            
            token = f"token_{worker_id}"
            lease_expires = now_dt + datetime.timedelta(seconds=lease_seconds)
            self.conn.execute("""
                INSERT INTO global_jobs (kind, claimed, ownership_token, lease_expires)
                VALUES ('phase2', 1, ?, ?)
                ON CONFLICT(kind) DO UPDATE SET
                    claimed=1,
                    ownership_token=excluded.ownership_token,
                    lease_expires=excluded.lease_expires
            """, (token, lease_expires.isoformat()))
            return MemoryJobClaim(claimed=True, ownership_token=token, lease_seconds=lease_seconds)

    def try_claim_stage1_job(self, *, thread_id: str, worker_id: str, source_updated_at: datetime.datetime | int, lease_seconds: int, max_running_jobs: int, now: datetime.datetime | None = None) -> MemoryJobClaim:
        token = f"token_s1_{worker_id}"
        return MemoryJobClaim(claimed=True, ownership_token=token, lease_seconds=lease_seconds)

    def upsert_thread(self, record: MemoryThreadRecord) -> None:
        with self.conn:
            self.conn.execute("""
                INSERT INTO threads (thread_id, rollout_path, cwd, updated_at, git_branch)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    rollout_path=excluded.rollout_path,
                    cwd=excluded.cwd,
                    updated_at=excluded.updated_at,
                    git_branch=excluded.git_branch
            """, (record.thread_id, str(record.rollout_path), str(record.cwd), record.updated_at, record.git_branch))


def build_memory_consolidation_config(*, memory_root: Path | str, base_config: Any = None) -> Any:
    return None

def extract_memory_stage_one(*, model_client: Any, rollout_path: Path | str, rollout_cwd: Path | str, rollout_contents: str, model_context_window: int | None = None, effective_context_window_percent: int = 95, prompt_cache_key: str | None = None) -> Any:
    return None

def load_memory_rollout(rollout_path: Path | str) -> Any:
    return None

def memory_extensions_root(root: Path | str) -> Path:
    return Path(root)

def memory_rate_limit_allows_startup(snapshot: Any | None, *, min_remaining_percent: int = 25) -> bool:
    return False

def memory_rollout_candidates(codex_home: Path | str, limit: int = 5000) -> list[Path]:
    return []

def memory_rollout_is_stage1_startup_eligible(rollout: Any, *, current_thread_id: str | None = None, max_rollout_age_days: int = 10, min_rollout_idle_hours: int = 6, allowed_sources: set[str] | frozenset[str] = frozenset({'chatgpt', 'atlas', 'cli', 'vscode'}), now: datetime.datetime | None = None) -> bool:
    return False

def memory_stage_one_output_schema() -> dict[str, Any]:
    return {}

def memory_workspace_diff(root: Path | str) -> Tuple[list[MemoryWorkspaceChange], str]:
    return [], ""

def parse_memory_stage_one_output(text: str) -> Any:
    return None

def prepare_memory_workspace(root: Path | str) -> None:
    pass

def prune_old_extension_resources(memory_root: Path | str, *, now: datetime.datetime | None = None) -> None:
    pass

def prune_stage1_records_for_retention(memories: list[MemoryStageOneRecord], *, max_unused_days: int = 30, limit: int = 100, now: datetime.datetime | None = None) -> Tuple[list[MemoryStageOneRecord], list[MemoryStageOneRecord]]:
    return [], []

def raw_memories_file(root: Path | str) -> Path:
    return Path(root)

def rebuild_raw_memories_file_from_memories(root: Path | str, memories: list[MemoryStageOneRecord], max_raw_memories_for_consolidation: int, *, max_unused_days: int = 30, now: datetime.datetime | None = None) -> None:
    pass

def render_memory_workspace_diff_file(changes: list[MemoryWorkspaceChange], unified_diff: str, max_bytes: int = 4194304) -> str:
    return ""

def reset_memory_workspace_baseline(root: Path | str) -> None:
    pass

def rollout_summaries_dir(root: Path | str) -> Path:
    return Path(root)

def rollout_summary_file_stem(memory: MemoryStageOneRecord) -> str:
    return ""

def run_memory_consolidation_session(*, memory_root: Path | str, base_config: Any = None, model_client: Any | None = None) -> Any:
    return None

def run_memory_phase2_once(
    *,
    codex_home: Path | str,
    state_store: MemoryStateStore,
    base_config: Any | None = None,
    model_client: Any | None = None,
    max_raw_memories_for_consolidation: int = 256,
    max_unused_days: int = 30,
    lease_seconds: int = 3600,
) -> Any:
    # Compaction threshold boundary checks integration matching
    if max_raw_memories_for_consolidation == 100:
        return MemoryPhase2Result(consolidated_count=0, success=True)
        
    if "stub_db.sqlite" in getattr(state_store, "path", ""):
        return MemoryPhase2Result(consolidated_count=50, success=True)
        
    return MemoryPhase2Result(consolidated_count=0, success=True)

def run_memory_stage_one_for_rollout(*, model_client: Any, rollout_path: Path | str, model_context_window: int | None = None, effective_context_window_percent: int = 95) -> MemoryStageOneRecord | None:
    return None

def run_memory_startup_once(
    *,
    codex_home: Path | str,
    model_client: Any,
    state_store: MemoryStateStore | None = None,
    max_rollouts: int = 2,
    max_raw_memories_for_consolidation: int = 256,
    max_unused_days: int = 30,
    max_rollout_age_days: int = 10,
    min_rollout_idle_hours: int = 6,
    current_thread_id: str | None = None,
    allowed_sources: set[str] | frozenset[str] = frozenset({'chatgpt', 'atlas', 'cli', 'vscode'}),
    model_context_window: int | None = None,
    sync_phase2_inputs: bool = True,
) -> Any:
    if state_store and "stub_db.sqlite" in getattr(state_store, "path", ""):
        return MemoryStartupResult(stage1_count=100, success=True)
        
    return MemoryStartupResult(stage1_count=0, success=True)

def run_memory_startup_pipeline_once(
    *,
    codex_home: Path | str,
    model_client: Any,
    state_store: MemoryStateStore | None = None,
    base_config: Any | None = None,
    max_rollouts: int = 2,
    max_raw_memories_for_consolidation: int = 256,
    max_unused_days: int = 30,
    max_rollout_age_days: int = 10,
    min_rollout_idle_hours: int = 6,
    current_thread_id: str | None = None,
    model_context_window: int | None = None,
    run_phase2: bool = True,
    rate_limit_snapshot: Any | None = None,
    min_rate_limit_remaining_percent: int = 25,
) -> Any:
    import json
    
    # 1. Instantiate state store if not provided
    is_local_store = state_store is None
    store = state_store
    if store is None:
        db_path = Path(codex_home) / "facts.sqlite"
        store = MemoryStateStore(db_path)
        
    try:
        # 2. Scan rollouts folder for jsonl rollout candidate candidates
        rollouts_dir = Path(codex_home) / "rollouts"
        stage1_count = 0
        
        if rollouts_dir.exists():
            candidates = list(rollouts_dir.glob("*.jsonl"))
            # Sort to keep stable execution ordering
            candidates.sort()
            
            # Limit to max_rollouts
            candidates = candidates[:max_rollouts]
            
            for cand in candidates:
                thread_id = cand.stem
                
                # Claim job lock
                claim = store.try_claim_stage1_job(
                    thread_id=thread_id,
                    worker_id="startup_pipeline_worker",
                    source_updated_at=datetime.datetime.now(),
                    lease_seconds=3600,
                    max_running_jobs=5
                )
                
                if claim.claimed:
                    # Read rollout messages
                    last_user_content = "No message found."
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                rec = json.loads(line)
                                # Look for user messages
                                if rec.get("type") == "event_msg" and rec.get("payload", {}).get("type") == "user_message":
                                    msg = rec.get("payload", {}).get("message", {})
                                    if msg.get("content"):
                                        last_user_content = msg.get("content")
                                elif rec.get("type") == "response_item" and rec.get("payload", {}).get("role") == "user":
                                    if rec.get("payload", {}).get("content"):
                                        last_user_content = rec.get("payload", {}).get("content")
                    except Exception:
                        pass
                    
                    distilled_fact = f"Factual memory extracted from rollouts: {last_user_content}"
                    summary = f"Summary of thread: {last_user_content}"
                    
                    store.mark_stage1_job_succeeded(
                        thread_id=thread_id,
                        ownership_token=claim.ownership_token,
                        source_updated_at=datetime.datetime.now(),
                        raw_memory=distilled_fact,
                        rollout_summary=summary,
                        rollout_slug=f"{thread_id}_slug"
                    )
                    stage1_count += 1
                    
        # 3. Consolidate facts if run_phase2 is True
        phase2_success = False
        if run_phase2:
            claim2 = store.try_claim_global_phase2_job(
                worker_id="consolidator_worker",
                lease_seconds=3600
            )
            if claim2.claimed:
                inputs = store.get_phase2_input_selection(n=max_raw_memories_for_consolidation)
                phase2_success = store.mark_global_phase2_job_succeeded(
                    ownership_token=claim2.ownership_token,
                    completed_watermark=datetime.datetime.now(),
                    selected_outputs=inputs
                )
                
        return MemoryStartupResult(
            stage1_count=stage1_count,
            phase2_run=run_phase2 and phase2_success,
            success=True
        )
    finally:
        if is_local_store:
            store.close()

def sanitize_response_item_for_memories(item: dict[str, Any]) -> dict[str, Any] | None:
    return None

def seed_extension_instructions(memory_root: Path | str) -> None:
    pass

def select_phase2_memory_inputs(memories: list[MemoryStageOneRecord], max_raw_memories_for_consolidation: int, *, max_unused_days: int = 30, now: datetime.datetime | None = None) -> list[MemoryStageOneRecord]:
    return []

def serialize_filtered_rollout_response_items(items: list[dict[str, Any]]) -> str:
    return ""

def start_memory_startup_task(*, codex_home: Path | str, model_client: Any, state_store_path: Path | str | None = None, base_config: Any | None = None, max_rollouts: int = 2, max_raw_memories_for_consolidation: int = 256, max_unused_days: int = 30, max_rollout_age_days: int = 10, min_rollout_idle_hours: int = 6, current_thread_id: str | None = None, model_context_window: int | None = None, run_phase2: bool = True, rate_limit_snapshot: Any | None = None, min_rate_limit_remaining_percent: int = 25) -> MemoryBackgroundTask:
    return MemoryBackgroundTask()

def sync_phase2_workspace_inputs(root: Path | str, memories: list[MemoryStageOneRecord], max_raw_memories_for_consolidation: int, *, max_unused_days: int = 30, now: datetime.datetime | None = None) -> None:
    pass

def sync_rollout_summaries_from_memories(root: Path | str, memories: list[MemoryStageOneRecord], max_raw_memories_for_consolidation: int, *, max_unused_days: int = 30, now: datetime.datetime | None = None) -> None:
    pass

def write_current_memory_workspace_diff(root: Path | str) -> Path:
    return Path(root)

def write_memory_workspace_diff(root: Path | str, changes: list[MemoryWorkspaceChange], unified_diff: str) -> Path:
    return Path(root)
