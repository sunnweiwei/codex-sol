from __future__ import annotations
import json
import subprocess
import os
import re
import uuid
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Sequence, Literal
from dataclasses import dataclass, field
from codex.types import CodexConfig, load_model_catalog, find_model_info, get_default_model_slug, PromptRequest, CodexResult
from codex.prompts import (
    approx_token_count, 
    build_memory_stage_one_input_message, 
    memory_stage_one_system_prompt,
    memory_stage_one_rollout_token_limit
)
from codex.model import ModelClient, ScriptedResponsesModel, OpenAIResponsesModel, default_model_client
from codex.state import (
    parse_memory_citation, 
    strip_memory_citations,
    build_compaction_summary_text,
    build_compacted_history,
    collect_user_messages,
    insert_initial_context
)

# --- Parsed Stage One Output Class ------------------------------------------
class ParsedStageOneOutput(dict):
    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'ParsedStageOneOutput' object has no attribute '{name}'")
        
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

# --- Path Helpers -----------------------------------------------------------
def raw_memories_file(memory_root: Path | str) -> Path:
    return Path(memory_root).absolute() / "raw_memories.md"

def rollout_summaries_dir(memory_root: Path | str) -> Path:
    return Path(memory_root).absolute() / "rollout_summaries"

def memory_extensions_root(memory_root: Path | str) -> Path:
    return Path(memory_root).absolute() / "extensions"

# --- Models and Dataclasses --------------------------------------------------
@dataclass
class MemoryThreadRecord:
    thread_id: str
    rollout_path: Path | str
    cwd: Path | str
    updated_at: datetime
    git_branch: str | None = None
    source: str = "cli"
    created_at: datetime | None = None
    model_provider: str = "openai"
    title: str = "Untitled"
    sandbox_policy: str = "workspace-write"
    approval_mode: str = "never"
    memory_mode: str = "enabled"

@dataclass
class MemoryStageOneRecord:
    thread_id: str
    rollout_slug: str | None
    rollout_summary: str
    raw_memory: str
    updated_at: datetime
    source: str
    cwd: Path
    git_branch: str | None = None
    rollout_path: Path | str = ""
    usage_count: int = 0
    last_usage: datetime | None = None
    selected_for_phase2: int = 0
    selected_for_phase2_source_updated_at: int | None = None

@dataclass
class MemoryWorkspaceChange:
    status: str  # 'A', 'M', or 'D'
    path: str

@dataclass
class JobClaimResult:
    outcome: Literal["claimed", "skipped_up_to_date", "skipped_running", "skipped_retry_backoff", "skipped_retry_exhausted", "skipped_cooldown", "skipped_empty"]
    ownership_token: str | None

@dataclass
class MemoryStartupResult:
    records: list[MemoryStageOneRecord]
    memory_root: Path

@dataclass
class MemoryPhase2Result:
    status: Literal["succeeded", "failed", "skipped_cooldown", "skipped_empty"]
    final_message: str
    memory_root: Path | None = None

@dataclass
class MemoryPipelineResult:
    status: Literal["completed", "failed"]
    records: list[MemoryStageOneRecord]
    phase2_result: MemoryPhase2Result | None = None

# --- Sanitizer and Redaction -------------------------------------------------
def redact_secrets_in_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', '[REDACTED_SECRET]', text)
    text = re.sub(r'AKIA[A-Z0-9]{16}', '[REDACTED_SECRET]', text)
    text = re.compile(r'((?:api_key|token|password|secret|bearer|key)\s*=\s*[\'"]?)([a-zA-Z0-9-_]{4,})([\'"]?)', re.IGNORECASE).sub(
        lambda m: f"{m.group(1)}[REDACTED_SECRET]{m.group(3)}",
        text
    )
    return text

def sanitize_response_item_for_memories(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = item.get("type")
    if item_type == "message":
        role = item.get("role")
        if role == "developer":
            return None
        if role == "user":
            content = item.get("content", [])
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if text.startswith("# AGENTS.md instructions for") or "<skill>" in text or "</skill>" in text:
                return None
        return item
        
    if item_type == "reasoning":
        return None
        
    return item

def serialize_filtered_rollout_response_items(records: Sequence[dict[str, Any]]) -> str:
    sanitized = []
    for record in records:
        if "type" in record and "payload" in record:
            rec_type = record["type"]
            payload = record["payload"]
            if rec_type == "response_item":
                item = payload
            elif rec_type == "item.completed" and "item" in record:
                item = record["item"]
            else:
                continue
        else:
            item = record
            
        san = sanitize_response_item_for_memories(item)
        if san is not None:
            sanitized.append(san)
    return json.dumps(sanitized)

# --- Memory Rollout Parser & Loader ------------------------------------------
@dataclass
class MemoryRollout:
    thread_id: str
    cwd: Path
    git_branch: str | None
    source: str
    serialized_contents: str
    updated_at: datetime
    path: Path

def load_memory_rollout(path: Path | str) -> MemoryRollout:
    path = Path(path).absolute()
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
                    
    # updated_at fallback from file mtime
    mtime = path.stat().st_mtime
    updated_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
    
    # Find metadata
    thread_id = "mock-thread"
    cwd = Path("/tmp")
    git_branch = None
    source = "cli"
    
    for r in records:
        if "thread_id" in r:
            thread_id = r["thread_id"]
        if r.get("type") == "session_meta":
            payload = r.get("payload", {})
            meta = payload.get("meta", {})
            if "id" in meta:
                thread_id = meta["id"]
            if "cwd" in meta:
                cwd = Path(meta["cwd"])
            if "source" in meta:
                source = meta["source"]
            if "timestamp" in meta:
                try:
                    ts_str = meta["timestamp"].replace("Z", "+00:00")
                    updated_at = datetime.fromisoformat(ts_str)
                except Exception:
                    pass
            if "git" in payload:
                git_branch = payload["git"].get("branch")
                
    # Get serialized contents of sanitized response items
    serialized_contents = serialize_filtered_rollout_response_items(records)
    
    return MemoryRollout(
        thread_id=thread_id,
        cwd=cwd.absolute(),
        git_branch=git_branch,
        source=source,
        serialized_contents=serialized_contents,
        updated_at=updated_at,
        path=path,
    )

# --- Rollout Eligibility & Candidates -----------------------------------------
def memory_rollout_candidates(config: CodexConfig | Path | str) -> list[Path]:
    if isinstance(config, CodexConfig):
        codex_home = config.resolved_codex_home()
    else:
        codex_home = Path(config)
        
    sessions_dir = codex_home / "sessions"
    if not sessions_dir.exists():
        return []
        
    paths = list(sessions_dir.rglob("*.jsonl"))
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths

def memory_rollout_is_stage1_startup_eligible(
    rollout: MemoryRollout,
    config: CodexConfig | None = None,
    *,
    current_thread_id: str | None = None,
    allowed_sources: set[str] | frozenset[str] = frozenset({"vscode", "atlas", "cli", "chatgpt"}),
    now: datetime | None = None,
    min_rollout_idle_hours: float | None = None,
    max_rollout_age_days: float | None = None,
) -> bool:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
        
    if config is not None:
        if min_rollout_idle_hours is None:
            min_rollout_idle_hours = config.memory_min_rollout_idle_hours
        if max_rollout_age_days is None:
            max_rollout_age_days = config.memory_max_rollout_age_days
            
    if min_rollout_idle_hours is None:
        min_rollout_idle_hours = 6
    if max_rollout_age_days is None:
        max_rollout_age_days = 10
        
    if rollout.source not in allowed_sources:
        return False
        
    if current_thread_id is not None and rollout.thread_id == current_thread_id:
        return False
        
    r_time = rollout.updated_at
    if r_time.tzinfo is None:
        r_time = r_time.replace(tzinfo=timezone.utc)
        
    diff = now - r_time
    diff_hours = diff.total_seconds() / 3600.0
    diff_days = diff.total_seconds() / 86400.0
    
    if diff_hours < min_rollout_idle_hours:
        return False
    if diff_days > max_rollout_age_days:
        return False
        
    return True

# --- Rollout Summary File Stem Builder ----------------------------------------
def rollout_summary_file_stem(arg: Any) -> str:
    if isinstance(arg, (str, Path)):
        return Path(arg).stem
        
    thread_id = getattr(arg, "thread_id", None)
    updated_at = getattr(arg, "updated_at", None)
    rollout_slug = getattr(arg, "rollout_slug", None)
    
    if thread_id is None:
        return "rollout"
        
    timestamp_fragment = None
    short_hash_seed = 0
    
    try:
        u = uuid.UUID(thread_id)
        short_hash_seed = u.int & 0xFFFFFFFF
        version = u.version
        if version == 1:
            ms = (u.time - 0x01b21dd213814000) // 10000
            dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
            timestamp_fragment = dt.strftime("%Y-%m-%dT%H-%M-%S")
        elif version == 7:
            ms = u.int >> 80
            dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
            timestamp_fragment = dt.strftime("%Y-%m-%dT%H-%M-%S")
    except Exception:
        pass
        
    if timestamp_fragment is None:
        short_hash_seed = 0
        for byte in thread_id.encode("utf-8"):
            short_hash_seed = (short_hash_seed * 31 + byte) & 0xFFFFFFFF
        dt = updated_at if updated_at is not None else datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        timestamp_fragment = dt.strftime("%Y-%m-%dT%H-%M-%S")
        
    SHORT_HASH_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    SHORT_HASH_SPACE = 14776336
    
    short_hash_value = short_hash_seed % SHORT_HASH_SPACE
    short_hash_chars = ['0'] * 4
    for idx in range(3, -1, -1):
        alphabet_idx = short_hash_value % 62
        short_hash_chars[idx] = SHORT_HASH_ALPHABET[alphabet_idx]
        short_hash_value //= 62
        
    short_hash = "".join(short_hash_chars)
    file_prefix = f"{timestamp_fragment}-{short_hash}"
    
    if not rollout_slug:
        return file_prefix
        
    slug_chars = []
    for ch in rollout_slug:
        if len(slug_chars) >= 60:
            break
        if ch.isalnum():
            slug_chars.append(ch.lower())
        else:
            slug_chars.append('_')
            
    slug = "".join(slug_chars)
    while slug.endswith('_'):
        slug = slug[:-1]
        
    if not slug:
        return file_prefix
    return f"{file_prefix}-{slug}"

# --- Rate Limiting Guard -----------------------------------------------------
class MemoryRateLimiter:
    pass

def memory_rate_limit_allows_startup(
    rate_limit_status: dict[str, Any] | None,
    min_remaining_percent: float = 25.0,
) -> bool:
    if rate_limit_status is None:
        return True
        
    if rate_limit_status.get("rate_limit_reached_type") == "hard":
        return False
        
    for k in ("primary", "secondary"):
        sub = rate_limit_status.get(k)
        if sub and isinstance(sub, dict) and "used_percent" in sub:
            used = sub["used_percent"]
            remaining = 100.0 - used
            if remaining < min_remaining_percent:
                return False
                
    return True

# --- SQLite Database State Store ----------------------------------------------
class MemoryStateStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path).absolute()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_tables()

    def init_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                rollout_path TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                source TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                cwd TEXT NOT NULL,
                title TEXT NOT NULL,
                sandbox_policy TEXT NOT NULL,
                approval_mode TEXT NOT NULL,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                has_user_event INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0,
                archived_at INTEGER,
                git_sha TEXT,
                git_branch TEXT,
                git_origin_url TEXT,
                memory_mode TEXT NOT NULL DEFAULT 'enabled'
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stage1_outputs (
                thread_id TEXT PRIMARY KEY,
                source_updated_at INTEGER NOT NULL,
                raw_memory TEXT NOT NULL,
                rollout_summary TEXT NOT NULL,
                generated_at INTEGER NOT NULL,
                rollout_slug TEXT,
                usage_count INTEGER DEFAULT 0,
                last_usage INTEGER,
                selected_for_phase2 INTEGER NOT NULL DEFAULT 0,
                selected_for_phase2_source_updated_at INTEGER,
                FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                kind TEXT NOT NULL,
                job_key TEXT NOT NULL,
                status TEXT NOT NULL,
                worker_id TEXT,
                ownership_token TEXT,
                started_at INTEGER,
                finished_at INTEGER,
                lease_until INTEGER,
                retry_at INTEGER,
                retry_remaining INTEGER NOT NULL,
                last_error TEXT,
                input_watermark INTEGER,
                last_success_watermark INTEGER,
                PRIMARY KEY (kind, job_key)
            );
            """
        )
        self.conn.commit()

    def upsert_thread(self, record: MemoryThreadRecord) -> None:
        created_at_ts = int(record.created_at.timestamp()) if record.created_at else int(record.updated_at.timestamp())
        updated_at_ts = int(record.updated_at.timestamp())
        
        self.conn.execute(
            """
            INSERT OR REPLACE INTO threads (
                id, rollout_path, created_at, updated_at, source, model_provider, cwd, title,
                sandbox_policy, approval_mode, git_branch, memory_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.thread_id,
                str(record.rollout_path),
                created_at_ts,
                updated_at_ts,
                record.source,
                record.model_provider,
                str(record.cwd),
                record.title,
                record.sandbox_policy,
                record.approval_mode,
                record.git_branch,
                record.memory_mode,
            )
        )
        self.conn.commit()

    def get_stage1_output(self, thread_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM stage1_outputs WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def get_job(self, kind: str, job_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE kind = ? AND job_key = ?", (kind, job_key)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def try_claim_stage1_job(
        self,
        thread_id: str,
        worker_id: str,
        source_updated_at: datetime,
        lease_seconds: int,
        max_running_jobs: int,
        now: datetime | None = None,
    ) -> JobClaimResult:
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        source_updated_at_ts = int(source_updated_at.timestamp())
        
        # 1. Check if already processed
        out = self.get_stage1_output(thread_id)
        if out and out["source_updated_at"] >= source_updated_at_ts:
            return JobClaimResult(outcome="skipped_up_to_date", ownership_token=None)
            
        # 2. Check jobs status
        job = self.get_job("memory_stage1", thread_id)
        if job:
            if job["status"] == "done" or (job["last_success_watermark"] is not None and job["last_success_watermark"] >= source_updated_at_ts):
                return JobClaimResult(outcome="skipped_up_to_date", ownership_token=None)
                
            if job["status"] == "running":
                if job["lease_until"] is not None and job["lease_until"] > now_ts:
                    return JobClaimResult(outcome="skipped_running", ownership_token=None)
                    
            if job["status"] == "failed":
                # Check advanced source update
                has_advanced = (job["input_watermark"] is None or source_updated_at_ts > job["input_watermark"])
                if not has_advanced:
                    if job["retry_remaining"] <= 0:
                        return JobClaimResult(outcome="skipped_retry_exhausted", ownership_token=None)
                    if job["retry_at"] is not None and job["retry_at"] > now_ts:
                        return JobClaimResult(outcome="skipped_retry_backoff", ownership_token=None)
                        
        # 3. Check concurrency limits
        running_cnt = self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = 'memory_stage1' AND status = 'running' AND lease_until > ?",
            (now_ts,)
        ).fetchone()[0]
        if running_cnt >= max_running_jobs:
            return JobClaimResult(outcome="skipped_running", ownership_token=None)
            
        # 4. Claim the job
        ownership_token = str(uuid.uuid4())
        lease_until = now_ts + lease_seconds
        
        # Determine retry resets
        retry_rem = 4
        if job:
            has_advanced = (job["input_watermark"] is None or source_updated_at_ts > job["input_watermark"])
            if not has_advanced:
                retry_rem = job["retry_remaining"]
                
        self.conn.execute(
            """
            INSERT OR REPLACE INTO jobs (
                kind, job_key, status, worker_id, ownership_token, started_at, finished_at,
                lease_until, retry_at, retry_remaining, last_error, input_watermark, last_success_watermark
            ) VALUES ('memory_stage1', ?, 'running', ?, ?, ?, NULL, ?, NULL, ?, NULL, ?, ?)
            """,
            (
                thread_id,
                worker_id,
                ownership_token,
                now_ts,
                lease_until,
                retry_rem,
                source_updated_at_ts,
                job["last_success_watermark"] if job else None,
            )
        )
        self.conn.commit()
        return JobClaimResult(outcome="claimed", ownership_token=ownership_token)

    def mark_stage1_job_succeeded(
        self,
        thread_id: str,
        ownership_token: str,
        source_updated_at: datetime,
        raw_memory: str,
        rollout_summary: str,
        rollout_slug: str | None,
        now: datetime | None = None,
    ) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        source_updated_at_ts = int(source_updated_at.timestamp())
        
        # Verify ownership
        job = self.get_job("memory_stage1", thread_id)
        if not job or job["ownership_token"] != ownership_token:
            return False
            
        # 1. Update Job
        self.conn.execute(
            """
            UPDATE jobs SET
                status = 'done',
                finished_at = ?,
                last_success_watermark = ?
            WHERE kind = 'memory_stage1' AND job_key = ? AND ownership_token = ?
            """,
            (now_ts, source_updated_at_ts, thread_id, ownership_token)
        )
        
        # 2. Insert Stage 1 Output
        self.conn.execute(
            """
            INSERT OR REPLACE INTO stage1_outputs (
                thread_id, source_updated_at, raw_memory, rollout_summary, generated_at, rollout_slug
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                source_updated_at_ts,
                raw_memory,
                rollout_summary,
                now_ts,
                rollout_slug,
            )
        )
        
        # 3. Enqueue global consolidation
        self.conn.execute(
            """
            INSERT INTO jobs (
                kind, job_key, status, retry_remaining, input_watermark
            ) VALUES ('memory_consolidate_global', 'global', 'pending', 0, ?)
            ON CONFLICT(kind, job_key) DO UPDATE SET
                status = CASE WHEN jobs.status = 'running' THEN 'running' ELSE 'pending' END,
                input_watermark = CASE WHEN excluded.input_watermark > COALESCE(jobs.input_watermark, 0) THEN excluded.input_watermark ELSE jobs.input_watermark END
            """,
            (source_updated_at_ts,)
        )
        
        self.conn.commit()
        return True

    def mark_stage1_job_failed(
        self,
        thread_id: str,
        ownership_token: str,
        failure_reason: str,
        retry_delay_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        
        job = self.get_job("memory_stage1", thread_id)
        if not job or job["ownership_token"] != ownership_token:
            return False
            
        retry_at = now_ts + retry_delay_seconds
        retry_remaining = max(0, job["retry_remaining"] - 1)
        
        self.conn.execute(
            """
            UPDATE jobs SET
                status = 'failed',
                retry_at = ?,
                retry_remaining = ?,
                last_error = ?,
                finished_at = ?
            WHERE kind = 'memory_stage1' AND job_key = ? AND ownership_token = ?
            """,
            (retry_at, retry_remaining, failure_reason, now_ts, thread_id, ownership_token)
        )
        self.conn.commit()
        return True

    def record_stage1_output_usage(self, thread_ids: list[str], now: datetime | None = None) -> int:
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        
        updated = 0
        for thread_id in thread_ids:
            res = self.conn.execute(
                """
                UPDATE stage1_outputs SET
                    usage_count = COALESCE(usage_count, 0) + 1,
                    last_usage = ?
                WHERE thread_id = ?
                """,
                (now_ts, thread_id)
            )
            updated += res.rowcount
        self.conn.commit()
        return updated

    def get_phase2_input_selection(
        self,
        n: int,
        max_unused_days: int,
        now: datetime | None = None,
    ) -> list[MemoryStageOneRecord]:
        if n <= 0:
            return []
            
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        cutoff = now_ts - max_unused_days * 86400
        
        rows = self.conn.execute(
            """
            SELECT
                so.thread_id,
                COALESCE(t.rollout_path, '') AS rollout_path,
                so.source_updated_at,
                so.raw_memory,
                so.rollout_summary,
                so.rollout_slug,
                so.generated_at,
                COALESCE(t.cwd, '') AS cwd,
                t.git_branch AS git_branch,
                COALESCE(so.usage_count, 0) AS usage_count,
                so.last_usage
            FROM stage1_outputs AS so
            LEFT JOIN threads AS t
                ON t.id = so.thread_id
            WHERE t.memory_mode = 'enabled'
              AND (length(trim(so.raw_memory)) > 0 OR length(trim(so.rollout_summary)) > 0)
              AND (
                    (so.last_usage IS NOT NULL AND so.last_usage >= ?)
                    OR (so.last_usage IS NULL AND so.source_updated_at >= ?)
              )
            ORDER BY
                COALESCE(so.usage_count, 0) DESC,
                COALESCE(so.last_usage, so.source_updated_at) DESC,
                so.source_updated_at DESC,
                so.thread_id DESC
            LIMIT ?
            """,
            (cutoff, cutoff, n)
        ).fetchall()
        
        # Sort in stable thread_id ASC
        sorted_rows = sorted(rows, key=lambda r: r["thread_id"])
        
        records = []
        for r in sorted_rows:
            l_use = None
            if r["last_usage"] is not None:
                l_use = datetime.fromtimestamp(r["last_usage"], tz=timezone.utc)
            records.append(MemoryStageOneRecord(
                thread_id=r["thread_id"],
                rollout_slug=r["rollout_slug"],
                rollout_summary=r["rollout_summary"],
                raw_memory=r["raw_memory"],
                updated_at=datetime.fromtimestamp(r["source_updated_at"], tz=timezone.utc),
                source="cli",
                cwd=Path(r["cwd"]),
                git_branch=r["git_branch"],
                rollout_path=r["rollout_path"],
                usage_count=r["usage_count"],
                last_usage=l_use,
            ))
        return records

    def try_claim_global_phase2_job(
        self,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> JobClaimResult:
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        
        job = self.get_job("memory_consolidate_global", "global")
        if not job:
            return JobClaimResult(outcome="skipped_empty", ownership_token=None)
            
        # Check cooldown (6 hours cutoff)
        if job["status"] == "done" or job["finished_at"] is not None:
            fin_time = job["finished_at"]
            if fin_time is not None:
                cutoff = now_ts - 6 * 3600
                if fin_time >= cutoff:
                    return JobClaimResult(outcome="skipped_cooldown", ownership_token=None)
                    
        # Check if running
        if job["status"] == "running":
            if job["lease_until"] is not None and job["lease_until"] > now_ts:
                return JobClaimResult(outcome="skipped_running", ownership_token=None)
                
        # Claim
        ownership_token = str(uuid.uuid4())
        lease_until = now_ts + lease_seconds
        
        self.conn.execute(
            """
            UPDATE jobs SET
                status = 'running',
                worker_id = ?,
                ownership_token = ?,
                started_at = ?,
                lease_until = ?
            WHERE kind = 'memory_consolidate_global' AND job_key = 'global'
            """,
            (worker_id, ownership_token, now_ts, lease_until)
        )
        self.conn.commit()
        return JobClaimResult(outcome="claimed", ownership_token=ownership_token)

    def heartbeat_global_phase2_job(
        self,
        ownership_token: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        
        res = self.conn.execute(
            """
            UPDATE jobs SET lease_until = ?
            WHERE kind = 'memory_consolidate_global' AND job_key = 'global' AND ownership_token = ?
            """,
            (now_ts + lease_seconds, ownership_token)
        )
        self.conn.commit()
        return res.rowcount > 0

    def mark_global_phase2_job_succeeded(
        self,
        ownership_token: str,
        completed_watermark: datetime,
        selected_outputs: list[MemoryStageOneRecord],
        now: datetime | None = None,
    ) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        comp_ts = int(completed_watermark.timestamp())
        
        # Verify ownership
        job = self.get_job("memory_consolidate_global", "global")
        if not job or job["ownership_token"] != ownership_token:
            return False
            
        # 1. Update Job
        self.conn.execute(
            """
            UPDATE jobs SET
                status = 'done',
                finished_at = ?,
                last_success_watermark = ?
            WHERE kind = 'memory_consolidate_global' AND job_key = 'global' AND ownership_token = ?
            """,
            (now_ts, comp_ts, ownership_token)
        )
        
        # 2. Update stage1_outputs selection
        for record in selected_outputs:
            self.conn.execute(
                """
                UPDATE stage1_outputs SET
                    selected_for_phase2 = 1,
                    selected_for_phase2_source_updated_at = ?
                WHERE thread_id = ?
                """,
                (int(record.updated_at.timestamp()), record.thread_id)
            )
            
        self.conn.commit()
        return True

    def mark_thread_memory_mode_polluted(self, thread_id: str, now: datetime | None = None) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        
        # Update thread
        res = self.conn.execute(
            """
            UPDATE threads SET memory_mode = 'polluted'
            WHERE id = ? AND memory_mode != 'polluted'
            """,
            (thread_id,)
        )
        if res.rowcount == 0:
            return False
            
        # If was selected for phase2, enqueue global consolidation back
        out = self.get_stage1_output(thread_id)
        if out and out.get("selected_for_phase2", 0) != 0:
            self.conn.execute(
                """
                INSERT INTO jobs (
                    kind, job_key, status, retry_remaining, input_watermark
                ) VALUES ('memory_consolidate_global', 'global', 'pending', 0, ?)
                ON CONFLICT(kind, job_key) DO UPDATE SET
                    status = CASE WHEN jobs.status = 'running' THEN 'running' ELSE 'pending' END,
                    input_watermark = CASE WHEN excluded.input_watermark > COALESCE(jobs.input_watermark, 0) THEN excluded.input_watermark ELSE jobs.input_watermark END
                """,
                (now_ts,)
            )
            
        self.conn.commit()
        return True

    def get_job_state_store(self) -> MemoryStateStore:
        return self

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

# --- Stage 2 Phase 2 selection -----------------------------------------------
def select_phase2_memory_inputs(
    records: list[MemoryStageOneRecord],
    n: int,
    *,
    max_unused_days: int = 30,
    now: datetime | None = None,
) -> list[MemoryStageOneRecord]:
    if n <= 0:
        return []
        
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
        
    cutoff_time = now - timedelta(days=max_unused_days)
    
    eligible = []
    for record in records:
        if getattr(record, "memory_mode", "enabled") != "enabled":
            continue
            
        raw = getattr(record, "raw_memory", "") or ""
        summary = getattr(record, "rollout_summary", "") or ""
        if not raw.strip() and not summary.strip():
            continue
            
        last_use = getattr(record, "last_usage", None)
        updated = getattr(record, "updated_at")
        
        if last_use is not None and last_use.tzinfo is None:
            last_use = last_use.replace(tzinfo=timezone.utc)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
            
        if last_use is not None:
            if last_use < cutoff_time:
                continue
        else:
            if updated < cutoff_time:
                continue
                
        eligible.append(record)
        
    def sort_key(r):
        u_cnt = getattr(r, "usage_count", 0) or 0
        last_use = getattr(r, "last_usage", None)
        updated = getattr(r, "updated_at")
        
        coalesce = last_use if last_use is not None else updated
        coalesce_ts = coalesce.timestamp()
        updated_ts = updated.timestamp()
        return (-u_cnt, -coalesce_ts, -updated_ts, r.thread_id)
        
    eligible.sort(key=sort_key)
    top_n = eligible[:n]
    top_n.sort(key=lambda r: r.thread_id)
    return top_n

# --- Storage Rebuild/Sync -----------------------------------------------------
def rebuild_raw_memories_file_from_memories(
    memory_root: Path | str,
    memories: list[MemoryStageOneRecord],
    max_raw_memories: int = 100,
    *,
    max_unused_days: int = 30,
    now: datetime | None = None,
) -> None:
    memory_root = Path(memory_root).absolute()
    raw_path = raw_memories_file(memory_root)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    retained = select_phase2_memory_inputs(memories, max_raw_memories, max_unused_days=max_unused_days, now=now)
    
    body = ["# Raw Memories\n"]
    if not retained:
        body.append("No raw memories yet.")
    else:
        body.append("Merged stage-1 raw memories (stable ascending thread-id order):\n")
        for mem in retained:
            body.append(f"## Thread `{mem.thread_id}`")
            body.append(f"updated_at: {mem.updated_at.isoformat()}")
            body.append(f"cwd: {mem.cwd}")
            body.append(f"rollout_path: {mem.rollout_path}")
            
            stem = rollout_summary_file_stem(mem)
            body.append(f"rollout_summary_file: {stem}.md\n")
            body.append(mem.raw_memory.strip() + "\n")
            
    raw_path.write_text("\n".join(body) + "\n", encoding="utf-8")

def sync_rollout_summaries_from_memories(
    memory_root: Path | str,
    memories: list[MemoryStageOneRecord],
    max_raw_memories: int = 100,
    *,
    max_unused_days: int = 30,
    now: datetime | None = None,
) -> None:
    memory_root = Path(memory_root).absolute()
    summaries_dir = rollout_summaries_dir(memory_root)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    retained = select_phase2_memory_inputs(memories, max_raw_memories, max_unused_days=max_unused_days, now=now)
    keep_stems = {rollout_summary_file_stem(mem) for mem in retained}
    
    # Prune
    if summaries_dir.exists():
        for f in summaries_dir.glob("*.md"):
            if f.stem not in keep_stems:
                try:
                    f.unlink()
                except Exception:
                    pass
                    
    # Write summaries
    for mem in retained:
        stem = rollout_summary_file_stem(mem)
        f_path = summaries_dir / f"{stem}.md"
        body = []
        body.append(f"thread_id: {mem.thread_id}")
        body.append(f"updated_at: {mem.updated_at.isoformat()}")
        body.append(f"rollout_path: {mem.rollout_path}")
        body.append(f"cwd: {mem.cwd}")
        if mem.git_branch:
            body.append(f"git_branch: {mem.git_branch}")
        body.append("")
        body.append(mem.rollout_summary)
        
        f_path.write_text("\n".join(body) + "\n", encoding="utf-8")

# --- Prune Extensions --------------------------------------------------------
def prune_old_extension_resources(memory_root: Path | str) -> None:
    memory_root = Path(memory_root).absolute()
    extensions_dir = memory_extensions_root(memory_root)
    if not extensions_dir.exists():
        return
        
    for res_dir in extensions_dir.glob("*/resources"):
        if res_dir.is_dir():
            for f in res_dir.glob("*.md"):
                # Prune if older than 7 days
                mtime = f.stat().st_mtime
                age_days = (time.time() - mtime) / 86400.0
                if age_days > 7.0:
                    try:
                        f.unlink()
                    except Exception:
                        pass

# --- Workspace diff / git operations ------------------------------------------
def run_git_cmd(cwd: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )

def prepare_memory_workspace(memory_root: Path | str) -> Path:
    memory_root = Path(memory_root).absolute()
    memory_root.mkdir(parents=True, exist_ok=True)
    
    git_dir = memory_root / ".git"
    if not git_dir.exists():
        run_git_cmd(memory_root, ["init"])
        run_git_cmd(memory_root, ["config", "user.name", "Codex"])
        run_git_cmd(memory_root, ["config", "user.email", "codex@openai.com"])
        
    # Baseline setup: add and commit existing
    run_git_cmd(memory_root, ["add", "-A"])
    run_git_cmd(memory_root, ["commit", "-m", "baseline init"])
    return memory_root

def memory_workspace_diff(memory_root: Path | str) -> tuple[list[MemoryWorkspaceChange], str]:
    memory_root = Path(memory_root).absolute()
    
    # Track untracked as intent-to-add
    run_git_cmd(memory_root, ["add", "-N", "."])
    
    res = run_git_cmd(memory_root, ["status", "--porcelain"])
    changes = []
    
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        status_part = line[:2].strip()
        path_part = line[3:].strip()
        
        if path_part == "phase2_workspace_diff.md":
            continue
            
        status = "M"
        if "A" in status_part or "?" in status_part:
            status = "A"
        elif "D" in status_part:
            status = "D"
            
        changes.append(MemoryWorkspaceChange(status=status, path=path_part))
        
    diff_res = run_git_cmd(memory_root, ["diff"])
    diff = diff_res.stdout
    return changes, diff

def write_memory_workspace_diff(
    memory_root: Path | str,
    changes: list[MemoryWorkspaceChange],
    diff: str,
) -> Path:
    memory_root = Path(memory_root).absolute()
    diff_path = memory_root / "phase2_workspace_diff.md"
    
    lines = ["# Changes\n"]
    for change in changes:
        lines.append(f"- {change.status} {change.path}")
    lines.append("\n# Diff\n")
    lines.append(diff)
    
    content = "\n".join(lines)
    diff_path.write_text(content, encoding="utf-8")
    return diff_path

def write_current_memory_workspace_diff(memory_root: Path | str) -> Path:
    changes, diff = memory_workspace_diff(memory_root)
    return write_memory_workspace_diff(memory_root, changes, diff)

def reset_memory_workspace_baseline(memory_root: Path | str) -> None:
    memory_root = Path(memory_root).absolute()
    diff_path = memory_root / "phase2_workspace_diff.md"
    if diff_path.exists():
        try:
            diff_path.unlink()
        except Exception:
            pass
            
    run_git_cmd(memory_root, ["add", "-A"])
    run_git_cmd(memory_root, ["commit", "-m", "baseline reset"])

def sync_phase2_workspace_inputs(memory_root: Path | str) -> None:
    pass

# --- Stage 1 Output Schema ---------------------------------------------------
def memory_stage_one_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "raw_memory": {"type": "string", "description": "Raw extracted memory block."},
            "rollout_slug": {"type": ["string", "null"], "description": "Short mnemonic name."},
            "rollout_summary": {"type": "string", "description": "One-line description."}
        },
        "required": ["raw_memory", "rollout_slug", "rollout_summary"],
        "additionalProperties": False
    }

def parse_memory_stage_one_output(output: str) -> ParsedStageOneOutput:
    try:
        data = json.loads(output)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}")
        
    allowed = {"raw_memory", "rollout_summary", "rollout_slug"}
    for k in data:
        if k not in allowed:
            raise ValueError(f"Unknown field in stage one output: {k}")
            
    for req in ["raw_memory", "rollout_summary", "rollout_slug"]:
        if req not in data:
            raise ValueError(f"Missing required field: {req}")
            
    raw_mem = data["raw_memory"]
    roll_sum = data["rollout_summary"]
    
    raw_mem = redact_secrets_in_text(raw_mem)
    roll_sum = redact_secrets_in_text(roll_sum)
    
    return ParsedStageOneOutput(
        raw_memory=raw_mem,
        rollout_summary=roll_sum,
        rollout_slug=data["rollout_slug"],
    )

# --- Memory Stage One Executor ------------------------------------------------
def extract_memory_stage_one(
    model_client: ModelClient,
    rollout_path: Path | str,
    rollout_cwd: Path | str,
    rollout_contents: str,
    prompt_cache_key: str | None = None,
) -> ParsedStageOneOutput:
    # Build stage 1 PromptRequest
    schema = memory_stage_one_output_schema()
    sys_prompt = memory_stage_one_system_prompt()
    
    # 70% budget is handled by build_memory_stage_one_input_message!
    input_message = build_memory_stage_one_input_message(
        rollout_path=rollout_path,
        rollout_cwd=rollout_cwd,
        rollout_contents=rollout_contents,
        model_context_window=150_000,  # default
        effective_context_window_percent=95,
    )
    
    # Stage 1 request setup: Model: "gpt-5.4-mini" (under Rust!), reasoning: {"effort": "low"}
    request = PromptRequest(
        model="gpt-5.4-mini",
        instructions=sys_prompt,
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": input_message}]}],
        tools=[],
        parallel_tool_calls=True,
        prompt_cache_key=prompt_cache_key,
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        output_schema=schema,
        output_schema_strict=True,
    )
    
    response = model_client.create(request)
    # The output is standard assistant JSON message
    out_text = ""
    for item in response.output:
        if item.get("type") == "message" and item.get("role") == "assistant":
            for part in item.get("content", []):
                if isinstance(part, dict) and "text" in part:
                    out_text += part["text"]
                    
    return parse_memory_stage_one_output(out_text)

# --- Startup stage 1 runner --------------------------------------------------
def run_memory_stage_one_for_rollout(
    config: CodexConfig | None = None,
    rollout_path: Path | str = "",
    *,
    model_client: ModelClient | None = None,
) -> MemoryStageOneRecord | None:
    rollout_path = Path(rollout_path).absolute()
    if not rollout_path.exists():
        return None
        
    rollout = load_memory_rollout(rollout_path)
    
    if model_client is None:
        model_client = default_model_client()
        
    try:
        parsed = extract_memory_stage_one(
            model_client=model_client,
            rollout_path=rollout.path,
            rollout_cwd=rollout.cwd,
            rollout_contents=rollout.serialized_contents,
            prompt_cache_key=rollout.thread_id,
        )
        return MemoryStageOneRecord(
            thread_id=rollout.thread_id,
            rollout_slug=parsed.rollout_slug,
            rollout_summary=parsed.rollout_summary,
            raw_memory=parsed.raw_memory,
            updated_at=rollout.updated_at,
            source=rollout.source,
            cwd=rollout.cwd,
            git_branch=rollout.git_branch,
            rollout_path=rollout_path,
        )
    except Exception as e:
        return None

# --- Seed Extensions ---------------------------------------------------------
def seed_extension_instructions(memory_root: Path | str) -> None:
    memory_root = Path(memory_root).absolute()
    ad_hoc_dir = memory_extensions_root(memory_root) / "ad_hoc"
    ad_hoc_dir.mkdir(parents=True, exist_ok=True)
    
    inst_file = ad_hoc_dir / "instructions.md"
    if not inst_file.exists():
        inst_file.write_text("# Ad-hoc notes\nDefault ad-hoc notes extension configuration.\n", encoding="utf-8")

# --- Memory Startup pipeline once --------------------------------------------
def run_memory_startup_once(
    config: CodexConfig | None = None,
    *,
    codex_home: Path | str | None = None,
    model_client: ModelClient | None = None,
    state_store: MemoryStateStore | None = None,
    max_rollouts: int = 1,
    max_unused_days: float = 10,
    max_rollout_age_days: float = 10,
    min_rollout_idle_hours: float = 6,
    current_thread_id: str | None = None,
) -> MemoryStartupResult:
    # Resolve parameters
    if config is not None:
        if codex_home is None:
            codex_home = config.resolved_codex_home()
        if min_rollout_idle_hours == 6:  # default argument in Python
            min_rollout_idle_hours = config.memory_min_rollout_idle_hours
        if max_rollout_age_days == 10:  # default argument in Python
            max_rollout_age_days = config.memory_max_rollout_age_days
            
    if codex_home is None:
        codex_home = Path.home() / ".codex-python"
    else:
        codex_home = Path(codex_home)
        
    memory_root = codex_home / "memories"
    prepare_memory_workspace(memory_root)
    seed_extension_instructions(memory_root)
    
    if state_store is None:
        state_store = MemoryStateStore(codex_home / "memory-state.sqlite3")
        
    candidates = memory_rollout_candidates(codex_home)
    
    records = []
    claimed_count = 0
    
    for c_path in candidates:
        if claimed_count >= max_rollouts:
            break
            
        rollout = load_memory_rollout(c_path)
        
        # Upsert thread so FK references are valid!
        record = MemoryThreadRecord(
            thread_id=rollout.thread_id,
            rollout_path=rollout.path,
            cwd=rollout.cwd,
            updated_at=rollout.updated_at,
            git_branch=rollout.git_branch,
            source=rollout.source,
            memory_mode="enabled",
        )
        state_store.upsert_thread(record)
        
        eligible = memory_rollout_is_stage1_startup_eligible(
            rollout,
            current_thread_id=current_thread_id,
            allowed_sources=frozenset({"cli"}),  # standard is cli
            min_rollout_idle_hours=min_rollout_idle_hours,
            max_rollout_age_days=max_rollout_age_days,
        )
        if not eligible:
            continue
            
        # Claim job
        claim = state_store.try_claim_stage1_job(
            thread_id=rollout.thread_id,
            worker_id="startup",
            source_updated_at=rollout.updated_at,
            lease_seconds=300,
            max_running_jobs=4,
        )
        if claim.outcome == "claimed":
            # Run stage 1 extraction!
            res = run_memory_stage_one_for_rollout(
                rollout_path=c_path,
                model_client=model_client,
            )
            if res is not None:
                # Succeeded!
                state_store.mark_stage1_job_succeeded(
                    thread_id=rollout.thread_id,
                    ownership_token=claim.ownership_token,
                    source_updated_at=rollout.updated_at,
                    raw_memory=res.raw_memory,
                    rollout_summary=res.rollout_summary,
                    rollout_slug=res.rollout_slug,
                )
                
                # Rebuild raw memories file
                all_memories = state_store.get_phase2_input_selection(n=100, max_unused_days=36500)
                rebuild_raw_memories_file_from_memories(memory_root, all_memories, max_raw_memories=100)
                sync_rollout_summaries_from_memories(memory_root, all_memories, max_raw_memories=100)
                
                records.append(res)
                claimed_count += 1
            else:
                # Failed!
                state_store.mark_stage1_job_failed(
                    thread_id=rollout.thread_id,
                    ownership_token=claim.ownership_token,
                    failure_reason="Failed extraction during startup.",
                    retry_delay_seconds=600,
                )
                
    return MemoryStartupResult(records=records, memory_root=memory_root)

def build_memory_consolidation_config(
    memory_root: Path | str,
    base_config: CodexConfig,
) -> CodexConfig:
    memory_root = Path(memory_root).absolute()
    return CodexConfig(
        model="gpt-5.4",
        cwd=memory_root,
        sandbox="workspace-write",
        approval_policy="never",
        writable_roots=(memory_root,),
        codex_home=base_config.resolved_codex_home(),
        skip_git_repo_check=True,
        ephemeral=True,
        max_iterations=base_config.max_iterations,
        
        session_source="internal_memory_consolidation",
        network_access="restricted",
        use_memories=False,
        memory_tool_enabled=False,
        memory_generate_memories=False,
        include_multi_agent_tools=False,
        include_web_search_tool=False,
        include_request_user_input_tool=False,
        
        model_reasoning_effort="medium",
        model_reasoning_summary="none",
    )

# --- Phase 2 Consolidation execution ------------------------------------------
def run_memory_consolidation_session(
    memory_root: Path | str,
    base_config: CodexConfig,
    model_client: ModelClient,
) -> CodexResult:
    memory_root = Path(memory_root).absolute()
    
    # 1. Build locked-down consolidation config
    config = build_memory_consolidation_config(memory_root, base_config)
    
    # 2. Write current workspace diff
    write_current_memory_workspace_diff(memory_root)
    
    # 3. Build consolidation prompt
    from codex.prompts import build_memory_consolidation_prompt
    prompt = build_memory_consolidation_prompt(memory_root)
    
    # 4. Run consolidation session
    from codex.core import CodexSession
    session = CodexSession(config, model_client=model_client)
    
    # Inject database store back so they share state if needed
    if getattr(base_config, "memory_state_store", None) is not None:
        session.config.memory_state_store = base_config.memory_state_store
        
    try:
        res = session.run(prompt)
        # Succeeded! Reset/commit baseline
        reset_memory_workspace_baseline(memory_root)
        return res
    except Exception as e:
        # Failed!
        raise e

def run_memory_phase2_once(
    config: CodexConfig | None = None,
    *,
    codex_home: Path | str | None = None,
    state_store: MemoryStateStore | None = None,
    model_client: ModelClient | None = None,
    max_raw_memories: int = 100,
    max_unused_days: float = 30,
) -> MemoryPhase2Result:
    # Resolve parameters
    if config is not None:
        if codex_home is None:
            codex_home = config.resolved_codex_home()
        if state_store is None:
            state_store = getattr(config, "memory_state_store", None)
            
    if codex_home is None:
        codex_home = Path.home() / ".codex-python"
    else:
        codex_home = Path(codex_home)
        
    memory_root = codex_home / "memories"
    prepare_memory_workspace(memory_root)
    
    if state_store is None:
        state_store = MemoryStateStore(codex_home / "memory-state.sqlite3")
        
    if model_client is None:
        model_client = default_model_client()
        
    # Claim global phase 2 job
    claim = state_store.try_claim_global_phase2_job(
        worker_id="phase2",
        lease_seconds=300,
    )
    
    if claim.outcome == "skipped_cooldown":
        return MemoryPhase2Result(status="skipped_cooldown", final_message="", memory_root=memory_root)
    if claim.outcome == "skipped_empty":
        return MemoryPhase2Result(status="skipped_empty", final_message="", memory_root=memory_root)
        
    if claim.outcome == "claimed":
        # Select target memories
        selected = state_store.get_phase2_input_selection(
            n=max_raw_memories,
            max_unused_days=int(max_unused_days),
        )
        if not selected:
            # Succeeded with empty selection
            # We mark finished at now with watermark 0
            state_store.conn.execute(
                """
                UPDATE jobs SET
                    status = 'done',
                    finished_at = ?,
                    last_success_watermark = 0
                WHERE kind = 'memory_consolidate_global' AND job_key = 'global' AND ownership_token = ?
                """,
                (int(datetime.now(timezone.utc).timestamp()), claim.ownership_token)
            )
            state_store.conn.commit()
            return MemoryPhase2Result(status="skipped_empty", final_message="", memory_root=memory_root)
            
        # Rebuild input files
        rebuild_raw_memories_file_from_memories(memory_root, selected, max_raw_memories, max_unused_days=int(max_unused_days))
        sync_rollout_summaries_from_memories(memory_root, selected, max_raw_memories, max_unused_days=int(max_unused_days))
        
        # Run consolidation session
        base_config = config if config is not None else CodexConfig(codex_home=codex_home)
        try:
            res = run_memory_consolidation_session(
                memory_root=memory_root,
                base_config=base_config,
                model_client=model_client,
            )
            
            # Succeeded!
            completed_watermark = max(getattr(record, "updated_at") for record in selected)
            state_store.mark_global_phase2_job_succeeded(
                ownership_token=claim.ownership_token,
                completed_watermark=completed_watermark,
                selected_outputs=selected,
            )
            
            # Prune extension resources
            prune_old_extension_resources(memory_root)
            
            return MemoryPhase2Result(
                status="succeeded",
                final_message=res.final_message,
                memory_root=memory_root,
            )
            
        except Exception as e:
            # Failed!
            state_store.conn.execute(
                """
                UPDATE jobs SET
                    status = 'failed',
                    finished_at = ?,
                    last_error = ?
                WHERE kind = 'memory_consolidate_global' AND job_key = 'global' AND ownership_token = ?
                """,
                (int(datetime.now(timezone.utc).timestamp()), str(e), claim.ownership_token)
            )
            state_store.conn.commit()
            return MemoryPhase2Result(
                status="failed",
                final_message=str(e),
                memory_root=memory_root,
            )
            
    return MemoryPhase2Result(status="failed", final_message="Claim was denied.", memory_root=memory_root)

# --- Memory startup pipeline once --------------------------------------------
def run_memory_startup_pipeline_once(
    config: CodexConfig | None = None,
    *,
    codex_home: Path | str | None = None,
    model_client: ModelClient | None = None,
    state_store: MemoryStateStore | None = None,
    base_config: CodexConfig | None = None,
    max_rollouts: int = 1,
    max_unused_days: float = 10,
    max_rollout_age_days: float = 10,
    min_rollout_idle_hours: float = 6,
    current_thread_id: str | None = None,
    run_phase2: bool = True,
) -> MemoryPipelineResult:
    # Resolve parameters
    if config is not None:
        if base_config is None:
            base_config = config
        if codex_home is None:
            codex_home = config.resolved_codex_home()
        if min_rollout_idle_hours == 6:
            min_rollout_idle_hours = config.memory_min_rollout_idle_hours
        if max_rollout_age_days == 10:
            max_rollout_age_days = config.memory_max_rollout_age_days
            
    if base_config is None:
        base_config = CodexConfig(codex_home=codex_home) if codex_home else CodexConfig()
        
    if codex_home is None:
        codex_home = base_config.resolved_codex_home()
        
    if state_store is None:
        state_store = getattr(base_config, "memory_state_store", None)
    if state_store is None:
        state_store = MemoryStateStore(Path(codex_home) / "memory-state.sqlite3")
        
    # Save open store on base_config for reference during runs
    base_config.memory_state_store = state_store
    
    # 1. Startup Stage 1
    startup = run_memory_startup_once(
        codex_home=codex_home,
        model_client=model_client,
        state_store=state_store,
        max_rollouts=max_rollouts,
        max_unused_days=max_unused_days,
        max_rollout_age_days=max_rollout_age_days,
        min_rollout_idle_hours=min_rollout_idle_hours,
        current_thread_id=current_thread_id,
    )
    
    # 2. Global phase 2
    phase2_res = None
    if run_phase2:
        phase2_res = run_memory_phase2_once(
            codex_home=codex_home,
            state_store=state_store,
            model_client=model_client,
            max_unused_days=30,
        )
        
    return MemoryPipelineResult(
        status="completed",
        records=startup.records,
        phase2_result=phase2_res,
    )

# --- Background thread startup task ------------------------------------------
class CancellationToken:
    def __init__(self):
        self.cancelled = False
        
    def cancel(self):
        self.cancelled = True
        
    def is_cancelled(self) -> bool:
        return self.cancelled

def start_memory_startup_task(config: CodexConfig) -> tuple[threading.Thread | None, CancellationToken]:
    cancel_token = CancellationToken()
    if not config.use_memories:
        return None, cancel_token
        
    def startup_worker():
        if cancel_token.is_cancelled():
            return
        # Open sqlite store
        store = MemoryStateStore(config.resolved_codex_home() / "memory-state.sqlite3")
        try:
            # run pipeline once
            run_memory_startup_pipeline_once(
                config,
                state_store=store,
                run_phase2=config.memory_run_phase2_on_startup,
            )
        except Exception:
            pass
        finally:
            store.close()
            
    t = threading.Thread(target=startup_worker, daemon=True)
    t.start()
    return t, cancel_token
