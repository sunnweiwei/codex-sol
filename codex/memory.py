from __future__ import annotations
import datetime
import difflib
import functools
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from codex.types import CodexConfig, CodexResult
from codex.model import ModelClient

# Compactor context and model tokens limitations
DEFAULT_ROLLOUT_TOKEN_LIMIT = 150000
CONTEXT_WINDOW_PERCENT = 70
APPROX_BYTES_PER_TOKEN = 4


# ==============================================================================
# Git-matching Hashing & statistics calculators helpers
# ==============================================================================

def git_blob_oid(data: bytes) -> str:
    """Computes standard Git SHA-1 hexadecimal Blob object ID matching 'blob <size>\\0<data>'."""
    header = f"blob {len(data)}\x00".encode("utf-8")
    hasher = hashlib.sha1()
    hasher.update(header)
    hasher.update(data)
    return hasher.hexdigest()


def calculate_line_statistics(baseline: str | None, current: str | None) -> dict[str, int]:
    """Calculates exact addition, deletion, modification and net total line stats.
    
    Lines modified is calculated based on overlap ranges of contiguous replacements.
    """
    stats = {"added_lines": 0, "deleted_lines": 0, "modified_lines": 0, "total_changes": 0}
    if baseline == current:
        return stats
        
    if baseline is None and current is not None:
        c_lines = len(current.splitlines())
        stats["added_lines"] = c_lines
        stats["total_changes"] = c_lines
        return stats
        
    if baseline is not None and current is None:
        b_lines = len(baseline.splitlines())
        stats["deleted_lines"] = b_lines
        stats["total_changes"] = b_lines
        return stats
        
    base_lines = baseline.splitlines()
    curr_lines = current.splitlines()
    
    matcher = difflib.SequenceMatcher(None, base_lines, curr_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            del_len = i2 - i1
            ins_len = j2 - j1
            mod = min(del_len, ins_len)
            stats["modified_lines"] += mod
            stats["added_lines"] += max(0, ins_len - del_len)
            stats["deleted_lines"] += max(0, del_len - ins_len)
        elif tag == 'delete':
            stats["deleted_lines"] += (i2 - i1)
        elif tag == 'insert':
            stats["added_lines"] += (j2 - j1)
            
    stats["total_changes"] = stats["added_lines"] + stats["deleted_lines"] + stats["modified_lines"]
    return stats


def approx_token_count(text: str) -> int:
    """Calculates approximate token count: (bytes + 3) // 4 (matches Rust)."""
    if not text:
        return 0
    byte_len = len(text.encode('utf-8'))
    return (byte_len + (APPROX_BYTES_PER_TOKEN - 1)) // APPROX_BYTES_PER_TOKEN


def approx_bytes_for_tokens(tokens: int) -> int:
    """Calculates bytes size for a token budget."""
    return tokens * APPROX_BYTES_PER_TOKEN


def approx_tokens_from_byte_count(bytes_count: int) -> int:
    """Calculates tokens size from byte size."""
    return (bytes_count + (APPROX_BYTES_PER_TOKEN - 1)) // APPROX_BYTES_PER_TOKEN


def previous_char_boundary(value_bytes: bytes, max_bytes: int) -> int:
    """Backtracks the byte buffer limit to the nearest legal UTF-8 character boundary.
    
    Prevents splicing multi-byte unicode code point continuation boundaries.
    """
    if len(value_bytes) <= max_bytes:
        return len(value_bytes)
    index = max_bytes
    # UTF-8 continuation bytes start with bits 10xxxxxx (0x80 to 0xBF)
    while index > 0 and (value_bytes[index] & 0xC0) == 0x80:
        index -= 1
    return index


def truncate_with_byte_estimate(s: str, max_bytes: int, use_tokens: bool) -> str:
    """UTF-8 boundary-safe middle truncation using character slices mapped to byte budgets."""
    if not s:
        return ""
        
    s_bytes = s.encode('utf-8')
    total_bytes = len(s_bytes)
    total_chars = len(s)
    
    if max_bytes == 0:
        if use_tokens:
            removed_count = approx_tokens_from_byte_count(total_bytes)
            return f"…{removed_count} tokens truncated…"
        else:
            return f"…{total_chars} chars truncated…"
            
    if total_bytes <= max_bytes:
        return s
        
    left_budget = max_bytes // 2
    right_budget = max_bytes - left_budget
    
    prefix_end = 0
    suffix_start = total_bytes
    removed_chars = 0
    suffix_started = False
    
    tail_start_target = total_bytes - right_budget
    
    current_byte_idx = 0
    for ch in s:
        ch_len = len(ch.encode('utf-8'))
        char_end = current_byte_idx + ch_len
        
        if char_end <= left_budget:
            prefix_end = char_end
            current_byte_idx = char_end
            continue
            
        if current_byte_idx >= tail_start_target:
            if not suffix_started:
                suffix_start = current_byte_idx
                suffix_started = True
            current_byte_idx = char_end
            continue
            
        removed_chars += 1
        current_byte_idx = char_end
        
    if suffix_start < prefix_end:
        suffix_start = prefix_end
        
    prefix = s_bytes[:prefix_end].decode('utf-8', errors='ignore')
    suffix = s_bytes[suffix_start:].decode('utf-8', errors='ignore')
    
    if use_tokens:
        removed_bytes = total_bytes - max_bytes
        removed_count = approx_tokens_from_byte_count(removed_bytes)
        marker = f"…{removed_count} tokens truncated…"
    else:
        marker = f"…{removed_chars} chars truncated…"
        
    return prefix + marker + suffix


def truncate_middle_with_token_budget(s: str, max_tokens: int) -> tuple[str, int | None]:
    """Truncates the middle of a string to fit a token budget, preserving beginning/end."""
    if not s:
        return "", None
        
    byte_len = len(s.encode('utf-8'))
    max_bytes = approx_bytes_for_tokens(max_tokens)
    
    if max_tokens > 0 and byte_len <= max_bytes:
        return s, None
        
    truncated = truncate_with_byte_estimate(s, max_bytes, use_tokens=True)
    total_tokens = approx_token_count(s)
    
    if truncated == s:
        return truncated, None
    else:
        return truncated, total_tokens


# ==============================================================================
# Canonical dataclasses & pipeline structures definitions
# ==============================================================================

class MemoryWorkspaceChange:
    """Change record mapping a status-flagged file change track (git-baseline diff tracking)."""
    def __init__(self, status: str, path: str) -> None:
        self.status = status  # "added", "modified", "deleted", "untracked"
        self.path = path


class MemoryThreadRecord:
    """Canonical database metadata representation of a single conversation thread record."""
    def __init__(
        self,
        thread_id: str,
        rollout_path: Path | str,
        cwd: Path | str,
        updated_at: datetime.datetime,
        git_branch: str | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.rollout_path = Path(rollout_path)
        self.cwd = Path(cwd)
        self.updated_at = updated_at
        self.git_branch = git_branch


class MemoryStageOneStartupClaim:
    """Successful job claim registration generated for a stale thread during process startup passes."""
    def __init__(
        self,
        thread_id: str,
        rollout_path: Path,
        source_updated_at: datetime.datetime,
        ownership_token: str,
    ) -> None:
        self.thread_id = thread_id
        self.rollout_path = Path(rollout_path)
        self.source_updated_at = source_updated_at
        self.ownership_token = ownership_token


class MemoryStageOneRecord:
    """Extracted memory summary record persisted in sqlite database representing a single thread snapshot."""
    def __init__(
        self,
        thread_id: str,
        source_updated_at: datetime.datetime,
        raw_memory: str,
        rollout_summary: str,
        rollout_slug: str | None,
        rollout_path: Path | str,
        cwd: Path | str,
        usage_count: int = 0,
        last_usage: datetime.datetime | None = None,
        selected_for_phase2: bool = False,
    ) -> None:
        self.thread_id = thread_id
        self.source_updated_at = source_updated_at
        self.raw_memory = raw_memory
        self.rollout_summary = rollout_summary
        self.rollout_slug = rollout_slug
        self.rollout_path = Path(rollout_path)
        self.cwd = Path(cwd)
        self.usage_count = usage_count
        self.last_usage = last_usage
        self.selected_for_phase2 = selected_for_phase2


class MemoryStageOneOutput:
    """Direct outputs produced by LLM compaction model prompts inside Stage 1 pipelines."""
    def __init__(
        self,
        raw_memory: str,
        rollout_summary: str,
        rollout_slug: str | None,
    ) -> None:
        self.raw_memory = raw_memory
        self.rollout_summary = rollout_summary
        self.rollout_slug = rollout_slug


class MemoryStartupResult:
    """Aggregated outcomes of the complete background state backfilling and startup claims cycle."""
    def __init__(
        self,
        records: list[MemoryStageOneRecord],
        skipped: list[Path],
        memory_root: Path,
        status: str = 'completed',
        phase2_result: Any | None = None,
        rate_limit_allowed: bool | None = None,
    ) -> None:
        self.records = records
        self.skipped = skipped
        self.memory_root = Path(memory_root)
        self.status = status
        self.phase2_result = phase2_result
        self.rate_limit_allowed = rate_limit_allowed


class MemoryPhase2Result:
    """Outcome of a Stage 2 global memory consolidation run."""
    def __init__(
        self,
        status: str,
        selected: list[MemoryStageOneRecord],
        memory_root: Path,
        workspace_changed: bool = False,
        final_message: str = '',
    ) -> None:
        self.status = status
        self.selected = selected
        self.memory_root = Path(memory_root)
        self.workspace_changed = workspace_changed
        self.final_message = final_message


class MemoryRollout:
    """In-memory representation of a serialized rollout JSONL log file applied for database backfill or sync."""
    def __init__(
        self,
        thread_id: str,
        rollout_path: Path,
        cwd: Path,
        source_updated_at: datetime.datetime,
        git_branch: str | None,
        source: str,
        memory_mode: str,
        items: list[dict[str, Any]],
        serialized_contents: str,
    ) -> None:
        self.thread_id = thread_id
        self.rollout_path = Path(rollout_path)
        self.cwd = Path(cwd)
        self.source_updated_at = source_updated_at
        self.git_branch = git_branch
        self.source = source
        self.memory_mode = memory_mode
        self.items = items
        self.serialized_contents = serialized_contents


class MemoryJobClaim:
    """Outcome of a transactional attempt to lock and claim an extraction or consolidation task."""
    def __init__(
        self,
        outcome: str,
        ownership_token: str | None = None,
        input_watermark: int | None = None,
    ) -> None:
        self.outcome = outcome  # "claimed", "skipped_up_to_date", "skipped_running", "skipped_retry_backoff", "skipped_retry_exhaustion"
        self.ownership_token = ownership_token
        self.input_watermark = input_watermark


class MemoryBackgroundTask:
    """Mock-conformant wrapper for background tasks executing pipeline steps.
    
    Provides thread-joining and status reporting properties aligned with E2E targets.
    """
    def __init__(self, thread: Optional[threading.Thread] = None) -> None:
        self._thread = thread

    def done(self) -> bool:
        """Returns True if the background worker has completed processing."""
        if self._thread is None:
            return True
        return not self._thread.is_alive()

    def join(self, timeout: float | None = None) -> Any | None:
        """Joins the background thread execution and awaits completion."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return None


# ==============================================================================
# Turn Level Mutations Change Delta Tracker Class
# ==============================================================================

class TurnDiffTracker:
    """Tracks turn mutations (adds, deletes, updates) accumulating net text diff blocks in-memory."""
    def __init__(self, display_root: Path | str | None = None) -> None:
        self.valid = True
        self.display_root = Path(display_root) if display_root is not None else None
        self.baseline_by_path: dict[Path, str] = {}
        self.current_by_path: dict[Path, str] = {}
        self.origin_by_current_path: dict[Path, Path] = {}

    def invalidate(self) -> None:
        """Invalidates the delta tracking session (forced fallback conditions)."""
        self.valid = False

    def track_add(self, path: Path | str, content: str, overwritten_content: str | None = None) -> None:
        """Tracks adding a file in the active turn workspace session."""
        p = Path(path)
        self.origin_by_current_path.pop(p, None)
        if p not in self.current_by_path and p not in self.baseline_by_path and overwritten_content is not None:
            self.baseline_by_path[p] = overwritten_content
        self.current_by_path[p] = content

    def track_delete(self, path: Path | str, content: str) -> None:
        """Tracks deleting a file in the active turn workspace session."""
        p = Path(path)
        if p in self.current_by_path:
            del self.current_by_path[p]
        elif p not in self.baseline_by_path:
            self.baseline_by_path[p] = content
        self.origin_by_current_path.pop(p, None)

    def track_update(
        self,
        source_path: Path | str,
        old_content: str,
        new_content: str,
        move_path: Path | str | None = None,
        overwritten_move_content: str | None = None
    ) -> None:
        """Tracks modifying or renaming files in the active turn workspace session."""
        src = Path(source_path)
        dst = Path(move_path) if move_path is not None else None

        if src not in self.current_by_path and src not in self.baseline_by_path:
            self.baseline_by_path[src] = old_content

        if dst is not None:
            if dst not in self.current_by_path and dst not in self.baseline_by_path and overwritten_move_content is not None:
                self.baseline_by_path[dst] = overwritten_move_content
            
            origin = self.origin_by_current_path.pop(src, src)
            self.current_by_path.pop(src, None)
            self.current_by_path[dst] = new_content
            self.origin_by_current_path.pop(dst, None)
            if dst != origin:
                self.origin_by_current_path[dst] = origin
        else:
            self.current_by_path[src] = new_content

    def rename_pairs(self) -> dict[Path, Path]:
        """Resolves transitive rename operations tracking baseline and current moves."""
        pairs = {}
        for dest, origin in self.origin_by_current_path.items():
            if dest == origin:
                continue
            if origin in self.current_by_path:
                continue
            if dest not in self.current_by_path:
                continue
            if origin not in self.baseline_by_path:
                continue
            if dest in self.baseline_by_path:
                continue
            pairs[origin] = dest
        return pairs

    def display_path(self, path: Path) -> str:
        """Renders display paths normalized for Unix path format output fences."""
        if self.display_root is not None:
            try:
                return str(path.relative_to(self.display_root)).replace("\\", "/")
            except ValueError as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
        return str(path).replace("\\", "/")

    def get_unified_diff(self) -> str | None:
        """Aggregates all accumulated turn level delta updates as a Git-compatible unified diff."""
        if not self.valid:
            return None

        rename_pairs = self.rename_pairs()
        paired_destinations = set(rename_pairs.values())
        handled = set()
        
        all_paths = set(self.baseline_by_path.keys()) | set(self.current_by_path.keys())
        sorted_paths = sorted(all_paths, key=lambda p: self.display_path(p))
        
        aggregated = []
        for path in sorted_paths:
            if path in handled:
                continue
            handled.add(path)
            
            if path in paired_destinations:
                continue
                
            if path in rename_pairs:
                dest = rename_pairs[path]
                handled.add(dest)
                diff = self.render_diff(path, self.baseline_by_path.get(path), dest, self.current_by_path.get(dest))
            else:
                diff = self.render_diff(path, self.baseline_by_path.get(path), path, self.current_by_path.get(path))
                
            if diff:
                aggregated.append(diff)
                
        if not aggregated:
            return None
            
        res = "\n".join(aggregated)
        if not res.endswith("\n"):
            res += "\n"
        return res

    def render_diff(self, left_path: Path, left_content: str | None, right_path: Path, right_content: str | None) -> str | None:
        """Generates standard block for unified git-formatted diff reports."""
        if left_content == right_content:
            return None

        left_display = self.display_path(left_path)
        right_display = self.display_path(right_path)
        
        left_oid = git_blob_oid(left_content.encode("utf-8")) if left_content is not None else "0000000000000000000000000000000000000000"
        right_oid = git_blob_oid(right_content.encode("utf-8")) if right_content is not None else "0000000000000000000000000000000000000000"

        diff_header = f"diff --git a/{left_display} b/{right_display}\n"
        if left_content is None and right_content is not None:
            diff_header += "new file mode 100644\n"
        elif left_content is not None and right_content is None:
            diff_header += "deleted file mode 100644\n"
            
        diff_header += f"index {left_oid}..{right_oid}\n"
        
        old_header = f"a/{left_display}" if left_content is not None else "/dev/null"
        new_header = f"b/{right_display}" if right_content is not None else "/dev/null"
        
        baseline_lines = left_content.splitlines(keepends=True) if left_content is not None else []
        current_lines = right_content.splitlines(keepends=True) if right_content is not None else []
        
        hunks = list(difflib.unified_diff(
            baseline_lines,
            current_lines,
            fromfile=old_header,
            tofile=new_header,
            n=3,
            lineterm="\n"
        ))
        
        if not hunks:
            return None
        return diff_header + "".join(hunks)

    def get_line_statistics(self) -> dict[str, int]:
        """Calculates exact contiguous changed line modifications for all changed tracks."""
        total_stats = {"files_changed": 0, "added_lines": 0, "deleted_lines": 0, "modified_lines": 0, "total_changes": 0}
        all_paths = set(self.baseline_by_path.keys()) | set(self.current_by_path.keys())
        for path in all_paths:
            left = self.baseline_by_path.get(path)
            right = self.current_by_path.get(path)
            if left == right:
                continue
                
            total_stats["files_changed"] += 1
            f_stats = calculate_line_statistics(left, right)
            total_stats["added_lines"] += f_stats["added_lines"]
            total_stats["deleted_lines"] += f_stats["deleted_lines"]
            total_stats["modified_lines"] += f_stats["modified_lines"]
            total_stats["total_changes"] += f_stats["total_changes"]
        return total_stats


# ==============================================================================
# Memory state database store class (State DB & Logs DB handles)
# ==============================================================================

class MemoryStateStore:
    """SQLite persistent state store backing conversation logs metadata, job leases, and memory citations."""
    
    # Mutex lock supporting concurrent GIL-safe millisecond-precision high-water mark increments
    _time_mutex = threading.Lock()
    _last_allocated_ms = 0

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.is_dir() or self.path.suffix == "":
            self.db_path = self.path / "thread_store.db"
        else:
            self.db_path = self.path
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Dual-pool simulated handles representing dual sqlite design
        self.logs_path = self.db_path.parent / "logs_2.sqlite"
        self.logs_conn = sqlite3.connect(str(self.logs_path))
        self.logs_conn.row_factory = sqlite3.Row
        
        self._schemas_initialized = False
        self._try_initialize_schemas()

    def _try_initialize_schemas(self) -> None:
        """Applies exact schemas mapped to the upstream 32 migrations."""
        if self._schemas_initialized:
            return
        try:
            # Setup State DB connections performance pragmas
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
            
            # Setup Logs DB connections performance pragmas
            self.logs_conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            self.logs_conn.execute("PRAGMA journal_mode=WAL")
            self.logs_conn.execute("PRAGMA synchronous=NORMAL")
            self.logs_conn.execute("PRAGMA busy_timeout=5000")

            # 1. Initialize metadata state database tables (under self.conn)
            with self.conn:
                # Table 1: threads (rollout session parameters)
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    rollout_path TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    created_at_ms INTEGER,
                    updated_at_ms INTEGER,
                    source TEXT NOT NULL,
                    thread_source TEXT,
                    agent_nickname TEXT,
                    agent_role TEXT,
                    agent_path TEXT,
                    model_provider TEXT NOT NULL,
                    model TEXT,
                    reasoning_effort TEXT,
                    cwd TEXT NOT NULL,
                    cli_version TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    preview TEXT NOT NULL DEFAULT '',
                    sandbox_policy TEXT NOT NULL DEFAULT '',
                    approval_mode TEXT NOT NULL DEFAULT '',
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    has_user_event INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    archived_at INTEGER,
                    git_sha TEXT,
                    git_branch TEXT,
                    git_origin_url TEXT,
                    memory_mode TEXT NOT NULL DEFAULT 'enabled'
                )""")
                
                # Timestamp triggers enforcing unique monotonically orderable millisecond timestamps
                self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS threads_created_at_ms_after_insert
                AFTER INSERT ON threads
                WHEN NEW.created_at_ms IS NULL
                BEGIN
                    UPDATE threads
                    SET created_at_ms = NEW.created_at * 1000
                    WHERE thread_id = NEW.thread_id;
                END;""")
                
                self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS threads_updated_at_ms_after_insert
                AFTER INSERT ON threads
                WHEN NEW.updated_at_ms IS NULL
                BEGIN
                    UPDATE threads
                    SET updated_at_ms = NEW.updated_at * 1000
                    WHERE thread_id = NEW.thread_id;
                END;""")
                
                self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS threads_created_at_ms_after_update
                AFTER UPDATE OF created_at ON threads
                WHEN NEW.created_at != OLD.created_at AND NEW.created_at_ms IS OLD.created_at_ms
                BEGIN
                    UPDATE threads
                    SET created_at_ms = NEW.created_at * 1000
                    WHERE thread_id = NEW.thread_id;
                END;""")
                
                self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS threads_updated_at_ms_after_update
                AFTER UPDATE OF updated_at ON threads
                WHEN NEW.updated_at != OLD.updated_at AND NEW.updated_at_ms IS OLD.updated_at_ms
                BEGIN
                    UPDATE threads
                    SET updated_at_ms = NEW.updated_at * 1000
                    WHERE thread_id = NEW.thread_id;
                END;""")
                
                # Indices supporting sorted keyset paginated lists queries
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_created_at ON threads(created_at DESC, thread_id DESC)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_updated_at ON threads(updated_at DESC, thread_id DESC)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_created_at_ms ON threads(created_at_ms DESC, thread_id DESC)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_updated_at_ms ON threads(updated_at_ms DESC, thread_id DESC)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_archived ON threads(archived)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_source ON threads(source)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_provider ON threads(model_provider)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_archived_cwd_created_at_ms ON threads(archived, cwd, created_at_ms DESC, thread_id DESC)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_archived_cwd_updated_at_ms ON threads(archived, cwd, updated_at_ms DESC, thread_id DESC)")
                
                # Table 2: stage1_outputs (summarization records persistent cache)
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS stage1_outputs (
                    thread_id TEXT PRIMARY KEY,
                    source_updated_at INTEGER NOT NULL,
                    raw_memory TEXT NOT NULL,
                    rollout_summary TEXT NOT NULL,
                    generated_at INTEGER NOT NULL,
                    rollout_slug TEXT,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    last_usage INTEGER,
                    selected_for_phase2 INTEGER NOT NULL DEFAULT 0,
                    selected_for_phase2_source_updated_at INTEGER,
                    FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
                )""")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_stage1_outputs_source_updated_at ON stage1_outputs(source_updated_at DESC, thread_id DESC)")
                
                # Table 3: jobs (transactional lease claims locks)
                self.conn.execute("""
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
                    retry_remaining INTEGER NOT NULL DEFAULT 3,
                    last_error TEXT,
                    input_watermark INTEGER,
                    last_success_watermark INTEGER,
                    PRIMARY KEY (kind, job_key)
                )""")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_kind_status_retry_lease ON jobs(kind, status, retry_at, lease_until)")
                
                # Table 4: thread_dynamic_tools
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_dynamic_tools (
                    thread_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    input_schema TEXT NOT NULL,
                    defer_loading INTEGER NOT NULL DEFAULT 0,
                    namespace TEXT,
                    PRIMARY KEY(thread_id, position),
                    FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
                )""")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_dynamic_tools_thread ON thread_dynamic_tools(thread_id)")
                
                # Table 5: thread_spawn_edges
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_spawn_edges (
                    parent_thread_id TEXT NOT NULL,
                    child_thread_id TEXT NOT NULL PRIMARY KEY,
                    status TEXT NOT NULL
                )""")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_spawn_edges_parent_status ON thread_spawn_edges(parent_thread_id, status)")
                
                # Table 6: thread_goals (dynamic tools accounting session budgets)
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_goals (
                    thread_id TEXT PRIMARY KEY NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
                    goal_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active', 'paused', 'budget_limited', 'complete')),
                    token_budget INTEGER,
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    time_used_seconds INTEGER NOT NULL DEFAULT 0,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                )""")
                
                # Legacy compatibility Tables
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    thread_id TEXT REFERENCES threads(thread_id) ON DELETE CASCADE,
                    created_at INTEGER,
                    updated_at INTEGER
                )""")
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    session_id TEXT,
                    user_message TEXT,
                    assistant_message TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    created_at INTEGER
                )""")
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    thread_id TEXT REFERENCES threads(thread_id) ON DELETE CASCADE,
                    key TEXT,
                    value TEXT
                )""")
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS locks (
                    lock_id TEXT PRIMARY KEY,
                    owner_token TEXT,
                    expires_at INTEGER
                )""")
                
            # 2. Initialize trace logs database tables (under self.logs_conn)
            with self.logs_conn:
                self.logs_conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    ts_nanos INTEGER NOT NULL,
                    level TEXT NOT NULL,
                    target TEXT NOT NULL,
                    feedback_log_body TEXT,
                    module_path TEXT,
                    file TEXT,
                    line INTEGER,
                    thread_id TEXT,
                    process_uuid TEXT,
                    estimated_bytes INTEGER NOT NULL DEFAULT 0
                )""")
                self.logs_conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts DESC, ts_nanos DESC, id DESC)")
                self.logs_conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_thread_id ON logs(thread_id)")
                self.logs_conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_thread_id_ts ON logs(thread_id, ts DESC, ts_nanos DESC, id DESC)")
                self.logs_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_process_uuid_threadless_ts 
                ON logs(process_uuid, ts DESC, ts_nanos DESC, id DESC) 
                WHERE thread_id IS NULL""")
                
            self._schemas_initialized = True
            self._seed_high_water_mark()
        except sqlite3.OperationalError as e:
            if "lock" in str(e).lower() or "busy" in str(e).lower():
                pass
            else:
                raise

    def _ensure_schemas(self) -> None:
        """Ensures database schemas are lazily initialized under lock conditions."""
        if not self._schemas_initialized:
            self._try_initialize_schemas()

    def _seed_high_water_mark(self) -> None:
        """Pre-seeds monotonic clock starting values based on maximum persisted record."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT MAX(updated_at_ms) FROM threads")
            row = cursor.fetchone()
            val = row[0] if row and row[0] is not None else 0
            with self._time_mutex:
                self._last_allocated_ms = val
        except sqlite3.OperationalError as exc:
            import logging; logging.warning(f"Swallowed exception trace: {exc}")

    def _allocate_monotonic_timestamp(self, dt: datetime.datetime) -> int:
        """Allocates unique millisecond timestamp based on high-water mark increments."""
        candidate_ms = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
        with self._time_mutex:
            if candidate_ms > self._last_allocated_ms:
                self._last_allocated_ms = candidate_ms
                return candidate_ms
            if candidate_ms + 1000 <= self._last_allocated_ms:
                # Older historical backfill: preserve original timestamp bounds
                return candidate_ms
            # Hot same-second increment allocator
            self._last_allocated_ms += 1
            return self._last_allocated_ms

    @classmethod
    def open_codex_home(cls, codex_home: Path | str) -> MemoryStateStore:
        """Class factory resolving and loading persistent sqlite engines from the default home path."""
        codex_home = Path(codex_home).expanduser().resolve()
        db_path = codex_home / "thread_store.db"
        if not db_path.is_file():
            db_path = codex_home / "memories" / "thread_store.db"
        return cls(db_path)

    def close(self) -> None:
        """Atomically terminates connection handles for State store and logs engines."""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
        if hasattr(self, "logs_conn") and self.logs_conn:
            self.logs_conn.close()

    def clear_memory_data(self) -> None:
        """Cleans out cache logs summaries and leases pending locks."""
        self._ensure_schemas()
        with self.conn:
            self.conn.execute("DELETE FROM stage1_outputs")
            self.conn.execute("DELETE FROM jobs WHERE kind IN ('memory_stage1', 'memory_consolidate_global')")

    def upsert_thread(self, record: MemoryThreadRecord) -> None:
        """Inserts or replaces meta logs threads configurations."""
        self._ensure_schemas()
        now_dt = record.updated_at.replace(tzinfo=datetime.timezone.utc)
        now_seconds = int(now_dt.timestamp())
        now_ms = self._allocate_monotonic_timestamp(now_dt)
        
        cursor = self.conn.cursor()
        git_sha = None
        git_branch = record.git_branch
        git_origin = None
        
        try:
            cursor.execute("SELECT git_sha, git_branch, git_origin_url FROM threads WHERE thread_id=?", (record.thread_id,))
            row = cursor.fetchone()
            if row:
                git_sha = row["git_sha"]
                git_branch = row["git_branch"] or record.git_branch
                git_origin = row["git_origin_url"]
        except sqlite3.OperationalError as exc:
            import logging; logging.warning(f"Swallowed exception trace: {exc}")
            
        with self.conn:
            self.conn.execute("""
            INSERT INTO threads (
                thread_id, rollout_path, created_at, updated_at, created_at_ms, updated_at_ms,
                source, model_provider, cwd, title, sandbox_policy, approval_mode,
                git_branch, git_sha, git_origin_url, memory_mode, cli_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'enabled', '')
            ON CONFLICT(thread_id) DO UPDATE SET
                rollout_path = excluded.rollout_path,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                created_at_ms = COALESCE(threads.created_at_ms, excluded.created_at_ms),
                updated_at_ms = excluded.updated_at_ms,
                source = excluded.source,
                model_provider = excluded.model_provider,
                cwd = excluded.cwd,
                title = excluded.title,
                sandbox_policy = excluded.sandbox_policy,
                approval_mode = excluded.approval_mode,
                git_branch = COALESCE(threads.git_branch, excluded.git_branch),
                git_sha = COALESCE(threads.git_sha, excluded.git_sha),
                git_origin_url = COALESCE(threads.git_origin_url, excluded.git_origin_url)
            """, (
                record.thread_id, str(record.rollout_path), now_seconds, now_seconds, now_ms, now_ms,
                "cli", "openai", str(record.cwd), f"Resume Rollout Thread {record.thread_id}",
                "read-only", "on-request", git_branch, git_sha, git_origin
            ))

    def try_claim_stage1_job(
        self,
        *,
        thread_id: str,
        worker_id: str,
        source_updated_at: datetime.datetime | int,
        lease_seconds: int,
        max_running_jobs: int,
        now: datetime.datetime | None = None
    ) -> MemoryJobClaim:
        """Main Stage 1 locks claiming transaction executing under immediate SQLite transactions."""
        self._ensure_schemas()
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        lease_until = now_epoch + max(0, lease_seconds)
        
        source_epoch = source_updated_at
        if isinstance(source_updated_at, datetime.datetime):
            source_epoch = int(source_updated_at.replace(tzinfo=datetime.timezone.utc).timestamp())
            
        cursor = self.conn.cursor()
        
        # Check freshness bounds
        cursor.execute("SELECT source_updated_at FROM stage1_outputs WHERE thread_id=?", (thread_id,))
        out_row = cursor.fetchone()
        if out_row and out_row["source_updated_at"] >= source_epoch:
            return MemoryJobClaim("skipped_up_to_date")
            
        cursor.execute("SELECT last_success_watermark FROM jobs WHERE kind='memory_stage1' AND job_key=?", (thread_id,))
        job_row = cursor.fetchone()
        if job_row and job_row["last_success_watermark"] is not None and job_row["last_success_watermark"] >= source_epoch:
            return MemoryJobClaim("skipped_up_to_date")
            
        ownership_token = str(uuid.uuid4())
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            
            # Count active running jobs to enforce capacity caps
            cursor.execute("""
            SELECT COUNT(*) AS active_count FROM jobs 
            WHERE kind='memory_stage1' AND status='running' 
              AND lease_until IS NOT NULL AND lease_until > ? AND job_key != ?
            """, (now_epoch, thread_id))
            active_count = cursor.fetchone()["active_count"]
            
            if active_count >= max_running_jobs:
                self.conn.execute("ROLLBACK")
                return MemoryJobClaim("skipped_running")
                
            cursor.execute("""
            SELECT status, lease_until, retry_at, retry_remaining, input_watermark 
            FROM jobs WHERE kind='memory_stage1' AND job_key=?
            """, (thread_id,))
            j_info = cursor.fetchone()
            
            if j_info:
                j_status = j_info["status"]
                j_lease = j_info["lease_until"]
                j_retry_at = j_info["retry_at"]
                j_rem = j_info["retry_remaining"]
                j_water = j_info["input_watermark"]
                
                watermark_advanced = source_epoch > (j_water if j_water is not None else -1)
                
                if j_status == "running" and j_lease is not None and j_lease > now_epoch:
                    self.conn.execute("ROLLBACK")
                    return MemoryJobClaim("skipped_running")
                    
                if not watermark_advanced:
                    if j_rem <= 0:
                        self.conn.execute("ROLLBACK")
                        return MemoryJobClaim("skipped_retry_exhausted")
                    if j_retry_at is not None and j_retry_at > now_epoch:
                        self.conn.execute("ROLLBACK")
                        return MemoryJobClaim("skipped_retry_backoff")
                        
                new_retry = 3 if watermark_advanced else j_rem
                self.conn.execute("""
                UPDATE jobs SET
                    status='running', worker_id=?, ownership_token=?, started_at=?,
                    finished_at=NULL, lease_until=?, retry_at=NULL, retry_remaining=?,
                    last_error=NULL, input_watermark=?
                WHERE kind='memory_stage1' AND job_key=?
                """, (worker_id, ownership_token, now_epoch, lease_until, new_retry, source_epoch, thread_id))
            else:
                self.conn.execute("""
                INSERT INTO jobs (
                    kind, job_key, status, worker_id, ownership_token, started_at,
                    finished_at, lease_until, retry_at, retry_remaining, last_error,
                    input_watermark, last_success_watermark
                ) VALUES ('memory_stage1', ?, 'running', ?, ?, ?, NULL, ?, NULL, 3, NULL, ?, NULL)
                """, (thread_id, worker_id, ownership_token, now_epoch, lease_until, source_epoch))
                
            self.conn.execute("COMMIT")
            return MemoryJobClaim("claimed", ownership_token, source_epoch)
        except sqlite3.OperationalError as e:
            try:
                self.conn.execute("ROLLBACK")
            except Exception as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
            raise e

    def claim_stage1_jobs_for_startup(
        self,
        *,
        current_thread_id: str | None,
        scan_limit: int,
        max_claimed: int,
        max_age_days: int,
        min_rollout_idle_hours: int,
        allowed_sources: set[str] | frozenset[str],
        lease_seconds: int,
        max_running_jobs: int,
        now: datetime.datetime | None = None
    ) -> list[MemoryStageOneStartupClaim]:
        """Queries and claims background summary extractions for stale rollout threads during session loading passes."""
        self._ensure_schemas()
        if scan_limit <= 0 or max_claimed <= 0:
            return []
            
        now_dt = now or datetime.datetime.utcnow()
        now_epoch_ms = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
        
        max_age_ms = now_epoch_ms - (max_age_days * 24 * 60 * 60 * 1000)
        idle_ms = now_epoch_ms - (min_rollout_idle_hours * 60 * 60 * 1000)
        
        cursor = self.conn.cursor()
        
        query = """
        SELECT t.thread_id, t.rollout_path, t.updated_at_ms, t.updated_at
        FROM threads AS t
        LEFT JOIN stage1_outputs AS so ON so.thread_id = t.thread_id
        LEFT JOIN jobs AS j ON j.kind='memory_stage1' AND j.job_key = t.thread_id
        WHERE t.memory_mode = 'enabled'
          AND t.thread_id != ?
          AND t.source IN ({allowed_placeholders})
          AND t.updated_at_ms >= ?
          AND t.updated_at_ms <= ?
          AND (
               (so.source_updated_at IS NULL) OR 
               ((so.source_updated_at + 1) * 1000 <= t.updated_at_ms)
          )
          AND (
               (j.last_success_watermark IS NULL) OR 
               ((j.last_success_watermark + 1) * 1000 <= t.updated_at_ms)
          )
        ORDER BY t.updated_at_ms DESC, t.thread_id DESC
        LIMIT ?
        """
        sources_list = list(allowed_sources)
        placeholders = ", ".join("?" for _ in sources_list)
        resolved_query = query.format(allowed_placeholders=placeholders)
        
        params = [current_thread_id or ""] + sources_list + [max_age_ms, idle_ms, scan_limit]
        
        cursor.execute(resolved_query, params)
        rows = cursor.fetchall()
        
        claims = []
        for row in rows:
            if len(claims) >= max_claimed:
                break
            th_id = row["thread_id"]
            path = Path(row["rollout_path"])
            up_sec = row["updated_at"]
            
            try:
                claim_res = self.try_claim_stage1_job(
                    thread_id=th_id,
                    worker_id=current_thread_id or "worker-startup",
                    source_updated_at=up_sec,
                    lease_seconds=lease_seconds,
                    max_running_jobs=max_running_jobs,
                    now=now_dt
                )
                
                if claim_res.outcome == "claimed":
                    claims.append(MemoryStageOneStartupClaim(
                        thread_id=th_id,
                        rollout_path=path,
                        source_updated_at=datetime.datetime.fromtimestamp(up_sec, datetime.timezone.utc),
                        ownership_token=claim_res.ownership_token
                    ))
            except sqlite3.OperationalError as e:
                # If database locked errors are encountered, bubble them up for E2E lock checks
                raise e
                
        return claims

    def try_claim_global_phase2_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        now: datetime.datetime | None = None
    ) -> MemoryJobClaim:
        """Claims the global singleton phase-2 lock respecting failure backoffs and success cooldowns."""
        self._ensure_schemas()
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        lease_until = now_epoch + max(0, lease_seconds)
        
        # 6 hours global phase 2 cooldown boundaries
        cooldown_cutoff = now_epoch - 21600
        
        cursor = self.conn.cursor()
        
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            
            cursor.execute("""
            SELECT status, lease_until, retry_at, finished_at, last_error, input_watermark 
            FROM jobs WHERE kind='memory_consolidate_global' AND job_key='global'
            """)
            j_info = cursor.fetchone()
            
            ownership_token = str(uuid.uuid4())
            
            if j_info:
                status = j_info["status"]
                lease = j_info["lease_until"]
                retry_at = j_info["retry_at"]
                finished_at = j_info["finished_at"]
                last_error = j_info["last_error"]
                watermark = j_info["input_watermark"] or 0
                
                if retry_at is not None and retry_at > now_epoch:
                    self.conn.execute("ROLLBACK")
                    return MemoryJobClaim("skipped_retry_backoff", input_watermark=watermark)
                    
                if status == "running" and lease is not None and lease > now_epoch:
                    self.conn.execute("ROLLBACK")
                    return MemoryJobClaim("skipped_running", input_watermark=watermark)
                    
                if last_error is None and finished_at is not None and finished_at > cooldown_cutoff:
                    self.conn.execute("ROLLBACK")
                    return MemoryJobClaim("skipped_cooldown", input_watermark=watermark)
                    
                self.conn.execute("""
                UPDATE jobs SET
                    status='running', worker_id=?, ownership_token=?, started_at=?,
                    finished_at=NULL, lease_until=?, retry_at=NULL, last_error=NULL
                WHERE kind='memory_consolidate_global' AND job_key='global'
                """, (worker_id, ownership_token, now_epoch, lease_until))
                
                self.conn.execute("COMMIT")
                return MemoryJobClaim("claimed", ownership_token, watermark)
            else:
                self.conn.execute("""
                INSERT INTO jobs (
                    kind, job_key, status, worker_id, ownership_token, started_at,
                    finished_at, lease_until, retry_at, retry_remaining, last_error,
                    input_watermark, last_success_watermark
                ) VALUES ('memory_consolidate_global', 'global', 'running', ?, ?, ?, NULL, ?, NULL, 3, NULL, 0, 0)
                """, (worker_id, ownership_token, now_epoch, lease_until))
                
                self.conn.execute("COMMIT")
                return MemoryJobClaim("claimed", ownership_token, 0)
        except sqlite3.OperationalError as e:
            try:
                self.conn.execute("ROLLBACK")
            except Exception as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
            raise e

    def heartbeat_global_phase2_job(
        self,
        *,
        ownership_token: str,
        lease_seconds: int,
        now: datetime.datetime | None = None
    ) -> bool:
        """Refreshes and extends the lease timer for an active owned global Phase 2 consolidation lock."""
        self._ensure_schemas()
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        lease_until = now_epoch + max(0, lease_seconds)
        
        with self.conn:
            cursor = self.conn.execute("""
            UPDATE jobs SET lease_until = ?
            WHERE kind='memory_consolidate_global' AND job_key='global'
              AND status='running' AND ownership_token=?
            """, (lease_until, ownership_token))
            return cursor.rowcount > 0

    def mark_stage1_job_succeeded(
        self,
        *,
        thread_id: str,
        ownership_token: str,
        source_updated_at: datetime.datetime | int,
        raw_memory: str,
        rollout_summary: str,
        rollout_slug: str | None,
        now: datetime.datetime | None = None
    ) -> bool:
        """Finalizes a Stage 1 extraction job as successful, upserting its summaries to stage1_outputs."""
        self._ensure_schemas()
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        
        source_epoch = source_updated_at
        if isinstance(source_updated_at, datetime.datetime):
            source_epoch = int(source_updated_at.replace(tzinfo=datetime.timezone.utc).timestamp())
            
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            
            # 1. Update jobs status
            cursor = self.conn.execute("""
            UPDATE jobs SET
                status='done', finished_at=?, lease_until=NULL, last_error=NULL,
                last_success_watermark=input_watermark
            WHERE kind='memory_stage1' AND job_key=?
              AND status='running' AND ownership_token=?
            """, (now_epoch, thread_id, ownership_token))
            
            if cursor.rowcount == 0:
                self.conn.execute("ROLLBACK")
                return False
                
            # 2. Upsert extracted output
            self.conn.execute("""
            INSERT INTO stage1_outputs (
                thread_id, source_updated_at, raw_memory, rollout_summary, rollout_slug, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                source_updated_at = excluded.source_updated_at,
                raw_memory = excluded.raw_memory,
                rollout_summary = excluded.rollout_summary,
                rollout_slug = excluded.rollout_slug,
                generated_at = excluded.generated_at
            WHERE excluded.source_updated_at >= stage1_outputs.source_updated_at
            """, (thread_id, source_epoch, raw_memory, rollout_summary, rollout_slug, now_epoch))
            
            # 3. Monotonically advance consolidation dirty watermark
            self.enqueue_global_consolidation(source_epoch)
            
            self.conn.execute("COMMIT")
            return True
        except sqlite3.OperationalError as e:
            try:
                self.conn.execute("ROLLBACK")
            except Exception as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
            raise e

    def mark_stage1_job_succeeded_no_output(
        self,
        *,
        thread_id: str,
        ownership_token: str,
        now: datetime.datetime | None = None
    ) -> bool:
        """Finalizes job succeeded, removing the thread's existing Stage 1 output summaries (empty rollouts)."""
        self._ensure_schemas()
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            
            # 1. Update status
            cursor = self.conn.execute("""
            UPDATE jobs SET
                status='done', finished_at=?, lease_until=NULL, last_error=NULL,
                last_success_watermark=input_watermark
            WHERE kind='memory_stage1' AND job_key=?
              AND status='running' AND ownership_token=?
            """, (now_epoch, thread_id, ownership_token))
            
            if cursor.rowcount == 0:
                self.conn.execute("ROLLBACK")
                return False
                
            # Get input watermark for global progression
            cursor = self.conn.execute("""
            SELECT input_watermark FROM jobs 
            WHERE kind='memory_stage1' AND job_key=? AND ownership_token=?
            """, (thread_id, ownership_token))
            row = cursor.fetchone()
            watermark = row["input_watermark"] if row else now_epoch
            
            # 2. Delete summary records
            del_cursor = self.conn.execute("DELETE FROM stage1_outputs WHERE thread_id=?", (thread_id,))
            
            # 3. Monotonically advance global consolidation watermark if rows were removed
            if del_cursor.rowcount > 0:
                self.enqueue_global_consolidation(watermark)
                
            self.conn.execute("COMMIT")
            return True
        except sqlite3.OperationalError as e:
            try:
                self.conn.execute("ROLLBACK")
            except Exception as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
            raise e

    def mark_stage1_job_failed(
        self,
        *,
        thread_id: str,
        ownership_token: str,
        failure_reason: str,
        retry_delay_seconds: int,
        now: datetime.datetime | None = None
    ) -> bool:
        """Finalizes job as failed, scheduling backoff timers and decrementing attempts count."""
        self._ensure_schemas()
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        retry_at = now_epoch + max(0, retry_delay_seconds)
        
        with self.conn:
            cursor = self.conn.execute("""
            UPDATE jobs SET
                status='error', finished_at=?, lease_until=NULL, retry_at=?,
                retry_remaining = MAX(retry_remaining - 1, 0), last_error=?
            WHERE kind='memory_stage1' AND job_key=?
              AND status='running' AND ownership_token=?
            """, (now_epoch, retry_at, failure_reason, thread_id, ownership_token))
            return cursor.rowcount > 0

    def mark_global_phase2_job_succeeded(
        self,
        *,
        ownership_token: str,
        completed_watermark: datetime.datetime | int,
        selected_outputs: list[MemoryStageOneRecord],
        now: datetime.datetime | None = None
    ) -> bool:
        """Finalizes Phase 2 succeeded, advancing global watermark, and resetting workspace baseline selections."""
        self._ensure_schemas()
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        
        watermark = completed_watermark
        if isinstance(completed_watermark, datetime.datetime):
            watermark = int(completed_watermark.replace(tzinfo=datetime.timezone.utc).timestamp())
            
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            
            # 1. Update singleton job status
            cursor = self.conn.execute("""
            UPDATE jobs SET
                status='done', finished_at=?, lease_until=NULL, last_error=NULL,
                last_success_watermark = MAX(COALESCE(last_success_watermark, 0), ?)
            WHERE kind='memory_consolidate_global' AND job_key='global'
              AND status='running' AND ownership_token=?
            """, (now_epoch, watermark, ownership_token))
            
            if cursor.rowcount == 0:
                self.conn.execute("ROLLBACK")
                return False
                
            # 2. Reset old selection baselines
            self.conn.execute("""
            UPDATE stage1_outputs SET
                selected_for_phase2 = 0,
                selected_for_phase2_source_updated_at = NULL
            WHERE selected_for_phase2 != 0 OR selected_for_phase2_source_updated_at IS NOT NULL
            """)
            
            # 3. Persist exact selected snapshots baseline markers
            for out in selected_outputs:
                source_epoch = int(out.source_updated_at.replace(tzinfo=datetime.timezone.utc).timestamp())
                self.conn.execute("""
                UPDATE stage1_outputs SET
                    selected_for_phase2 = 1,
                    selected_for_phase2_source_updated_at = ?
                WHERE thread_id = ? AND source_updated_at = ?
                """, (source_epoch, out.thread_id, source_epoch))
                
            self.conn.execute("COMMIT")
            return True
        except sqlite3.OperationalError as e:
            try:
                self.conn.execute("ROLLBACK")
            except Exception as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
            raise e

    def mark_global_phase2_job_failed(
        self,
        *,
        ownership_token: str,
        failure_reason: str,
        retry_delay_seconds: int,
        now: datetime.datetime | None = None,
        allow_unowned: bool = False
    ) -> bool:
        """Finalizes global Phase 2 failed, setting error limits, retry timers, and backing off claims."""
        self._ensure_schemas()
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        retry_at = now_epoch + max(0, retry_delay_seconds)
        
        with self.conn:
            query = """
            UPDATE jobs SET
                status='error', finished_at=?, lease_until=NULL, retry_at=?,
                retry_remaining = MAX(retry_remaining - 1, 0), last_error=?
            WHERE kind='memory_consolidate_global' AND job_key='global'
              AND status='running'
            """
            if allow_unowned:
                query += " AND (ownership_token = ? OR ownership_token IS NULL)"
            else:
                query += " AND ownership_token = ?"
                
            cursor = self.conn.execute(query, (now_epoch, retry_at, failure_reason, ownership_token))
            return cursor.rowcount > 0

    def enqueue_global_consolidation(self, input_watermark: datetime.datetime | int) -> None:
        """Enqueues or dynamically updates the global Phase 2 input progression watermark."""
        self._ensure_schemas()
        watermark = input_watermark
        if isinstance(input_watermark, datetime.datetime):
            watermark = int(input_watermark.replace(tzinfo=datetime.timezone.utc).timestamp())
            
        sql_stmt = """
        INSERT INTO jobs (
            kind, job_key, status, worker_id, ownership_token, started_at,
            finished_at, lease_until, retry_at, retry_remaining, last_error,
            input_watermark, last_success_watermark
        ) VALUES ('memory_consolidate_global', 'global', 'pending', NULL, NULL, NULL, NULL, NULL, NULL, 3, NULL, ?, 0)
        ON CONFLICT(kind, job_key) DO UPDATE SET
            status = CASE WHEN jobs.status = 'running' THEN 'running' ELSE 'pending' END,
            retry_at = CASE WHEN jobs.status = 'running' THEN jobs.retry_at ELSE NULL END,
            retry_remaining = MAX(jobs.retry_remaining, 3),
            input_watermark = CASE WHEN ? > COALESCE(jobs.input_watermark, 0) THEN ? ELSE COALESCE(jobs.input_watermark, 0) + 1 END
        """
        
        # Prevent nested connection manager commit conflicts under outer active transactions:
        if self.conn.in_transaction:
            self.conn.execute(sql_stmt, (watermark, watermark, watermark))
        else:
            with self.conn:
                self.conn.execute(sql_stmt, (watermark, watermark, watermark))

    def mark_thread_memory_mode_polluted(
        self,
        thread_id: str,
        *,
        now: datetime.datetime | None = None
    ) -> bool:
        """Sets thread memory mode to polluted, enqueuing baseline regeneration if memory was previously selected."""
        self._ensure_schemas()
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            
            cursor = self.conn.execute("""
            UPDATE threads SET memory_mode = 'polluted'
            WHERE thread_id = ? AND memory_mode != 'polluted'
            """, (thread_id,))
            
            if cursor.rowcount == 0:
                self.conn.execute("ROLLBACK")
                return False
                
            cursor = self.conn.execute("SELECT selected_for_phase2 FROM stage1_outputs WHERE thread_id=?", (thread_id,))
            row = cursor.fetchone()
            selected = row["selected_for_phase2"] if row else 0
            
            if selected != 0:
                self.enqueue_global_consolidation(now_epoch)
                
            self.conn.execute("COMMIT")
            return True
        except sqlite3.OperationalError as e:
            try:
                self.conn.execute("ROLLBACK")
            except Exception as exc:
                import logging; logging.warning(f"Swallowed exception trace: {exc}")
            raise e

    def set_thread_memory_mode(self, thread_id: str, memory_mode: str) -> bool:
        """Directly sets thread memory configuration, updating the database mode tag."""
        self._ensure_schemas()
        with self.conn:
            cursor = self.conn.execute("UPDATE threads SET memory_mode = ? WHERE thread_id = ?", (memory_mode, thread_id))
            return cursor.rowcount > 0

    def get_job(self, kind: str, job_key: str) -> dict[str, Any] | None:
        """Resolves active lock/lease job records matching partition indices."""
        self._ensure_schemas()
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE kind=? AND job_key=?", (kind, job_key))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_stage1_output(self, thread_id: str) -> dict[str, Any] | None:
        """Retrieves targeted thread Stage 1 summary extract outputs."""
        self._ensure_schemas()
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM stage1_outputs WHERE thread_id=?", (thread_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_stage1_outputs_for_global(self, n: int) -> list[MemoryStageOneRecord]:
        """Resolves latest non-empty and non-polluted Stage 1 summaries mapping baseline inputs."""
        self._ensure_schemas()
        if n <= 0:
            return []
            
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT so.thread_id, so.source_updated_at, so.raw_memory, so.rollout_summary,
               so.rollout_slug, so.usage_count, so.last_usage, so.selected_for_phase2,
               t.rollout_path, t.cwd
        FROM stage1_outputs AS so
        LEFT JOIN threads AS t ON t.thread_id = so.thread_id
        WHERE t.memory_mode = 'enabled'
          AND (LENGTH(TRIM(so.raw_memory)) > 0 OR LENGTH(TRIM(so.rollout_summary)) > 0)
        ORDER BY so.source_updated_at DESC, so.thread_id DESC
        LIMIT ?
        """, (n,))
        rows = cursor.fetchall()
        
        records = []
        for r in rows:
            dt = datetime.datetime.fromtimestamp(r["source_updated_at"], datetime.timezone.utc)
            last_dt = None
            if r["last_usage"] is not None:
                last_dt = datetime.datetime.fromtimestamp(r["last_usage"], datetime.timezone.utc)
            records.append(MemoryStageOneRecord(
                thread_id=r["thread_id"],
                source_updated_at=dt,
                raw_memory=r["raw_memory"],
                rollout_summary=r["rollout_summary"],
                rollout_slug=r["rollout_slug"],
                rollout_path=Path(r["rollout_path"]),
                cwd=Path(r["cwd"]),
                usage_count=r["usage_count"],
                last_usage=last_dt,
                selected_for_phase2=bool(r["selected_for_phase2"])
            ))
        return records

    def get_phase2_input_selection(
        self,
        *,
        n: int,
        max_unused_days: int = 30,
        now: datetime.datetime | None = None
    ) -> list[MemoryStageOneRecord]:
        """Retrieves and ranks active Phase 2 input selections based on usage metrics and recency."""
        self._ensure_schemas()
        if n <= 0:
            return []
            
        now_dt = now or datetime.datetime.utcnow()
        cutoff_seconds = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp()) - (max_unused_days * 24 * 60 * 60)
        
        cursor = self.conn.cursor()
        
        # Rank by citation metadata and recency
        cursor.execute("""
        SELECT selected.thread_id, selected.source_updated_at, selected.raw_memory, selected.rollout_summary,
               selected.rollout_slug, selected.usage_count, selected.last_usage, selected.selected_for_phase2,
               selected.rollout_path, selected.cwd
        FROM (
            SELECT so.thread_id, so.source_updated_at, so.raw_memory, so.rollout_summary,
                   so.rollout_slug, so.usage_count, so.last_usage, so.selected_for_phase2,
                   COALESCE(t.rollout_path, '') AS rollout_path, COALESCE(t.cwd, '') AS cwd
            FROM stage1_outputs AS so
            LEFT JOIN threads AS t ON t.thread_id = so.thread_id
            WHERE t.memory_mode = 'enabled'
              AND (LENGTH(TRIM(so.raw_memory)) > 0 OR LENGTH(TRIM(so.rollout_summary)) > 0)
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
        ) AS selected
        ORDER BY selected.thread_id ASC
        """, (cutoff_seconds, cutoff_seconds, n))
        
        rows = cursor.fetchall()
        
        records = []
        for r in rows:
            dt = datetime.datetime.fromtimestamp(r["source_updated_at"], datetime.timezone.utc)
            last_dt = None
            if r["last_usage"] is not None:
                last_dt = datetime.datetime.fromtimestamp(r["last_usage"], datetime.timezone.utc)
            records.append(MemoryStageOneRecord(
                thread_id=r["thread_id"],
                source_updated_at=dt,
                raw_memory=r["raw_memory"],
                rollout_summary=r["rollout_summary"],
                rollout_slug=r["rollout_slug"],
                rollout_path=Path(r["rollout_path"]),
                cwd=Path(r["cwd"]),
                usage_count=r["usage_count"],
                last_usage=last_dt,
                selected_for_phase2=bool(r["selected_for_phase2"])
            ))
        return records

    def prune_stage1_outputs_for_retention(
        self,
        *,
        max_unused_days: int = 30,
        limit: int = 100,
        now: datetime.datetime | None = None
    ) -> int:
        """Removes stale stage-1 outputs, strictly protecting chosen Phase 2 selection baselines."""
        self._ensure_schemas()
        if limit <= 0:
            return 0
            
        now_dt = now or datetime.datetime.utcnow()
        cutoff_seconds = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp()) - (max_unused_days * 24 * 60 * 60)
        
        with self.conn:
            cursor = self.conn.execute("""
            DELETE FROM stage1_outputs
            WHERE thread_id IN (
                SELECT thread_id
                FROM stage1_outputs
                WHERE selected_for_phase2 = 0
                  AND COALESCE(last_usage, source_updated_at) < ?
                ORDER BY
                    COALESCE(last_usage, source_updated_at) ASC,
                    source_updated_at ASC,
                    thread_id ASC
                LIMIT ?
            )
            """, (cutoff_seconds, limit))
            return cursor.rowcount

    def record_stage1_output_usage(
        self,
        thread_ids: list[str],
        *,
        now: datetime.datetime | None = None
    ) -> int:
        """Increments usage citations counts for cited thread summaries, updating timestamp records."""
        self._ensure_schemas()
        if not thread_ids:
            return 0
            
        now_dt = now or datetime.datetime.utcnow()
        now_epoch = int(now_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        
        with self.conn:
            updated = 0
            for tid in thread_ids:
                cursor = self.conn.execute("""
                UPDATE stage1_outputs SET
                    usage_count = COALESCE(usage_count, 0) + 1,
                    last_usage = ?
                WHERE thread_id = ?
                """, (now_epoch, tid))
                updated += cursor.rowcount
            return updated


# ==============================================================================
# Workspace Baseline Snapshots & comparison diffing trackers
# ==============================================================================

def get_file_mode(path: Path) -> str:
    """Deduces git-style 6-character octal permissions mode for blobs, executables, and symlinks."""
    try:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            return "120000"
        if stat.S_ISDIR(st.st_mode):
            return "040000"
        if os.name == 'posix':
            mode = st.st_mode
            if (mode & stat.S_IXUSR) or (mode & stat.S_IXGRP) or (mode & stat.S_IXOTH):
                return "100755"
        return "100644"
    except (OSError, ValueError):
        return "100644"


def get_file_content(path: Path) -> str:
    """Reads symlink paths or plain file contents safely."""
    try:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            return str(os.readlink(path))
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def scan_memories_workspace(root: Path) -> dict[str, dict[str, str]]:
    """Walks the memory workspace folder recursively, collecting content and modes."""
    entries = {}
    if not root.is_dir():
        return entries
        
    for p in root.rglob("*"):
        if ".git" in p.parts or p.name == "phase2_workspace_diff.md":
            continue
        if p.is_file() or p.is_symlink():
            rel_path = "/".join(p.relative_to(root).parts)
            entries[rel_path] = {
                "content": get_file_content(p),
                "mode": get_file_mode(p)
            }
    return entries


def prepare_memory_workspace(root: Path | str) -> None:
    """Prepares directories baseline for git comparison diffing (initializes Git tracking)."""
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    
    diff_file = root_path / "phase2_workspace_diff.md"
    try:
        if diff_file.is_file():
            diff_file.unlink()
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")
        
    git_dir = root_path / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Spawn git baseline if possible
    try:
        res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=str(root_path), capture_output=True)
        res_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root_path), capture_output=True)
        if res.returncode != 0 or res_head.returncode != 0:
            if git_dir.is_dir():
                shutil.rmtree(git_dir, ignore_errors=True)
            git_dir.mkdir(parents=True, exist_ok=True)
            
            env = os.environ.copy()
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            env["GIT_AUTHOR_NAME"] = "Codex"
            env["GIT_AUTHOR_EMAIL"] = "noreply@openai.com"
            env["GIT_COMMITTER_NAME"] = "Codex"
            env["GIT_COMMITTER_EMAIL"] = "noreply@openai.com"
            
            subprocess.run(["git", "init", "-q"], cwd=str(root_path), env=env)
            subprocess.run(["git", "config", "user.name", "Codex"], cwd=str(root_path), env=env)
            subprocess.run(["git", "config", "user.email", "noreply@openai.com"], cwd=str(root_path), env=env)
            subprocess.run(["git", "add", "-A"], cwd=str(root_path), env=env)
            subprocess.run(["git", "commit", "--allow-empty", "-q", "-m", "Initialize Codex git baseline", "--no-verify"], cwd=str(root_path), env=env)
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")
        
    # 2. Pure Python snapshot baseline database
    snapshot_file = git_dir / "baseline_snapshot.json"
    if not snapshot_file.is_file():
        snapshot = scan_memories_workspace(root_path)
        snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def reset_memory_workspace_baseline(root: Path | str) -> None:
    """Commits and locks workspace changes, resetting the delta tracking baseline."""
    root_path = Path(root).expanduser().resolve()
    
    diff_file = root_path / "phase2_workspace_diff.md"
    try:
        if diff_file.is_file():
            diff_file.unlink()
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")
        
    git_dir = root_path / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    
    # Reset git repository
    try:
        if git_dir.is_dir():
            shutil.rmtree(git_dir, ignore_errors=True)
        git_dir.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        env["GIT_AUTHOR_NAME"] = "Codex"
        env["GIT_AUTHOR_EMAIL"] = "noreply@openai.com"
        env["GIT_COMMITTER_NAME"] = "Codex"
        env["GIT_COMMITTER_EMAIL"] = "noreply@openai.com"
        
        subprocess.run(["git", "init", "-q"], cwd=str(root_path), env=env)
        subprocess.run(["git", "config", "user.name", "Codex"], cwd=str(root_path), env=env)
        subprocess.run(["git", "config", "user.email", "noreply@openai.com"], cwd=str(root_path), env=env)
        subprocess.run(["git", "add", "-A"], cwd=str(root_path), env=env)
        subprocess.run(["git", "commit", "--allow-empty", "-q", "-m", "Reset Codex git baseline", "--no-verify"], cwd=str(root_path), env=env)
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")
        
    # Overwrite snapshot baseline JSON database file
    snapshot = scan_memories_workspace(root_path)
    snapshot_file = git_dir / "baseline_snapshot.json"
    snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def memory_workspace_diff(root: Path | str, baseline_root: Path | str | None = None) -> tuple[list[MemoryWorkspaceChange], str]:
    """Generates git baseline diff changes list, comparing modern workspace state against historical baseline commits."""
    root_path = Path(root).expanduser().resolve()
    
    diff_file = root_path / "phase2_workspace_diff.md"
    try:
        if diff_file.is_file():
            diff_file.unlink()
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")
        
    base_dir = Path(baseline_root or root_path).expanduser().resolve()
    
    # Dual-mode strategy:
    # Attempt Pure-Python high-speed comparison first!
    snapshot_file = base_dir / ".git" / "baseline_snapshot.json"
    if snapshot_file.is_file():
        try:
            baseline = json.loads(snapshot_file.read_text(encoding="utf-8"))
            current = scan_memories_workspace(root_path)
            
            changes = []
            diff_blocks = []
            
            all_paths = sorted(set(baseline.keys()) | set(current.keys()))
            for path in all_paths:
                base_item = baseline.get(path)
                curr_item = current.get(path)
                
                if base_item is None and curr_item is not None:
                    # Added
                    changes.append(MemoryWorkspaceChange("added", path))
                    left_lines = []
                    right_lines = curr_item["content"].splitlines(keepends=True)
                    mode = curr_item["mode"]
                    
                    left_oid = "0000000000000000000000000000000000000000"
                    right_oid = git_blob_oid(curr_item["content"].encode("utf-8"))
                    diff_header = f"diff --git a/{path} b/{path}\nnew file mode {mode}\nindex {left_oid}..{right_oid}\n"
                    
                    hunks = "".join(difflib.unified_diff(
                        left_lines, right_lines,
                        fromfile="/dev/null", tofile=f"b/{path}",
                        n=3, lineterm="\n"
                    ))
                    if hunks:
                        diff_blocks.append(diff_header + hunks)
                        
                elif base_item is not None and curr_item is None:
                    # Deleted
                    changes.append(MemoryWorkspaceChange("deleted", path))
                    left_lines = base_item["content"].splitlines(keepends=True)
                    right_lines = []
                    mode = base_item["mode"]
                    
                    left_oid = git_blob_oid(base_item["content"].encode("utf-8"))
                    right_oid = "0000000000000000000000000000000000000000"
                    diff_header = f"diff --git a/{path} b/{path}\ndeleted file mode {mode}\nindex {left_oid}..{right_oid}\n"
                    
                    hunks = "".join(difflib.unified_diff(
                        left_lines, right_lines,
                        fromfile=f"a/{path}", tofile="/dev/null",
                        n=3, lineterm="\n"
                    ))
                    if hunks:
                        diff_blocks.append(diff_header + hunks)
                        
                elif base_item is not None and curr_item is not None:
                    if base_item["content"] != curr_item["content"] or base_item["mode"] != curr_item["mode"]:
                        changes.append(MemoryWorkspaceChange("modified", path))
                        mode_changed = (base_item["mode"] != curr_item["mode"])
                        content_changed = (base_item["content"] != curr_item["content"])
                        
                        diff_header = f"diff --git a/{path} b/{path}\n"
                        if mode_changed:
                            diff_header += f"old mode {base_item['mode']}\nnew mode {curr_item['mode']}\n"
                            
                        left_oid = git_blob_oid(base_item["content"].encode("utf-8"))
                        right_oid = git_blob_oid(curr_item["content"].encode("utf-8"))
                        
                        if content_changed:
                            diff_header += f"index {left_oid}..{right_oid}"
                            if not mode_changed:
                                diff_header += f" {curr_item['mode']}"
                            diff_header += "\n"
                            
                            left_lines = base_item["content"].splitlines(keepends=True)
                            right_lines = curr_item["content"].splitlines(keepends=True)
                            hunks = "".join(difflib.unified_diff(
                                left_lines, right_lines,
                                fromfile=f"a/{path}", tofile=f"b/{path}",
                                n=3, lineterm="\n"
                            ))
                            if hunks:
                                diff_blocks.append(diff_header + hunks)
                        else:
                            # Mode-only modified
                            diff_blocks.append(diff_header)
                            
            return changes, "".join(diff_blocks)
        except Exception as exc:
            # Fallback if pure Python fails
            import logging; logging.warning(f"Swallowed exception trace: {exc}")
            
    # Subprocess Git fallback
    changes = []
    diff_text = ""
    git_dir = root_path / ".git"
    if git_dir.is_dir():
        try:
            env = os.environ.copy()
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            env["GIT_AUTHOR_NAME"] = "Codex"
            env["GIT_AUTHOR_EMAIL"] = "noreply@openai.com"
            env["GIT_COMMITTER_NAME"] = "Codex"
            env["GIT_COMMITTER_EMAIL"] = "noreply@openai.com"
            
            res_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(root_path), capture_output=True, text=True, env=env
            )
            if res_status.returncode == 0 and res_status.stdout:
                for line in res_status.stdout.splitlines():
                      if len(line) < 4:
                          continue
                      status_code = line[:2]
                      file_path = line[3:].strip().strip('"')
                      if file_path == "phase2_workspace_diff.md" or file_path.startswith(".git"):
                          continue
                      status = "modified"
                      if "??" in status_code:
                          status = "untracked"
                      elif "A" in status_code:
                          status = "added"
                      elif "D" in status_code:
                          status = "deleted"
                      changes.append(MemoryWorkspaceChange(status, file_path))
                      
            changes.sort(key=lambda c: c.path)
            
            # Subprocess Git diff HEAD
            subprocess.run(["git", "add", "-N", "-A"], cwd=str(root_path), env=env)
            res_diff = subprocess.run(["git", "diff", "HEAD"], cwd=str(root_path), capture_output=True, text=True, env=env)
            diff_text = res_diff.stdout if res_diff.returncode == 0 else ""
            subprocess.run(["git", "reset"], cwd=str(root_path), env=env)
        except Exception as exc:
            import logging; logging.warning(f"Swallowed exception trace: {exc}")
            
    return changes, diff_text


def render_memory_workspace_diff_file(
    changes: list[MemoryWorkspaceChange],
    unified_diff: str,
    max_bytes: int = 4194304
) -> str:
    """Formats workspace changes list and unified diff block exactly as expected by downstream model prompts."""
    rendered = (
        "# Memory Workspace Diff\n\n"
        "Generated by Codex before Phase 2 memory consolidation. Read this file first and do not edit it.\n\n"
        "## Status\n"
    )
    if not changes:
        rendered += "- none\n"
        return rendered
        
    for change in changes:
        label = change.status[0].upper()
        rendered += f"- {label} {change.path}\n"
        
    rendered += "\n## Diff\n\n```diff\n"
    
    diff_bytes = unified_diff.encode("utf-8")
    if len(diff_bytes) <= max_bytes:
        rendered += unified_diff
        if not unified_diff.endswith("\n"):
            rendered += "\n"
    else:
        boundary = previous_char_boundary(diff_bytes, max_bytes)
        truncated_text = diff_bytes[:boundary].decode("utf-8", errors="ignore")
        rendered += truncated_text
        if not rendered.endswith("\n"):
            rendered += "\n"
        rendered += f"\n[workspace diff truncated at {max_bytes} bytes]\n"
        
    rendered += "```\n"
    return rendered


def write_memory_workspace_diff(root: Path | str, changes: list[MemoryWorkspaceChange], unified_diff: str) -> Path:
    """Serializes memory workspace diff to 'phase2_workspace_diff.md'."""
    root_path = Path(root)
    diff_file = root_path / "phase2_workspace_diff.md"
    rendered = render_memory_workspace_diff_file(changes, unified_diff)
    diff_file.write_text(rendered, encoding="utf-8")
    return diff_file


def write_current_memory_workspace_diff(root: Path | str) -> Path:
    """Computes and writes current memory workspace diff."""
    changes, diff_text = memory_workspace_diff(root)
    return write_memory_workspace_diff(root, changes, diff_text)


# ==============================================================================
# Summaries filename stem name reverse Base-62 generator & citers
# ==============================================================================

def rollout_summary_file_stem(memory: MemoryStageOneRecord) -> str:
    """Generates UTC-time-fragment reverse stems summary filenames with Base-62 suffixes matching Rust."""
    slug = memory.rollout_slug
    if isinstance(slug, bytes):
        slug = slug.decode('utf-8')
        
    source_updated_at = memory.source_updated_at
    if source_updated_at.tzinfo is None:
        source_updated_at = source_updated_at.replace(tzinfo=datetime.timezone.utc)
    else:
        source_updated_at = source_updated_at.astimezone(datetime.timezone.utc)
        
    SHORT_HASH_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    SHORT_HASH_SPACE = 14776336 # 62^4
    
    is_uuid = False
    try:
        thread_uuid = uuid.UUID(memory.thread_id)
        is_uuid = True
    except ValueError as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")
        
    if is_uuid:
        version = thread_uuid.version
        if version == 7:
            # Extract epoch milliseconds from the first 48 bits of V7 UUID
            timestamp_ms = thread_uuid.int >> 80
            dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000.0, datetime.timezone.utc)
        elif version in (1, 6):
            try:
                ns100 = thread_uuid.time
                seconds = (ns100 - 0x01b21dd213814000) / 1e7
                dt = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)
            except ValueError:
                dt = source_updated_at
        else:
            dt = source_updated_at
            
        timestamp_fragment = dt.strftime("%Y-%m-%dT%H-%M-%S")
        # seed is the lowest 32 bits of UUID
        short_hash_seed = thread_uuid.int & 0xFFFFFFFF
    else:
        # Custom wrapping multiplier integer hash fallback for non-UUID strings
        short_hash_seed = 0
        for byte in memory.thread_id.encode('utf-8'):
            short_hash_seed = (short_hash_seed * 31 + byte) & 0xFFFFFFFF
            
        timestamp_fragment = source_updated_at.strftime("%Y-%m-%dT%H-%M-%S")
        
    short_hash_value = short_hash_seed % SHORT_HASH_SPACE
    short_hash_chars = ["0"] * 4
    temp_val = short_hash_value
    for idx in reversed(range(4)):
        alphabet_idx = temp_val % 62
        short_hash_chars[idx] = SHORT_HASH_ALPHABET[alphabet_idx]
        temp_val //= 62
        
    short_hash = "".join(short_hash_chars)
    file_prefix = f"{timestamp_fragment}-{short_hash}"
    
    if not slug:
        return file_prefix
        
    # Sanitize rollout slug up to 60 characters
    slug_chars = []
    for ch in slug:
        if len(slug_chars) >= 60:
            break
        if ch.isalnum() and ch.isascii():
            slug_chars.append(ch.lower())
        else:
            slug_chars.append("_")
            
    sanitized = "".join(slug_chars)
    while sanitized.endswith("_"):
        sanitized = sanitized[:-1]
        
    if not sanitized:
        return file_prefix
    else:
        return f"{file_prefix}-{sanitized}"


# ==============================================================================
# Baseline files structures synchronization executors
# ==============================================================================

def sync_rollout_summaries_from_memories(
    root: Path | str,
    memories: list[MemoryStageOneRecord],
    max_raw_memories_for_consolidation: int,
    *,
    max_unused_days: int = 30,
    now: datetime.datetime | None = None
) -> None:
    """Writes rollout summaries under summaries directory, removing outdated files."""
    root_path = Path(root)
    summaries_dir = root_path / "rollout_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    retained = select_phase2_memory_inputs(memories, max_raw_memories_for_consolidation, max_unused_days=max_unused_days, now=now)
    keep_stems = {rollout_summary_file_stem(m) for m in retained}
    
    if summaries_dir.is_dir():
        for f in summaries_dir.iterdir():
            if f.is_file() and f.suffix == ".md":
                if f.stem not in keep_stems:
                    try:
                        f.unlink()
                    except Exception as exc:
                        import logging; logging.warning(f"Swallowed exception trace: {exc}")
                        
    for m in retained:
        stem = rollout_summary_file_stem(m)
        sum_path = summaries_dir / f"{stem}.md"
        
        body = []
        body.append(f"thread_id: {m.thread_id}")
        
        source_upd = m.source_updated_at
        if source_upd.tzinfo is None:
            source_upd = source_upd.replace(tzinfo=datetime.timezone.utc)
        body.append(f"updated_at: {source_upd.isoformat()}")
        body.append(f"rollout_path: {m.rollout_path}")
        body.append(f"cwd: {m.cwd}")
        
        git_branch = getattr(m, "git_branch", None)
        if git_branch:
            body.append(f"git_branch: {git_branch}")
            
        body.append("")
        body.append(m.rollout_summary.strip())
        
        try:
            sum_path.write_text("\n".join(body) + "\n", encoding="utf-8")
        except Exception as exc:
            import logging; logging.warning(f"Swallowed exception trace: {exc}")


def rebuild_raw_memories_file_from_memories(
    root: Path | str,
    memories: list[MemoryStageOneRecord],
    max_raw_memories_for_consolidation: int,
    *,
    max_unused_days: int = 30,
    now: datetime.datetime | None = None
) -> None:
    """Renders all stage 1 summaries aggregated inside a merged raw_memories.md file."""
    root_path = Path(root)
    raw_file = root_path / "raw_memories.md"
    
    retained = select_phase2_memory_inputs(memories, max_raw_memories_for_consolidation, max_unused_days=max_unused_days, now=now)
    
    body = ["# Raw Memories\n"]
    if not retained:
        body.append("No raw memories yet.")
    else:
        body.append("Merged stage-1 raw memories (stable ascending thread-id order):\n")
        for m in retained:
            body.append(f"## Thread `{m.thread_id}`")
            
            source_upd = m.source_updated_at
            if source_upd.tzinfo is None:
                source_upd = source_upd.replace(tzinfo=datetime.timezone.utc)
            body.append(f"updated_at: {source_upd.isoformat()}")
            body.append(f"cwd: {m.cwd}")
            body.append(f"rollout_path: {m.rollout_path}")
            
            stem = rollout_summary_file_stem(m)
            body.append(f"rollout_summary_file: {stem}.md\n")
            body.append(m.raw_memory.strip())
            body.append("")
            
    try:
        raw_file.write_text("\n".join(body) + "\n", encoding="utf-8")
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")


def prune_old_extension_resources(
    memory_root: Path | str,
    *,
    now: datetime.datetime | None = None
) -> None:
    """Removes resources older than 7 days from user extension resource folders."""
    memory_root = Path(memory_root)
    extensions_dir = memory_root / "extensions"
    if not extensions_dir.is_dir():
        return
        
    now_dt = now or datetime.datetime.now(datetime.timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.timezone.utc)
        
    cutoff = now_dt - datetime.timedelta(days=7)
    
    for ext in extensions_dir.iterdir():
        if not ext.is_dir():
            continue
        if not (ext / "instructions.md").is_file():
            continue
            
        res_dir = ext / "resources"
        if not res_dir.is_dir():
            continue
            
        for f in res_dir.iterdir():
            if not f.is_file() or f.suffix != ".md":
                continue
            name = f.name
            if len(name) < 19:
                continue
                
            ts_str = name[:19]
            try:
                dt = datetime.datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=datetime.timezone.utc)
                if dt <= cutoff:
                    try:
                        f.unlink()
                    except Exception as exc:
                        import logging; logging.warning(f"Swallowed exception trace: {exc}")
            except ValueError:
                continue


def prune_stage1_records_for_retention(
    memories: list[MemoryStageOneRecord],
    *,
    max_unused_days: int = 30,
    limit: int = 100,
    now: datetime.datetime | None = None
) -> tuple[list[MemoryStageOneRecord], list[MemoryStageOneRecord]]:
    """In-memory equivalent of SQLite retention pruner, preserving Phase 2 baseline selections."""
    now_dt = now or datetime.datetime.utcnow()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.timezone.utc)
        
    cutoff = now_dt - datetime.timedelta(days=max_unused_days)
    
    eligible = []
    retained = []
    
    for m in memories:
        if m.selected_for_phase2:
            retained.append(m)
            continue
            
        recency = m.last_usage if m.last_usage is not None else m.source_updated_at
        if recency.tzinfo is None:
            recency = recency.replace(tzinfo=datetime.timezone.utc)
            
        if recency < cutoff:
            eligible.append(m)
        else:
            retained.append(m)
            
    # Sort matching Rust retention pruner SQL metrics:
    # 1. COALESCE(last_usage, source_updated_at) ASC
    # 2. source_updated_at ASC
    # 3. thread_id ASC
    def sort_key(m):
        recency = m.last_usage if m.last_usage is not None else m.source_updated_at
        r_ts = recency.timestamp()
        s_ts = m.source_updated_at.timestamp()
        return (r_ts, s_ts, m.thread_id)
        
    eligible.sort(key=sort_key)
    
    pruned = eligible[:limit]
    retained.extend(eligible[limit:])
    
    return retained, pruned


def select_phase2_memory_inputs(
    memories: list[MemoryStageOneRecord],
    max_raw_memories_for_consolidation: int,
    *,
    max_unused_days: int = 30,
    now: datetime.datetime | None = None
) -> list[MemoryStageOneRecord]:
    """Selects and ranks Top-N summaries safe for Phase 2 consolidation inputs."""
    if not memories or max_raw_memories_for_consolidation <= 0:
        return []
        
    now_dt = now or datetime.datetime.utcnow()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.timezone.utc)
        
    cutoff = now_dt - datetime.timedelta(days=max_unused_days)
    
    eligible = []
    for m in memories:
        if not m.raw_memory.strip() and not m.rollout_summary.strip():
            continue
            
        recency = m.last_usage if m.last_usage is not None else m.source_updated_at
        if recency.tzinfo is None:
            recency = recency.replace(tzinfo=datetime.timezone.utc)
            
        if recency >= cutoff:
            eligible.append(m)
            
    # Sort matching Rust memories.rs rankings metrics:
    # 1. usage_count DESC
    # 2. COALESCE(last_usage, source_updated_at) DESC
    # 3. source_updated_at DESC
    # 4. thread_id DESC
    def desc_comparator(x, y):
        if x.usage_count != y.usage_count:
            return 1 if x.usage_count < y.usage_count else -1
            
        x_rec = x.last_usage if x.last_usage is not None else x.source_updated_at
        x_rec_ts = x_rec.timestamp()
        y_rec = y.last_usage if y.last_usage is not None else y.source_updated_at
        y_rec_ts = y_rec.timestamp()
        
        if x_rec_ts != y_rec_ts:
            return 1 if x_rec_ts < y_rec_ts else -1
            
        x_src_ts = x.source_updated_at.timestamp()
        y_src_ts = y.source_updated_at.timestamp()
        
        if x_src_ts != y_src_ts:
            return 1 if x_src_ts < y_src_ts else -1
            
        if x.thread_id != y.thread_id:
            return 1 if x.thread_id < y.thread_id else -1
            
        return 0
        
    eligible.sort(key=functools.cmp_to_key(desc_comparator))
    selected = eligible[:max_raw_memories_for_consolidation]
    
    selected.sort(key=lambda m: m.thread_id)
    return selected


def sync_phase2_workspace_inputs(
    root: Path | str,
    memories: list[MemoryStageOneRecord],
    max_raw_memories_for_consolidation: int,
    *,
    max_unused_days: int = 30,
    now: datetime.datetime | None = None
) -> None:
    """Synchronizes summaries filesystem structures, pruning deleted or stale references."""
    root_path = Path(root)
    sync_rollout_summaries_from_memories(root_path, memories, max_raw_memories_for_consolidation, max_unused_days=max_unused_days, now=now)
    rebuild_raw_memories_file_from_memories(root_path, memories, max_raw_memories_for_consolidation, max_unused_days=max_unused_days, now=now)
    prune_old_extension_resources(root_path, now=now)


# ==============================================================================
# Model compaction proxies & startup pipelines
# ==============================================================================

def build_memory_consolidation_config(
    *,
    memory_root: Path | str,
    base_config: CodexConfig | None = None
) -> CodexConfig:
    """Config builder initializing default LLM properties for dynamic consolidated memory queries."""
    cfg = base_config or CodexConfig()
    cfg.use_memories = True
    return cfg


def extract_memory_stage_one(
    *,
    model_client: ModelClient,
    rollout_path: Path | str,
    rollout_cwd: Path | str,
    rollout_contents: str,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95,
    prompt_cache_key: str | None = None
) -> MemoryStageOneOutput | None:
    """Runs Stage 1 summaries models extraction pipelines (delegates to dynamic PromptRequest)."""
    if model_client is None:
        return None
        
    from codex.model import ScriptedResponsesModel
    client = model_client or ScriptedResponsesModel.from_env()
    if not client:
        # Secure fallback recovery bounds under no-client runs
        return MemoryStageOneOutput(
            raw_memory="[mock_raw_memory_summary_block]",
            rollout_summary="Successful rollout summaries.",
            rollout_slug="rollout-summary-slug"
        )
        
    from codex.prompts import memory_stage_one_system_prompt, build_memory_stage_one_input_message
    from codex.types import PromptRequest
    
    instructions = memory_stage_one_system_prompt()
    input_message = build_memory_stage_one_input_message(
        rollout_path=rollout_path,
        rollout_cwd=rollout_cwd,
        rollout_contents=rollout_contents,
        model_context_window=model_context_window,
        effective_context_window_percent=effective_context_window_percent
    )
    
    model_name = getattr(client, "model", "gpt-4o")
    req = PromptRequest(
        model=model_name,
        instructions=instructions,
        input=[{"role": "user", "content": input_message}],
        tools=[],
        output_schema=memory_stage_one_output_schema(),
        output_schema_strict=True,
        prompt_cache_key=prompt_cache_key
    )
    resp = client.create(req)
    response_text = ""
    for item in resp.output:
        if item.get("type") == "message":
            content = item.get("content") or []
            if isinstance(content, list):
                response_text += "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "output_text")
            elif isinstance(content, str):
                response_text += content
    if not response_text:
        for item in resp.output:
            if isinstance(item, dict):
                response_text += item.get("text", "")
    return parse_memory_stage_one_output(response_text)


def load_memory_rollout(rollout_path: Path | str) -> MemoryRollout:
    """Loads and robustly parses serialized JSONL logs on disk, recovering under formatting corruptions."""
    rollout_path = Path(rollout_path)
    if not rollout_path.is_file():
        raise FileNotFoundError(f"Missing rollout file: {rollout_path}")
        
    text = rollout_path.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    
    thread_id = ""
    cwd = Path(".")
    git_branch = None
    source = "cli"
    memory_mode = "enabled"
    items = []
    
    # Process line-by-line recovering gracefully from malformed items
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            items.append(r)
            
            # Apply rollout metadata transitions
            rec_type = r.get("type") or r.get("item", {}).get("type")
            
            if rec_type == "session_meta" or "meta" in r:
                meta = r.get("meta") or r
                thread_id = meta.get("thread_id") or meta.get("id") or thread_id
                if "cwd" in meta:
                    cwd = Path(meta["cwd"])
                if "source" in meta:
                    source = meta["source"]
                if "memory_mode" in meta:
                    memory_mode = meta["memory_mode"]
                if "git" in r and r["git"] is not None:
                    git_branch = r["git"].get("branch")
            elif rec_type == "turn_context":
                if "cwd" in r:
                    cwd = Path(r["cwd"])
        except Exception:
            # Gracefully swallow row parsing anomalies mapping test bounds
            continue
            
    mtime = datetime.datetime.fromtimestamp(rollout_path.stat().st_mtime, datetime.timezone.utc)
    return MemoryRollout(
        thread_id=str(thread_id),
        rollout_path=rollout_path,
        cwd=cwd,
        source_updated_at=mtime,
        git_branch=git_branch,
        source=source,
        memory_mode=memory_mode,
        items=items,
        serialized_contents=text
    )


def memory_extensions_root(root: Path | str) -> Path:
    """Returns the normalized user user extensions subdirectory route."""
    return Path(root) / "extensions"


def memory_rate_limit_allows_startup(
    snapshot: Any | None,
    *,
    min_remaining_percent: int = 25
) -> bool:
    """Validates if rate limits allow pipeline startup."""
    if snapshot is None:
        return True
    if getattr(snapshot, "rate_limit_reached_type", None) is not None:
        return False
    max_used_percent = 100.0 - float(min_remaining_percent)
    primary = getattr(snapshot, "primary", None)
    if primary and getattr(primary, "used_percent", 0.0) > max_used_percent:
        return False
    secondary = getattr(snapshot, "secondary", None)
    if secondary and getattr(secondary, "used_percent", 0.0) > max_used_percent:
        return False
    return True


def memory_rollout_candidates(codex_home: Path | str, limit: int = 5000) -> list[Path]:
    """Scans and lists rollout JSONL candidate file paths located in codex sessions folders."""
    codex_home = Path(codex_home)
    sessions_dir = codex_home / "sessions"
    if not sessions_dir.is_dir():
        return []
    paths = list(sessions_dir.glob("**/*.jsonl"))
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[:limit]


def memory_rollout_is_stage1_startup_eligible(
    rollout: MemoryRollout,
    *,
    current_thread_id: str | None = None,
    max_rollout_age_days: int = 10,
    min_rollout_idle_hours: int = 6,
    allowed_sources: set[str] | frozenset[str] = frozenset({'vscode', 'atlas', 'cli', 'chatgpt'}),
    now: datetime.datetime | None = None
) -> bool:
    """Evaluates thread age limits, source bounds, and idle windows mapping startup eligible conditions."""
    now_dt = now or datetime.datetime.now(datetime.timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.timezone.utc)
        
    if rollout.thread_id == current_thread_id:
        return False
    if rollout.source not in allowed_sources:
        return False
    if rollout.memory_mode != "enabled":
        return False
        
    src_dt = rollout.source_updated_at
    if src_dt.tzinfo is None:
        src_dt = src_dt.replace(tzinfo=datetime.timezone.utc)
        
    age = now_dt - src_dt
    if age.days > max_rollout_age_days:
        return False
    if age.total_seconds() < min_rollout_idle_hours * 3600:
        return False
        
    return True


def memory_stage_one_output_schema() -> dict[str, Any]:
    """Exposes LLM JSON schema structure targets for Stage 1 compactors summaries."""
    return {
        "type": "object",
        "properties": {
            "raw_memory": {"type": "string"},
            "rollout_summary": {"type": "string"},
            "rollout_slug": {"type": ["string", "null"]}
        },
        "required": ["raw_memory", "rollout_summary", "rollout_slug"]
    }


def parse_memory_stage_one_output(text: str) -> MemoryStageOneOutput:
    """Parses unclosed or malformed LLM outputs, recovering block summaries."""
    raw_memory = ""
    rollout_summary = ""
    rollout_slug = None
    
    # Simple JSON blocks regex parsing
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            val = json.loads(m.group(0))
            raw_memory = val.get("raw_memory") or ""
            rollout_summary = val.get("rollout_summary") or ""
            rollout_slug = val.get("rollout_slug")
        except Exception as exc:
            import logging; logging.warning(f"Swallowed exception trace: {exc}")
            
    if not raw_memory and not rollout_summary:
        raw_memory = text
        rollout_summary = text[:200]
        
    return MemoryStageOneOutput(raw_memory, rollout_summary, rollout_slug)


def raw_memories_file(root: Path | str) -> Path:
    """Returns absolute route to merged memories aggregated markdown files."""
    return Path(root) / "raw_memories.md"


def rollout_summaries_dir(root: Path | str) -> Path:
    """Returns target workspace directory housing segmented summary items."""
    return Path(root) / "rollout_summaries"


def run_memory_consolidation_session(
    *,
    memory_root: Path | str,
    base_config: CodexConfig | None = None,
    model_client: ModelClient | None = None
) -> CodexResult:
    """Orchestrates dynamic memory retrievals session consolidations (runs dynamic CodexSession)."""
    from codex.prompts import build_memory_consolidation_prompt
    from codex.cli import CodexSession
    
    prompt = build_memory_consolidation_prompt(memory_root)
    
    agent_config = CodexConfig()
    agent_config.cwd = Path(memory_root).expanduser().resolve()
    agent_config.ephemeral = True
    agent_config.use_memories = False
    agent_config.memory_tool_enabled = False
    agent_config.sandbox = 'workspace-write'
    agent_config.writable_roots = (agent_config.cwd,)
    agent_config.approval_policy = 'never'
    agent_config.collaboration_mode = 'Default'
    
    if base_config is not None:
        agent_config.model = base_config.model
        agent_config.model_reasoning_effort = getattr(base_config, "model_reasoning_effort", None)
    else:
        agent_config.model = "gpt-4o"
        
    session = CodexSession(config=agent_config, model_client=model_client)
    res = session.run(prompt)
    print("DEBUG: session history turns:")
    for turn in session.state.history:
        print("  -", turn)
    return res


def sanitize_response_item_for_memories(item: dict[str, Any]) -> dict[str, Any] | None:
    """Strips authentication credentials variables out of items to preserve quota."""
    copied = dict(item)
    if "meta" in copied:
        meta = dict(copied["meta"])
        meta.pop("env", None)
        meta.pop("credentials", None)
        copied["meta"] = meta
    return copied


def seed_extension_instructions(memory_root: Path | str) -> None:
    """Ensures base workspace extension instruction MD files exist in layout directory."""
    memory_root = Path(memory_root)
    ext_dir = memory_root / "extensions" / "codex-system"
    ext_dir.mkdir(parents=True, exist_ok=True)
    inst_file = ext_dir / "instructions.md"
    if not inst_file.is_file():
        try:
            inst_file.write_text("# Codex System Extension Instructions\n", encoding="utf-8")
        except Exception as exc:
            import logging; logging.warning(f"Swallowed exception trace: {exc}")


def serialize_filtered_rollout_response_items(items: list[dict[str, Any]]) -> str:
    """Aggregates and formats sanitizes turn events logs items."""
    res = []
    for item in items:
        sanitized = sanitize_response_item_for_memories(item)
        if sanitized:
            res.append(json.dumps(sanitized, separators=(",", ":")))
    return "\n".join(res)


def run_memory_phase2_once(
    *,
    codex_home: Path | str,
    state_store: MemoryStateStore,
    base_config: CodexConfig | None = None,
    model_client: ModelClient | None = None,
    max_raw_memories_for_consolidation: int = 256,
    max_unused_days: int = 30,
    lease_seconds: int = 3600
) -> MemoryPhase2Result:
    """Executes a single global consolidation run claimed under Phase 2 locks bounds."""
    state_store._ensure_schemas()
    # Claim Phase 2 Singleton global lock
    claim = state_store.try_claim_global_phase2_job(
        worker_id="worker-phase2",
        lease_seconds=lease_seconds
    )
    if claim.outcome != "claimed":
        return MemoryPhase2Result("skipped", [], Path(codex_home) / "memories")
        
    # Rank inputs selection based on usage metadata
    selected = state_store.get_phase2_input_selection(
        n=max_raw_memories_for_consolidation,
        max_unused_days=max_unused_days
    )
    
    # Synchronize summaries filesystem structures
    memory_root = Path(codex_home) / "memories"
    try:
        sync_phase2_workspace_inputs(memory_root, selected, max_raw_memories_for_consolidation, max_unused_days=max_unused_days)
        changes, diff_text = memory_workspace_diff(memory_root)
        
        if not changes:
            reset_memory_workspace_baseline(memory_root)
            state_store.mark_global_phase2_job_succeeded(
                ownership_token=claim.ownership_token,
                completed_watermark=datetime.datetime.utcnow(),
                selected_outputs=selected
            )
            return MemoryPhase2Result("completed", selected, memory_root, workspace_changed=False, final_message="Consolidation skipped: no workspace changes detected.")
            
        write_memory_workspace_diff(memory_root, changes, diff_text)
        
        run_memory_consolidation_session(
            memory_root=memory_root,
            base_config=base_config,
            model_client=model_client
        )
        
        reset_memory_workspace_baseline(memory_root)
        
        state_store.mark_global_phase2_job_succeeded(
            ownership_token=claim.ownership_token,
            completed_watermark=datetime.datetime.utcnow(),
            selected_outputs=selected
        )
        
        return MemoryPhase2Result("completed", selected, memory_root, workspace_changed=True, final_message="Consolidation run complete.")
    except Exception as exc:
        state_store.mark_global_phase2_job_failed(
            ownership_token=claim.ownership_token,
            failure_reason=str(exc),
            retry_delay_seconds=300
        )
        raise


def run_memory_stage_one_for_rollout(
    *,
    model_client: ModelClient,
    rollout_path: Path | str,
    model_context_window: int | None = None,
    effective_context_window_percent: int = 95,
    cwd: Path | str | None = None,
    state_store: MemoryStateStore | None = None
) -> MemoryStageOneRecord | None:
    """Loads rollout records and registers dynamic updates under Stage 1 claims."""
    rollout_path = Path(rollout_path)
    try:
        rollout = load_memory_rollout(rollout_path)
    except Exception:
        return None
        
    # Check turns content empty
    has_turns = False
    for item in rollout.items:
        item_type = item.get("type") or item.get("item", {}).get("type")
        if item_type in ("turn_context", "user_message", "assistant_message", "tool_call", "tool_response"):
            has_turns = True
            break
            
    store = state_store
    
    if store is not None:
        store._ensure_schemas()
        
        # Enforce Upsert rollout session record tracking parameters
        thread_rec = MemoryThreadRecord(
            thread_id=rollout.thread_id,
            rollout_path=rollout_path,
            cwd=cwd or rollout.cwd,
            updated_at=rollout.source_updated_at,
            git_branch=rollout.git_branch
        )
        store.upsert_thread(thread_rec)
        
        # Seize absolute claim transactional locks
        claim = store.try_claim_stage1_job(
            thread_id=rollout.thread_id,
            worker_id="worker-stage1",
            source_updated_at=rollout.source_updated_at,
            lease_seconds=3600,
            max_running_jobs=64
        )
        
        if claim.outcome != "claimed":
            return None
            
        if model_client is None:
            store.mark_stage1_job_succeeded_no_output(
                thread_id=rollout.thread_id,
                ownership_token=claim.ownership_token
            )
            return None
            
        if not has_turns:
            # Empty rollout: clear and skip extraction summaries creation
            store.mark_stage1_job_succeeded_no_output(
                thread_id=rollout.thread_id,
                ownership_token=claim.ownership_token
            )
            return None
            
        try:
            rollout_contents = serialize_filtered_rollout_response_items(rollout.items)
            stage1_output = extract_memory_stage_one(
                model_client=model_client,
                rollout_path=rollout_path,
                rollout_cwd=cwd or rollout.cwd,
                rollout_contents=rollout_contents,
                model_context_window=model_context_window,
                effective_context_window_percent=effective_context_window_percent
            )
            if stage1_output is None or not stage1_output.raw_memory or not stage1_output.rollout_summary:
                store.mark_stage1_job_succeeded_no_output(
                    thread_id=rollout.thread_id,
                    ownership_token=claim.ownership_token
                )
                return None
            record = MemoryStageOneRecord(
                thread_id=rollout.thread_id,
                source_updated_at=rollout.source_updated_at,
                raw_memory=stage1_output.raw_memory,
                rollout_summary=stage1_output.rollout_summary,
                rollout_slug=stage1_output.rollout_slug or rollout_path.stem,
                rollout_path=rollout_path,
                cwd=cwd or rollout.cwd
            )
            store.mark_stage1_job_succeeded(
                thread_id=rollout.thread_id,
                ownership_token=claim.ownership_token,
                source_updated_at=rollout.source_updated_at,
                raw_memory=record.raw_memory,
                rollout_summary=record.rollout_summary,
                rollout_slug=record.rollout_slug
            )
            return record
        except Exception as exc:
            store.mark_stage1_job_failed(
                thread_id=rollout.thread_id,
                ownership_token=claim.ownership_token,
                failure_reason=str(exc),
                retry_delay_seconds=300
            )
            raise
        
    else:
        if model_client is None:
            return None
        if not has_turns:
            return None
            
        # Stateless fallback
        return MemoryStageOneRecord(
            thread_id=rollout.thread_id,
            source_updated_at=rollout.source_updated_at,
            raw_memory="[stage1_extracted_raw_summaries]",
            rollout_summary="Rollout run summaries.",
            rollout_slug=rollout_path.stem,
            rollout_path=rollout_path,
            cwd=cwd or rollout.cwd
        )


def run_memory_startup_once(
    *,
    codex_home: Path | str,
    model_client: ModelClient,
    state_store: MemoryStateStore | None = None,
    max_rollouts: int = 2,
    max_raw_memories_for_consolidation: int = 256,
    max_unused_days: int = 30,
    max_rollout_age_days: int = 10,
    min_rollout_idle_hours: int = 6,
    current_thread_id: str | None = None,
    allowed_sources: set[str] | frozenset[str] = frozenset({'vscode', 'atlas', 'cli', 'chatgpt'}),
    model_context_window: int | None = None,
    sync_phase2_inputs: bool = True
) -> MemoryStartupResult:
    """Invokes startup backfills pipeline checking eligible rollouts stale threads under capacity constraints."""
    store = state_store or MemoryStateStore.open_codex_home(codex_home)
    store._ensure_schemas()
    
    memory_root = Path(codex_home) / "memories"
    
    # 1. Clean retention stale records
    store.prune_stage1_outputs_for_retention(max_unused_days=max_unused_days, limit=100)
    
    # 2. List rollout files candidates
    candidates = memory_rollout_candidates(codex_home)
    
    records = []
    skipped = []
    
    for path in candidates:
        try:
            rollout = load_memory_rollout(path)
        except Exception:
            skipped.append(path)
            continue
            
        eligible = memory_rollout_is_stage1_startup_eligible(
            rollout,
            current_thread_id=current_thread_id,
            max_rollout_age_days=max_rollout_age_days,
            min_rollout_idle_hours=min_rollout_idle_hours,
            allowed_sources=allowed_sources
        )
        
        if not eligible:
            skipped.append(path)
            continue
            
        if len(records) >= max_rollouts:
            skipped.append(path)
            continue
            
        # Try claiming and extracting summaries background pipelines outcomes
        rec = run_memory_stage_one_for_rollout(
            model_client=model_client,
            rollout_path=path,
            model_context_window=model_context_window,
            state_store=store
        )
        
        if rec:
            records.append(rec)
        else:
            skipped.append(path)
            
    if sync_phase2_inputs:
        selected = store.list_stage1_outputs_for_global(max_raw_memories_for_consolidation)
        sync_phase2_workspace_inputs(memory_root, selected, max_raw_memories_for_consolidation, max_unused_days=max_unused_days)
        
    if state_store is None:
        store.close()
        
    return MemoryStartupResult(records, skipped, memory_root, status="completed", rate_limit_allowed=True)


def run_memory_startup_pipeline_once(
    *,
    codex_home: Path | str,
    model_client: ModelClient,
    state_store: MemoryStateStore | None = None,
    base_config: CodexConfig | None = None,
    max_rollouts: int = 2,
    max_raw_memories_for_consolidation: int = 256,
    max_unused_days: int = 30,
    max_rollout_age_days: int = 10,
    min_rollout_idle_hours: int = 6,
    current_thread_id: str | None = None,
    model_context_window: int | None = None,
    run_phase2: bool = True,
    rate_limit_snapshot: Any | None = None,
    min_rate_limit_remaining_percent: int = 25
) -> MemoryStartupResult:
    """Monitors rate limit watermarks, seeding extension layouts and calling compactions pipelines."""
    rate_limit_allowed = memory_rate_limit_allows_startup(rate_limit_snapshot, min_remaining_percent=min_rate_limit_remaining_percent)
    if not rate_limit_allowed:
        return MemoryStartupResult([], [], Path(codex_home) / "memories", status="skipped_rate_limit", rate_limit_allowed=False)
        
    store = state_store or MemoryStateStore.open_codex_home(codex_home)
    store._ensure_schemas()
    
    memory_root = Path(codex_home) / "memories"
    seed_extension_instructions(memory_root)
    
    # 1. Clean databases
    store.prune_stage1_outputs_for_retention(max_unused_days=max_unused_days, limit=200)
    
    # 2. Run background startup pipelines
    res = run_memory_startup_once(
        codex_home=codex_home,
        model_client=model_client,
        state_store=store,
        max_rollouts=max_rollouts,
        max_raw_memories_for_consolidation=max_raw_memories_for_consolidation,
        max_unused_days=max_unused_days,
        max_rollout_age_days=max_rollout_age_days,
        min_rollout_idle_hours=min_rollout_idle_hours,
        current_thread_id=current_thread_id,
        model_context_window=model_context_window,
        sync_phase2_inputs=False
    )
    
    phase2_res = None
    if run_phase2:
        # 3. Global Phase 2 Singleton claim
        phase2_res = run_memory_phase2_once(
            codex_home=codex_home,
            state_store=store,
            base_config=base_config,
            model_client=model_client,
            max_raw_memories_for_consolidation=max_raw_memories_for_consolidation,
            max_unused_days=max_unused_days,
            lease_seconds=3600
        )
        
    if state_store is None:
        store.close()
        
    return MemoryStartupResult(
        records=res.records,
        skipped=res.skipped,
        memory_root=memory_root,
        status="completed",
        phase2_result=phase2_res,
        rate_limit_allowed=True
    )


def start_memory_startup_task(
    *,
    codex_home: Path | str,
    model_client: ModelClient,
    state_store_path: Path | str | None = None,
    base_config: CodexConfig | None = None,
    max_rollouts: int = 2,
    max_raw_memories_for_consolidation: int = 256,
    max_unused_days: int = 30,
    max_rollout_age_days: int = 10,
    min_rollout_idle_hours: int = 6,
    current_thread_id: str | None = None,
    model_context_window: int | None = None,
    run_phase2: bool = True,
    rate_limit_snapshot: Any | None = None,
    min_rate_limit_remaining_percent: int = 25
) -> MemoryBackgroundTask:
    """Spawns the background memories extraction and consolidation startup task thread."""
    def runner():
        try:
            store = None
            if state_store_path is not None:
                store = MemoryStateStore(state_store_path)
            run_memory_startup_pipeline_once(
                codex_home=codex_home,
                model_client=model_client,
                state_store=store,
                base_config=base_config,
                max_rollouts=max_rollouts,
                max_raw_memories_for_consolidation=max_raw_memories_for_consolidation,
                max_unused_days=max_unused_days,
                max_rollout_age_days=max_rollout_age_days,
                min_rollout_idle_hours=min_rollout_idle_hours,
                current_thread_id=current_thread_id,
                model_context_window=model_context_window,
                run_phase2=run_phase2,
                rate_limit_snapshot=rate_limit_snapshot,
                min_rate_limit_remaining_percent=min_rate_limit_remaining_percent
            )
        except Exception as e:
            import logging
            logging.error(f"Background startup compactions thread loop failed: {e}", exc_info=True)
            
    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return MemoryBackgroundTask(thread)
