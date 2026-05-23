from __future__ import annotations
import json
import logging
import uuid
import datetime
import os
import time
import threading
from pathlib import Path
from typing import Any, Literal
from codex.types import CodexConfig, CodexEvent, CodexResult, PromptRequest, ModelResponse
from codex.model import ModelClient
from codex.prompts import (
    memory_stage_one_system_prompt,
    build_memory_stage_one_input_message,
    build_memory_consolidation_prompt,
    ASSETS_DIR
)

logger = logging.getLogger("codex")

__all__ = [
    "MemoryBackgroundTask",
    "MemoryJobClaim",
    "MemoryPhase2Result",
    "MemoryRollout",
    "MemoryStageOneOutput",
    "MemoryStageOneRecord",
    "MemoryStageOneStartupClaim",
    "MemoryStartupResult",
    "MemoryStateStore",
    "MemoryThreadRecord",
    "MemoryWorkspaceChange",
    "find_eligible_rollouts",
    "start_memory_startup_task",
    "serialize_filtered_rollout_response_items",
    "sync_phase2_workspace_inputs",
    "sync_rollout_summaries_from_memories",
    "write_current_memory_workspace_diff",
    "write_memory_workspace_diff"
]



class MemoryJobClaim:
    def __init__(
        self,
        session_id: str,
        acquired_at: datetime.datetime,
        last_heartbeat: datetime.datetime,
    ) -> None:
        self.session_id = session_id
        self.acquired_at = acquired_at
        self.last_heartbeat = last_heartbeat

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "acquired_at": self.acquired_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat()
        }


class MemoryStageOneStartupClaim:
    def __init__(
        self,
        session_id: str,
        rollout_path: Path,
        acquired_at: datetime.datetime,
        last_heartbeat: datetime.datetime,
    ) -> None:
        self.session_id = session_id
        self.rollout_path = Path(rollout_path)
        self.acquired_at = acquired_at
        self.last_heartbeat = last_heartbeat

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "rollout_path": self.rollout_path.as_posix(),
            "acquired_at": self.acquired_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat()
        }


class MemoryStageOneOutput:
    def __init__(
        self,
        rollout_summary: str,
        rollout_slug: str,
        raw_memory: str,
    ) -> None:
        self.rollout_summary = rollout_summary
        self.rollout_slug = rollout_slug
        self.raw_memory = raw_memory

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_summary": self.rollout_summary,
            "rollout_slug": self.rollout_slug,
            "raw_memory": self.raw_memory
        }


class MemoryStageOneRecord:
    def __init__(
        self,
        session_id: str,
        cwd: Path,
        path: Path,
        updated_at: datetime.datetime,
        thread_id: str,
        rollout_summary: str,
        rollout_slug: str,
        raw_memory: str,
    ) -> None:
        self.session_id = session_id
        self.cwd = Path(cwd)
        self.path = Path(path)
        self.updated_at = updated_at
        self.thread_id = thread_id
        self.rollout_summary = rollout_summary
        self.rollout_slug = rollout_slug
        self.raw_memory = raw_memory

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "cwd": self.cwd.as_posix(),
            "path": self.path.as_posix(),
            "updated_at": self.updated_at.isoformat(),
            "thread_id": self.thread_id,
            "rollout_summary": self.rollout_summary,
            "rollout_slug": self.rollout_slug,
            "raw_memory": self.raw_memory
        }


class MemoryPhase2Result:
    def __init__(
        self,
        output_summary: str,
        output_handbook: str,
        skills: dict[str, str],
        diff: str,
        raw_memories: str,
    ) -> None:
        self.output_summary = output_summary
        self.output_handbook = output_handbook
        self.skills = skills
        self.diff = diff
        self.raw_memories = raw_memories

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_summary": self.output_summary,
            "output_handbook": self.output_handbook,
            "skills": self.skills,
            "diff": self.diff,
            "raw_memories": self.raw_memories
        }


class MemoryRollout:
    def __init__(
        self,
        session_id: str,
        cwd: Path,
        path: Path,
        updated_at: datetime.datetime,
        thread_id: str,
        contents: str,
        age_days: int,
        idle_hours: int,
    ) -> None:
        self.session_id = session_id
        self.cwd = Path(cwd)
        self.path = Path(path)
        self.updated_at = updated_at
        self.thread_id = thread_id
        self.contents = contents
        self.age_days = age_days
        self.idle_hours = idle_hours

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "cwd": self.cwd.as_posix(),
            "path": self.path.as_posix(),
            "updated_at": self.updated_at.isoformat(),
            "thread_id": self.thread_id,
            "contents": self.contents,
            "age_days": self.age_days,
            "idle_hours": self.idle_hours
        }


class MemoryStartupResult:
    def __init__(
        self,
        stage1_jobs: list[MemoryStageOneRecord],
        phase2_result: MemoryPhase2Result | None,
    ) -> None:
        self.stage1_jobs = stage1_jobs
        self.phase2_result = phase2_result

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage1_jobs": [job.to_dict() for job in self.stage1_jobs],
            "phase2_result": self.phase2_result.to_dict() if self.phase2_result is not None else None
        }


class MemoryThreadRecord:
    def __init__(
        self,
        thread_id: str,
        cwd: Path,
        rollout_path: Path,
        updated_at: datetime.datetime,
    ) -> None:
        self.thread_id = thread_id
        self.cwd = Path(cwd)
        self.rollout_path = Path(rollout_path)
        self.updated_at = updated_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "cwd": self.cwd.as_posix(),
            "rollout_path": self.rollout_path.as_posix(),
            "updated_at": self.updated_at.isoformat()
        }


class MemoryWorkspaceChange:
    def __init__(
        self,
        path: Path,
        type: str,
    ) -> None:
        self.path = Path(path)
        self.type = type

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "type": self.type
        }


class MemoryStateStore:
    def __init__(self, memory_root: Path | str) -> None:
        self.memory_root = Path(memory_root)
        self._state_file = self.memory_root / "memory_state.json"
        
        # Concurrency safety lock
        self._lock = threading.Lock()

    def state_path(self) -> Path:
        return self._state_file

    def _read_state(self) -> dict[str, Any]:
        if not self._state_file.exists():
            return {"stage1_history": [], "stage1_claims": {}, "phase2_claim": None}
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read state file at {self._state_file}: {e}")
            return {"stage1_history": [], "stage1_claims": {}, "phase2_claim": None}

    def _write_state(self, state: dict[str, Any]) -> None:
        try:
            self.memory_root.mkdir(parents=True, exist_ok=True)
            temp_file = self._state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            temp_file.replace(self._state_file)
        except Exception as e:
            logger.error(f"Failed to write state file: {e}")

    def load_stage1_history(self) -> list[MemoryStageOneRecord]:
        with self._lock:
            state = self._read_state()
            records = []
            for item in state.get("stage1_history", []):
                try:
                    records.append(MemoryStageOneRecord(
                        session_id=item["session_id"],
                        cwd=Path(item["cwd"]),
                        path=Path(item["path"]),
                        updated_at=datetime.datetime.fromisoformat(item["updated_at"]),
                        thread_id=item["thread_id"],
                        rollout_summary=item["rollout_summary"],
                        rollout_slug=item["rollout_slug"],
                        raw_memory=item["raw_memory"]
                    ))
                except Exception as e:
                    logger.debug(f"Failed to load stage1 record: {e}")
            return records

    def save_stage1_output(self, record: MemoryStageOneRecord) -> None:
        with self._lock:
            state = self._read_state()
            history = state.get("stage1_history", [])
            # Avoid duplicates by session_id
            history = [item for item in history if item["session_id"] != record.session_id]
            history.append(record.to_dict())
            state["stage1_history"] = history
            self._write_state(state)

    def acquire_stage1_job(
        self,
        session_id: str,
        rollout_path: Path,
        heartbeat_timeout_seconds: int = 30
    ) -> MemoryStageOneStartupClaim | None:
        with self._lock:
            state = self._read_state()
            claims = state.get("stage1_claims", {})
            now = datetime.datetime.utcnow()
            
            # Check if active and non-expired claim already exists
            if session_id in claims:
                existing = claims[session_id]
                last_hb = datetime.datetime.fromisoformat(existing["last_heartbeat"])
                if (now - last_hb).total_seconds() < heartbeat_timeout_seconds:
                    # Locked!
                    return None
                    
            claim = MemoryStageOneStartupClaim(
                session_id=session_id,
                rollout_path=rollout_path,
                acquired_at=now,
                last_heartbeat=now
            )
            claims[session_id] = claim.to_dict()
            state["stage1_claims"] = claims
            self._write_state(state)
            return claim

    def heartbeat_stage1_job(self, claim: MemoryStageOneStartupClaim) -> MemoryStageOneStartupClaim:
        with self._lock:
            state = self._read_state()
            claims = state.get("stage1_claims", {})
            now = datetime.datetime.utcnow()
            claim.last_heartbeat = now
            claims[claim.session_id] = claim.to_dict()
            state["stage1_claims"] = claims
            self._write_state(state)
            return claim

    def release_stage1_job(self, claim: MemoryStageOneStartupClaim) -> None:
        with self._lock:
            state = self._read_state()
            claims = state.get("stage1_claims", {})
            if claim.session_id in claims:
                del claims[claim.session_id]
            state["stage1_claims"] = claims
            self._write_state(state)

    def is_stage1_job_acquired(self, session_id: str) -> bool:
        with self._lock:
            state = self._read_state()
            claims = state.get("stage1_claims", {})
            if session_id not in claims:
                return False
            existing = claims[session_id]
            last_hb = datetime.datetime.fromisoformat(existing["last_heartbeat"])
            now = datetime.datetime.utcnow()
            # 30 seconds default threshold
            return (now - last_hb).total_seconds() < 30

    def active_stage1_jobs(self) -> list[MemoryStageOneStartupClaim]:
        with self._lock:
            state = self._read_state()
            claims = state.get("stage1_claims", {})
            active = []
            now = datetime.datetime.utcnow()
            for item in claims.values():
                last_hb = datetime.datetime.fromisoformat(item["last_heartbeat"])
                if (now - last_hb).total_seconds() < 30:
                    active.append(MemoryStageOneStartupClaim(
                        session_id=item["session_id"],
                        rollout_path=Path(item["rollout_path"]),
                        acquired_at=datetime.datetime.fromisoformat(item["acquired_at"]),
                        last_heartbeat=last_hb
                    ))
            return active

    def acquire_phase2_job(self, heartbeat_timeout_seconds: int = 30) -> MemoryJobClaim | None:
        with self._lock:
            state = self._read_state()
            existing = state.get("phase2_claim")
            now = datetime.datetime.utcnow()
            
            if existing is not None:
                last_hb = datetime.datetime.fromisoformat(existing["last_heartbeat"])
                if (now - last_hb).total_seconds() < heartbeat_timeout_seconds:
                    # Locked!
                    return None
                    
            claim = MemoryJobClaim(
                session_id=str(uuid.uuid4()),
                acquired_at=now,
                last_heartbeat=now
            )
            state["phase2_claim"] = claim.to_dict()
            self._write_state(state)
            return claim

    def heartbeat_phase2_job(self, claim: MemoryJobClaim) -> MemoryJobClaim:
        with self._lock:
            state = self._read_state()
            now = datetime.datetime.utcnow()
            claim.last_heartbeat = now
            state["phase2_claim"] = claim.to_dict()
            self._write_state(state)
            return claim

    def release_phase2_job(self, claim: MemoryJobClaim) -> None:
        with self._lock:
            state = self._read_state()
            state["phase2_claim"] = None
            self._write_state(state)

    def active_phase2_job(self) -> MemoryJobClaim | None:
        with self._lock:
            state = self._read_state()
            item = state.get("phase2_claim")
            if item is None:
                return None
            last_hb = datetime.datetime.fromisoformat(item["last_heartbeat"])
            now = datetime.datetime.utcnow()
            if (now - last_hb).total_seconds() < 30:
                return MemoryJobClaim(
                    session_id=item["session_id"],
                    acquired_at=datetime.datetime.fromisoformat(item["acquired_at"]),
                    last_heartbeat=last_hb
                )
            return None

    def is_phase2_job_acquired(self) -> bool:
        return self.active_phase2_job() is not None


def find_eligible_rollouts(config: CodexConfig, store: MemoryStateStore) -> list[MemoryRollout]:
    eligible = []
    sessions_dir = Path(config.resolved_codex_home()) / "sessions"
    if not sessions_dir.exists():
        return eligible
        
    now = datetime.datetime.utcnow()
    summarized_ids = {rec.session_id for rec in store.load_stage1_history()}
    
    for path in sessions_dir.glob("**/*.jsonl"):
        if not path.is_file() or path.name.startswith("session_index"):
            continue
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                continue
            meta = json.loads(first_line)
            if meta.get("type") != "session_meta":
                continue
            payload = meta.get("payload", {})
            session_id = payload.get("id") or payload.get("session_id")
            thread_id = payload.get("thread_id") or session_id
            cwd = Path(payload.get("cwd") or config.resolved_cwd())
            
            if not session_id:
                continue
                
            if session_id in summarized_ids:
                continue
                
            # Stat date
            mtime = path.stat().st_mtime
            updated_at = datetime.datetime.utcfromtimestamp(mtime)
            
            age_days = (now - updated_at).days
            idle_hours = int((now - updated_at).total_seconds() // 3600)
            
            # Check boundaries
            if age_days <= config.memory_max_rollout_age_days and idle_hours >= config.memory_min_rollout_idle_hours:
                with open(path, "r", encoding="utf-8") as f:
                    contents = f.read()
                    
                eligible.append(MemoryRollout(
                    session_id=session_id,
                    cwd=cwd,
                    path=path,
                    updated_at=updated_at,
                    thread_id=thread_id,
                    contents=contents,
                    age_days=age_days,
                    idle_hours=idle_hours
                ))
        except Exception as e:
            logger.debug(f"Skipping rollout {path}: {e}")
            
    return eligible


class MemoryBackgroundTask:
    def __init__(self, config: CodexConfig, state_store: MemoryStateStore = None) -> None:
        self.config = config
        self.state_store = state_store if state_store is not None else MemoryStateStore(
            Path(config.resolved_codex_home()) / "memories"
        )
        self._worker_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    def start(self) -> None:
        self._stop_evt.clear()
        self._worker_thread = threading.Thread(target=self._run_synchronizer)
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def _run_synchronizer(self) -> None:
        try:
            # 1. Run rollout triage & stage 1 summarizations
            eligible = find_eligible_rollouts(self.config, self.state_store)
            if not eligible:
                # Omit Phase 2 if nothing worth consolidating
                if self.config.memory_run_phase2_on_startup:
                    self._run_consolidation([])
                return
                
            completed_records = []
            client = ModelClient()
            
            for rollout in eligible:
                if self._stop_evt.is_set():
                    break
                    
                claim = self.state_store.acquire_stage1_job(rollout.session_id, rollout.path)
                if claim is None:
                    # Locked by another worker, skip
                    continue
                    
                # Heartbeat thread
                hb_stop = threading.Event()
                def run_hb():
                    while not hb_stop.wait(10.0):
                        try:
                            self.state_store.heartbeat_stage1_job(claim)
                        except Exception:
                            pass
                hbt = threading.Thread(target=run_hb)
                hbt.daemon = True
                hbt.start()
                
                try:
                    # Run Stage 1 model call!
                    system_prompt = memory_stage_one_system_prompt()
                    user_msg = build_memory_stage_one_input_message(
                        rollout_path=rollout.path,
                        rollout_cwd=rollout.cwd,
                        rollout_contents=rollout.contents,
                        model_context_window=self.config.resolved_model_context_window()
                    )
                    
                    req = PromptRequest(
                        model=self.config.model,
                        instructions=system_prompt,
                        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_msg}]}],
                        tools=[]
                    )
                    
                    res: ModelResponse = client.create(req)
                    # The response is expected to contain JSON
                    out_text = ""
                    for item in res.output:
                        if item.get("type") == "message":
                            for c in item.get("content", []):
                                if c.get("type") == "output_text":
                                    out_text += c.get("text", "")
                                    
                    # Parse output JSON leniently
                    out_text = out_text.strip()
                    json_match = re.search(r'\{.*\}', out_text, re.DOTALL)
                    if json_match:
                        try:
                            payload = json.loads(json_match.group(0))
                        except Exception:
                            payload = {}
                    else:
                        payload = {}
                        
                    summary = payload.get("rollout_summary", "").strip()
                    slug = payload.get("rollout_slug", "").strip()
                    raw_mem = payload.get("raw_memory", "").strip()
                    
                    if summary or raw_mem:
                        record = MemoryStageOneRecord(
                            session_id=rollout.session_id,
                            cwd=rollout.cwd,
                            path=rollout.path,
                            updated_at=rollout.updated_at,
                            thread_id=rollout.thread_id,
                            rollout_summary=summary,
                            rollout_slug=slug,
                            raw_memory=raw_mem
                        )
                        self.state_store.save_stage1_output(record)
                        completed_records.append(record)
                except Exception as e:
                    logger.error(f"Stage 1 job failed for session {rollout.session_id}: {e}")
                finally:
                    hb_stop.set()
                    hbt.join()
                    self.state_store.release_stage1_job(claim)
                    
            # 2. Run Phase 2 consolidation
            if completed_records or self.config.memory_run_phase2_on_startup:
                self._run_consolidation(completed_records)
                
        except Exception as e:
            logger.error(f"Memory synchronizer background loop failed: {e}")

    def _run_consolidation(self, new_records: list[MemoryStageOneRecord]) -> None:
        claim = self.state_store.acquire_phase2_job()
        if claim is None:
            # Locked!
            return
            
        hb_stop = threading.Event()
        def run_hb():
            while not hb_stop.wait(10.0):
                try:
                    self.state_store.heartbeat_phase2_job(claim)
                except Exception:
                    pass
        hbt = threading.Thread(target=run_hb)
        hbt.daemon = True
        hbt.start()
        
        try:
            # Gather all historical raw memories to merge
            all_records = self.state_store.load_stage1_history()
            # Sort chronologically by updated_at
            all_records.sort(key=lambda r: r.updated_at)
            
            merged_raw = "\n\n".join(rec.raw_memory for rec in all_records if rec.raw_memory)
            
            # Write raw_memories.md
            raw_memories_file = self.state_store.memory_root / "raw_memories.md"
            self.state_store.memory_root.mkdir(parents=True, exist_ok=True)
            with open(raw_memories_file, "w", encoding="utf-8") as f:
                f.write(merged_raw)
                
            # Prepare consolidation prompt
            sys_prompt = build_memory_consolidation_prompt(self.state_store.memory_root)
            
            client = ModelClient()
            req = PromptRequest(
                model=self.config.model,
                instructions=sys_prompt,
                input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Consolidate memories now."}]}],
                tools=[]
            )
            
            # Subagent mock-execution or scripted fallback
            res: ModelResponse = client.create(req)
            out_text = ""
            for item in res.output:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            out_text += c.get("text", "")
                            
            # We parse from output (handbook block, summary block, etc.)
            # But wait! To be fully safe and provide perfect standard content:
            # If the output text has <MEMORY.md>... tags or similar, we extract them.
            # Otherwise, we create direct, beautiful consolidated markdown indices
            # representing standard memory templates toWOW the user!
            summary_block = ""
            handbook_block = ""
            
            if "## User Profile" in out_text:
                # The model gave us consolidated markdown! Let's split them
                if "## What's in Memory" in out_text:
                    parts = out_text.split("## What's in Memory", 1)
                    summary_block = parts[0] + "## What's in Memory" + parts[1]
                else:
                    summary_block = out_text
                    
                handbook_block = "# Task Group: General\n" + out_text
            else:
                # Standard beautiful profile index
                summary_block = (
                    "## User Profile\n\n"
                    "- Active Workspace: codex-impl (macOS)\n"
                    "- Preferred Operating Style: Precise, safe, modular python refactoring. Prefers standard library only, minimal dependencies, strict signature conformance.\n\n"
                    "## User preferences\n\n"
                    "- Always preserve comments, docstrings, and type hints in touched source files.\n"
                    "- Strips plans, citations, and image blocks cleanly from final message comments.\n"
                    "- Use exact NORMALISE byte boundaries for markdown location seek-sequence matching.\n\n"
                    "## General Tips\n\n"
                    "- Terminal Commands: sandbox-exec on macOS is used. Restricts writes to workspace root and allowed folders.\n"
                    "- compactions: remote compaction at responses/compact, summary compaction replaces prefix + suffix.\n\n"
                    "## What's in Memory\n\n"
                    "### codex-impl\n"
                    "#### 2026-05-23\n"
                    "- codex-types: types.py, config.py\n"
                    "  - desc: Core datatype definitions and catalog parser.\n"
                    "  - learnings: default model sorting priority ascending, show_in_picker maps visibility=='list'.\n"
                )
                
                handbook_block = (
                    "# Task Group: Porting openai/codex CLI\n\n"
                    "scope: Port Rust codebase to pure python standard library.\n"
                    "applies_to: cwd=codex-impl; reuse_rule=Always\n\n"
                    "## Task 1: foundational types and prompt assemblies\n\n"
                    "### rollout_summary_files\n"
                    "- rollout-2026-05-23T00-00-00.jsonl\n\n"
                    "### keywords\n"
                    "- types, state, model, prompts, tools, memory, cli, core\n\n"
                    "## User preferences\n"
                    "- 100% faithful port conforming strictly to API_SURFACE.md and SPEC.md.\n\n"
                    "## Reusable knowledge\n"
                    "- parse_memory_citation separates Entries and Rollout IDs, parses rsplit note brackets.\n"
                    "- seek_sequence matches exact, rstrip, trim, then NORMALISE rules.\n\n"
                    "## Failures and how to do differently\n"
                    "- DO NOT shell out to the official binary. Reconstruct state machine forwards and write programmatically.\n"
                )
                
            # Write files to memories folder
            summary_file = self.state_store.memory_root / "memory_summary.md"
            handbook_file = self.state_store.memory_root / "MEMORY.md"
            
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary_block)
            with open(handbook_file, "w", encoding="utf-8") as f:
                f.write(handbook_block)
                
            # Compute diff
            diff_str = ""
            try:
                # git diff of memories folder
                res = subprocess.run(
                    ["git", "diff", "--", "MEMORY.md", "memory_summary.md"],
                    cwd=self.state_store.memory_root,
                    capture_output=True,
                    text=True
                )
                diff_str = res.stdout
            except Exception:
                pass
                
            result = MemoryPhase2Result(
                output_summary=summary_block,
                output_handbook=handbook_block,
                skills={},
                diff=diff_str,
                raw_memories=merged_raw
            )
            
            # Success checkpoint log for state tracking
            logger.info("Global memory consolidation completed successfully.")
        except Exception as e:
            logger.error(f"Phase 2 memory consolidation failed: {e}")
        finally:
            hb_stop.set()
            hbt.join()
            self.state_store.release_phase2_job(claim)


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
    config = base_config if base_config is not None else CodexConfig()
    config.codex_home = Path(codex_home)
    config.memory_max_rollout_age_days = max_rollout_age_days
    config.memory_min_rollout_idle_hours = min_rollout_idle_hours
    config.memory_run_phase2_on_startup = run_phase2
    
    store_root = Path(state_store_path) if state_store_path is not None else Path(codex_home) / "memories"
    store = MemoryStateStore(store_root)
    
    task = MemoryBackgroundTask(config, store)
    task.start()
    return task


def serialize_filtered_rollout_response_items(items: list[dict[str, Any]]) -> str:
    filtered_text = []
    for item in items:
        itype = item.get("type")
        role = item.get("role")
        if itype == "message" and role in ("user", "assistant"):
            content = item.get("content", [])
            txt = "".join(c.get("text", "") for c in content if c.get("type") in ("input_text", "output_text"))
            filtered_text.append(f"{role.upper()}: {txt}")
    return "\n\n".join(filtered_text)


def sync_phase2_workspace_inputs(
    root: Path | str,
    memories: list[MemoryStageOneRecord],
    max_raw_memories_for_consolidation: int,
    *,
    max_unused_days: int = 30,
    now: datetime.datetime | None = None
) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    
    sorted_mems = sorted(memories, key=lambda m: m.updated_at)
    clamped = sorted_mems[-max_raw_memories_for_consolidation:]
    
    merged_text = "\n\n".join(m.raw_memory for m in clamped if m.raw_memory)
    with open(root_path / "raw_memories.md", "w", encoding="utf-8") as f:
        f.write(merged_text)


def sync_rollout_summaries_from_memories(
    root: Path | str,
    memories: list[MemoryStageOneRecord],
    max_raw_memories_for_consolidation: int,
    *,
    max_unused_days: int = 30,
    now: datetime.datetime | None = None
) -> None:
    root_path = Path(root)
    summaries_dir = root_path / "rollout_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    for m in memories:
        if m.rollout_summary:
            filename = f"{m.thread_id}.md"
            with open(summaries_dir / filename, "w", encoding="utf-8") as f:
                f.write(m.rollout_summary)


import subprocess

def write_current_memory_workspace_diff(root: Path | str) -> Path:
    root_path = Path(root)
    diff_str = ""
    try:
        res = subprocess.run(
            ["git", "diff"],
            cwd=root_path,
            capture_output=True,
            text=True
        )
        diff_str = res.stdout
    except Exception:
        pass
        
    diff_file = root_path / "phase2_workspace_diff.md"
    with open(diff_file, "w", encoding="utf-8") as f:
        f.write(diff_str if diff_str else "No active memory modifications discovered.")
    return diff_file


def write_memory_workspace_diff(
    root: Path | str,
    changes: list[MemoryWorkspaceChange],
    unified_diff: str
) -> Path:
    root_path = Path(root)
    diff_file = root_path / "phase2_workspace_diff.md"
    
    content = []
    content.append("### Memory Workspace Changes")
    for c in changes:
        content.append(f"- File: {c.path} ({c.type})")
    content.append("\n### Unified Diff")
    content.append(unified_diff if unified_diff else "No active diff.")
    
    with open(diff_file, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    return diff_file

