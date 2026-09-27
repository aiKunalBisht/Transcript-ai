# api/async_processor.py
# Async Processing Layer for TranscriptAI v2.0
#
# Architecture: ThreadPoolExecutor over asyncio because analyze_transcript
# calls sync C extensions (MeCab, sentence-transformers) that don't release
# the GIL cleanly. Thread pool isolates them from the event loop.
#
# Production upgrade path:
#   1. Replace JobStore._store (dict) with Redis client.
#   2. Serialize AnalysisJob to/from JSON.
#   3. Use Redis TTL instead of background cleanup thread.
#   Everything outside JobStore stays identical — callers don't change.
#
# Interview answer for "how would you scale to 10k/day":
#   - This file: in-process ThreadPoolExecutor, good for ~100 req/min
#   - Next step: Celery + Redis broker, multiple worker processes
#   - At scale: vLLM for batch inference, horizontal worker scaling
#   - Observability: swap _metrics dict for Prometheus Counter/Gauge

import concurrent.futures
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger("transcriptai.async")

# ── Config ─────────────────────────────────────────────────────────────────────
_MAX_WORKERS   = 3      # concurrent LLM threads (Groq handles ~5 rps per key)
_MAX_JOBS      = 1000   # hard cap on in-memory store
_JOB_TTL_SEC   = 7200   # 2 h — completed jobs expire after this
_STUCK_JOB_SEC = 600    # 10 min — running jobs older than this get reaped
_CLEANUP_EVERY = 300    # background cleanup cadence in seconds


# ── Job model ──────────────────────────────────────────────────────────────────
@dataclass
class AnalysisJob:
    job_id:      str
    status:      str               # queued | running | done | failed
    transcript:  str
    language:    str
    result:      Optional[dict] = None
    error:       Optional[str]  = None
    created_at:  float = field(default_factory=time.time)
    started_at:  Optional[float] = None
    finished_at: Optional[float] = None
    duration_ms: Optional[float] = None

    def running_sec(self) -> float:
        return (time.time() - self.started_at) if self.started_at else 0.0

    def age_sec(self) -> float:
        return time.time() - self.created_at

    def to_status_dict(self) -> dict:
        def _iso(ts: Optional[float]) -> Optional[str]:
            return datetime.fromtimestamp(ts).isoformat() if ts else None
        return {
            "job_id":      self.job_id,
            "status":      self.status,
            "created_at":  _iso(self.created_at),
            "started_at":  _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "duration_ms": self.duration_ms,
            "has_result":  self.result is not None,
            "error":       self.error,
        }


# ── Job store ──────────────────────────────────────────────────────────────────
class JobStore:
    """
    Thread-safe, bounded in-memory job store.

    Bounded at _MAX_JOBS: when full, evicts the oldest done/failed jobs
    (10% of capacity at a time) before inserting a new one.
    If the store is full of running/queued jobs, the submit call is rejected
    with a RuntimeError — this is intentional back-pressure.

    Redis upgrade: replace _store with a Redis client that implements
    the same get / put / update / delete / values interface.
    """

    def __init__(self, max_size: int = _MAX_JOBS):
        self._store:    dict[str, AnalysisJob] = {}
        self._lock:     threading.Lock = threading.Lock()
        self._max_size: int = max_size

    def get(self, job_id: str) -> Optional[AnalysisJob]:
        with self._lock:
            return self._store.get(job_id)

    def put(self, job: AnalysisJob) -> None:
        with self._lock:
            if len(self._store) >= self._max_size:
                # Evict oldest completed jobs
                evictable = sorted(
                    [j for j in self._store.values() if j.status in ("done", "failed")],
                    key=lambda j: j.created_at,
                )
                batch = evictable[: max(1, self._max_size // 10)]
                for j in batch:
                    del self._store[j.job_id]
                log.info("Evicted %d old jobs (store at capacity)", len(batch))

                if len(self._store) >= self._max_size:
                    raise RuntimeError(
                        f"Job store full ({self._max_size} active jobs). "
                        "All worker slots are busy — try again shortly."
                    )
            self._store[job.job_id] = job

    def update(self, job: AnalysisJob) -> None:
        """Write back a mutated job. No-op if the job was evicted."""
        with self._lock:
            if job.job_id in self._store:
                self._store[job.job_id] = job

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._store.pop(job_id, None)

    def values(self) -> list[AnalysisJob]:
        with self._lock:
            return list(self._store.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ── Global state ───────────────────────────────────────────────────────────────
_store:    JobStore = JobStore()
_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_cleaner:  Optional[threading.Thread] = None
_shutdown: threading.Event = threading.Event()

# Lightweight counters — drop-in replaceable with Prometheus Counter/Gauge
_metrics: dict[str, float] = {
    "submitted": 0, "completed": 0, "failed": 0, "total_duration_ms": 0
}
_metrics_lock = threading.Lock()


def _inc(key: str, value: float = 1.0) -> None:
    with _metrics_lock:
        _metrics[key] = _metrics.get(key, 0.0) + value


# ── Background cleanup ─────────────────────────────────────────────────────────
def _cleanup_loop() -> None:
    """Daemon thread: expires old jobs and reaps stuck workers."""
    while not _shutdown.wait(timeout=_CLEANUP_EVERY):
        _do_cleanup()


def _do_cleanup() -> None:
    now = time.time()
    expired = reaped = 0

    for job in _store.values():
        # Expire completed jobs past TTL
        if job.status in ("done", "failed") and job.age_sec() > _JOB_TTL_SEC:
            _store.delete(job.job_id)
            expired += 1
            continue

        # Reap stuck running jobs
        if job.status == "running" and job.running_sec() > _STUCK_JOB_SEC:
            job.status      = "failed"
            job.error       = f"Reaped: exceeded max running time ({_STUCK_JOB_SEC}s)"
            job.finished_at = now
            _store.update(job)
            _inc("failed")
            reaped += 1
            log.warning("Reaped stuck job %s (ran %.0fs)", job.job_id, job.running_sec())

    if expired or reaped:
        log.info(
            "Cleanup cycle: expired=%d reaped=%d store_size=%d",
            expired, reaped, len(_store),
        )


# ── Lifecycle ──────────────────────────────────────────────────────────────────
def startup() -> None:
    """
    Initialize executor and cleanup thread.
    Call once from FastAPI lifespan startup.
    Safe to call multiple times — idempotent.
    """
    global _executor, _cleaner
    if _executor is not None:
        return  # already running

    _shutdown.clear()
    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_MAX_WORKERS,
        thread_name_prefix="tai-worker",
    )
    _cleaner = threading.Thread(
        target=_cleanup_loop, daemon=True, name="tai-cleanup"
    )
    _cleaner.start()
    log.info(
        "AsyncProcessor started (workers=%d, max_jobs=%d, ttl=%ds)",
        _MAX_WORKERS, _MAX_JOBS, _JOB_TTL_SEC,
    )


def shutdown() -> None:
    """
    Drain in-flight jobs and stop the cleanup thread.
    Call from FastAPI lifespan shutdown.
    """
    global _executor
    _shutdown.set()
    if _executor:
        # cancel_futures=False so in-flight jobs finish cleanly
        _executor.shutdown(wait=True, cancel_futures=False)
        _executor = None
    log.info("AsyncProcessor shut down")


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        startup()  # lazy init (e.g. dev/test without calling startup explicitly)
    return _executor


# ── Worker ─────────────────────────────────────────────────────────────────────
def _run_job(job_id: str) -> None:
    """
    Runs in a thread-pool worker.
    Imports analyzer lazily to avoid circular imports at module load.
    Sets _cleaned_transcript and _detected_language on the result so that
    downstream consumers (soft-rejection, rag_evaluator) have the source text.
    """
    job = _store.get(job_id)
    if not job:
        return

    job.status     = "running"
    job.started_at = time.time()
    _store.update(job)
    t0 = time.monotonic()

    try:
        from analysis.analyzer import analyze_transcript
        result = analyze_transcript(job.transcript, job.language)

        # Ensure downstream modules always find these keys
        result.setdefault("_cleaned_transcript", job.transcript)
        result.setdefault("_detected_language",  job.language)

        job.result      = result
        job.status      = "done"
        job.finished_at = time.time()
        job.duration_ms = round((time.monotonic() - t0) * 1000, 1)
        _store.update(job)

        _inc("completed")
        _inc("total_duration_ms", job.duration_ms)
        log.info(
            "Job %s done in %.0fms provider=%s",
            job_id, job.duration_ms, result.get("_provider", "?"),
        )

    except Exception as exc:
        job.status      = "failed"
        job.error       = str(exc)
        job.finished_at = time.time()
        job.duration_ms = round((time.monotonic() - t0) * 1000, 1)
        _store.update(job)
        _inc("failed")
        log.error("Job %s failed: %s", job_id, exc, exc_info=True)


# ── Public API ─────────────────────────────────────────────────────────────────
def submit_job(transcript: str, language: str = "en") -> str:
    """
    Submit a transcript for async analysis.
    Returns job_id immediately — never blocks.

    Raises:
        RuntimeError: if the job store is at capacity with no evictable slots.
    """
    job_id = uuid.uuid4().hex[:8]
    job = AnalysisJob(
        job_id=job_id, status="queued",
        transcript=transcript, language=language,
    )
    _store.put(job)                           # raises RuntimeError if full
    _get_executor().submit(_run_job, job_id)
    log.debug("Submitted job %s lang=%s", job_id, language)
    return job_id


def get_job_status(job_id: str) -> dict:
    """Return current status without waiting."""
    job = _store.get(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}
    return job.to_status_dict()


def get_job_result(job_id: str, timeout_sec: float = 30.0) -> dict:
    """
    Return the result for a job.

    Does an IMMEDIATE check first — if the job is already done this returns
    in microseconds regardless of timeout_sec. The poll loop only runs when
    the job is still queued/running.

    Args:
        timeout_sec: Max seconds to wait. Pass 0 to get the result only if
                     it's already available (raises TimeoutError otherwise).

    Raises:
        ValueError:    Job ID not found.
        RuntimeError:  Job failed.
        TimeoutError:  Job not done within timeout_sec.
    """
    job = _store.get(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    # ── Immediate path ─────────────────────────────────────────────────────
    if job.status == "done":
        return job.result
    if job.status == "failed":
        raise RuntimeError(f"Job {job_id} failed: {job.error}")

    # ── Poll path ──────────────────────────────────────────────────────────
    if timeout_sec <= 0:
        raise TimeoutError(
            f"Job {job_id} is {job.status} (pass timeout_sec > 0 to wait)"
        )

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        time.sleep(0.25)
        job = _store.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} disappeared (evicted during wait)")
        if job.status == "done":
            return job.result
        if job.status == "failed":
            raise RuntimeError(f"Job {job_id} failed: {job.error}")

    raise TimeoutError(f"Job {job_id} still {job.status} after {timeout_sec}s")


def process_batch(transcripts: list[dict]) -> list[dict]:
    """
    Submit N transcripts concurrently and wait for all to finish.

    Args:
        transcripts: [{"transcript": str, "language": str}, ...]

    Returns:
        [{"status": "success"|"failed", "result": dict|None, "error": str|None}, ...]
        in the same order as input.
    """
    job_ids = [
        submit_job(item.get("transcript", ""), item.get("language", "en"))
        for item in transcripts
    ]
    results = []
    for job_id in job_ids:
        try:
            results.append({"status": "success", "result": get_job_result(job_id, timeout_sec=300)})
        except Exception as exc:
            results.append({"status": "failed", "result": None, "error": str(exc)})
    return results


def get_queue_stats() -> dict:
    """Current queue state + lifetime counters."""
    all_jobs  = _store.values()
    done_jobs = [j for j in all_jobs if j.status == "done" and j.duration_ms]

    with _metrics_lock:
        m = dict(_metrics)

    return {
        "total":   len(all_jobs),
        "queued":  sum(1 for j in all_jobs if j.status == "queued"),
        "running": sum(1 for j in all_jobs if j.status == "running"),
        "done":    sum(1 for j in all_jobs if j.status == "done"),
        "failed":  sum(1 for j in all_jobs if j.status == "failed"),
        "avg_duration_ms": (
            round(sum(j.duration_ms for j in done_jobs) / len(done_jobs), 1)
            if done_jobs else 0
        ),
        "lifetime_submitted": int(m.get("submitted",    0)),
        "lifetime_completed": int(m.get("completed",    0)),
        "lifetime_failed":    int(m.get("failed",       0)),
        "store_size":         len(_store),
        "max_store_size":     _MAX_JOBS,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    startup()

    samples = [
        {"transcript": "Tanaka: Q3 report ready.\nSato: Let's review by Thursday.", "language": "mixed"},
        {"transcript": "田中: 検討いたします。難しいかもしれません。\n鈴木: 承知しました。", "language": "ja"},
        {"transcript": "Client: This delay is unacceptable.\nKenji: 大変申し訳ございません。We resolve today.", "language": "mixed"},
    ]

    print(f"Submitting {len(samples)} jobs...")
    t0 = time.time()
    results = process_batch(samples)
    elapsed = round(time.time() - t0, 1)

    for i, r in enumerate(results):
        if r["status"] == "success":
            summary  = (r["result"].get("summary") or [""])[0][:60]
            provider = r["result"].get("_provider", "?")
            print(f"  [{i+1}] ✅ {provider} | {summary}...")
        else:
            print(f"  [{i+1}] ❌ {r['error']}")

    print(f"\nCompleted in {elapsed}s")
    print(json.dumps(get_queue_stats(), indent=2))
    shutdown()