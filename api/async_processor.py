# async_processor.py
# Async Processing Layer for TranscriptAI
#
# ── WHY THIS EXISTS ──────────────────────────────────────────────────────────
#
# Streamlit is synchronous — one user blocks all others.
# If 50 users submit transcripts at the same time, they queue up.
#
# This module adds:
#   1. Concurrent processing — multiple transcripts at once
#   2. Job queue — submit and poll (non-blocking)
#   3. Progress tracking — know which jobs are running/done
#
# ── SCALING ANSWER FOR INTERVIEW ────────────────────────────────────────────
#   "For 10,000 transcripts/day:
#    1. async_processor.py handles concurrent requests via ThreadPoolExecutor
#    2. FastAPI endpoint is already async-ready
#    3. For true scale: Redis Queue (RQ) as job broker,
#       vLLM for high-throughput LLM inference,
#       multiple worker processes on separate machines.
#    The JSON schema contract means nothing downstream changes."
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import concurrent.futures
import uuid
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field


# ── JOB TRACKING ─────────────────────────────────────────────────────────────
@dataclass
class AnalysisJob:
    job_id:     str
    status:     str           # "queued" | "running" | "done" | "failed"
    transcript: str
    language:   str
    result:     Optional[dict] = None
    error:      Optional[str]  = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str]  = None
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None


# In-memory job store (replace with Redis for production)
_jobs: dict[str, AnalysisJob] = {}

# Thread pool — max 3 concurrent LLM calls (Ollama handles 1 at a time anyway)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


# ── ASYNC JOB SUBMISSION ──────────────────────────────────────────────────────
def submit_job(transcript: str, language: str = "en") -> str:
    """
    Submit a transcript for async analysis.
    Returns job_id immediately — does NOT block.

    Usage:
        job_id = submit_job(transcript, language)
        # Do other things...
        result = get_job_result(job_id)  # poll until done
    """
    job_id = str(uuid.uuid4())[:8]
    job    = AnalysisJob(
        job_id=job_id,
        status="queued",
        transcript=transcript,
        language=language
    )
    _jobs[job_id] = job

    # Submit to thread pool — non-blocking
    _executor.submit(_run_job, job_id)
    return job_id


def _run_job(job_id: str):
    """Worker function — runs in thread pool."""
    job = _jobs.get(job_id)
    if not job:
        return

    job.status     = "running"
    job.started_at = datetime.now().isoformat()
    start_time     = time.time()

    try:
        from analysis.analyzer import analyze_transcript
        result = analyze_transcript(job.transcript, job.language)

        job.result      = result
        job.status      = "done"
        job.finished_at = datetime.now().isoformat()
        job.duration_ms = round((time.time() - start_time) * 1000, 1)

    except Exception as e:
        job.status      = "failed"
        job.error       = str(e)
        job.finished_at = datetime.now().isoformat()
        job.duration_ms = round((time.time() - start_time) * 1000, 1)


def get_job_status(job_id: str) -> dict:
    """Check the status of a submitted job."""
    job = _jobs.get(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}

    return {
        "job_id":      job.job_id,
        "status":      job.status,
        "created_at":  job.created_at,
        "started_at":  job.started_at,
        "finished_at": job.finished_at,
        "duration_ms": job.duration_ms,
        "has_result":  job.result is not None,
        "error":       job.error
    }


def get_job_result(job_id: str, timeout_sec: int = 300) -> dict:
    """
    Wait for a job to complete and return the result.
    Blocks until done or timeout.
    """
    start = time.time()
    while time.time() - start < timeout_sec:
        job = _jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if job.status == "done":
            return job.result
        if job.status == "failed":
            raise RuntimeError(f"Job failed: {job.error}")
        time.sleep(0.5)

    raise TimeoutError(f"Job {job_id} timed out after {timeout_sec}s")


def process_batch(transcripts: list[dict], max_concurrent: int = 3) -> list[dict]:
    """
    Process multiple transcripts concurrently.

    Args:
        transcripts: List of {"transcript": str, "language": str}
        max_concurrent: Max parallel jobs (default 3)

    Returns:
        List of results in same order as input
    """
    job_ids = []
    for item in transcripts:
        job_id = submit_job(
            item.get("transcript", ""),
            item.get("language", "en")
        )
        job_ids.append(job_id)

    results = []
    for job_id in job_ids:
        try:
            result = get_job_result(job_id)
            results.append({"status": "success", "result": result})
        except Exception as e:
            results.append({"status": "failed", "error": str(e)})

    return results


def get_queue_stats() -> dict:
    """Returns current queue statistics."""
    all_jobs = list(_jobs.values())
    return {
        "total":   len(all_jobs),
        "queued":  sum(1 for j in all_jobs if j.status == "queued"),
        "running": sum(1 for j in all_jobs if j.status == "running"),
        "done":    sum(1 for j in all_jobs if j.status == "done"),
        "failed":  sum(1 for j in all_jobs if j.status == "failed"),
        "avg_duration_ms": (
            round(sum(j.duration_ms for j in all_jobs if j.duration_ms) /
                  max(sum(1 for j in all_jobs if j.duration_ms), 1), 1)
        )
    }


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    transcripts = [
        {
            "transcript": "Tanaka: Good morning. Q3 report is ready.\nSato: そうですね。Let's review by Thursday.",
            "language": "mixed"
        },
        {
            "transcript": "田中: 検討いたします。難しいかもしれません。\n鈴木: 承知しました。",
            "language": "ja"
        },
        {
            "transcript": "Client: This delay is unacceptable.\nKenji: 大変申し訳ございません。We will resolve this today.",
            "language": "mixed"
        }
    ]

    print(f"Submitting {len(transcripts)} transcripts concurrently...")
    start = time.time()

    results = process_batch(transcripts)

    elapsed = round(time.time() - start, 1)
    stats   = get_queue_stats()

    print(f"\nCompleted in {elapsed}s")
    print(f"Queue stats: {json.dumps(stats, indent=2)}")
    print(f"\nResults:")
    for i, r in enumerate(results):
        if r["status"] == "success":
            summary = r["result"].get("summary", [""])[0][:60]
            provider = r["result"].get("_provider", "unknown")
            print(f"  [{i+1}] ✅ Provider:{provider} | {summary}...")
        else:
            print(f"  [{i+1}] ❌ {r['error']}")