"""Ingest LongMemEval haystack sessions into memdio StorageManager."""

import json
import os
import shutil
import tempfile

from memdio.core.storage import StorageManager


def format_session(session: list[dict], session_date: str | None = None) -> str:
    """Format a session's turns into a single text block for storage."""
    lines = []
    if session_date:
        lines.append(f"[Date: {session_date}]")
    for turn in session:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def ingest_question(question: dict, base_dir: str | None = None) -> tuple[StorageManager, str]:
    """Ingest all haystack sessions for a single question into an isolated StorageManager.

    Returns (storage_manager, db_path) so caller can clean up.
    """
    if base_dir is None:
        base_dir = tempfile.mkdtemp(prefix="memdio_bench_")

    qid = question["question_id"]
    db_dir = os.path.join(base_dir, qid)

    storage = StorageManager(base_path=db_dir)

    sessions = question.get("haystack_sessions", [])
    dates = question.get("haystack_dates", [])

    for i, session in enumerate(sessions):
        date = dates[i] if i < len(dates) else None
        text = format_session(session, session_date=date)
        if text.strip():
            storage.store(text, document_date=date)

    return storage, db_dir


def cleanup_question_db(db_dir: str):
    """Remove temporary database directory."""
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir, ignore_errors=True)
