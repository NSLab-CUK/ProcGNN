"""Append-only NDJSON debug logging (under logs/ + optional per-run session file)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

# Repo root = parent of `src/`
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NDJSON_LOG_PATH = _PROJECT_ROOT / "logs" / "debug-8909e4.log"
_DEFAULT_LOG = DEFAULT_NDJSON_LOG_PATH
_session_log: Optional[Path] = None


def set_session_ndjson_path(path: Optional[Path]) -> None:
    """Also write every event to this file (in addition to the repo-wide default log)."""
    global _session_log
    if path is None:
        _session_log = None
        return
    _session_log = Path(path).resolve()
    _session_log.parent.mkdir(parents=True, exist_ok=True)


def get_session_ndjson_path() -> Optional[Path]:
    return _session_log


def append_ndjson_line(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=True) + "\n"
    _DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _DEFAULT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)
    if _session_log is not None:
        with _session_log.open("a", encoding="utf-8") as handle:
            handle.write(line)


def write_training_debug_event(
    *,
    session_id: str,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: Mapping[str, Any],
) -> None:
    append_ndjson_line(
        {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": dict(data),
            "timestamp": int(datetime.now().timestamp() * 1000),
        }
    )
