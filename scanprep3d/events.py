"""NDJSON event protocol for the 3D batch service.

Every event is one compact JSON object per stdout line so the Electron
``child_process`` host can parse line-by-line without framing logic.
"""

import json
import sys
import threading
from datetime import datetime

_print_lock = threading.Lock()


def emit(event_type, **payload):
    """Emit one NDJSON event on stdout (thread-safe)."""
    payload["type"] = event_type
    payload.setdefault("time", datetime.now().isoformat(timespec="seconds"))
    line = json.dumps(payload, ensure_ascii=False)
    with _print_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


# Event names kept as constants so both sides can grep a single source of truth.
BATCH_STARTED = "batch_started"
BATCH_COMPLETED = "batch_completed"
BATCH_CANCELLED = "batch_cancelled"
TASK_QUEUED = "task_queued"
TASK_STARTED = "task_started"
TASK_PROGRESS = "task_progress"
TASK_COMPLETED = "task_completed"
TASK_FAILED = "task_failed"
TASK_CANCELLED = "task_cancelled"
TEXTURES_PREPARED = "textures_prepared"
STATUS = "status"
ERROR = "error"
