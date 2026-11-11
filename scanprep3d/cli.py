"""Command-line entry point for the 3D batch service.

Usage::

    python -m scanprep3d.cli config.json

The config JSON schema::

    {
      "input_path": "C:/scans",          # file or folder (folders are recursive)
      "output_dir": "optional override", # otherwise sibling *_processed folders
      "recursive": true,
      "concurrency": 1,                  # default 1, clamped to max 2
      "decimate_ratio": 0.5,             # 0 disables decimation
      "texture_format": "webp",          # webp | ktx2 | none
      "webp_quality": 90
    }

Events are NDJSON on stdout; writing ``cancel`` (or ``stop``/``quit``) to
stdin requests cooperative cancellation.
"""

import json
import sys

from . import events
from . import service


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        events.emit(events.ERROR,
                    message="usage: python -m scanprep3d.cli <config.json>")
        return 1
    config_path = argv[0]
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        events.emit(events.ERROR,
                    message=f"could not read batch config {config_path}: {exc}")
        return 1

    service.start_stdin_cancel_watcher()
    return service.run_batch(config)


if __name__ == "__main__":
    sys.exit(main())
