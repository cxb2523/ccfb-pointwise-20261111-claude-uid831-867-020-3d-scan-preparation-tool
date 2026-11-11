"""Thin launcher for the ScanPrep 3D batch service.

Kept separate from ``scan_prep_engine.py`` so child processes serving 3D
batches do not pay for the image-pipeline's AI/torch import chain. The
implementation lives entirely in the :mod:`scanprep3d` package; the engine
exposes the same service through ``python scan_prep_engine.py --batch-service``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanprep3d.cli import main

if __name__ == "__main__":
    sys.exit(main())
