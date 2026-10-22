#!/usr/bin/env python3
"""Run a small scope-label repair and a blinded paired check, with resume."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from tkh.label_repair import main
if __name__ == '__main__':
    raise SystemExit(main())
