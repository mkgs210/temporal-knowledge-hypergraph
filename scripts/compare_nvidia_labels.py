#!/usr/bin/env python3
"""Compare two NVIDIA models on three identical groups; six inference calls."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh.nvidia_labels import main
if __name__=='__main__':raise SystemExit(main())
