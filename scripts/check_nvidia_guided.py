#!/usr/bin/env python3
"""Check hosted schema constraints with one short request, before full labeling."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh.nvidia_guided_check import main
if __name__=='__main__':raise SystemExit(main())
