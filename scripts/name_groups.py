#!/usr/bin/env python3
"""Label saved groups through NVIDIA by default; credentials are read from .env."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tkh.gemini_labels import main
if __name__=='__main__':
    if not any(arg=='--provider' or arg.startswith('--provider=') for arg in sys.argv[1:]):
        sys.argv.extend(['--provider','nvidia'])
    raise SystemExit(main())
