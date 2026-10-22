#!/usr/bin/env python3
"""Standalone persisted-output validator. Does not import the hierarchy builder."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from tkh.validation import main


def entry(argv=None):
    args=sys.argv[1:] if argv is None else argv
    if len(args)==1:
        root=Path(args[0])
        # CPU runs contain raw and labelled structures; check both, never hide
        # an invalid raw output behind a valid published one.
        if not (root/'protocol.json').exists():
            directories=[root/name for name in ('hierarchy','labelled_hierarchy') if (root/name).is_dir()]
            if directories:return max(main([str(directory)]) for directory in directories)
    return main(args)


if __name__ == '__main__':
    raise SystemExit(entry())
