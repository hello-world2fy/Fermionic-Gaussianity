#!/usr/bin/env python3
"""Regenerate a saved FK run's report and figures."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fk.report import make_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Existing results directory")
    args = parser.parse_args()
    print(json.dumps(make_report(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

