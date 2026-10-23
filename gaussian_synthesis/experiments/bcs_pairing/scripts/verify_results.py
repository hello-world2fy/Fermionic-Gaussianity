#!/usr/bin/env python3
"""Read-only recomputation of saved checks; no fitting or dynamics."""
import _bootstrap
import argparse
import json
from bcs_pairing.verification import verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    result = verify(args.directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
