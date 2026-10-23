#!/usr/bin/env python3
"""Archive regression logs and independently recheck one completed pilot."""
import _bootstrap
from _bootstrap import EXPERIMENT_ROOT, PROJECT_ROOT
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from bcs_pairing.provenance import write_json, digest
from bcs_pairing.verification import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    output = args.directory.resolve()
    validation = output/'validation'
    validation.mkdir(exist_ok=False)
    commands = [
        ('library_tests', [sys.executable, '-W', 'error::RuntimeWarning', str(PROJECT_ROOT/'run_tests.py')], 60),
        ('bcs_tests', [sys.executable, '-W', 'error::RuntimeWarning', str(EXPERIMENT_ROOT/'scripts/run_tests.py')], 90),
        ('saved_results', [sys.executable, '-W', 'error::RuntimeWarning', str(EXPERIMENT_ROOT/'scripts/verify_results.py'), str(output)], 60),
    ]
    reports = {}
    for name, command, timeout in commands:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout)
        (validation/(name+'.stdout.log')).write_text(completed.stdout)
        (validation/(name+'.stderr.log')).write_text(completed.stderr)
        summary = json.loads(completed.stdout) if completed.returncode == 0 else None
        reports[name] = {'command': command, 'exit_code': completed.returncode, 'summary': summary}
        print(name, 'passed' if completed.returncode == 0 else 'FAILED', flush=True)
    manifest = json.loads((output/'manifest.json').read_text())
    changed = [name for name, expected in manifest['environment']['source_sha256'].items()
               if digest(PROJECT_ROOT/name) != expected]
    reports['numeric_source_matches_snapshot'] = {'passed': not changed, 'changed': changed}
    # Alter copies only: verify that incomplete tables and altered scalars cannot pass.
    with tempfile.TemporaryDirectory(prefix='bcs-verifier-negative-') as tmp:
        copied = Path(tmp)/'result'
        shutil.copytree(output, copied, ignore=shutil.ignore_patterns('validation', 'figures'))
        path = copied/'baselines.csv'
        original = path.read_text()
        path.write_text('\n'.join(original.splitlines()[:-1])+'\n')
        rejected = False
        try:
            verify(copied, check_hashes=False)
        except ValueError as error:
            rejected = 'Missing quench rows' in str(error)
        reports['missing_row_rejected'] = {'passed': rejected}
        rows = list(csv.DictReader(original.splitlines()))
        rows[1]['F_PBCS_found'] = '0.123'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        altered = verify(copied)
        reports['altered_scalar_rejected'] = {'passed': altered['status'] == 'failed_acceptance'
             and not altered['checks']['recomputed_fidelities']['passed'] and not altered['checks']['saved_hashes']['passed']}
    success = all(row['exit_code'] == 0 for row in reports.values() if 'exit_code' in row)
    success = success and all(row['passed'] for row in reports.values() if 'passed' in row)
    write_json(validation/'summary.json', {'status': 'passed' if success else 'failed', 'reports': reports})
    shutil.copyfile(__file__, validation/'validate_all.py')
    write_json(validation/'manifest.json', {p.name: digest(p) for p in sorted(validation.iterdir()) if p.is_file()})
    print(json.dumps({'status': 'passed' if success else 'failed', 'validation_directory': str(validation)}))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
