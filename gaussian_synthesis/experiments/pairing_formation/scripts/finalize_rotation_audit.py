#!/usr/bin/env python3
"""Validate the eleventh-round documents and create a current file manifest."""
import hashlib
import json
import os
from pathlib import Path
import re
import time


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/assessment_rotation_geometry_01/checkpoint_manifest.json'
    if output.exists():
        raise FileExistsError(output)
    status = json.loads((root / 'STATUS.json').read_text())
    assert status['goal_status'] == 'active'
    assert status['latest_report'] == 'REPORT_ROTATION_GEOMETRY.md'
    assert root / status['latest_checkpoint'] == output
    jobs = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [part.decode(errors='replace') for part in
                    (entry / 'cmdline').read_bytes().split(b'\0') if part]
            cwd = (entry / 'cwd').resolve()
        except (OSError, PermissionError):
            continue
        if argv and any(arg.endswith('.py') and ('pairing_formation' in arg or
                        (cwd.is_relative_to(root) and (cwd / arg).exists()))
                        for arg in argv[1:]):
            jobs.append({'pid': int(entry.name), 'argv': argv})
    assert not jobs, jobs
    links = 0
    for filename in ('REPORT_ROTATION_GEOMETRY.md', 'ROTATION_GEOMETRY.md', 'README.md'):
        path = root / filename
        content = path.read_text()
        assert not any(ord(char) < 32 and char not in '\n\t' for char in content), filename
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if '://' in target or target.startswith('#'):
                continue
            location = (path.parent / target.split('#')[0]).resolve()
            assert location.exists() or location == output, (filename, target)
            links += 1
    log = root / 'results/assessment_rotation_geometry_01/tests.log'
    assert 'Ran 84 tests' in log.read_text() and log.read_text().rstrip().endswith('OK')
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path == output or '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                      'sha256': digest(path)})
    manifest = {
        'created_unix': time.time(),
        'scope': 'Eleventh-round current repository-local file manifest: local Gaussian modes, conventional nonlinear-field geometry, same-dimension Hamiltonian controls, one-transfer-mode controls and exact dyadic saved-matrix certificates. Earlier evidence retained. Original research goal remains active. Excludes this manifest and Python bytecode caches.',
        'files': files, 'file_count': len(files),
        'total_bytes': sum(row['bytes'] for row in files),
        'running_jobs': jobs, 'document_links_checked': links,
    }
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    for row in files:
        path = root / row['path']
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], path
    print(json.dumps({key: value for key, value in manifest.items() if key != 'files'}), flush=True)


if __name__ == '__main__':
    main()
