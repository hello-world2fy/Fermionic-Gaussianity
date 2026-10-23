"""Source snapshots and artifact hashes for this isolated experiment."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import numpy as np
import scipy


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(output, experiment_root, project_root):
    selected = list((project_root/'src/gaussian_synthesis').glob('*.py'))
    selected += list((project_root/'tests').glob('*.py'))
    for folder, suffix in [('bcs_pairing', '*.py'), ('scripts', '*.py'), ('tests', '*.py'), ('configs', '*.json')]:
        selected += list((experiment_root/folder).glob(suffix))
    selected += [project_root/'pyproject.toml', project_root/'run_tests.py', experiment_root/'requirements.txt']
    hashes = {}
    for source in sorted(selected):
        relative = source.relative_to(project_root)
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[str(relative)] = digest(destination)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
    (output/'requirements-freeze.txt').write_text(freeze, encoding='utf-8')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=project_root,
                            capture_output=True, text=True, check=True).stdout.strip()
    return {'python': sys.version, 'python_executable': sys.executable, 'numpy': np.__version__,
            'scipy': scipy.__version__, 'platform': platform.platform(), 'base_commit': commit,
            'source_sha256': hashes, 'source_note': 'Snapshot includes uncommitted implementation; base_commit alone is insufficient.'}
