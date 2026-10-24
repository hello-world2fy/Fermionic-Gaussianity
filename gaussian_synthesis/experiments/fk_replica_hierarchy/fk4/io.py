"""Atomic checkpoints and source fingerprints for append-only experiment runs."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .paths import ROOT, EXPERIMENTS


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def source_manifest(snapshot_dir=None):
    sources = {}
    for folder in (ROOT, EXPERIMENTS/"fk_two_cluster", EXPERIMENTS/"fk_two_cluster_followup", EXPERIMENTS/"fk_canonical_structure"):
        for p in folder.rglob("*.py"):
            if "results" not in p.parts and ".venv" not in p.parts:
                relative = str(p.relative_to(EXPERIMENTS))
                content = p.read_bytes()
                sources[relative] = hashlib.sha256(content).hexdigest()
                if snapshot_dir is not None:
                    destination = Path(snapshot_dir)/relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": sources,
            "spec_sha256": hashlib.sha256((ROOT/"EXPERIMENT_SPEC.txt").read_bytes()).hexdigest()}
