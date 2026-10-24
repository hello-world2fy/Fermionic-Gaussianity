import hashlib
from pathlib import Path
from .paths import ROOT
from fk5.io import source_manifest as previous_manifest
from fk4.io import write_json


def source_manifest(snapshot=None):
    result = previous_manifest(snapshot)
    for p in ROOT.rglob('*.py'):
        if 'results' in p.parts:
            continue
        name = str(p.relative_to(ROOT.parent))
        data = p.read_bytes()
        result['source_sha256'][name] = hashlib.sha256(data).hexdigest()
        if snapshot is not None:
            target = Path(snapshot)/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    result['energy_composition_spec_sha256'] = hashlib.sha256((ROOT/'EXPERIMENT_SPEC.txt').read_bytes()).hexdigest()
    result['preregistration_sha256'] = hashlib.sha256((ROOT/'configs/preregistration.json').read_bytes()).hexdigest()
    return result
