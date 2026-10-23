"""Explicit resource budgets and reproducible, non-overwriting outputs."""
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time


class ResourceLimit(RuntimeError):
    pass


def peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == 'darwin' else 1024)


class Budget:
    def __init__(self, max_seconds=180, max_memory_mib=512, total_deadline=None):
        self.max_seconds = float(max_seconds)
        self.max_memory_mib = float(max_memory_mib)
        self.total_deadline = total_deadline
        self.started = time.monotonic()

    def check(self):
        now = time.monotonic()
        if now - self.started > self.max_seconds:
            raise ResourceLimit('stage_time_limit')
        if self.total_deadline is not None and now >= self.total_deadline:
            raise ResourceLimit('total_time_limit')
        if peak_mib() > self.max_memory_mib:
            raise ResourceLimit('memory_limit')

    def __enter__(self):
        self.started = time.monotonic()
        # __exit__ is not called if __enter__ raises. Never leave an armed
        # timer behind when an already exhausted budget rejects entry.
        self.check()
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        seconds = self.max_seconds
        if self.total_deadline is not None:
            seconds = min(seconds, max(.001, self.total_deadline-self.started))
        def timeout(signum, frame):
            raise ResourceLimit('stage_or_total_time_limit')
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    def usage(self):
        return {'seconds': time.monotonic()-self.started, 'peak_mib': peak_mib()}


def jsonable(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_arrays(path, **arrays):
    import numpy as np
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,**arrays)
    temporary.replace(path)


def source_hashes(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(p in {'.venv','results','__pycache__'} for p in path.relative_to(root).parts):
            continue
        if path.suffix in {'.py','.json','.md','.txt'}:
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
