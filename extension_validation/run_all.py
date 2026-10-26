#!/usr/bin/env python3
"""Safe snapshot-reproduction driver for extension_validation.

Design contract
---------------
* READ-ONLY against the three producer trees (hartree/, slater/, signed_gaussian/).
  We never edit, move or delete a producer file.
* DEFAULT = dry-run. A dry-run only VERIFIES that the entry scripts exist and
  prints the plan; it does NOT reproduce any numbers and copies nothing.
* With --run: copy each producer tree WHOLE into a fresh
  runs/run_<UTC-timestamp>/<producer>/ directory (cache/venv/__pycache__ excluded),
  then run the documented stages inside the snapshot. Every producer script writes
  relative to its own __file__ (results/ or artifacts/ next to the code), so all
  new output lands inside the snapshot. Committed results/ artifacts/ are never
  touched. Matplotlib cache is redirected into the run dir (MPLCONFIGDIR) so the
  process never writes into $HOME.
* runs/run_<ts>/ refuses to be created if it already exists (no overwrite).
* On the first non-zero stage exit we STOP, record success=false in the manifest,
  and propagate a non-zero exit code (we do NOT silently return 0).

Stage entry points (verified against current producers; producers are still being
revised, so if a script is renamed this list must be updated):
  A hartree           : exp1 -> exp2 -> exp3 -> make_figures -> aggregate -> tests
  B slater            : run_benchmark (dimer + Hubbard + H2/STO-3G)
                        -> run_complete (Fock-complete feasibility, the decisive
                           test: full Fock basis + 40 non-orth atoms, budget 8/12)
                        -> plot_results -> make_summary (CSV->summary.json) -> tests
  C signed_gaussian   : run_magic -> run_hubbard -> run_convex (MC gradient-check,
                        kept as a failed-convergence record)
                        -> run_exact_convex (FINITE-POOL exact deterministic J_k,
                        the final bounded convex solve)
                        -> make_figs -> assemble_summary

Usage
-----
    .venv/bin/python run_all.py --list      # print stages, execute nothing
    .venv/bin/python run_all.py             # dry-run: verify entries + print plan
    .venv/bin/python run_all.py --run       # snapshot + execute all stages
    .venv/bin/python run_all.py --only B_slater --run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / "bin" / "python"
RUNS = ROOT / "runs"

# (stage_id, producer_dir, [scripts to run IN ORDER, relative to producer_dir])
STAGES = [
    ("A_hartree", "hartree", [
        "code/exp1_moments.py",
        "code/exp2_convex.py",
        "code/exp3_bosehubbard.py",
        "code/exp4_rdm.py",     # gamma1 RDM -> true condensate fraction (PSD, trace=N) + entropy
        "code/make_figures.py",
        "code/aggregate_summary.py",
        "code/run_tests.py",
    ]),
    ("B_slater", "slater", [
        "run_benchmark.py",    # dimer analytic + Hubbard L4/6 + H2/STO-3G -> results/
        "run_complete.py",     # Fock-complete feasibility (full Fock + 40 non-orth, n=8/12)
        "plot_results.py",     # results/ -> artifacts/*.png
        "timed_bench.py",      # fixed_n + fixed_time budgets (Hubbard n8/12, H2 n2/3/4; 3 starts; checkpoint/rollback)
        "make_summary.py",     # CSV -> summary.json (refreshed headline, derived from CSV)
        "tests/test_slater.py",
    ]),
    ("C_signed_gaussian", "signed_gaussian", [
        "scripts/run_magic.py",
        "scripts/run_hubbard.py",
        "scripts/run_convex.py",        # MC projected-gradient: kept record (KKT not converged)
        "scripts/run_exact_convex.py",  # FINAL finite-pool exact deterministic J_k solve
        "scripts/make_figs.py",
        "scripts/assemble_summary.py",
    ]),
]

# Never copy environment/caches into the snapshot. The shared .venv lives at
# extension_validation/.venv (NOT inside any producer dir), so no interpreter tree
# is copied; old producer caches are left in place and simply not copied.
COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".DS_Store",
    ".matplotlib", "mplconfig*", "*.egg-info", ".pytest_cache",
)


def pinned_env(run_dir: Path) -> dict:
    """Single-thread, bytecode-free env with matplotlib cache confined to run_dir."""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    mpl_cache = run_dir / "mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_cache)
    return env


def snapshot_producer(src: Path, dst: Path) -> None:
    if dst.exists():
        return  # already snapshotted this stage in the current run
    shutil.copytree(src, dst, ignore=COPY_IGNORE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true",
                    help="actually snapshot + execute (default: dry-run, safe)")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--only", choices=[s[0] for s in STAGES], help="restrict to one stage")
    args = ap.parse_args()

    if not PY.exists():
        print(f"ERROR: venv python not found at {PY}", file=sys.stderr)
        print("Create it first:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt",
              file=sys.stderr)
        return 2

    if args.list:
        for sid, sub, scripts in STAGES:
            print(f"[{sid}] {sub}/")
            for s in scripts:
                print(f"    - {s}")
        return 0

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rundir = RUNS / f"run_{ts}"

    plan = []  # (stage_id, producer_dir, script_relpath)
    missing = []
    for sid, sub, scripts in STAGES:
        if args.only and sid != args.only:
            continue
        for s in scripts:
            plan.append((sid, sub, s))
            if not (ROOT / sub / s).is_file():
                missing.append(f"{sub}/{s}")

    print("=" * 72)
    print(f"mode        : {'RUN (will execute)' if args.run else 'DRY-RUN (verify entries only; NO reproduction)'}")
    print(f"fresh run dir: {rundir}")
    print("guarantee   : committed hartree/results slater/results|artifacts "
          "signed_gaussian/artifacts are NEVER modified")
    print("=" * 72)
    for sid, sub, s in plan:
        tag = "ok " if (ROOT / sub / s).is_file() else "MISSING"
        print(f"  [{tag}] [{sid:<18}] (in snapshot {sub}/)  {Path(PY).name} {s}")

    if missing:
        print("\nWARNING: entry script(s) not found on disk:", file=sys.stderr)
        for m in missing:
            print("   -", m, file=sys.stderr)
        print("Producers may have renamed a script; update STAGES before --run.", file=sys.stderr)
        if args.run:
            return 6  # refuse to run a plan that references nonexistent entries

    if not args.run:
        print("\nDRY-RUN complete. Nothing executed, nothing copied. "
              "This only confirms entry points exist; it does NOT reproduce numbers.")
        print("Re-run with --run to snapshot into runs/ and execute.")
        return 0

    # --- real execution: snapshot then run, all output confined to rundir ---
    if rundir.exists():
        print(f"ERROR: run dir already exists, refusing to overwrite: {rundir}", file=sys.stderr)
        return 3
    RUNS.mkdir(exist_ok=True)
    rundir.mkdir(parents=False, exist_ok=False)

    env = pinned_env(rundir)
    manifest = {
        "run_dir": str(rundir.relative_to(ROOT)),
        "utc": ts,
        "python": str(PY),
        "success": True,
        "stopped_at": None,
        "env": {k: v for k, v in env.items()
                if k.startswith(("PY", "OMP", "OPEN", "MKL", "MPLCONFIGDIR"))},
        "stages": {},
    }

    rc_out = 0
    for sid, sub, s in plan:
        src = ROOT / sub
        if not src.is_dir():
            print(f"ERROR: missing producer dir {src}", file=sys.stderr)
            manifest["success"] = False
            manifest["stopped_at"] = f"{sub}/{s}"
            rc_out = 4
            break
        dst = rundir / sub
        snapshot_producer(src, dst)
        script = dst / s
        if not script.exists():
            print(f"ERROR: script missing in snapshot: {script}", file=sys.stderr)
            manifest["success"] = False
            manifest["stopped_at"] = f"{sub}/{s}"
            rc_out = 5
            break
        print(f"\n>>> [{sid}] {sub}/{s}")
        t0 = time.perf_counter()
        proc = subprocess.run([str(PY), s], cwd=str(dst), env=env)
        elapsed = time.perf_counter() - t0
        manifest["stages"].setdefault(sid, []).append(
            {"script": f"{sub}/{s}", "returncode": proc.returncode, "walltime_s": round(elapsed, 3)})
        if proc.returncode != 0:
            print(f"!! [{sid}] {s} FAILED rc={proc.returncode}; STOPPING. "
                  f"Committed results are untouched; inspect {rundir}.", file=sys.stderr)
            manifest["success"] = False
            manifest["stopped_at"] = f"{sub}/{s}"
            rc_out = proc.returncode or 1
            break

    (rundir / "run_all_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    status = "SUCCESS" if manifest["success"] else f"STOPPED ({manifest['stopped_at']})"
    print(f"Finished: {status}")
    print(f"Outputs confined to: {rundir}")
    print(f"Manifest: {rundir / 'run_all_manifest.json'}")
    print("=" * 72)
    return rc_out


if __name__ == "__main__":
    raise SystemExit(main())
