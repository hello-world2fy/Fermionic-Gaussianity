"""Reproducible single-threaded runner: moments -> convex -> BH -> RDM -> figures -> tests -> summary.
Fixed seed, deterministic. Writes only into results/ and tests/."""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.normpath(os.path.join(HERE, "..", "..", ".venv", "bin", "python"))
PY = VENV if os.path.exists(VENV) else sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1")


def run(script):
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([PY, os.path.join(HERE, script)], check=True, env=ENV)


if __name__ == "__main__":
    for s in ["exp1_moments.py", "exp2_convex.py", "exp3_bosehubbard.py",
              "exp4_rdm.py", "make_figures.py", "run_tests.py",
              "aggregate_summary.py"]:
        run(s)
    print("\nALL DONE")
