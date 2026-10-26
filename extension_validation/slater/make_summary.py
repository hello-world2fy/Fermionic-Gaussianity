"""Regenerate summary.json FROM the existing result CSVs (no experiment re-run).

Reads results/complete_ci.csv and results/dimer_extent.csv and emits
summary.json with accurate mean/median labels. Run after any benchmark:
    $PY make_summary.py
"""
import csv
import json
import collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

METHODS = ["cipsi", "greedy_noci", "replica_coef", "replica_energy"]
TAGS = ["L4_U1", "L4_U8", "L6_U1", "L6_U8"]


def stats():
    rows = list(csv.DictReader(open(RES / "complete_ci.csv")))
    nonorth = sum(1 for r in rows if r["picked_nonorth"] in ("True", "true", "1"))
    out = {}
    for budget in (8, 12):
        agg = collections.defaultdict(list)
        for r in rows:
            if int(r["n_det"]) == budget:
                agg[(r["tag"], r["method"])].append(float(r["dE"]))
        out[budget] = {tag: {m: dict(mean=float(np.mean(agg[(tag, m)])),
                                      median=float(np.median(agg[(tag, m)])))
                             for m in METHODS} for tag in TAGS}
    # KKT
    ks = [float(r["kkt_err"]) for r in rows
          if r["method"].startswith("replica") and r["kkt_err"] not in ("", "nan")]
    return out, nonorth, float(np.median(ks)), float(max(ks))


def dimer():
    rows = list(csv.DictReader(open(RES / "dimer_extent.csv")))
    return max(float(r["relerr"]) for r in rows)


def timed_table():
    """Fixed-walltime median dE across 3 seeds, per system/method/budget."""
    p = RES / "timed_bench.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    out = {}
    for sys in sorted(set(r["system"] for r in rows)):
        out[sys] = {}
        for mode in ["fixed_n", "timed"]:
            for b in sorted(set(r["budget"] for r in rows
                                if r["system"] == sys and r["mode"] == mode)):
                key = f"{mode}:{b}"
                out[sys][key] = {}
                for m in ["cipsi", "greedy_noci", "orbopt_noci", "replica_energy"]:
                    v = [float(r["dE"]) for r in rows
                         if r["system"] == sys and r["mode"] == mode
                         and r["budget"] == b and r["method"] == m
                         and r["dE"] not in ("nan", "NaN", "")]
                    n = [int(r["completed_n"]) for r in rows
                         if r["system"] == sys and r["mode"] == mode
                         and r["budget"] == b and r["method"] == m]
                    out[sys][key][m] = dict(
                        dE_median=float(np.median(v)) if v else None,
                        n_median=float(np.median(n)))
    return out


def main():
    st, nonorth, kmed, kmax = stats()
    drelerr = dimer()

    def cell(tag, m, b, key):
        return st[b][tag][m][key]

    # walltime ratios (replica / cipsi), summed over all steps, per tag
    rows = list(csv.DictReader(open(RES / "complete_ci.csv")))
    tot = collections.defaultdict(float)
    for r in rows:
        tot[(r["tag"], r["method"])] += (float(r["t_pool"]) + float(r["t_residual"])
                                         + float(r["t_optimize"]) + float(r["t_ritz"]))
    ratios = {tag: {m: tot[(tag, m)] / tot[(tag, "cipsi")]
                    for m in ("greedy_noci", "replica_coef", "replica_energy")}
              for tag in TAGS}
    headline = (
        "CURRENT DATA = results/timed_bench.csv (run_version ritzfix_v2, 456 runs: "
        "fixed-n 204 + timed 252, 3 seeds). The 2D Ritz objective was corrected to a "
        "variational QR-projection eigh; ALL PRIOR B PERFORMANCE NUMBERS (complete_ci 576, "
        "earlier timed curves, walltime ratios) are INVALID_RITZ_OBJECTIVE and retracted. "
        "Per-cell median dE (3 seeds) in experiments.fixed_walltime_bench: no single method "
        "wins across all systems. At fixed n, orbital-optimized Slater is best on Hubbard "
        "(L6 U1 @n=12 med 4.6e-3 vs CIPSI 2.85, replica 2.85); replica convex-J tracks CIPSI "
        "almost exactly on Hubbard but is slightly better on H2 mid-n. At fixed walltime "
        "1.5s L6 U1 orbopt med 2.9e-3 beats CIPSI 0.34, but L6 U8 CIPSI 0.015 beats orbopt 0.23. "
        "Throughput != energy; no universal winner and no advertised speedup.")
    wt_str = "(prior 3-12x walltime ratio RETRACTED -- invalid Ritz objective)"

    summary = {
        "conclusion": {
            "headline": headline,
            "current_data": {
                "csv": "results/timed_bench.csv",
                "csv_sha256": "80ef0d85592a30bbbf926682ff5e9714af83caabd8d928b140ab5b1a4e4e4292",
                "run_version": "ritzfix_v2",
                "n_rows": 456,
                "checkpoints": "results/ckpt/*.npz (456; keys H,psi,E,n,stage,residual,E_hist,n_hist,t_hist,r_hist)",
            },
            "advantage_demonstrated": False,
            "speedup_observed": False,
            "nonorth_atoms_selected_total": None,
            "nonorth_note": ("current ritzfix_v2 456-run did not tally this metric; "
                             "the old complete_ci count is INVALID_RITZ_OBJECTIVE and not migrated."),
            "mean_vs_median_note": ("mean and median differ strongly because one "
                                    "seed (L6_U8 seed2) is an outlier at dE~7.5; "
                                    "both are reported separately per cell."),
        },
        "source_audit": {
            "documents_read_only": ["Gaussian_Convex_Synthesis_Algorithm.pdf",
                                    "paper_v2/sections/primal_core.tex",
                                    "paper_v2/sections/dual_core.tex",
                                    "gaussian_synthesis/src/gaussian_synthesis/*.py"],
            "mechanism_confirmed": "Strictly convex dual witness; KKT makes tilted atom sum = residual. Secant Y_R is a signed linear superposition; the convex object is the phase-balanced gauge tau=sum|c_a| (NOT a non-negative convex hull).",
        },
        "experiments": {
            "dimer": {"max_relerr_xi_S": drelerr, "status": "ACCEPTED analytic identity"},
            "fock_complete_feasibility": {
                "classification": "INVALID_RITZ_OBJECTIVE",
                "status": "HISTORICAL INVALID. Old complete_ci 576 used the non-variational eigvalsh(S^-1 H) 2D objective; its numbers are retracted and NOT current. Archive file retained at results/complete_ci.csv for provenance only. Current results = fixed_walltime_bench (ritzfix_v2).",
                "kkt_residual_median": kmed, "kkt_residual_max": kmax,
            },
            "local_projected_dictionary": {
                "classification": "historical_diagnostic",
                "status": "NOT CURRENT. Kept at results/diagnostics/local_projected_benchmark.csv only to document the first rejected (projected-local) run. Its proj_residual ~0.95 meant the local dictionary did not span the residual; do not cite as a current result.",
            },
            "fixed_walltime_bench": {
                "setup": "preload models + FCI outside clock; hard deadline 0.3/0.8/1.5s; checkpoint only on completed Ritz <= deadline; rollback on overrun; E=current Ritz Rayleigh for residual, FCI scores only.",
                "systems": ["L4 U1/U8", "L6 U1/U8", "H2 STO-3G R=0.74/1.5/2.0"],
                "methods": ["cipsi PT2", "greedy_noci finite-pool", "orbopt_noci real Thouless L-BFGS", "replica_energy convex-J"],
                "result": timed_table(),
                "run_version": "ritzfix_v2 (variational QR-projection 2D Ritz; old eigvalsh(S^-1 H) invalid)",
                "finding": ("After correcting the 2D Ritz to a variational QR-projection eigh (old non-Hermitian "
                    "eigvalsh(S^-1 H) gave fake below-ground energies, e.g. -1.867 vs true -0.414): "
                    "(a) replica convex-J TRACKS CIPSI almost exactly at matched n (L6 U1 @n=12: both 2.85) -- NO advantage. "
                    "(b) The real orbital-optimized Slater baseline is now dramatically stronger at matched n: "
                    "L6 U1 @n=12 orbopt med dE=6e-3 vs CIPSI 2.85, replica 2.85; L4 U1 @n=12 orbopt=2e-4 vs CIPSI 1.02. "
                    "(c) H2 dim=4: orbopt reaches ~FCI (1e-10..1e-12); CIPSI/replica reach FCI at n=4. "
                    "(d) Fixed walltime 1.5s L6: CIPSI still wins on raw determinant throughput (n=160, dE=0.32) "
                    "because each orbopt step is an L-BFGS; orbopt reaches dE=2.5e-3 in ~few steps but slower per det. "
                    "Net: no demonstrated advantage for replica convex-J; the corrected orbital-optimized Slater is the "
                    "strongest matched-n baseline; CIPSI wins raw walltime throughput."),
            },
            "h2_demo": {
                "basis": "STO-3G, neutral singlet, 2e / 2 spatial orbitals",
                "geometries_A": [0.74, 1.5, 2.0],
                "FCI_check_vs_pyscf": "<1e-9 electronic",
                "gauge_tau_reported": [1.898, 1.834, 1.789],
                "gauge_note": "UPPER bound on a restricted 20-atom dictionary, NOT the exact Takagi extent (N=2e/m=4 spin-orbitals: tau <= sqrt(2)=1.414).",
            },
        },
        "tests": {
            "unit_tests_passed": 7,
            "list": ["hubbard_dimer_energy", "dimer_extent_matches_analytic",
                     "dimer_extent_brackets", "convex_j_orthonormal_closedform",
                     "convex_j_k1_frame_inverse", "slater_det_is_single_fock_state",
                     "h2_matches_pyscf_fci"],
            "reviewer_b_noci_test": "true KKT residual ~1e-7 (was 0.79-1.0 in v1)",
        },
        "limitations": [
            "Finite local Grassmannian dictionary; not a continuous-Grassmannian certificate.",
            "Non-orth atoms were never selected; the experiment does not test NOCI selection advantage.",
            "One outlier seed inflates means; medians are the robust comparison.",
            "H2 is a 2e toy; not a QC scaling claim.",
        ],
        "files": {
            "summary": str(HERE / "summary.json"),
            "code": ["hubbard.py", "dimer_extent.py", "noci.py", "selected_ci.py",
                     "complete_ci.py", "molecule.py", "run_complete.py",
                     "make_summary.py", "README.md"],
            "data": ["results/complete_ci.csv", "results/dimer_extent.csv",
                     "results/h2_demo.csv",
                     "results/diagnostics/local_projected_benchmark.csv"],
        },
    }
    json.dump(summary, open(HERE / "summary.json", "w"), indent=2)
    print("summary.json regenerated; nonorth picks =", nonorth,
          "kkt median=%.1e" % kmed)


if __name__ == "__main__":
    main()
