"""Assemble the master summary.json with the unified fields for the master report."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
magic = json.load(open(os.path.join(OUT, "magic_summary.json")))
hub = json.load(open(os.path.join(OUT, "hubbard_summary.json")))
conv_path = os.path.join(OUT, "convex_summary.json")
conv = json.load(open(conv_path)) if os.path.exists(conv_path) else None
exact_path = os.path.join(OUT, "exact_convex_summary.json")
exact = json.load(open(exact_path)) if os.path.exists(exact_path) else None

# analytic witness proof statement (closed form, WITH the 1/2 prefactor)
witness_proof = (
    "For even Gaussian ket with pairing Z, singular values mu1,mu2 and Pfaffian P: "
    "N^2=1/((1+mu1^2)(1+mu2^2)), |<M|g>|^2 = (1/2) N^2 |1+P|^2.  Triangle + |P|<=mu1 mu2 "
    "gives |<M|g>|^2 <= (1/2)(1+mu1 mu2)^2/((1+mu1^2)(1+mu2^2)) <= 1/2, since "
    "(1+mu1^2)(1+mu2^2)-(1+mu1 mu2)^2 = (mu1-mu2)^2 >= 0.  Hence Tr(W sigma_g)<=1 for even g; "
    "the odd sector gives Tr(W sigma_g)=0 exactly (both rho_M and P_+ annihilate odd kets), "
    "so the bound |Tr(W sigma_g)|<=1 holds.  Equality on product BCS g(t,t) for ALL t (mu1=mu2)."
)

summary = {
    "experiment": "C: signed Gaussian operator convex synthesis (independent re-implementation)",
    "seed": 20261026,
    "conclusion": {
        "rho_M_R": 3.0,
        "rho_M_R_status": "BRACKETED, not exact: analytic continuous lower=3, finite-dictionary LP upper=%.6f (~2e-5 gap)" % magic["rho_M_R_LP"],
        "rho_M_R_LP_numerical": magic["rho_M_R_LP"],
        "rho_M_R_lower_witness": magic["rho_M_R_lower_witness"],
        "white_noise_R_curve": "CONSISTENT with max(1, 3-7p/2), pc=4/7 (LP upper matches witness lower line to ~4 digits; plateau needs continuum hull argument)",
        "witness_W_valid": True,
        "witness_sup_continuous": 1.0,
        "hubbard_few_observable_witness_rigorous": False,
        "hubbard_few_observable_witness_finite_pool_only": True,
        "one_line": ("R(|M><M|) bracketed between analytic witness lower 3 and LP upper 3.000006; "
                     "white-noise curve consistent with max(1,3-7p/2); real convex J_k(W) program "
                     "solved (gradient verified to 2e-11); Hubbard U=0 baseline=1 at all beta after "
                     "adding Slater atoms, interaction excess dictionary-dependent; no rigorous "
                     "few-observable witness (finite-pool ratio shrinks as pool refines).")
    },
    "source_audit": {
        "pdf": "Gaussian_Convex_Synthesis_Algorithm.pdf (8 pp): pure-state ket synthesis, "
               "C_k(u) reproducing kernel, q_k(M)=2^-k b_k. This is the KET extent problem "
               "(M extent xi_G=2), NOT the density-matrix signed operator TV cost R tested here.",
        "reference_code": "gaussian_synthesis/src/gaussian_synthesis/: even-parity-only (8-dim), "
                          "pairing chart + spin_to_fock Haar sampler, projected-SGD C_k(u).",
        "paper_v2": "sections/dual_core.tex L89-94: gauge tau(v)=max_{h(y)<=1} Re<y|v>.",
        "difference_from_reference": "Reference: psi=integral f(g)|g>dg (vector, signed). "
                                     "Here: rho=integral w(g)|g><g|dg (operator, TV norm).",
        "independent_review": "extension_validation/review/c_findings.md: odd=0, 1/2 prefactor, "
                              "R=3 bracketed not exact, Hubbard U=0 dictionary bias, grand-canonical, "
                              "pool-unstable witness, white-noise plateau needs hull argument."
    },
    "experiments": {
        "rho_M": {
            "R_LP": magic["rho_M_R_LP"], "residual": magic["rho_M_residual"],
            "n_nonzero_atoms": magic["rho_M_n_nonzero"],
            "positive_mass": magic["rho_M_positive_mass"],
            "negative_mass": magic["rho_M_negative_mass"],
            "witness_analytic": witness_proof,
            "witness_sup_numeric": magic["witness_sup_numeric"],
            "odd_sector_value": 0.0,
        },
        "convex_Jk_MC_failed_attempt": conv,
        "convex_Jk_exact_finite_pool": exact,
        "triality_identity": ("rho_M = 2 sigma_plus - sigma_perp holds to ||.||=1.9e-16 for an "
                              "arbitrary real ONB (TV=3 algebraically), but our basis does NOT "
                              "make g_ij±=(e_i+i e_j)/sqrt2 satisfy the Gaussian quadric (it!=0); "
                              "the special antiunitary particle-hole basis is unverified, so the "
                              "exact upper bound 3 is NOT certified. Bracketed statement retained."),
        "white_noise": "see magic_curve.csv; R_LP consistent with max(1,3-7p/2); plateau not proven from continuum hull",
        "hubbard": {
            "grid": hub["grid"],
            "grand_canonical": True,
            "u0_baseline_Rnorm": hub["u0_baseline_Rnorm"],
            "slater_atoms_added": hub["slater_atoms_added"],
            "trend_note": ("U=0 baseline R_norm=1.0 at all beta after adding quadratic-Gibbs Slater "
                           "eigenstates (fixes the ~2x random-ray dictionary bias). Interaction signal "
                           "is excess=R(U,beta)-1 and remains dictionary-dependent; NOT a certified "
                           "physical interaction effect."),
            "beta_0p5_not_maxmixed": True,
            "witness": hub["witness"],
            "witness_note": ("Finite-pool ratio ~5.7 (was 7.4 at 20k pool) SHRINKS as pool refines -> "
                             "pool-unstable, not a certificate. Rigorous ||W|| bound gives ratio<1. "
                             "Matching measured f(g)=(1,<O_a>) does not reconstruct the full state; "
                             "MC variance cost is not the sign problem.")
        }
    },
    "tests": {
        "pairing_chart_normalization": "||g(Z)||=1 to machine precision",
        "quadric_even": "g^T J g = 0 for even Gaussian, = 1.0 for M",
        "witness_sup_analytic_equals_numeric": "0.5 analytic == 0.500000 numeric",
        "odd_sector_value_zero": "Tr(W sigma_odd)=0.0 exactly (review C-F1)",
        "rho_M_trace_reconstruction": "residual 1.8e-15",
        "convex_gradient_finite_diff": conv["gradient_finite_diff_max_err"] if conv else None,
        "exact_finite_pool_KKT": {
            "rhoM_k1": exact["rho_M"]["1"]["kkt_proj"] if exact else None,
            "rhoM_k2": exact["rho_M"]["2"]["kkt_proj"] if exact else None,
            "hubbard_feat_k1": exact["hubbard_feature"]["1"]["kkt"] if exact else None,
            "hubbard_feat_k2": exact["hubbard_feature"]["2"]["kkt"] if exact else None,
        },
        "convex_KKT": "C_k(W_k)-rho residual reported for k=1,2,4",
        "u0_sanity_after_slater": "R_norm=1.0000 at beta=0.5..8 for U=0",
    },
    "limitations": [
        "R(rho_M)=3 is BRACKETED (analytic lower 3, LP upper 3.000006), not an exact decomposition; "
        "no finite analytic triality projector combination was found.",
        "White-noise R=1 plateau for p>=4/7 is only CONSISTENT with the hull argument, not proven.",
        "LP over a finite dictionary is an UPPER bound; Hubbard interaction excess is dictionary-dependent "
        "(U=0 baseline=1 after Slater atoms, but residual discretization remains at low T).",
        "Gibbs state is GRAND-canonical (mu=U/2, N=0..4); beta=0.5 is not maximally mixed.",
        "Few-observable witness: finite-pool ratio shrinks with pool size; the ||W|| bound not certifying "
        "does NOT prove no witness exists, only that our crude bound cannot certify one.",
        "Monte-Carlo convex J_k: KKT residual 0.04-0.1 is MC noise; single-threaded fixed seed, float64.",
    ],
    "files": {
        "master_summary": "artifacts/summary.json",
        "magic_csv": "artifacts/magic_curve.csv",
        "magic_json": "artifacts/magic_summary.json",
        "magic_npz": "artifacts/magic_artifacts.npz",
        "convex_json": "artifacts/convex_summary.json",
        "hubbard_csv": "artifacts/hubbard_curve.csv",
        "hubbard_json": "artifacts/hubbard_summary.json",
        "hubbard_npz": "artifacts/hubbard_star.npz",
        "figs": ["artifacts/figs/fig1_white_noise_curve.png",
                 "artifacts/figs/fig2_hubbard_heatmap.png",
                 "artifacts/figs/fig3_witness_pool_vs_rigorous.png"],
        "logs": ["artifacts/logs/magic_run2.log", "artifacts/logs/hubbard_run.log"],
        "code": ["sggen/gaussian.py", "sggen/synthesis.py", "sggen/convex.py", "sggen/hubbard.py",
                 "scripts/run_magic.py", "scripts/run_hubbard.py", "scripts/run_convex.py",
                 "scripts/make_figs.py", "scripts/assemble_summary.py"]
    },
    "walltime_s": magic["walltime_s"] + hub["walltime_s"] + (conv["walltime_s"] if conv else 0),
}

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("wrote", os.path.join(OUT, "summary.json"))
