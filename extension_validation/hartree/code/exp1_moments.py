"""Exp A1: exact bosonic-frame moments, formula vs independent Gamma integral.

Verifies for m=2,3,4 multi-occupancy Fock states:
  q_k(n) = multinomial(N,n)^k / multinomial(kN, k n)   (closed form)
       vs d_{N,k} * E|<g^N|n>|^{2k}                    (independent Gamma route)
  F_1(n) = multinomial(n) prod(n_j/N)^{n_j}
  xi(n)  = 1/F_1(n)
  B_k^2 -> xi(n)  with predicted asymptotic.
Deterministic; records walltime and errors.
"""
import os, sys, json, time, math, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boson_frame as bf

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
os.makedirs(RES, exist_ok=True)

CASES = [
    # (m, N, label of multi-occupancy states to highlight)
    (2, 4, [(2, 2), (1, 3)]),
    (2, 6, [(3, 3), (2, 4)]),
    (3, 3, [(1, 1, 1), (2, 1, 0)]),
    (3, 4, [(2, 1, 1), (2, 2, 0), (1, 1, 2)]),
    (4, 4, [(1, 1, 1, 1), (2, 1, 1, 0), (2, 2, 0, 0)]),
]
KMAX = 12


def main():
    t_start = time.time()
    rows = []
    convergence = []
    for (m, N, highlights) in CASES:
        basis = bf.fock_basis(m, N)
        for n in basis:
            is_hi = list(n) in [list(h) for h in highlights]
            f1 = bf.F1_fock(n)
            xi = 1.0 / f1
            for k in range(1, KMAX + 1):
                qf = bf.qk_formula(n, k)
                qi = bf.qk_integral(n, k, m)
                rel = abs(qf - qi) / max(1e-300, abs(qf))
                rows.append(dict(m=m, N=N, n=str(n), highlight=is_hi, k=k,
                                 q_formula=qf, q_integral=qi, relerr=rel,
                                 F1=f1, xi=xi))
            # B_k^2 convergence for highlighted multi-occupancy states
            if is_hi:
                for k in range(1, 41):
                    Bk = bf.Bk_closed(n, k, m)
                    tk = bf.tk_closed(n, k)
                    Gam = bf.d_dim(m, N, k) ** (1.0 / k)
                    convergence.append(dict(m=m, N=N, n=str(n), k=k,
                                            Bk2=Bk ** 2, tk2=tk ** 2, xi=xi,
                                            Gamma=Gam,
                                            upper_violation=Bk ** 2 - xi,
                                            lower_violation=xi - tk ** 2))
    worst = max(r["relerr"] for r in rows)
    # write csv
    with open(os.path.join(RES, "exp1_moments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(RES, "exp1_convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(convergence[0].keys()))
        w.writeheader(); w.writerows(convergence)

    # summary numbers
    summary = {
        "walltime_s": time.time() - t_start,
        "n_cases": len(rows),
        "worst_qk_relerr": worst,
        "all_qk_match": worst < 1e-11,
        "Bk_convergence_checks": [],
    }
    # check certified sandwich  tk^2 <= xi <= Bk^2  for highlighted states
    for nkey in set(c["n"] for c in convergence):
        sub = [c for c in convergence if c["n"] == nkey]
        ok_low = all(c["tk2"] <= c["xi"] * (1 + 1e-9) for c in sub)
        ok_up = all(c["Bk2"] >= c["xi"] * (1 - 1e-9) for c in sub)
        last = sub[-1]
        summary["Bk_convergence_checks"].append(dict(
            n=nkey, xi=last["xi"],
            tk2_at_k40=last["tk2"], Bk2_at_k40=last["Bk2"],
            gamma_at_k40=last["Gamma"],
            sandwich_lower_ok=ok_low, sandwich_upper_ok=ok_up,
            rel_gap_at_k40=(last["Bk2"] - last["tk2"]) / last["xi"]))
    with open(os.path.join(RES, "exp1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
