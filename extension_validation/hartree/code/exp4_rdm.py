"""Exp A4 (format-fixed): true one-body RDM gamma_ij = <a_i^dag a_j> for BH ground states.

Fixes vs prior version:
  * rdm() returns the RAW (unsymmetrized) accumulation. We verify Hermiticity on the raw
    matrix first, THEN symmetrize for stable eigensolve. trace/PSD reported raw vs processed.
  * JSON is strict (allow_nan=False): Mott endpoint uses UoverJ=null + endpoint_label='Mott_J0'.
  * CSV writes 'Mott_J0' (never inf).
  * Certified xi bounds are JOINED read-only from exp3 (same M,N,U/J, k=5) -- no re-optimization.
    U=0 (pure Hartree) and Mott endpoints report the analytic exact xi, flagged analytic.
"""
import os, sys, json, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boson_frame as bf

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
os.makedirs(RES, exist_ok=True)


def build_bh(basis, m, UoverJ=1.0, J=1.0):
    D = len(basis); idx = {p: i for i, p in enumerate(basis)}
    H = np.zeros((D, D), dtype=complex)
    for p, i in idx.items():
        H[i, i] += 0.5 * UoverJ * sum(nj * (nj - 1) for nj in p)
        for a in range(m - 1):
            if p[a] > 0:
                q = list(p); q[a] -= 1; q[a + 1] += 1; q = tuple(q)
                H[idx[q], i] += -J * math.sqrt(p[a] * (p[a + 1] + 1))
            if p[a + 1] > 0:
                q = list(p); q[a + 1] -= 1; q[a] += 1; q = tuple(q)
                H[idx[q], i] += -J * math.sqrt(p[a + 1] * (p[a] + 1))
    return (H + H.conj().T) / 2


def rdm_raw(psi, basis, m):
    """RAW gamma_ij = <a_i^dag a_j> accumulation (no symmetrization)."""
    idx = {p: i for i, p in enumerate(basis)}
    gam = np.zeros((m, m), dtype=complex)
    for nprime, ip in idx.items():
        for j in range(m):
            if nprime[j] == 0:
                continue
            for i in range(m):
                n = list(nprime); n[j] -= 1; n[i] += 1; n = tuple(n)
                if n not in idx:
                    continue
                amp = math.sqrt(nprime[j] * (nprime[i] - (1 if i == j else 0) + 1))
                gam[i, j] += np.conj(psi[idx[n]]) * psi[ip] * amp
    return gam  # unsymmetrized


def one_site_entropy(psi, basis, m):
    maxn1 = max(p[0] for p in basis)
    rho = np.zeros((maxn1 + 1, maxn1 + 1), dtype=complex)
    for i, p in enumerate(basis):
        for j, q in enumerate(basis):
            if p[1:] == q[1:]:
                rho[p[0], q[0]] += psi[i] * np.conj(psi[j])
    ev = np.linalg.eigvalsh(rho); ev = ev[ev > 1e-12]
    return -float(np.sum(ev * np.log2(ev)))


def get_state(m, UoverJ, J=1.0):
    N = m; basis = bf.fock_basis(m, N)
    H = build_bh(basis, m, UoverJ=UoverJ, J=J)
    E, vecs = np.linalg.eigh(H)
    psi = vecs[:, 0]; psi /= np.linalg.norm(psi)
    psi = psi * np.exp(-1j * np.angle(psi[np.argmax(np.abs(psi))]))
    return basis, psi, float(E[0])


def join_exp3_bounds():
    """read-only map (m, UoverJ) -> (xi_lower, xi_upper) at k=5 from exp3."""
    d = json.load(open(os.path.join(RES, "exp3_bosehubbard.json")))
    out = {}
    for r in d["results"]:
        w = r["witness"][-1]  # k=5
        out[(r["m"], r["UoverJ"])] = (w["xi_lower_certified"], w["xi_upper_certified"])
    return out


def analyze(psi, basis, m, N, bounds, endpoint_label=None, UoverJ=None, J=1.0, E0=0.0):
    g_raw = rdm_raw(psi, basis, m)
    trace_raw = float(np.trace(g_raw).real)
    herm_err = float(np.max(np.abs(g_raw - g_raw.conj().T)))
    g = (g_raw + g_raw.conj().T) / 2  # symmetrize for stable eigensolve
    ev = np.linalg.eigvalsh(g)
    min_eig_raw = float(ev[0]); lam_max = float(ev[-1])
    F1 = bf.max_overlap_g(psi, basis, m, N, nstart=400, seed=1)
    S = one_site_entropy(psi, basis, m)
    row = dict(m=m, N=N, UoverJ=UoverJ, J=J, endpoint_label=endpoint_label, E0=E0,
               trace_raw=trace_raw, trace_err=abs(trace_raw - N),
               herm_err_raw=herm_err, min_eig_processed=min_eig_raw,
               lam_max=lam_max, condensate_fraction=lam_max / N,
               F1_hartree_estimate=F1, entropy_site1_bits=S)
    # xi bounds: join exp3 for finite-U ED cases; analytic for endpoints
    if endpoint_label == "Mott_J0":
        row["xi_lower"] = row["xi_upper"] = 1.0 / bf.F1_fock(tuple([1] * m))
        row["xi_bound_kind"] = "analytic_Fock"
    elif UoverJ == 0.0:
        row["xi_lower"] = row["xi_upper"] = 1.0
        row["xi_bound_kind"] = "analytic_pure_Hartree"
    elif (m, UoverJ) in bounds:
        row["xi_lower"], row["xi_upper"] = bounds[(m, UoverJ)]
        row["xi_bound_kind"] = "certified_from_exp3_k5"
    else:
        row["xi_lower"] = row["xi_upper"] = None
        row["xi_bound_kind"] = "none"
    return row


def main():
    t0 = time.time()
    bounds = join_exp3_bounds()
    rows = []
    saved = {}
    for m in [2, 3]:
        for UJ in [0.0, 0.5, 2.0, 6.0]:
            N = m
            basis, psi, E0 = get_state(m, UJ)
            row = analyze(psi, basis, m, N, bounds, UoverJ=UJ, J=1.0, E0=E0)
            row["tag"] = f"M{m}_UJ{UJ}_J1"
            saved[row["tag"]] = (np.array(basis, dtype=int), psi)
            rows.append(row)
            print(f"{row['tag']}: trace_err={row['trace_err']:.1e} herm_raw={row['herm_err_raw']:.1e} "
                  f"cond={row['condensate_fraction']:.4f} F1={row['F1_hartree_estimate']:.4f} xi={row['xi_bound_kind']}")
    # analytic Mott endpoint |1,...,1>
    for m in [2, 3]:
        N = m; basis = bf.fock_basis(m, N)
        psi = np.zeros(len(basis), dtype=complex); psi[basis.index(tuple([1] * m))] = 1.0
        row = analyze(psi, basis, m, N, bounds, endpoint_label="Mott_J0", UoverJ=None, J=0.0, E0=0.0)
        row["tag"] = f"M{m}_Mott_J0"
        saved[row["tag"]] = (np.array(basis, dtype=int), psi)
        rows.append(row)
        print(f"{row['tag']}: herm_raw={row['herm_err_raw']:.1e} cond={row['condensate_fraction']:.4f} xi={row['xi_lower']:.4f}(analytic)")
    np.savez(os.path.join(RES, "bh_states.npz"),
             **{f"{k}_basis": v[0] for k, v in saved.items()},
             **{f"{k}_psi": v[1] for k, v in saved.items()})
    cols = ["tag", "m", "N", "UoverJ", "endpoint_label", "J", "E0", "trace_raw",
            "trace_err", "herm_err_raw", "min_eig_processed", "lam_max",
            "condensate_fraction", "F1_hartree_estimate", "entropy_site1_bits",
            "xi_lower", "xi_upper", "xi_bound_kind"]
    with open(os.path.join(RES, "exp4_rdm.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    with open(os.path.join(RES, "exp4_rdm.json"), "w") as f:
        json.dump(dict(walltime_s=time.time() - t0, rows=rows), f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
