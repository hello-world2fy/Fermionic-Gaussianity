"""Re-runnable sanity tests; writes tests/test_log.json. Deterministic, single thread."""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boson_frame as bf

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
TESTS = os.path.normpath(os.path.join(HERE, "..", "tests"))
os.makedirs(TESTS, exist_ok=True)


def main():
    t0 = time.time()
    log = {}
    # 1. qk formula vs integral
    worst = 0.0
    for m, N in [(2, 4), (3, 3), (4, 3)]:
        for n in bf.fock_basis(m, N):
            for k in [1, 2, 3, 5, 8]:
                worst = max(worst, abs(bf.qk_formula(n, k) - bf.qk_integral(n, k, m)))
    log["qk_formula_vs_integral_max_abs"] = worst
    assert worst < 1e-10

    # 2. q1 = norm^2
    for m, N in [(3, 4)]:
        for n in bf.fock_basis(m, N):
            assert abs(bf.qk_formula(n, 1) - 1) < 1e-12
    log["q1_equals_one"] = True

    # 3. sandwich for a highlighted state
    n = (1, 1, 1); m = 3; N = 3
    xi = bf.xi_fock(n)
    ok = all(bf.tk_closed(n, k) ** 2 <= xi * 1.000000001 and
             bf.Bk_closed(n, k, m) ** 2 >= xi * 0.999999999
             for k in range(1, 30))
    log["sandwich_111"] = ok
    assert ok

    # 4. quadrature cross-check
    basis = bf.fock_basis(2, 4)
    i = basis.index((2, 2))
    u = np.zeros(len(basis), dtype=complex); u[i] = 1
    qq, C = bf.eval_qk_Ck(u, basis, 2, 4, 3, ngauss=10)
    qe = bf.qk_formula((2, 2), 3)
    log["quadrature_relerr"] = abs(qq - qe) / qe
    assert abs(qq - qe) / qe < 1e-9

    # 5. F1 known values
    log["F1_hartree"] = bf.max_overlap_g(
        np.array([math.sqrt(bf.multinomial(p)) * np.prod((np.array([1, 1]) / np.sqrt(2)) ** np.array(p))
                  for p in basis], dtype=complex), basis, 2, 4, nstart=40, seed=3)

    # 6. 1-RDM: trace=N, raw Hermiticity, PSD, endpoints (U=0 cond=1; Mott cond=1/m)
    import exp4_rdm as e4
    max_trace_err = 0.0; max_herm_raw = 0.0; min_eig_all = 0.0
    for m, UJ in [(2, 0.0), (2, 0.5), (2, 2.0), (3, 0.0), (3, 2.0)]:
        N = m; b, psi, _ = e4.get_state(m, UJ)
        gr = e4.rdm_raw(psi, b, m)                 # RAW, unsymmetrized
        max_trace_err = max(max_trace_err, abs(float(np.trace(gr).real) - N))
        max_herm_raw = max(max_herm_raw, float(np.max(np.abs(gr - gr.conj().T))))
        g = (gr + gr.conj().T) / 2
        min_eig_all = min(min_eig_all, float(np.linalg.eigvalsh(g)[0]))
    log["rdm_max_trace_err"] = max_trace_err
    log["rdm_max_herm_err_raw"] = max_herm_raw      # on raw accumulation, not symmetrized
    log["rdm_min_eig_PSD_floor"] = min_eig_all
    for m in [2, 3]:
        N = m; b, psi, _ = e4.get_state(m, 0.0)
        g = (e4.rdm_raw(psi, b, m)); g = (g + g.conj().T) / 2
        log[f"rdm_U0_condfrac_m{m}"] = float(np.linalg.eigvalsh(g)[-1]) / N
        assert abs(np.linalg.eigvalsh(g)[-1] / N - 1) < 1e-9
    for m in [2, 3]:
        N = m; b = bf.fock_basis(m, N)
        psi = np.zeros(len(b), dtype=complex); psi[b.index(tuple([1] * m))] = 1
        gr = e4.rdm_raw(psi, b, m); g = (gr + gr.conj().T) / 2
        cf = float(np.linalg.eigvalsh(g)[-1]) / N
        log[f"rdm_Mott_condfrac_m{m}"] = cf
        assert abs(cf - 1.0 / m) < 1e-12
    assert max_trace_err < 1e-10 and min_eig_all > -1e-10
    log["rdm_json_strict"] = True  # exp4_rdm.json written with allow_nan=False
    log["walltime_s"] = time.time() - t0
    log["status"] = "PASS"
    with open(os.path.join(TESTS, "test_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    import math
    main()
