"""Unit tests (run with the project venv)."""
import sys
import math
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from hubbard import HubbardModel
from dimer_extent import dimer_extent_numerical, xi_S_analytic
import noci


def test_hubbard_dimer_energy():
    for U in [0.0, 1.0, 4.0, 8.0]:
        m = HubbardModel(L=2, t=1.0, U=U)
        E, _ = m.exact_ground()
        E_an = (U - math.sqrt(U ** 2 + 16)) / 2
        assert abs(E - E_an) < 1e-12


def test_dimer_extent_matches_analytic():
    for U in [0.0, 1.0, 4.0, 16.0]:
        r = dimer_extent_numerical(U, 1.0)
        xa = xi_S_analytic(U, 1.0)
        assert abs(r["xi_S"] - xa) < 1e-10
        assert abs(r["lam1"] ** 2 + r["lam2"] ** 2 - 1) < 1e-10


def test_dimer_extent_brackets():
    assert abs(dimer_extent_numerical(0.0)["xi_S"] - 1.0) < 1e-9
    assert dimer_extent_numerical(64.0)["xi_S"] > 1.99


def test_convex_j_orthonormal_closedform():
    """On an orthonormal dictionary the KKT C(u)=r must hold and the
    unconstrained optimum must be u*=sign(r)|r|^{1/(2k-1)}."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((10, 5)))
    r = Q @ rng.standard_normal(5)
    out = noci.convex_j_synthesize(Q, r, k=2.0)
    assert out["kkt_err"] < 1e-5, out["kkt_err"]
    # closed form in the 5-dim row space
    rr = Q.T @ r
    ustar = Q @ (np.sign(rr) * np.abs(rr) ** (1.0 / 3))
    assert np.linalg.norm(out["w"] / np.linalg.norm(out["w"])
                          - ustar / np.linalg.norm(ustar)) < 1e-4


def test_convex_j_k1_frame_inverse():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 4))
    ctrue = rng.standard_normal(4)
    r = A @ ctrue; r /= np.linalg.norm(r)
    out = noci.convex_j_synthesize(A, r, k=1.0)
    assert out["kkt_err"] < 1e-6


def test_slater_det_is_single_fock_state():
    m = HubbardModel(L=4, t=1.0, U=4.0)
    Uup, Udn = noci.reference_U(m, m.neel_determinant())
    v = noci.slater_fock_vector(m, Uup, Udn)
    assert (np.abs(v) > 1e-12).sum() == 1


def test_h2_matches_pyscf_fci():
    warnings.filterwarnings("ignore")
    from molecule import h2_active
    ham, fci_e, rhf_e, mf = h2_active(0.74)
    E, _ = ham.exact_ground()
    assert abs(E - (fci_e - mf.energy_nuc())) < 1e-9


def run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    run_all()
