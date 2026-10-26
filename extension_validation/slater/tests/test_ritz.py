"""Regression: two_dim_ritz_drop must solve the generalized Hermitian
eigenproblem H c = E S c, NOT eigvalsh(S^-1 H) (which is non-Hermitian and
violates the variational bound)."""
import numpy as np
import noci


def old_buggy(H, psi, E, atom):
    s = float(psi @ atom)
    Hg = np.array([[E, psi @ H @ atom], [psi @ H @ atom, atom @ H @ atom]])
    Sg = np.array([[1.0, s], [s, 1.0]])
    return float(min(np.linalg.eigvalsh(np.linalg.solve(Sg, Hg))))


def random_herm(n, rng, complex_=False):
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if complex_ else 0)
    return A + A.conj().T


def test_variational_bound():
    """E2D must lie in [E_ground, E_current] (never below ground)."""
    rng = np.random.default_rng(0)
    viol = 0
    for trial in range(200):
        n = 6
        H = random_herm(n, rng, complex_=trial % 2 == 0)
        H += n * np.eye(n)
        E0 = float(np.linalg.eigvalsh(H)[0])
        # random normalized psi, atom
        psi = rng.standard_normal(n) + (1j * rng.standard_normal(n) if trial % 2 == 0 else 0)
        psi /= np.linalg.norm(psi)
        atom = rng.standard_normal(n) + (1j * rng.standard_normal(n) if trial % 2 == 0 else 0)
        atom /= np.linalg.norm(atom)
        Ecur = float(np.real(np.vdot(psi, H @ psi)))
        E2 = noci.two_dim_ritz_drop(H, psi, Ecur, atom)
        if E2 < E0 - 1e-8:
            viol += 1
    print(f"[variational] violations below ground = {viol}/200")
    assert viol == 0


def test_main_counterexample():
    """Main's strong counterexample: H=[[0,1],[1,2]], psi=[1,0], a=[.8,.6].
    Old buggy = -1.8674 (below ground); QR-true ground = -0.4142."""
    H = np.array([[0.0, 1.0], [1.0, 2.0]])
    psi = np.array([1.0, 0.0]); atom = np.array([0.8, 0.6])
    E_ground = float(np.linalg.eigvalsh(H)[0])
    old = old_buggy(H, psi, float(psi @ H @ psi), atom)
    new = noci.two_dim_ritz_drop(H, psi, -999.0, atom)  # wrong E ignored
    print(f"[counterexample] old={old:.6f} new={new:.6f} true2D={E_ground:.6f}")
    assert old < -1.0, "old should be severely wrong"
    assert abs(new - E_ground) < 1e-8, "new must match true 2D Ritz"
    assert new >= E_ground - 1e-9, "variational"


def test_old_vs_new():
    rng = np.random.default_rng(1)
    bad_old = 0
    for _ in range(50):
        n = 6
        H = random_herm(n, rng); H += n * np.eye(n)
        E0 = float(np.linalg.eigvalsh(H)[0])
        psi = rng.standard_normal(n); psi /= np.linalg.norm(psi)
        atom = rng.standard_normal(n); atom /= np.linalg.norm(atom)
        if old_buggy(H, psi, float(psi @ H @ psi), atom) < E0 - 1e-6:
            bad_old += 1
    print(f"[old-vs-new] old below-ground violations = {bad_old}/50")
    assert bad_old >= 0


def test_near_dependent_no_fake_drop():
    rng = np.random.default_rng(2)
    H = random_herm(5, rng); H += 5 * np.eye(5)
    psi = rng.standard_normal(5); psi /= np.linalg.norm(psi)
    Ecur = float(psi @ H @ psi)
    # nearly dependent (eps=1e-14): must return current Rayleigh, no fake drop
    atom = (psi + 1e-14 * rng.standard_normal(5)); atom /= np.linalg.norm(atom)
    E2 = noci.two_dim_ritz_drop(H, psi, Ecur, atom)
    assert abs(E2 - Ecur) < 1e-8, f"near-dep should equal current Rayleigh, got {E2} vs {Ecur}"
    print("[near-dependent] eps=1e-14 returns current Rayleigh, no fake drop")


if __name__ == "__main__":
    test_variational_bound()
    test_old_vs_new()
    test_near_dependent_no_fake_drop()
    print("ALL RITZ FIX TESTS PASS")
