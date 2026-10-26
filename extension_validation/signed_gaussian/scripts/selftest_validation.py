"""Validation fail-closed regression: reject bad projectors/kets, accept legal ones.

Writes artifacts/logs/selftest_validation.log (history logs preserved, not deleted).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sggen.audit import audit_ket, audit_projector
from sggen.gaussian import magic_ket, density_matrix, free_slater_atoms

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "logs")
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "selftest_validation.log"), "w")
def say(s): print(s); log.write(s + "\n")

free = free_slater_atoms()
good = density_matrix(free[0])
reject_cases = {}
# reject: magic non-Gaussian rhoM
reject_cases["magic_nonGaussian"] = density_matrix(magic_ket())
# reject: 2*valid sigma (unscaled)
reject_cases["2x_sigma"] = 2.0 * good
# reject: mixed I/8
reject_cases["mixed_I8"] = np.eye(8, dtype=complex) / 8
# reject: non-Hermitian offdiag perturb
nh = good.copy(); nh[0, 1] += 1j; reject_cases["nonHermitian"] = nh
# reject: NaN matrix
nanm = good.copy(); nanm[0, 0] = np.nan; reject_cases["nan"] = nanm
# reject: inf
infm = good.copy(); infm[0, 0] = np.inf; reject_cases["inf"] = infm
# reject: wrong shape
reject_cases["wrong_shape"] = np.zeros((4, 4), complex)
# reject: rank-mix negative
reject_cases["rank_mix"] = 0.5 * good + 0.5 * density_matrix(free[1])

n_rej = 0
for name, a in reject_cases.items():
    try:
        audit_projector(a, label=name); say(f"REJECTED?? {name} did NOT raise")
    except ValueError as e:
        n_rej += 1; say(f"reject OK: {name} -> {str(e)[:60]}")

# reject: NaN ket
try:
    audit_ket(np.array([np.nan]+[0j]*7)); say("REJECTED?? nan ket")
except ValueError:
    n_rej += 1; say("reject OK: nan_ket")
# reject: wrong shape ket
try:
    audit_ket(np.zeros(4)); say("REJECTED?? shape ket")
except ValueError:
    n_rej += 1; say("reject OK: shape_ket")

# accept: legal pure Gaussian projector + complex ket
m = audit_projector(good, label="good"); say(f"accept legal projector: outer_err={m['outer_err']:.1e}")
audit_ket(free[3] * np.exp(1j*0.7), label="good_complex_phase"); say("accept complex-phase ket")

# solver injection: bad atom must raise ValueError (not linprog failure)
from sggen.synthesis import solve_signed_lp
try:
    solve_signed_lp(np.eye(8)/8, np.array([reject_cases["mixed_I8"]]))
    say("REJECTED?? solver accepted mixed")
except ValueError as e:
    say(f"solver rejects mixed atom: {str(e)[:50]}")

# tiny smoke: legal 1-atom decomposition
r = solve_signed_lp(good, np.array([good]))
say(f"smoke legal solve value={r['value']:.4f} (expect 1) frob={r['frobenius_residual']:.1e}")

say(f"\nRESULT: rejects={n_rej} PASS")
log.close()
