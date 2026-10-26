"""Fail-closed audit unit test: legal atoms pass, deliberately broken atoms RAISE.

Writes results to artifacts/logs/selftest_audit.log.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sggen.audit import audit_ket, audit_pool
from sggen.gaussian import (magic_ket, free_slater_atoms, contact_kets,
                            product_family_kets, sample_even_gaussian_rays)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "logs")
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "selftest_audit.log"), "w")
def say(s): print(s); log.write(s + "\n")

fails = []
# 1. legal atoms
free = free_slater_atoms()
for i, k in enumerate(free):
    m = audit_ket(k, label=f"free#{i}"); say(f"free#{i} quadric={m['quadric']:.1e} norm={m['norm']:.3f} OK")
rng = np.random.default_rng(0)
for i, k in enumerate(contact_kets(rng, 20)):
    audit_ket(k, label=f"contact#{i}")
say("contact(20): all pass")
for i, k in enumerate(product_family_kets([0.5, 1.0, 2.0])):
    audit_ket(k, label=f"product#{i}")
say("product: all pass")
for i, k in enumerate(sample_even_gaussian_rays(rng, 20)):
    audit_ket(k, label=f"haar#{i}")
say("haar(20): all pass")

# 2. ILLEGAL atoms must RAISE (fail closed)
# (a) magic M itself: NOT Gaussian (quadric != 0)
M = magic_ket()
try:
    audit_ket(M, label="M-must-fail"); fails.append("M did NOT raise")
except ValueError as e:
    say(f"M correctly raises: {e}")
# (b) unnormalized
try:
    audit_ket(2.0 * free[0], label="unnorm"); fails.append("unnorm did NOT raise")
except ValueError as e:
    say(f"unnormalized correctly raises: {e}")
# (c) random non-Gaussian vector
try:
    audit_ket(rng.standard_normal(8) + 1j*rng.standard_normal(8), label="random")
    fails.append("random did NOT raise")
except ValueError as e:
    say(f"random vector correctly raises: {e}")

say(f"\nRESULT: fails={fails if fails else 'none'}  audit_fail_closed={'PASS' if not fails else 'FAIL'}")
log.close()
print("wrote", os.path.join(OUT, "selftest_audit.log"))
