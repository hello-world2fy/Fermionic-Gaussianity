#!/usr/bin/env python3
"""Independent ground eigensolve and finite-span checks of B1/B2 artifacts."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from fk5.io import write_json
from fk5.two_orbit import actual_arc_atoms
from fk.model import build_model
from fk2.orbits import covariance_from_kets


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    s=json.loads((args.run/'summary.json').read_text());c=s['tighter_all_decomposition_certificate']
    mp.dps=110
    H=mp.matrix([[-56,20*mp.sqrt(2),0],[20*mp.sqrt(2),-14,20*mp.sqrt(14)],[0,20*mp.sqrt(14),-8]])
    e,V=mp.eigsy(H);z=[V[i,0] for i in range(3)]
    if z[0]<0:z=[-x for x in z]
    a=mp.mpf(2000)/2713
    A=[(2-a*a)/4,-a/mp.sqrt(2),(2+5*a*a)/(4*mp.sqrt(7))]
    norm2=sum(x*x for x in A)
    F=sum(x*y for x,y in zip(A,z))**2/norm2
    lo,hi=map(mp.mpf,s['contact_only_fidelity_ceiling_interval']);assert lo<=F<=hi<1
    lo,hi=map(mp.mpf,s['contact_only_ceiling_attaining_cost_interval']);assert lo<=1/mp.sqrt(norm2)<=hi
    S=z[0]-mp.sqrt(8)*z[1]+mp.sqrt(7)*z[2];L0=2*z[0]
    U=mp.mpf(c['cost_upper']);lam=mp.mpf(4713)/1000
    low=(mp.sqrt(S/U)-1)**2;upper=2*(1-L0/U)
    assert mp.mpf(c['all_cost_le_U_mean_eta_interval'][0])<=low
    assert mp.mpf(c['all_cost_le_U_mean_eta_interval'][1])>=upper
    Hsupport=1+lam+2/(lam-2)
    variance=(Hsupport-(S+lam*L0)/U)/((lam-2)/2)
    assert mp.mpf(c['all_cost_le_U_mean_squared_sqrt_eta_displacement_upper'])>=variance
    model=build_model();g=actual_arc_atoms(model,float(a));cov=covariance_from_kets(g)
    assert np.max(abs(cov@cov+np.eye(16)))<1e-12
    expected=np.array([float(x) for x in A]);projected=g@model.S.conj()
    np.testing.assert_allclose(projected,np.repeat(expected[None,:],8,axis=0),atol=1e-14,rtol=0)
    # Optimize fidelity over the complete complex span of these actual atoms.
    # Its optimum must equal the analytic ceiling, attained by their average.
    Umat,sv,_=np.linalg.svd(g.T,full_matrices=False);basis=Umat[:,sv>1e-12]
    target=model.target(20);best=float(np.linalg.norm(basis.conj().T@target)**2)
    assert abs(best-float(F))<3e-14
    write_json(args.run/'verification.json',dict(artifact_verification_passed=True,
        independent_ground_eigensolve_digits=110,
        independent_covariance_and_full_complex_span_checked=True,
        independent_exact_target_fidelity_ceiling=mp.nstr(F,80),
        tighter_eta_and_concentration_bounds_checked=True,
        global_contact_classification_source='JOINT_CONTACT_THEORY.md; equality-condition proof, not numerical maximization.',
        global_quotient_distance_stability_constant_certified=False))
    print('Verified contact-only ceiling and all-decomposition eta bounds',flush=True)


if __name__=='__main__':main()
