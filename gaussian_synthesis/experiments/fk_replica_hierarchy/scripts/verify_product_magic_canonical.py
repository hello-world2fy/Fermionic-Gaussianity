#!/usr/bin/env python3
"""Verify every saved t0 particle and the complete factored 256N expansion."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','12')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4.io import write_json
from fk4.product_magic_twirl import proof_and_actions
from fk4.distributions import full_distribution
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    assert summary['status']=='complete'
    manifest=json.loads((args.run/'manifest.json').read_text())
    for rel,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((args.run/'source_snapshot'/rel).read_bytes()).hexdigest()==digest
    proof,perms,phases,target=proof_and_actions()
    saved=np.load(args.run/'finite_projector.npz')
    np.testing.assert_array_equal(saved['permutations'],perms)
    np.testing.assert_array_equal(saved['phases'],phases)
    np.testing.assert_array_equal(saved['target'],target)
    model=build_model()
    checks=[]
    for row in summary['rows']:
        for obs in row['observations']:
            data=np.load(args.run/f"{obs['key']}.npz")
            kets=data['P_kets'];factors=data['branch_coefficients_per_particle']
            a=kets.conj()@target
            np.testing.assert_allclose(data['aligned_phases'],a/abs(a),atol=1e-12)
            np.testing.assert_allclose(data['P_log_r'],2*np.log(abs(a)),atol=1e-11)
            cost=256*np.sum(abs(factors))
            np.testing.assert_allclose(cost,obs['B'],atol=1e-11)
            np.testing.assert_allclose(factors,obs['B']*a/abs(a)/(256*len(kets)),atol=1e-16)
            # Sum all 256N physical branches by linearity, with no omission
            # or interpretation of the averaged vector as a Gaussian atom.
            aggregate=factors@kets
            reconstructed=np.sum(phases*aggregate[perms],axis=0)
            residual=float(np.linalg.norm(reconstructed-target))
            assert residual<1e-11
            assert obs['finite_realization']['actual_Gaussian_terms']==256*len(kets)
            features,quality=full_distribution(kets,model.j_matrices)
            for name,values in features.items():np.testing.assert_allclose(values,data[name],atol=1e-10)
            eta=features['features'][:,3]
            envelope_error=float(np.max(2*abs(a)-(1-eta/2)))
            assert envelope_error<1e-9
            assert eta.mean()<=2*(1-2/cost)+1e-10
            checks.append({'key':obs['key'],'actual_terms':256*len(kets),
                'all_branch_reconstruction_residual':residual,
                'all_particle_purity_error':quality['max_purity_error_all_particles'],
                'coefficient_cost':float(cost),'eta_mean':float(eta.mean()),
                'eta_necessity_upper':float(2*(1-2/cost)),'support_envelope_max_excess':envelope_error})
            print('verified',obs['key'],'rho',residual,flush=True)
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,
        'integer_projector_proof':proof,'observations':checks,
        'q_integral_error_certified':False,'canonical_B_integral_error_certified':False,
        'floating_particle_Gaussianity_gate':True,'actual_branch_reconstruction_checked':True})


if __name__=='__main__':main()
