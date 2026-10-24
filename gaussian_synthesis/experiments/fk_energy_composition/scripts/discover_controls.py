#!/usr/bin/env python3
"""Freeze cheap, strong Hamiltonian/known-state controls before certification."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk6.paths import ROOT
from fk6.io import source_manifest,write_json
from fk6.blind_discovery import direct_dual_krylov
from fk.model import build_model
from fk3.discovery import covariance_from_kets


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    source=json.loads((args.input/'summary.json').read_text());model=build_model();rows=[]
    for src in source['rows']:
        if src['k']!=2:continue  # The HFB and finite oracle are shared between k=2,4.
        saved=args.input/(src['key']+'_frozen.npz')
        assert hashlib.sha256(saved.read_bytes()).hexdigest()==src['frozen_sha256']
        f=np.load(saved);ref=f['reference'];H=f['hamiltonian'];atoms=f['low_atoms']
        psi=model.target(src['t']);cov=covariance_from_kets(atoms)
        purity=float(np.max(abs(cov@cov+np.eye(16))))
        energy_max=float(np.max(np.einsum('ni,ij,nj->n',atoms.conj(),H,atoms).real))
        assert purity<1e-9 and energy_max<=np.vdot(ref,H@ref).real+src['delta']+1e-8
        candidates={}
        for name,target in [('physical_HFB_residual',H@ref),('known_state_HFB_residual',psi)]:
            start=time.monotonic();ell=target-ref*np.vdot(ref,target);ell/=np.linalg.norm(ell)
            ell*=np.exp(1j*np.angle(np.vdot(ell,psi)))
            candidates[name]=(ell,dict(elapsed_seconds=time.monotonic()-start,
                uses_distribution_samples=False,uses_exact_target_wavefunction=name.startswith('known'),
                physical_residual_phase_aligned_for_reporting_only=True,
                uses_hand_excitation_directions=False,empirical_support_is_global_upper=False))
        candidates['direct_dual_krylov']=direct_dual_krylov(H,psi,ref,atoms,src['budget'])
        candidates['direct_dual_krylov_orthogonal']=direct_dual_krylov(H,psi,ref,atoms,src['budget'],remove_reference=True)
        for method,(ell,info) in candidates.items():
            key=f"t{src['t']:g}_{method}_seed{src['seed']}_fraction{src['fraction']:g}"
            frozen=args.output/(key+'_frozen.npz')
            np.savez_compressed(frozen,witness=ell,reference=ref,low_atoms=atoms,hamiltonian=H)
            hfb_seconds=src['hfb']['elapsed_seconds'];oracle_seconds=src['low_oracle']['elapsed_seconds']
            row=dict(key=key,t=src['t'],seed=src['seed'],fraction=src['fraction'],delta=src['delta'],
                budget=src['budget'],method=method,discovery=info,
                frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                hfb_generation_seconds_charged=hfb_seconds,finite_oracle_seconds_charged=oracle_seconds,
                total_discovery_seconds=hfb_seconds+oracle_seconds+info['elapsed_seconds'],
                finite_oracle_purity_error=purity,finite_oracle_max_energy=energy_max,
                reused_Hamiltonian_only_reference_and_oracle=str(saved.resolve()),
                information_permission='known-state; all methods may read the target',
                distribution_artifact_values_used=False,certificate_has_not_run=True,
                timing_limit='Discovery accounting only; not an end-to-end R/B performance comparison')
            rows.append(row);write_json(args.output/(key+'.json'),row)
    write_json(args.output/'summary.json',dict(status='frozen_before_global_certification',rows=rows,
        direction_freeze_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        no_distribution_samples_used=True,independent_gain_claimed=False))
    print('frozen',len(rows),'control directions',flush=True)


if __name__=='__main__':main()
