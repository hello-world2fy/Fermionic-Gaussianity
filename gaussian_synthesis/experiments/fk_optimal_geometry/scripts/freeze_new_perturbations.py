#!/usr/bin/env python3
"""Freeze genuinely new hopping/interaction tests and dictionaries before ED."""
from pathlib import Path
import argparse
import datetime
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk5.io import source_manifest,write_json
from fk5.perturbation_transfer import sha
from fk5.joint_dual_certificates import exact_schmidt_polar
from fk.gaussian import sample_gaussian
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    gates={}
    for run in ['initial_certificates_02','two_orbit_02','window_sharp_bounds_02','window_variational_01','energy_exponents_01','structure_ablation_01','coherent_orbits_01']:
        path=Path('results')/run/'verification.json';assert json.loads(path.read_text())['artifact_verification_passed'];gates[run]=sha(path)
    dictionaries=json.loads(Path('results/structure_ablation_01/dictionaries.json').read_text())['rows']
    arrays={};sources={};tick=time.monotonic()
    for row in dictionaries:
        if row['method'] not in ['two_orbit','physics_prior','dual_columns','canonical_Lambda_k8','canonical_Lambda_k16','canonical_Lambda_k32']:continue
        file=Path('results/structure_ablation_01/dictionaries')/(row['key']+'.npz')
        assert sha(file)==row['file_sha256'];arrays[row['key']]=np.load(file)['atoms'][:32]
        sources[row['key']]=dict(source=str(file.resolve()),source_sha256=sha(file),
            recorded_upstream_seconds=row.get('source_cost_seconds'),
            upstream_scope=('Recorded predecessor dependency graph' if 'source_cost_seconds' in row else 'Analytic construction; upstream discovery/proof cost not separately timed'),
            exact_original_target_only=True)
    model=build_model();patterns=[];cases=[]
    for i in range(1,6):
        seed=83000+i;rng=np.random.default_rng(seed)
        r=rng.normal(size=8);r-=r.mean();r/=np.sqrt(np.mean(r*r))
        s=rng.normal(size=(2,7));s-=s.mean(axis=1)[:,None];s/=np.sqrt(np.mean(s*s))
        assert not np.allclose(r,[1,1,1,1,-1,-1,-1,-1])
        patterns.append(dict(id=i,seed=seed,hopping=r.tolist(),interaction=s.tolist()))
        for t in [9,10.9225,11,20]:
            setup_rng=np.random.default_rng(np.random.SeedSequence([84000+i,int(round(t*10000))]))
            starts=sample_gaussian(8,30,setup_rng).kets
            arrays[f't{t:g}_pattern{i}_HFB_starts']=np.concatenate([(exact_schmidt_polar()[model.even_masks]/4)[None,:],arrays[f't{t:g}_rep0_two_orbit'][:1],starts])
            for kind,pattern in [('hopping',r),('interaction',s)]:
                for amplitude in [.02,.05,.1,.2]:
                    key=f't{t:g}_{kind}_pattern{i}_amplitude{amplitude:g}'
                    cases.append(dict(key=key,t=t,kind=kind,amplitude=amplitude,pattern_id=i,pattern=pattern.tolist(),
                        seed=850000+i*1000+int(t*10)+int(amplitude*100)))
    np.savez_compressed(args.output/'frozen_inputs.npz',**arrays)
    registration=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gates=gates,
        new_exact_ground_states_accessed=False,patterns=patterns,cases=cases,dictionary_sources=sources,
        frozen_input_sha256=sha(args.output/'frozen_inputs.npz'),setup_seconds=time.monotonic()-tick,
        amplitudes=[.02,.05,.1,.2],t_values=[9,10.9225,11,20],pattern_replicates=5,
        predictions_per_case=15,terms=[16,32],single_Gaussian_terms=1,
        canonical_measure='Lambda proportional to |overlap|^(2k-1), the original coefficient-sampling baseline called P upstream',
        new_hopping='H0 + (i t amplitude/2) sum r_a gamma_Aa gamma_Ba; r has zero mean and RMS one',
        new_interaction='H0 - amplitude sum s_xa Q_xa^2; zero mean within each cluster, global RMS one',
        adaptive_columns='Ritz-energy residual pricing by the reused Gaussian-overlap optimizer; 16 starts, 40 sweeps, up to 32 distinct atoms; HFB initialization',
        HFB='32 frozen Gaussian starts, 100 coordinate sweeps, gradient tolerance 1e-8',
        evaluation='No full/new ED before all 160 cases and their prediction files are complete and hashed. Then full parity-resolved ED only for independent evaluation.',
        endpoints=['fidelity','energy_error','mass_correlation','Wick_deviation','coefficient_l1','actual_Gaussian_terms','prediction_seconds','recorded_upstream_cost'],
        advantage_rule='Report paired differences at matched actual-term budgets and coefficient costs. No computational advantage from fidelity alone or from unmeasured upstream setup. All failures and every perturbation are retained.',
        scope='M1-M3 support geometry and nontrivial physical barriers. Matching optimal dual and global quotient-distance constants remain unresolved; F tests transfer of explicit certified constructions, not unknown optimality.')
    write_json(args.output/'preregistration.json',registration)
    (args.output/'PREREGISTRATION.md').write_text('# New FK perturbations frozen before exact validation\n\n160 Hamiltonians: four original couplings, five independent frozen patterns, four amplitudes, and two distinct perturbation classes. Each case compares contact construction, physical prior, frozen columns, canonical k8/k16/k32 at 16 and 32 actual terms, plus multistart HFB and adaptive energy-residual columns. All 2400 predictions must finish and be hashed before new exact diagonalization.\n\nThe full machine-readable protocol, seeds, source costs, limitations and success criteria are in preregistration.json. Canonical candidates use the coefficient measure Lambda; the historical source named it P. This experiment does not silently reinterpret it as overlap^(2k) sampling.\n')
    print('Frozen',len(cases),'new Hamiltonian specifications and',len(arrays),'input arrays; no new ED.',flush=True)


if __name__=='__main__':main()
