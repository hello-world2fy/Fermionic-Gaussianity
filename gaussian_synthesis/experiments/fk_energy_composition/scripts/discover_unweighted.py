#!/usr/bin/env python3
"""Freeze blind energy-bin directions before invoking any excitation certificate."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk6.paths import ROOT,PREVIOUS
from fk6.io import source_manifest,write_json
from fk6.blind_discovery import hfb_reference,generic_window_atoms,discover
from fk.model import build_model


def one(task):
    source,out,t,seed,budget=task;source=Path(source);out=Path(out);started=time.monotonic()
    model=build_model();ids=model.even_masks;H=model.full_hamiltonian(t)[np.ix_(ids,ids)]
    reference,hfb=hfb_reference(H,seed)
    input_summary=json.loads((source/f't{t:g}_k2_seed{seed}.json').read_text())
    correlation=input_summary['correlation_energy'];EG=input_summary['EG']
    assert abs(np.vdot(reference,H@reference).real-EG)<1e-8
    pools={};pool_info={}
    for fraction in [.1,.25,.5,1.]:
        atoms,info=generic_window_atoms(H,reference,fraction*correlation,seed)
        pools[fraction]=atoms;pool_info[fraction]=info
        assert np.max(np.einsum('ni,ij,nj->n',atoms.conj(),H,atoms).real-EG)<fraction*correlation+1e-8
    rows=[]
    for k in [2,4]:
        path=source/f't{t:g}_k{k}_seed{seed}_particles.npz';f=np.load(path)
        # Symmetry averaging is an integration step already audited in M1.
        # The discovery function itself sees full Hilbert vectors, not S or e_r.
        g=f['kets'];phases=f['aligned_phases'];B=float(f['coefficient_mass']);energies=f['energy_excess']
        overlaps=g@model.S.conj()
        edges=np.quantile(energies,np.linspace(0,1,17));labels=np.searchsorted(edges[1:-1],energies,side='right')
        z=np.array([(phases[labels==j]@overlaps[labels==j])*B/len(g) for j in range(16)])
        vectors=z.real@model.S.T
        assert np.linalg.norm(vectors.sum(axis=0)-f['total_rb'])<3e-12
        base=f't{t:g}_k{k}_seed{seed}'
        np.savez_compressed(out/f'{base}_energy_bins.npz',energy_edges=edges,bin_counts=np.bincount(labels,minlength=16),
            conditional_vectors=vectors,reference=reference,H=H)
        del g,overlaps,f
        for fraction in [.1,.25,.5,1.]:
            witness,info=discover(vectors,reference,pools[fraction],budget)
            key=base+f'_fraction{fraction:g}'
            frozen=out/f'{key}_frozen.npz'
            np.savez_compressed(frozen,witness=witness,conditional_vectors=vectors,reference=reference,
                low_atoms=pools[fraction],hamiltonian=H)
            row=dict(t=t,k=k,seed=seed,fraction=fraction,delta=fraction*correlation,budget=budget,
                key=key,frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                witness_sha256=hashlib.sha256(witness.tobytes()).hexdigest(),discovery=info,
                hfb=hfb,low_oracle=pool_info[fraction],certificate_has_not_run=True,
                source_particles=str(path.resolve()),source_distribution_count=1048576,
                source_sampling_seconds=json.loads((source/f't{t:g}_k{k}_seed{seed}.json').read_text())['elapsed_seconds'],
                discovery_workflow_seconds=time.monotonic()-started,
                claimed_independent_gain=False)
            write_json(out/f'{key}.json',row);rows.append(row)
        print('frozen',base,'sixteen coherent bins; four windows',flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'results/m1_unweighted_n1048576_s5_02')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seeds',type=int,nargs='+',default=[25101,25102,25103,25104,25105]);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);write_json(out/'manifest.json',source_manifest(out/'source_snapshot'))
    initial=json.loads((PREVIOUS/'results/initial_certificates_02/summary.json').read_text())
    two=json.loads((PREVIOUS/'results/two_orbit_02/summary.json').read_text())['construction']['exact_construction_cost_interval'][1]
    U={11:next(r for r in initial['A1_rows'] if r['t']==11)['cost_upper'],20:two}
    work=[(str(args.source.resolve()),str(out),t,seed,U[t]) for t in [11,20] for seed in args.seeds]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from multiprocessing import get_context
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(one,x) for x in work]):
            rows.extend(future.result());write_json(out/'partial_summary.json',dict(status='running',rows=rows))
    write_json(out/'summary.json',dict(status='frozen_before_global_certification',rows=rows,
        direction_freeze_completed_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        discovery_has_no_excitation_coordinates_input=True,independent_gain_claimed=False))


if __name__=='__main__':main()
