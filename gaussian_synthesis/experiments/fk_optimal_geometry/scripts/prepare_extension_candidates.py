#!/usr/bin/env python3
"""Prepare both F2 candidates and select by an explicit, pre-ED theorem test."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import diags,save_npz
from fk5.io import source_manifest,write_json
from fk5.extension_models import three_cluster_fk,number_conserving_pairing,quasiparticle_Fock_transform
from fk.model import build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    model=build_model();three=[]
    for t in [1,9,20]:
        H=three_cluster_fk(model,t);assert H.shape==(4096,4096) and not (H-H.conj().T).nnz
        parity=diags([(-1)**i.bit_count() for i in range(4096)])
        assert not (H@parity-parity@H).nnz
        save_npz(args.output/f'three_cluster_t{t}.npz',H)
        three.append(dict(t=t,complex_modes=12,full_dimension=4096,even_dimension=2048,nnz=H.nnz,
            sparse_Hermitian_and_parity_checks=True,ED_performed=False))
    pairing=[]
    for g in [.05,.1,.2]:
        H,info=number_conserving_pairing(g);N=diags([i.bit_count() for i in range(256)])
        assert not (H-H.conj().T).nnz and not (H@N-N@H).nnz
        V=info['interaction'];row_norm=np.asarray(abs(V).sum(axis=1)).ravel()
        assert max(row_norm)==4
        masks=info['seniority_zero_masks'];K=H[np.ix_(masks,masks)].toarray()
        off=K-np.diag(np.diag(K));assert np.all(off.real<=0) and not off.imag.any()
        assert np.all(np.sum(off!=0,axis=1)==4)
        reference=info['reference_mask'];assert info['kinetic'][reference]==-8
        assert sorted(info['kinetic'])[1]==-7
        U=quasiparticle_Fock_transform(reference)
        qmask=[int(np.argmax(abs(U[m]))) for m in masks]
        assert sorted(m.bit_count() for m in qmask)==[0,4,4,4,4,8]
        save_npz(args.output/f'pairing_g{g:g}.npz',H)
        np.savez_compressed(args.output/f'pairing_g{g:g}_structure.npz',seniority_zero_masks=masks,quasiparticle_masks=qmask,quasiparticle_transform=U)
        pairing.append(dict(g=g,complex_modes=8,full_dimension=256,physical_particle_number=4,
            seniority_zero_dimension=6,reference_Gaussian_energy=-8,
            interaction_operator_norm_upper=4*g,full_ground_gap_lower=1-4*g,
            Gaussian_energy_coercivity_lower=1-4.5*g,Gaussian_energy_growth_upper=3+4.5*g,
            expected_exact_cost_exponent=2,proposed_finite_Gaussian_terms=18,
            all_parameter_theory_requires_independent_check=True,ED_performed=False))
    write_json(args.output/'selection.json',dict(status='prepared',three_cluster_candidate=three,pairing_candidate=pairing,
        selected_model='number_conserving_pairing',selection_before_new_ED=True,
        go_reason='A global Gaussian coercivity bound and required 4/8-quasiparticle amplitudes offer an explicit independent test of the energy-curvature/excitation-order theorem, including a possible approximation-error crossover.',
        three_cluster_no_go_for_full_scan='The 4096-dimensional sparse model is prepared, but no global Gaussian energy-window coercivity or support bound is presently available. A larger ED scan alone would not test the new theorem.',
        next='Independently prove the covariance coercivity inequality; isolate exact ground state; certify nonzero eighth-excitation amplitude and 18-term Gaussian constructions at four windows.',
        no_assumption_of_efficient_global_Gaussian_rank_with_cluster_number=True))
    print('Prepared three 4096-dimensional FK matrices and three pairing matrices. Selected pairing for a theorem test before new ED.',flush=True)


if __name__=='__main__':main()
