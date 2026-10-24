#!/usr/bin/env python3
"""Full-Fock coherent kernels and rigorous whole-orbit deletion ceilings."""
from pathlib import Path
from itertools import combinations
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk5.io import source_manifest,write_json
from fk5.two_orbit import two_orbit_parameters,two_orbit_atoms,interval_fraction,ground_root_enclosure
from fk5.joint_dual_certificates import exact_schmidt_polar
from fk4.certificates import _bounds
from fk.model import build_model
from fk.observables import mass_correlation_wick


def orbit_ceiling(t,spectrum,kind,parameter):
    iv.dps=90;lo,hi=ground_root_enclosure(t,spectrum['ground_energy_interval'])
    e=iv.mpf([interval_fraction(lo).a,interval_fraction(hi).b]);T=iv.mpf(float(t))
    r1=(e+56)/(iv.sqrt(2)*T);r2=iv.sqrt(14)*T*r1/(e+8)
    z0=1/iv.sqrt(1+r1*r1+r2*r2);z=[z0,r1*z0,r2*z0]
    a=iv.mpf(parameter)
    v=([(2-a*a)/4,-a/iv.sqrt(2),(2+5*a*a)/(4*iv.sqrt(7))] if kind=='A'
       else [(2*a*a-1)/4,-a/iv.sqrt(2),(5+2*a*a)/(4*iv.sqrt(7))])
    norm2=sum(x*x for x in v);F=sum(x*y for x,y in zip(z,v))**2/norm2
    return dict(parameter_interval=_bounds(a),singlet_projection_intervals=[_bounds(x) for x in v],
        fidelity_ceiling_interval=_bounds(F),attaining_8term_cost_interval=_bounds(1/iv.sqrt(norm2)),
        energy_error_lower=max(0.,_bounds(iv.mpf(spectrum['full_spectral_gap_lower'])*(1-F))[0]),
        arbitrary_terms_and_coefficient_cost=True,
        scope='Every normalized superposition from this entire fixed-parameter Spin(7) orbit; not every possible physical replacement orbit')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    model=build_model();initial=json.loads(Path('results/initial_certificates_02/summary.json').read_text())
    pairs=np.array(list(combinations(range(16),2)));quartets=np.array(list(combinations(range(16),4)))
    perms=np.argmax(abs(model.gammas),axis=2);phases=np.take_along_axis(model.gammas,perms[:,:,None],axis=2)[:,:,0]
    rows=[]
    for t in [9,10.9225,11,20]:
        spectrum=next(r for r in initial['spectra'] if r['t']==t);joint=next(r for r in initial['A1_rows'] if r['t']==t)
        parameters,a,cp,cm=two_orbit_parameters(t,spectrum['ground_energy_interval'])
        atoms,c=two_orbit_atoms(model,a,cp,cm);full=np.zeros((16,256),complex);full[:,model.even_masks]=atoms
        def operator_kernel(indices):
            acted=full.copy()
            for j in reversed(indices):acted=acted[:,perms[j]]*phases[j]
            if len(indices)==2:acted*=1j
            return full.conj()@acted.T
        K2=np.array([operator_kernel(q) for q in pairs]);K4=np.array([operator_kernel(q) for q in quartets])
        assert np.max(abs(K2-K2.conj().transpose(0,2,1)))<2e-14
        assert np.max(abs(K4-K4.conj().transpose(0,2,1)))<2e-14
        kernels={'norm':full.conj()@full.T,'energy':full.conj()@model.full_hamiltonian(t)@full.T,
                 'mass_correlation':full.conj()@model.mass_correlation_operator@full.T}
        indices=np.arange(16);same_main=(indices[:,None]<8)&(indices[None,:]<8)
        same_secondary=(indices[:,None]>=8)&(indices[None,:]>=8);diagonal=np.eye(16,dtype=bool)
        masks={'diagonal':diagonal,'within_main_offdiagonal':same_main&~diagonal,
               'within_secondary_offdiagonal':same_secondary&~diagonal,'between_orbits':~(same_main|same_secondary)}
        assert np.array_equal(sum(m.astype(int) for m in masks.values()),np.ones((16,16),int))
        weights=np.outer(c.conj(),c);denom=float(np.sum(weights*kernels['norm']).real);assert abs(denom-1)<1e-12
        parts={};two_parts=[];four_parts=[]
        for name,mask in masks.items():
            values={key:float(np.sum(weights*mask*K).real/denom) for key,K in kernels.items()}
            two=np.einsum('ij,qij->q',weights*mask,K2).real/denom
            four=np.einsum('ij,qij->q',weights*mask,K4).real/denom
            parts[name]=values;two_parts.append(two);four_parts.append(four)
        two_parts=np.array(two_parts);four_parts=np.array(four_parts)
        density_cases={}
        density_masks={'coherent':np.ones((16,16),bool),'dephased_between_orbits':same_main|same_secondary,'incoherent_atoms':diagonal}
        for name,mask in density_masks.items():
            norm=float(np.sum(weights*mask*kernels['norm']).real);assert norm>0
            two=np.einsum('ij,qij->q',weights*mask,K2).real/norm
            four=np.einsum('ij,qij->q',weights*mask,K4).real/norm
            cov=np.zeros((16,16));cov[pairs[:,0],pairs[:,1]]=two;cov-=cov.T
            aa,bb,cc,dd=quartets.T
            wick4=-cov[aa,bb]*cov[cc,dd]+cov[aa,cc]*cov[bb,dd]-cov[aa,dd]*cov[bb,cc]
            mass=float(np.sum(weights*mask*kernels['mass_correlation']).real/norm)
            wickmass=float(mass_correlation_wick(cov,model.j_matrices))
            density_cases[name]=dict(energy=float(np.sum(weights*mask*kernels['energy']).real/norm),
                mass_correlation=mass,wick_mass_prediction=wickmass,wick_mass_deviation=mass-wickmass,
                all_distinct_four_point_connected_l2=float(np.linalg.norm(four-wick4)),
                all_distinct_four_point_connected_max=float(np.max(abs(four-wick4))),
                raw_density_trace=norm,normalized_positive_density=True)
        state=c@atoms;assert np.linalg.norm(state-model.target(t))<2e-12
        lam=joint['lambda_numerator']/joint['lambda_denominator']
        deletion={
            'main_only':orbit_ceiling(t,spectrum,'A',parameters['main_arc_parameter_interval']),
            'secondary_only':orbit_ceiling(t,spectrum,'B',parameters['secondary_rotation_cosine_interval']),
            'joint_contacts_only':orbit_ceiling(t,spectrum,'A',2/(lam-2))}
        # The joint orbit here is defined at the frozen binary64 parameter;
        # interval roundoff does not silently turn it into an exact rational one.
        row=dict(t=t,parameters=parameters,full_Fock_reconstruction_error=float(np.linalg.norm(state-model.target(t))),
            actual_terms=16,coefficient_cost=float(sum(abs(c))),kernel_parts=parts,density_cases=density_cases,
            two_point_observables=120,four_point_observables=1820,whole_orbit_deletion_certificates=deletion,
            wick_deviation_grouped_as_linear_observable=False,
            necessity_link='Single-orbit fidelity ceilings hold independent of term count and cost. D1/D2 additionally bound every low-energy-window dictionary under its explicit coefficient budget.')
        np.savez_compressed(args.output/f't{t:g}_kernels.npz',atoms=atoms,coefficients=c,
            pairs=pairs,quartets=quartets,two_point_kernels=K2,four_point_kernels=K4,
            two_point_parts=two_parts,four_point_parts=four_parts,**kernels)
        write_json(args.output/f't{t:g}.json',row);rows.append(row)
        print(t,'exact mass',density_cases['coherent']['mass_correlation'],'between energy',parts['between_orbits']['energy'],flush=True)
    write_json(args.output/'summary.json',dict(status='complete',rows=rows,independent_verification_required=True))


if __name__=='__main__':main()
