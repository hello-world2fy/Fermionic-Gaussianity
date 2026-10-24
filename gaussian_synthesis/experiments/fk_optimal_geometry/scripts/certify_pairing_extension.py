#!/usr/bin/env python3
"""Exact rational ground isolation and finite Gaussian window constructions."""
from pathlib import Path
from fractions import Fraction
from itertools import combinations
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk5.io import source_manifest,write_json
from fk5.extension_models import number_conserving_pairing,quasiparticle_Fock_transform,annihilators
from fk5.two_orbit import interval_fraction
from fk4.certificates import _bounds


def exact_matrix(g):
    occupied=list(combinations(range(4),2));epsilon=[-3,-1,1,3]
    return [[Fraction(2*sum(epsilon[k] for k in a)) if i==j else -g if len(set(a)&set(b))==1 else Fraction(0)
             for j,b in enumerate(occupied)] for i,a in enumerate(occupied)]


def positive_shift(A,e):
    n=len(A);D=[line.copy() for line in A]
    for i in range(n):D[i][i]-=e
    for k in range(n):
        if D[k][k]<=0:return False
        for i in range(k+1,n):
            for j in range(i,n):
                D[j][i]=D[i][j]=D[i][j]-D[i][k]*D[j][k]/D[k][k]
    return True


def isolated_target(g):
    A=exact_matrix(g);lo=Fraction(-8)-4*g;hi=Fraction(-8)
    assert positive_shift(A,lo) and not positive_shift(A,hi)
    for _ in range(140):
        middle=(lo+hi)/2
        if positive_shift(A,middle):lo=middle
        else:hi=middle
    iv.dps=100;E=iv.mpf([interval_fraction(lo).a,interval_fraction(hi).b])
    # Solve the lower 5x5 block with v0=1, then normalize. The lowest
    # eigenvalue lies below every eigenvalue of this principal submatrix.
    M=[[interval_fraction(A[i][j])-(E if i==j else 0) for j in range(1,6)] for i in range(1,6)]
    rhs=[-interval_fraction(A[i][0]) for i in range(1,6)]
    for k in range(5):
        assert _bounds(M[k][k])[0]>0
        for i in range(k+1,5):
            ratio=M[i][k]/M[k][k]
            for j in range(k+1,5):M[i][j]-=ratio*M[k][j]
            rhs[i]-=ratio*rhs[k]
    v=[iv.mpf(0)]*5
    for i in reversed(range(5)):v[i]=(rhs[i]-sum(M[i][j]*v[j] for j in range(i+1,5)))/M[i][i]
    v=[iv.mpf(1),*v];norm=iv.sqrt(sum(a*a for a in v));q=[a/norm for a in v]
    assert all(_bounds(a)[0]>0 for a in q)
    return A,lo,hi,E,q


def wick_structure_proof():
    U=quasiparticle_Fock_transform(15);ops=annihilators(8);errors=[];cases=[];counts=np.zeros(8,int)
    for i,j in combinations(range(4),2):
        modes=[2*i,2*i+1,2*j+1,2*j];dagger=[True,True,False,False]
        transformed=[];original=[];relative_dagger=[]
        for mode,dag in zip(modes,dagger):
            physical=(ops[mode].conj().T if dag else ops[mode]).toarray()
            qdag=bool(dag)^bool(15>>mode&1);relative=(ops[mode].conj().T if qdag else ops[mode]).toarray()
            assert np.array_equal(U.conj().T@physical@U,relative)
            original.append(physical);transformed.append(relative);relative_dagger.append(qdag)
        assert sum(relative_dagger) in [0,2,4]
        counts[modes]+=1
        cases.append(dict(orbitals=[i,j],relative_modes=modes,relative_creation_flags=relative_dagger,
            three_Wick_products_use_four_distinct_modes=True))
    assert np.all(counts==3)
    return dict(exact_particle_hole_operator_identities=True,unordered_transfer_cases=cases,
        mode_multiplicity=counts.tolist(),each_Wick_product_bound='sum of four occupations / 4',
        unordered_transfer_plus_adjoint_bound='3/2 times sum of four occupations',
        global_interaction_expectation_bound='(9/2) g <N_quasiparticle>',
        normal_bound='|rho_ab| <= sqrt(n_a n_b)',anomalous_bound='|kappa_ab| <= sqrt(min(n_a,n_b))')


def component_atoms(U,qmask,p):
    modes=[i for i in range(8) if qmask>>i&1];m=len(modes)//2;zeta=np.exp(2j*np.pi/(m+1));atoms=[];weights=[]
    for j in range(m+1):
        vector=np.zeros(256,complex)
        for subset in range(1<<m):
            mask=sum((1<<modes[2*i])+(1<<modes[2*i+1]) for i in range(m) if subset>>i&1)
            degree=subset.bit_count();vector[mask]=(1-p)**((m-degree)/2)*p**(degree/2)*zeta**(j*degree)
        atoms.append(U@vector);weights.append(zeta**(-m*j)/((m+1)*p**(m/2)))
    return np.array(atoms),np.array(weights)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    selection=json.loads(Path('results/extension_candidates_01/selection.json').read_text())
    assert selection['selected_model']=='number_conserving_pairing' and selection['selection_before_new_ED']
    write_json(args.output/'exact_Wick_structure.json',wick_structure_proof())
    U=quasiparticle_Fock_transform(15);rows=[];targets=[]
    for gtext in ['.05','.1','.2']:
        g=Fraction(gtext);A,lo,hi,E,q=isolated_target(g);G=interval_fraction(g)
        c=1-iv.mpf(9)*G/2;C=3+iv.mpf(9)*G/2
        H,info=number_conserving_pairing(float(g));masks=info['seniority_zero_masks']
        qmask=[int(np.argmax(abs(U[mask]))) for mask in masks]
        signs=[float(U[mask,bitmask].real) for mask,bitmask in zip(masks,qmask)]
        target=np.zeros(256,complex);target[masks]=[float(a.mid) for a in q]
        targets.append(dict(g_numerator=g.numerator,g_denominator=g.denominator,
            energy_rational_bracket=[[a.numerator,a.denominator] for a in [lo,hi]],energy_interval=_bounds(E),
            positive_ground_amplitudes_intervals=[_bounds(a) for a in q],physical_Fock_masks=masks.tolist(),
            full_spectral_gap_lower=_bounds(1-4*G)[0],Gaussian_coercivity_lower=_bounds(c)[0],
            Gaussian_growth_upper=_bounds(C)[1],unique_Gaussian_minimum_reference_mask=15))
        for delta in ['.001','.01','.1','1']:
            D=interval_fraction(Fraction(delta));lower=64*c*c*q[-1]/(D*D)
            upper=q[0]+4*C/D*sum(q[1:5])+64*C*C/(D*D)*q[5]
            approximate_lower=4*c*(q[1]-q[5])/(iv.sqrt(iv.mpf(3)/2)*D)
            assert _bounds(approximate_lower)[0]>0
            atom_list=[U[:,0]];coefficient_list=[float(q[0].mid)]
            for idx in range(1,6):
                m=qmask[idx].bit_count()//2;prob=D/(2*m*C);pfloat=float(prob.mid)
                atoms,coeff=component_atoms(U,qmask[idx],pfloat)
                atom_list.extend(atoms);coefficient_list.extend(coeff*float(q[idx].mid)*signs[idx])
            atoms=np.array(atom_list);coeff=np.array(coefficient_list)
            assert len(atoms)==18
            residual=float(np.linalg.norm(coeff@atoms-target));cost=float(sum(abs(coeff)))
            energies=np.einsum('ni,in->n',atoms.conj(),H@atoms.T).real
            assert max(energies+8)<=float(delta)+1e-10
            np.savez_compressed(args.output/f'g{float(g):g}_delta{float(delta):g}.npz',atoms=atoms,coefficients=coeff,target=target)
            rows.append(dict(g=float(g),delta=float(delta),lower_cost_interval=_bounds(lower),upper_cost_interval=_bounds(upper),
                exact_Gaussian_terms=18,exact_divergence_exponent=2,
                maximum_float_atom_energy_excess=float(max(energies+8)),numerical_reconstruction_residual=residual,
                numerical_cost=cost,
                exact_construction_proof='For each 4/8-excitation component use the 3/5-root Fourier filter; coefficient intervals refer to the exact rational Hamiltonian ground state.',
                float_reconstruction_is_diagnostic_not_certificate=True,
                drop_eighth_component_Hilbert_error_interval=_bounds(q[5]),
                drop_eighth_component_cost_upper_interval=_bounds(q[0]+4*C/D*sum(q[1:5])),
                error_q8_cost_lower_interval=_bounds(approximate_lower),error_q8_matching_exponent=1,
                drop_eighth_component_actual_terms=13,
                approximate_vector_normalized=False))
        print('Certified pairing g',float(g),'q8',_bounds(q[5]),flush=True)
    write_json(args.output/'summary.json',dict(status='complete',targets=targets,rows=rows,
        exact_couplings_defined_as_decimal_rationals=True,independent_verification_required=True,
        distinct_number_conserving_model=True,whole_Gaussian_dictionary_coercivity_not_HFB_only=True))


if __name__=='__main__':main()
