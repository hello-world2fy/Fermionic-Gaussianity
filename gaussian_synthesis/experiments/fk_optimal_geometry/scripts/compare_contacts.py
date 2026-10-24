#!/usr/bin/env python3
"""B1/B2: actual atoms, active projected columns, exact contact-only ceiling."""
from pathlib import Path
import argparse
import csv
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import iv
from fk5.paths import ROOT,PREVIOUS
from fk5.io import source_manifest,write_json
from fk5.two_orbit import actual_arc_atoms,scaled_basis
from fk5.joint_dual_certificates import certify_joint
from fk4.certificates import _bounds
from fk4.distributions import full_distribution
from fk3.discovery import covariance_from_kets
from fk.conventions import quadratic_operator
from fk.model import build_model


def orbit_distance(cov,J,parameter,kind):
    if kind=='A':
        v=np.einsum('jab,ab->j',J,cov[:8,:8]-cov[8:,8:])/8
    else:
        v=np.einsum('jab,ab->j',J,cov[:8,8:])/8
    n=v/np.linalg.norm(v) if np.linalg.norm(v)>1e-14 else np.eye(7)[0]
    j=np.einsum('j,jab->ab',n,J)
    b=np.sqrt(1-parameter**2)
    pred=np.zeros((16,16))
    if kind=='A':
        pred[:8,:8]=b*j;pred[8:,8:]=-b*j
        pred[:8,8:]=-parameter*np.eye(8)
    else:
        pred[:8,8:]=-parameter*np.eye(8)+b*j
    pred[8:,:8]=-pred[:8,8:].T
    return float(np.linalg.norm(cov-pred))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    m=build_model();ids=m.even_masks
    proof=json.loads((ROOT/'results/two_orbit_02/summary.json').read_text())['construction']
    verified=json.loads((ROOT/'results/two_orbit_02/verification.json').read_text())
    assert verified['artifact_verification_passed']
    a=sum(proof['main_arc_parameter_interval'])/2
    astar=2000/2713
    cert,raw=certify_joint(m,20,verified['certified_cost_upper'],lambda_value='4713/1000')
    w=raw['joint_witness_display'];H=cert['support_upper']
    d=np.load(ROOT/'results/two_orbit_02/construction_t20.npz')
    groups=[('joint_contact',actual_arc_atoms(m,astar),np.ones(8)/8),
            ('certified_main',d['atoms'][:8],abs(d['coefficients'][:8])),
            ('certified_secondary',d['atoms'][8:],abs(d['coefficients'][8:]))]
    for seed in [25101,25102,25103,25104,25105]:
        c=np.load(PREVIOUS/f'results/columns_n32_s5_01/columns_t20_seed{seed}.npz')
        keep=abs(c['coefficients'])>1e-7
        groups.append((f'active_projected_columns_{seed}',c['atoms'][keep],abs(c['coefficients'][keep])))
    rows=[];summaries=[]
    ham=m.full_hamiltonian(20)[np.ix_(ids,ids)]
    for name,atoms,weights in groups:
        f,quality=full_distribution(atoms,m.j_matrices)
        cov=covariance_from_kets(atoms)
        overlaps=abs(atoms.conj()@w)
        energy=np.einsum('ia,ab,ib->i',atoms.conj(),ham,atoms).real
        for i,g in enumerate(atoms):
            rows.append(dict(source=name,index=i,absolute_coefficient=float(weights[i]),
                joint_overlap=float(overlaps[i]),joint_normalized_deficit=float(1-overlaps[i]/H),
                energy=float(energy[i]),mass_a_squared=float(f['features'][i,0]),
                mass_b_squared=float(f['features'][i,1]),mass_dot=float(f['features'][i,2]),
                eta=float(f['features'][i,3]),
                distance_to_joint_covariance_orbit=orbit_distance(cov[i],m.j_matrices,astar,'A'),
                distance_to_main_covariance_orbit=orbit_distance(cov[i],m.j_matrices,a,'A'),
                distance_to_secondary_covariance_orbit=orbit_distance(cov[i],m.j_matrices,-a,'B')))
        summaries.append(dict(source=name,count=len(atoms),coefficient_mass=float(weights.sum()),
            weighted_features=(weights@f['features']/weights.sum()).tolist(),
            max_purity_error=quality['max_purity_error_all_particles']))
    with (args.output/'atoms.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Exact Gaussian-integer identities for all 21 diagonal Spin(7) generators.
    D=scaled_basis()
    for i in range(7):
        for j in range(i+1,7):
            q=quadratic_operator(m.j_matrices[i]@m.j_matrices[j],m.local_gammas)
            generator=np.kron(q,np.eye(16))+np.kron(np.eye(16),q)
            assert np.array_equal(generator@D,np.zeros((256,3)))
    iv.dps=80
    x=iv.mpf(2000)/2713
    A=[(2-x*x)/4,-x/iv.sqrt(2),(2+5*x*x)/(4*iv.sqrt(7))]
    z=[iv.mpf(v) for v in proof['physical_singlet_coordinates_intervals']]
    norm2=sum(v*v for v in A)
    fidelity=sum(a*b for a,b in zip(A,z))**2/norm2
    cost=1/iv.sqrt(norm2)
    F=_bounds(fidelity);assert F[1]<1
    weights=abs(d['coefficients']);weights/=weights.sum()
    f=d['features']
    observed=float(weights@((np.sqrt(f[:,3])-astar)**2))
    report=dict(status='complete',groups=summaries,tighter_all_decomposition_certificate=cert,
                exact_diagonal_Spin7_invariance_checks=21,
                contact_only_fidelity_ceiling_interval=F,
                contact_only_ceiling_attaining_cost_interval=_bounds(cost),
                contact_only_exact_reconstruction_impossible=True,
                actual_construction_mean_eta=float(weights@f[:,3]),
                actual_construction_sqrt_eta_squared_displacement=observed,
                scalar_concentration_bound_slack_ratio=observed/cert['all_cost_le_U_mean_squared_sqrt_eta_displacement_upper'],
                covariance_distance='Unnormalized Frobenius distance to the full fixed-parameter S6 orbit, minimized analytically over n.',
                active_column_coefficients='Projected orbit coefficients from prior column generation; these rows do not claim direct full-Fock reconstruction.',
                joint_contact_classification_source='JOINT_CONTACT_THEORY.md',
                matching_two_orbit_dual_globally_certified=False,
                global_quotient_distance_stability_constant_certified=False)
    write_json(args.output/'summary.json',report)
    print('contact-only fidelity ceiling',F,flush=True)
    print('new all-cost<=U eta interval',cert['all_cost_le_U_mean_eta_interval'],flush=True)
    print('actual mean eta',report['actual_construction_mean_eta'],'scalar slack ratio',report['scalar_concentration_bound_slack_ratio'],flush=True)


if __name__=='__main__':main()
