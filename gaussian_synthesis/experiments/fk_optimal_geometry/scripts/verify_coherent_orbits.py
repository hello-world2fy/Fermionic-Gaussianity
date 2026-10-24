#!/usr/bin/env python3
"""Independent sparse-operator audit of every saved two/four-point kernel."""
from pathlib import Path
from itertools import combinations
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import csr_matrix
from mpmath import mp
from fk5.io import write_json
from fk.model import build_model
from fk.observables import exact_metrics,mass_correlation_wick
from verify_window_physics import digest,enclosed,ground
from verify_initial_certificates import independent_basis_and_polar


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();mp.dps=110
    model=build_model();D,W=independent_basis_and_polar(model);gamma=[csr_matrix(g) for g in model.gammas]
    two_ops={pair:gamma[pair[0]]@gamma[pair[1]] for pair in combinations(range(16),2)}
    V=mp.matrix([[mp.mpf(1)/4,mp.sqrt(14)/4,mp.mpf(1)/4],[-mp.sqrt(2)/2,0,mp.sqrt(2)/2],[mp.sqrt(7)/4,-mp.sqrt(2)/4,mp.sqrt(7)/4]])
    initial=json.loads(Path('results/initial_certificates_02/summary.json').read_text());summary=json.loads((args.run/'summary.json').read_text())
    checks=[];table=[]
    for row in summary['rows']:
        t=row['t'];data=np.load(args.run/f't{t:g}_kernels.npz');atoms=data['atoms'];c=data['coefficients']
        full=np.zeros((16,256),complex);full[:,model.even_masks]=atoms
        pairs=data['pairs'];quartets=data['quartets'];K2=data['two_point_kernels'];K4=data['four_point_kernels']
        assert len(pairs)==120 and len(quartets)==1820
        maxerror=0.
        for j,(a,b) in enumerate(pairs):
            expected=full.conj()@(1j*two_ops[a,b]@full.T)
            maxerror=max(maxerror,float(np.max(abs(expected-K2[j]))))
        for j,(a,b,d,e) in enumerate(quartets):
            expected=full.conj()@(two_ops[a,b]@(two_ops[d,e]@full.T))
            maxerror=max(maxerror,float(np.max(abs(expected-K4[j]))))
        assert maxerror<3e-14
        cov=np.zeros((16,16,16));cov[:,pairs[:,0],pairs[:,1]]=np.diagonal(K2,axis1=1,axis2=2).T.real;cov-=cov.transpose(0,2,1)
        assert np.max(abs(cov@cov+np.eye(16)))<2e-12
        a,b,d,e=quartets.T
        wick=-cov[:,a,b]*cov[:,d,e]+cov[:,a,d]*cov[:,b,e]-cov[:,a,e]*cov[:,b,d]
        atom_four_error=float(np.max(abs(wick-np.diagonal(K4,axis1=1,axis2=2).T.real)))
        assert atom_four_error<3e-12
        state=c@atoms;target=model.target(t);assert np.linalg.norm(state-target)<2e-12
        e0,q,h=ground(mp.mpf(t));z=V*q
        # Independently recover the signed two-orbit parameters via a 2x2
        # eigendecomposition, rather than the producer's scalar formulas.
        m0=(5*z[0]+mp.sqrt(7)*z[2])/3;m1=-mp.sqrt(2)*z[1];m2=2*m0-4*z[0]
        eigen,vec=mp.eigsy(mp.matrix([[m0,m1],[m1,m2]]));a=vec[1,1]/vec[0,1]
        cp=eigen[1]/(1+a*a);cm=-eigen[0]/(1+a*a)
        for value,key in [(a,'main_arc_parameter_interval'),(cp,'positive_orbit_mass_interval'),(cm,'negative_orbit_mass_interval'),(cp+cm,'exact_construction_cost_interval')]:enclosed(value,row['parameters'][key])
        spec=next(r for r in initial['spectra'] if r['t']==t)
        for name,certificate in row['whole_orbit_deletion_certificates'].items():
            u=(-a if name=='secondary_only' else a)
            if name=='joint_contacts_only':u=mp.mpf(certificate['parameter_interval'][0])
            pvec=mp.matrix([(2*u*u-1)/4,-u/mp.sqrt(2),(5+2*u*u)/(4*mp.sqrt(7))] if name=='secondary_only'
                          else [(2-u*u)/4,-u/mp.sqrt(2),(2+5*u*u)/(4*mp.sqrt(7))])
            F=(z.T*pvec)[0]**2/(pvec.T*pvec)[0]
            # Endpoint arithmetic for the frozen joint parameter may widen
            # by a few ulps. The exact orbit parameters are enclosed directly.
            if name!='joint_contacts_only':enclosed(F,certificate['fidelity_ceiling_interval'])
            else:assert float(F)<=certificate['fidelity_ceiling_interval'][1]+2e-15
            assert certificate['energy_error_lower']<=float(mp.mpf(spec['full_spectral_gap_lower'])*(1-F))+2e-13
        vectors={'coherent':[state],'dephased_between_orbits':[c[:8]@atoms[:8],c[8:]@atoms[8:]],
                 'incoherent_atoms':[coefficient*atom for coefficient,atom in zip(c,atoms)]}
        direct_totals={}
        for name,states in vectors.items():
            rho=sum(np.outer(v,v.conj()) for v in states);norm=float(np.trace(rho).real);rho/=norm
            fullrho=np.zeros((256,256),complex);fullrho[np.ix_(model.even_masks,model.even_masks)]=rho
            E=float(np.trace(fullrho@model.full_hamiltonian(t)).real)
            mass=float(np.trace(fullrho@model.mass_correlation_operator).real)
            two=np.array([(1j*op).multiply(fullrho.T).sum().real for op in two_ops.values()])
            four=np.array([(two_ops[a,b]@two_ops[d,e]).multiply(fullrho.T).sum().real for a,b,d,e in quartets])
            covariance=np.zeros((16,16));covariance[pairs[:,0],pairs[:,1]]=two;covariance-=covariance.T
            a,b,d,e=quartets.T
            wick=-covariance[a,b]*covariance[d,e]+covariance[a,d]*covariance[b,e]-covariance[a,e]*covariance[b,d]
            masswick=float(mass_correlation_wick(covariance,model.j_matrices))
            r=row['density_cases'][name]
            assert abs(E-r['energy'])<3e-12 and abs(mass-r['mass_correlation'])<3e-13
            assert abs(mass-masswick-r['wick_mass_deviation'])<3e-13
            assert abs(np.linalg.norm(four-wick)-r['all_distinct_four_point_connected_l2'])<3e-13
            assert abs(norm-r['raw_density_trace'])<3e-14
            if name=='coherent':
                np.testing.assert_allclose(two,data['two_point_parts'].sum(axis=0),atol=3e-14,rtol=0)
                np.testing.assert_allclose(four,data['four_point_parts'].sum(axis=0),atol=3e-14,rtol=0)
                direct_totals=dict(norm=norm,energy=E,mass_correlation=mass)
        # Audit the grouping itself, not just its summed result.
        masks=[]
        for name in row['kernel_parts']:
            mask=np.array([[i==j if name=='diagonal' else (i<8 and j<8 and i!=j) if name=='within_main_offdiagonal'
                            else (i>=8 and j>=8 and i!=j) if name=='within_secondary_offdiagonal' else (i<8)!=(j<8)
                            for j in range(16)] for i in range(16)])
            masks.append(mask);weight=np.outer(c.conj(),c)*mask
            for key in ['norm','energy','mass_correlation']:
                value=np.sum(weight*data[key]).real/direct_totals['norm']
                assert abs(value-row['kernel_parts'][name][key])<3e-13
        for i,mask in enumerate(masks):
            np.testing.assert_allclose(np.einsum('ij,qij->q',np.outer(c.conj(),c)*mask,K2).real/direct_totals['norm'],data['two_point_parts'][i],atol=3e-14,rtol=0)
            np.testing.assert_allclose(np.einsum('ij,qij->q',np.outer(c.conj(),c)*mask,K4).real/direct_totals['norm'],data['four_point_parts'][i],atol=3e-14,rtol=0)
        checks.append(dict(t=t,independent_sparse_kernel_error_max=maxerror,Gaussian_atomic_four_point_Wick_error_max=atom_four_error,
            kernels_checked=1940,whole_orbit_certificates_checked=3,positive_density_cases_checked=3))
        table.append(dict(t=t,main_ceiling=row['whole_orbit_deletion_certificates']['main_only']['fidelity_ceiling_interval'][1],
                          main_energy_lower=row['whole_orbit_deletion_certificates']['main_only']['energy_error_lower'],
                          exact=row['density_cases']['coherent'],dephased=row['density_cases']['dephased_between_orbits'],parts=row['kernel_parts']))
    write_json(args.run/'verification.json',dict(artifact_verification_passed=True,independent_decimal_digits=110,
        checks=checks,summary_sha256=digest(args.run/'summary.json'),verification_script_sha256=digest(__file__),
        exact_all_parameter_eight_atom_average_identity_source='results/two_orbit_02/verification.json',
        numerical_kernel_validation_not_interval_arithmetic=True))
    lines=['# Orbit deletion and coherent physical correlations','',
        'Four 16-Gaussian constructions reproduce the physical FK ground states. Every one of the 120 two-point and 1820 distinct four-point kernels at each t was independently recomputed with sparse operators. Each atom satisfies Wick factorization; the coherent state generally does not.','',
        '| t | Entire main-orbit fidelity ceiling | Main-only energy-error lower | Exact mass correlation | Mass after dephasing between orbits |',
        '|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['t']} | {r['main_ceiling']:.10f} | {r['main_energy_lower']:.8g} | {r['exact']['mass_correlation']:.9f} | {r['dephased']['mass_correlation']:.9f} |")
    lines+=['','The single-orbit ceilings hold for arbitrary term counts and costs on that entire fixed-parameter orbit. They do not prove that every good decomposition must use precisely the displayed secondary orbit.','',
        'Kernel contributions are separated into diagonal, within-main, within-secondary and between-orbit terms. These raw contributions share the coherent-state normalization and need not be expectation values of physical states individually. By contrast, the orbit-dephased and atom-incoherent comparisons are positive density matrices normalized separately.','',
        'Wick deviations are formed only after summing the two-point and four-point moments. They are never added as if they were linear observables. Removing the small secondary orbit affects the exact interacting correlations even when main-only fidelity is close to one. The independent D1/D2 bounds give the stronger term-count-independent energy obstruction for all low-energy-window dictionaries under their stated budgets.','']
    (args.run/'REPORT.md').write_text('\n'.join(lines))
    print('Verified 7760 coherent kernels, all atom Wick identities, 12 whole-orbit bounds and 12 positive-density comparisons.',flush=True)


if __name__=='__main__':main()
