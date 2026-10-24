#!/usr/bin/env python3
"""Independent full-Fock, 110-digit and rational audit of D1--D3 artifacts.

Does not call either certificate producer or the variational optimizer.
"""
from pathlib import Path
from fractions import Fraction
from itertools import combinations
import argparse
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from fk5.io import write_json
from fk.model import build_model
from verify_initial_certificates import independent_basis_and_polar


def enclosed(value, interval, label=''):
    assert mp.mpf(interval[0]) <= value <= mp.mpf(interval[1]), (label, str(value), interval)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ground(t):
    a = 3 * mp.sqrt(14)
    h = mp.matrix([[-14-4*t, -a, 0], [-a, -50, -a], [0, -a, -14+4*t]])
    ev, v = mp.eigsy(h)
    q = v[:, 0] * mp.sign(v[0, 0])
    return ev[0], q, h


def window_cap(t, delta):
    return (mp.sqrt(16*(t-9)**2+72*delta)-4*(t-9))/36


def independent_geometry(model):
    D, W = independent_basis_and_polar(model)
    U = np.zeros((256, 256), complex)
    U[:, 0] = W/4
    for mask in range(1, 256):
        bit = mask.bit_length()-1
        rest = mask ^ (1 << bit)
        U[:, mask] = (-1)**rest.bit_count() * ((model.gammas[bit]-1j*model.gammas[bit+8]) @ U[:, rest])/2
    assert np.array_equal(U.conj().T@U, np.eye(256))
    assert np.array_equal(U.conj().T@model.h_hop@U, np.diag([i.bit_count()-4 for i in range(256)]))
    omega = U.conj().T @ ((7*D[:, 0]-D[:, 2])/4)
    assert np.array_equal(omega, np.rint(omega.real))
    assert np.count_nonzero(omega) == 14 and np.vdot(omega, omega) == 14
    # Build the bilinear form by expanding the three Pfaffian pairings of
    # each occupied four-mode mask, independently of the producer's wedge loop.
    pairs = list(combinations(range(8), 2)); lookup = {p:i for i,p in enumerate(pairs)}
    T = np.zeros((28, 28), dtype=int)
    for mask in np.flatnonzero(omega):
        a,b,c,d = [i for i in range(8) if mask >> i & 1]
        for left,right,sign in [((a,b),(c,d),1),((a,c),(b,d),-1),((a,d),(b,c),1)]:
            i,j = lookup[left],lookup[right]
            T[i,j] = T[j,i] = sign*int(omega[mask].real)
    M = T.T@T; I = np.eye(28, dtype=int)
    assert np.array_equal((M-I)@(M-9*I), np.zeros((28,28), dtype=int))
    assert np.trace(M) == 84
    e4 = U@omega/np.sqrt(14)
    e8 = D[:,0]/8+D[:,1]/4+D[:,2]/8
    assert abs(abs(np.vdot(U[:,255], e8))-1) < 1e-15
    Q = np.column_stack([W/4, e4, e8])
    return Q, U, T, {str(i):int(omega[i].real) for i in np.flatnonzero(omega)}


def atom_checks(model, atoms, t, delta, Q, U, T):
    full = np.zeros((len(atoms),256), complex); full[:,model.even_masks] = atoms
    norms = np.sum(abs(full)**2, axis=1)
    assert max(abs(norms-1)) < 2e-12
    perms = np.argmax(abs(model.gammas), axis=2)
    phases = np.take_along_axis(model.gammas, perms[:,:,None], axis=2)[:,:,0]
    acted = full[:,perms]*phases
    cov = (1j*(acted.conj() @ acted.transpose(0,2,1))).real
    purity = float(np.max(abs(cov@cov+np.eye(16))))
    assert purity < 2e-10
    energies = np.einsum('ni,ij,nj->n',full.conj(),model.full_hamiltonian(t),full).real
    assert max(energies+14+4*t) <= delta+2e-10
    hopping = full@U.conj(); pairs = list(combinations(range(8),2))
    z = np.array([hopping[:,(1<<a)+(1<<b)]/hopping[:,0] for a,b in pairs]).T
    predicted = hopping[:,0]*np.einsum('ni,ij,nj->n',z,T,z)/(2*np.sqrt(14))
    actual = full@Q[:,1].conj()
    pfaffian_error = float(max(abs(predicted-actual)))
    assert pfaffian_error < 2e-12
    X = float(window_cap(mp.mpf(t),mp.mpf(str(delta))))
    h4 = min(np.sqrt(1.5)*X,3*X/np.sqrt(14*(1-2*X)))
    assert max(abs(actual)) <= h4+2e-12
    assert max(abs(full@Q[:,2].conj())) <= X*X/4+2e-12
    return dict(count=len(atoms),maximum_purity_error=purity,
                maximum_energy_excess=float(max(energies+14+4*t)),
                maximum_independent_Pfaffian_amplitude_error=pfaffian_error)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bounds',type=Path,default=Path('results/window_sharp_bounds_02'))
    p.add_argument('--variational',type=Path,default=Path('results/window_variational_01'))
    p.add_argument('--exponents',type=Path,default=Path('results/energy_exponents_01'))
    args=p.parse_args();mp.dps=110;model=build_model()
    Q,U,T,omega=independent_geometry(model)
    initial=json.loads(Path('results/initial_certificates_02/summary.json').read_text())
    assert json.loads(Path('results/initial_certificates_02/verification.json').read_text())['artifact_verification_passed']
    assert json.loads(Path('results/two_orbit_02/verification.json').read_text())['exact_full16_term_reconstruction_formula']
    assert json.loads((args.bounds/'exact_Cayley_proof.json').read_text())['exact_integer_Cayley_coefficients']==omega
    bounds=json.loads((args.bounds/'summary.json').read_text())['rows'];assert len(bounds)==48
    spectra={r['t']:r for r in initial['spectra']};old={(r['t'],r['delta'],r['budget_multiplier']):r for r in initial['D1_D2_rows']}
    cert={};gcache={t:ground(mp.mpf(t)) for t in [9,11,20]}
    for r in bounds:
        t=r['t'];d=mp.mpf(str(r['delta']));B=mp.mpf(r['coefficient_budget']);X=window_cap(mp.mpf(t),d)
        h4=min(mp.sqrt(mp.mpf(3)/2)*X,3*X/mp.sqrt(14*(1-2*X)));caps=[B*h4,B*X*X/4]
        e,q,h=gcache[t];spec=spectra[t]
        assert mp.mpf(spec['ground_energy_interval'][0])<e<mp.mpf(spec['ground_energy_interval'][1])
        assert np.max(abs(model.full_hamiltonian(t)@Q-Q@np.array(h.tolist(),float)))<3e-14
        for x,b in zip(caps,r['excitation_caps_intervals']):enclosed(x,b,'cap')
        for x,b in zip(q,r['physical_hopping_coordinates_intervals']):enclosed(x,b,'physical target')
        lam=[mp.mpf(x) for x in r['fidelity_test_weights']];mu=[mp.mpf(x) for x in r['energy_penalties']]
        assert min(lam+mu)>=0
        f=(mp.sqrt(q[0]**2+(q[1]-lam[0])**2+(q[2]-lam[1])**2)+sum(a*b for a,b in zip(lam,caps)))**2
        assert mp.mpf(r['fidelity_upper'])>=min(1,f)
        hp=h+mp.diag([0,*mu]);lowest=mp.eigsy(hp,eigvals_only=True)[0]
        lo,hi=[Fraction(*v) for v in r['penalized_singlet_eigenvalue_rational_bracket']]
        enclosed(lowest,[mp.mpf(lo.numerator)/lo.denominator,mp.mpf(hi.numerator)/hi.denominator],'penalty eigenvalue')
        diag=[Fraction(-14-4*t),Fraction(-50)+Fraction(r['energy_penalties'][0]),Fraction(-14+4*t)+Fraction(r['energy_penalties'][1])]
        def positive(x):
            a=diag[0]-x;b=a*(diag[1]-x)-126;c=b*(diag[2]-x)-126*a
            return a>0 and b>0 and c>0
        assert positive(lo) and not positive(hi)
        assert r['non_ground_full_spectrum_lower']<=spec['ground_energy_interval'][0]+spec['full_spectral_gap_lower']
        assert r['non_ground_full_spectrum_lower']>-14-4*t
        direct=lowest-sum(a*b*b for a,b in zip(mu,caps))
        assert mp.mpf(r['direct_energy_lower'])<=direct
        assert mp.mpf(r['direct_energy_error_lower'])<=max(0,direct-e)
        assert mp.mpf(r['spectral_gap_energy_error_lower'])<=max(0,mp.mpf(spec['full_spectral_gap_lower'])*(1-mp.mpf(r['fidelity_upper'])))
        key=(t,r['delta'],r['budget_multiplier'])
        assert r['coefficient_budget']==old[key]['coefficient_budget']
        assert r['best_energy_error_lower']==max(r['direct_energy_error_lower'],r['spectral_gap_energy_error_lower'],old[key]['ground_energy_error_lower'])
        cert[key]=r
    print('Verified 48 global normalized window certificates, exact Cayley matrix and rational penalty eigenvalues.',flush=True)
    comparisons=[];dictionary_checks=[];warnings=0
    for t in [9,11,20]:
        e,q,h=gcache[t];target=Q@np.array(list(q),float)
        for delta in [.001,.01,.1,1.]:
            path=args.variational/f't{t}_delta{delta:g}';atoms=np.load(path/'dictionary.npz')['atoms']
            dictionary_checks.append(dict(t=t,delta=delta,**atom_checks(model,atoms,t,delta,Q,U,T)))
            arrays=np.load(path/'solutions.npz');rows=json.loads((path/'summary.json').read_text())['rows'];assert len(rows)==4
            H=model.full_hamiltonian(t)[np.ix_(model.even_masks,model.even_masks)]
            for index,r in enumerate(rows):
                b=cert[t,delta,r['budget_multiplier']];assert r['coefficient_budget']==b['coefficient_budget']
                for objective in ['fidelity','energy']:
                    c=arrays[f'budget{index}_{objective}_coefficients'];state=c@atoms
                    norm=float(np.linalg.norm(state));cost=float(sum(abs(c)));F=float(abs(np.vdot(target[model.even_masks],state))**2);E=float(np.vdot(state,H@state).real)
                    assert abs(norm-1)<2e-12 and cost<=r['coefficient_budget']
                    assert abs(cost-r[f'{objective}_solution_cost'])<1e-11
                    assert abs(F-r['fidelity_optimized' if objective=='fidelity' else 'energy_solution_fidelity'])<2e-12
                    assert abs(E-r['fidelity_solution_energy' if objective=='fidelity' else 'energy_optimized'])<2e-11
                    assert F<=b['fidelity_upper']+2e-12 and E-float(e)>=b['best_energy_error_lower']-2e-11
                warnings+=r['fidelity_convex_solver_status']!='optimal'
                warnings+=sum(x['status']!='optimal' for history in r['energy_multistart_histories'] for x in history)
                comparisons.append(dict(t=t,delta=delta,budget_multiplier=r['budget_multiplier'],budget=r['coefficient_budget'],
                    fidelity_feasible=r['fidelity_optimized'],fidelity_global_upper=b['fidelity_upper'],
                    fidelity_gap=b['fidelity_upper']-r['fidelity_optimized'],
                    energy_error_feasible=r['energy_error_optimized'],energy_error_global_lower=b['best_energy_error_lower'],
                    energy_error_gap=r['energy_error_optimized']-b['best_energy_error_lower']))
    assert len(comparisons)==48
    print('Verified 1164 actual atoms and 96 normalized, cost-feasible physical trial states.',flush=True)
    exponent=json.loads((args.exponents/'summary.json').read_text());assert len(exponent['FK_rows'])==36
    checked_atoms=set()
    for r in exponent['FK_rows']:
        t=r['t'];d=mp.mpf(str(r['delta']));eps=mp.mpf(str(r['epsilon']));X=window_cap(mp.mpf(t),d);x=mp.mpf('0.99')*X
        _,q,_=gcache[t];desired=q.copy();desired[2]-=eps
        nodes=[mp.mpf(0),x/2,x]
        P=mp.matrix([[(1-y/2)**2 for y in nodes],[3*(2*y-y*y)/(2*mp.sqrt(14)) for y in nodes],[y*y/4 for y in nodes]])
        c=mp.lu_solve(P,desired)
        for value,interval in zip(c,r['orbit_coefficient_intervals']):enclosed(value,interval,'inverse coefficients')
        enclosed(sum(abs(a) for a in c),r['upper_cost_interval'],'cost upper')
        enclosed(4*(q[2]-eps)/X**2,r['lower_cost_interval'],'cost lower')
        enclosed(x,r['x_interval'],'x')
        assert max(abs(z) for z in P*c-desired)<mp.mpf('1e-90')
        assert 18*x*x+4*(t-9)*x<=mp.mpf(r['maximum_atom_energy_excess_upper'])<d
        assert r['asymptotic_exponent']==(1 if t==9 else 2) and r['actual_Gaussian_terms']==17
        assert r['approximate_vector_normalized']==(eps==0)
        key=(t,r['delta'])
        if key not in checked_atoms:
            name={.001:'.001',.01:'.01',.1:'.1',1.:'1'}[r['delta']]
            atoms=np.load(args.exponents/f't{t}_delta{name}_atoms.npz')['atoms'];assert len(atoms)==17
            atom_checks(model,atoms,t,float(d),Q,U,T);checked_atoms.add(key)
            # Check both finite orbit averages in full Fock space. The exact
            # all-parameter identity is separately proved by local16 algebra.
            for group,y in [(atoms[1:9],x/2),(atoms[9:17],x)]:
                projection=np.array([(1-float(y)/2)**2,3*(2*float(y)-float(y)**2)/(2*np.sqrt(14)),float(y)**2/4])
                assert max(abs(group.mean(axis=0)-(Q@projection)[model.even_masks]))<2e-13
    toy_checks=[]
    for r in exponent['toy_rows']:
        delta=mp.mpf(str(r['delta']));p=delta/8;zeta=mp.exp(2j*mp.pi/5)
        states=[];coeff=[]
        for j in range(5):
            state=mp.matrix(256,1)
            for mask in range(16):
                n=mask.bit_count();fock=sum(3<<(2*i) for i in range(4) if mask>>i&1)
                state[fock]=(1-p)**(mp.mpf(4-n)/2)*p**(mp.mpf(n)/2)*zeta**(j*n)
            states.append(state);coeff.append((3/(mp.sqrt(10)*(1-p)**2)+zeta**(-4*j)/(mp.sqrt(10)*p*p))/5)
            assert abs(sum(abs(v)**2 for v in state)-1)<mp.mpf('1e-100')
            assert abs(sum(mask.bit_count()*abs(state[mask])**2 for mask in range(256))-delta)<mp.mpf('1e-100')
        phi=sum((c*g for c,g in zip(coeff,states)),mp.matrix(256,1));phi[0]-=3/mp.sqrt(10);phi[255]-=1/mp.sqrt(10)
        assert max(abs(v) for v in phi)<mp.mpf('1e-95')
        lower=1/(mp.sqrt(10)*p*p);upper=lower+3/(mp.sqrt(10)*(1-p)**2)
        enclosed(lower,r['lower_cost_interval']);enclosed(upper,r['upper_cost_interval'])
        assert sum(abs(c) for c in coeff)<=upper and upper/lower<=mp.mpf(r['ratio_upper'])
        toy_checks.append(dict(delta=r['delta'],actual_cost=float(sum(abs(c) for c in coeff)),residual_max=float(max(abs(v) for v in phi))))
    common=dict(artifact_verification_passed=True,independent_decimal_digits=110,verification_script_sha256=digest(__file__))
    write_json(args.bounds/'verification.json',dict(**common,bounds_checked=48,exact_integer_Cayley_operator_checked=True,
        full_Fock_singlet_invariance_checked=True,exact_rational_penalized_spectral_minors_checked=True,
        summary_sha256=digest(args.bounds/'summary.json')))
    write_json(args.variational/'verification.json',dict(**common,dictionaries=dictionary_checks,normalized_trial_states_checked=96,
        solver_precision_warning_records=warnings,global_energy_optimizer_claimed=False,
        floating_feasibility_audit_not_interval_atom_certificate=True,comparisons=comparisons,
        summary_sha256=digest(args.variational/'summary.json')))
    write_json(args.exponents/'verification.json',dict(**common,FK_constructions_checked=36,actual_FK_atoms_checked=204,
        independent_3x3_inverse=True,exact_all_parameter_orbit_identity_source='results/two_orbit_02/verification.json',
        root_of_unity_full_Fock_constructions=toy_checks,positive_error_FK_vectors_may_be_unnormalized=True,
        summary_sha256=digest(args.exponents/'summary.json')))
    lines=['# Normalized energy-window bounds and actual variational states','',
        'All 48 global bounds passed independent 110-digit and exact-rational checks. All 96 saved trial states satisfy normalization and coefficient budgets. The 12 dictionaries contain 1164 actual Gaussian atoms checked in full Fock space.',
        '',f'There are {warnings} recorded non-optimal solver status entries (including repeated subproblems). Numerical trial states are feasible upper benchmarks; no global energy optimum is claimed. Budgets use the original plan reference U.','',
        'At Delta=0.01 and budget U:','',
        '| t | Feasible fidelity | Global fidelity upper | Global energy-error lower | Feasible energy error |',
        '|---|---:|---:|---:|---:|']
    for r in sorted(comparisons,key=lambda x:x['t']):
        if r['delta']==.01 and r['budget_multiplier']=='1':
            lines.append(f"| {r['t']} | {r['fidelity_feasible']:.10f} | {r['fidelity_global_upper']:.10f} | {r['energy_error_global_lower']:.10f} | {r['energy_error_feasible']:.10f} |")
    lines+=['','The global bound applies to any number of Gaussian terms. The small remaining gap at t11 and t20 is a bound on the room for improvement by any energy-window ansatz under the same budget, not just this 97-atom dictionary.','',
        'D3 independently checks 36 finite FK constructions (three couplings, four windows, three errors), with matching divergence powers, and four five-Gaussian Fourier constructions attaining the toy leading constant. Positive-error FK approximants are not required to be normalized.','']
    (args.variational/'REPORT.md').write_text('\n'.join(lines))
    print('Verified 36 FK exponent constructions and four five-term full-Fock Fourier constructions; reports saved.',flush=True)


if __name__=='__main__':main()
