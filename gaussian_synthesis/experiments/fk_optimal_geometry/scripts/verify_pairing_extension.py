#!/usr/bin/env python3
"""Independent rational determinants, 110-digit ED and Fourier constructions."""
from pathlib import Path
from fractions import Fraction
from itertools import combinations
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from scipy.sparse import load_npz
from fk5.io import write_json
from fk5.extension_models import annihilators
from fk.model import build_model
from verify_window_physics import enclosed,digest


def determinant(A):
    A=[r.copy() for r in A];sign=1;value=Fraction(1)
    for i in range(len(A)):
        pivot=next((j for j in range(i,len(A)) if A[j][i]),None)
        if pivot is None:return Fraction(0)
        if pivot!=i:A[i],A[pivot]=A[pivot],A[i];sign*=-1
        value*=A[i][i]
        for j in range(i+1,len(A)):
            factor=A[j][i]/A[i][i]
            for k in range(i+1,len(A)):A[j][k]-=factor*A[i][k]
    return sign*value


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();mp.dps=110
    s=json.loads((args.run/'summary.json').read_text());assert len(s['targets'])==3 and len(s['rows'])==12
    operators=annihilators(8);pair=[operators[2*j+1]@operators[2*j] for j in range(4)]
    V=-sum(pair[i].conj().T@pair[j] for i in range(4) for j in range(4) if i!=j)
    assert max(np.asarray(abs(V).sum(axis=1)).ravel())==4
    kinetic=np.array([sum([-3,-3,-1,-1,1,1,3,3][j] for j in range(8) if mask>>j&1) for mask in range(256)])
    counts=np.zeros(8,int);proof=json.loads((args.run/'exact_Wick_structure.json').read_text())
    for row in proof['unordered_transfer_cases']:
        counts[row['relative_modes']]+=1;assert len(set(row['relative_modes']))==4
        flags=row['relative_creation_flags'];assert sum(flags) in [0,2,4]
        # A Gaussian four-point contraction has exactly three disjoint-pair
        # products. Each is either two anomalous or two normal contractions.
        for pairs in [[(0,1),(2,3)],[(0,2),(1,3)],[(0,3),(1,2)]]:
            types=[flags[i]==flags[j] for i,j in pairs];assert types[0]==types[1]
    assert counts.tolist()==[3]*8
    checks=[];target_cache={};gamma=build_model().gammas
    perms=np.argmax(abs(gamma),axis=2);phases=np.take_along_axis(gamma,perms[:,:,None],axis=2)[:,:,0]
    for r in s['targets']:
        g=Fraction(r['g_numerator'],r['g_denominator']);gg=mp.mpf(g.numerator)/g.denominator
        masks=r['physical_Fock_masks'];sub=V[np.ix_(masks,masks)].toarray();A=[[Fraction(int(kinetic[masks[i]])) if i==j else g*int(sub[i,j].real) for j in range(6)] for i in range(6)]
        M=mp.matrix([[mp.mpf(x.numerator)/x.denominator for x in line] for line in A]);ev,vec=mp.eigsy(M)
        q=vec[:,0]*mp.sign(vec[0,0]);lo,hi=[Fraction(*x) for x in r['energy_rational_bracket']]
        for endpoint,positive in [(lo,True),(hi,False)]:
            minors=[determinant([[A[i][j]-(endpoint if i==j else 0) for j in range(k)] for i in range(k)]) for k in range(1,7)]
            assert all(x>0 for x in minors)==positive
        assert mp.mpf(lo.numerator)/lo.denominator<ev[0]<mp.mpf(hi.numerator)/hi.denominator
        for x,b in zip(q,r['positive_ground_amplitudes_intervals']):enclosed(x,b)
        assert all(x>0 for x in q)
        assert mp.mpf(r['full_spectral_gap_lower'])<=1-4*gg
        assert mp.mpf(r['Gaussian_coercivity_lower'])<=1-mp.mpf(9)*gg/2
        assert mp.mpf(r['Gaussian_growth_upper'])>=3+mp.mpf(9)*gg/2
        H=np.diag(kinetic)+float(g)*V.toarray();values,vectors=np.linalg.eigh(H)
        assert abs(values[0]-float(ev[0]))<1e-12 and values[1]-values[0]>=r['full_spectral_gap_lower']
        assert np.linalg.norm(vectors[np.setdiff1d(np.arange(256),masks),0])<1e-12
        structure=np.load(Path('results/extension_candidates_01')/f'pairing_g{float(g):g}_structure.npz');U=structure['quasiparticle_transform']
        assert np.array_equal(U.conj().T@U,np.eye(256))
        # Independent explicit particle/hole vacuum convention.
        assert np.argmax(abs(U[:,0]))==15
        for j in range(8):
            expected=operators[j].conj().T if 15>>j&1 else operators[j]
            assert np.array_equal(U.conj().T@operators[j]@U,expected.toarray())
        target_cache[float(g)]=(gg,q,U,H,masks,structure['quasiparticle_masks'])
    for r in s['rows']:
        g,q,U,H,masks,qmasks=target_cache[r['g']];delta=mp.mpf(str(r['delta']));c=1-mp.mpf(9)*g/2;C=3+mp.mpf(9)*g/2
        lower=64*c*c*q[5]/delta**2;upper=q[0]+4*C/delta*sum(q[1:5])+64*C*C*q[5]/delta**2
        enclosed(lower,r['lower_cost_interval']);enclosed(upper,r['upper_cost_interval'])
        approx_lower=4*c*(q[1]-q[5])/(mp.sqrt(mp.mpf(3)/2)*delta)
        enclosed(approx_lower,r['error_q8_cost_lower_interval']);assert approx_lower>0
        enclosed(q[0]+4*C/delta*sum(q[1:5]),r['drop_eighth_component_cost_upper_interval'])
        enclosed(q[5],r['drop_eighth_component_Hilbert_error_interval'])
        # Reconstruct entirely in independent MP quasiparticle coordinates.
        phi=mp.matrix(256,1);phi[0]=q[0];mass=abs(q[0]);atomcount=1
        for index in range(1,6):
            qmask=int(qmasks[index]);m=qmask.bit_count()//2;p=delta/(2*m*C);root=mp.exp(2j*mp.pi/(m+1))
            modes=[i for i in range(8) if qmask>>i&1];sign=int(U[masks[index],qmask].real)
            for j in range(m+1):
                vector=mp.matrix(256,1)
                for subset in range(1<<m):
                    mask=sum((1<<modes[2*k])+(1<<modes[2*k+1]) for k in range(m) if subset>>k&1)
                    degree=subset.bit_count();vector[mask]=(1-p)**(mp.mpf(m-degree)/2)*p**(mp.mpf(degree)/2)*root**(j*degree)
                assert abs(sum(abs(x)**2 for x in vector)-1)<mp.mpf('1e-100')
                number=sum(mask.bit_count()*abs(vector[mask])**2 for mask in range(256))
                assert abs(number-2*m*p)<mp.mpf('1e-100')
                coefficient=q[index]*sign*root**(-m*j)/((m+1)*p**(mp.mpf(m)/2))
                phi+=coefficient*vector;mass+=abs(coefficient);atomcount+=1
        desired=mp.matrix(256,1)
        for amplitude,mask,qmask in zip(q,masks,qmasks):desired[int(qmask)]=amplitude*int(U[mask,qmask].real)
        error=max(abs(x) for x in phi-desired);assert error<mp.mpf('1e-95') and atomcount==18
        assert abs(mass-upper)<mp.mpf('1e-95')
        data=np.load(args.run/f"g{r['g']:g}_delta{r['delta']:g}.npz");atoms=data['atoms'];coef=data['coefficients']
        assert atoms.shape==(18,256)
        acted=atoms[:,perms]*phases;cov=(1j*(acted.conj()@acted.transpose(0,2,1))).real
        purity=float(np.max(abs(cov@cov+np.eye(16))));assert purity<2e-12
        energies=np.einsum('ni,ij,nj->n',atoms.conj(),H,atoms).real
        assert max(energies+8)<=r['delta']+1e-11
        observed=float(np.linalg.norm(coef@atoms-data['target']))
        assert observed<1e-9 and abs(observed-r['numerical_reconstruction_residual'])<1e-13
        checks.append(dict(g=r['g'],delta=r['delta'],MP_reconstruction_error=float(error),
            actual_atoms=18,maximum_Gaussian_purity_error=purity,maximum_atom_energy_excess=float(max(energies+8))))
    write_json(args.run/'verification.json',dict(artifact_verification_passed=True,independent_decimal_digits=110,
        physical_targets_checked=3,finite_window_constructions_checked=12,exact_rational_principal_minor_checks=True,
        all_six_Wick_transfer_cases_and_mode_counts_checked=True,global_covariance_bound_proof='PAIRING_EXTENSION_PLAN.md',
        actual_atom_checks=checks,exact_cost_exponent=2,at_error_q8_cost_exponent=1,
        summary_sha256=digest(args.run/'summary.json'),verification_script_sha256=digest(__file__)))
    lines=['# Number-conserving pairing: a second physical mechanism test','',
        'The selected four-orbital pairing model has a rigorously unique Gaussian minimum (the filled two-lowest-pair Slater state) at all three couplings. Global Gaussian coercivity, full-Fock spectral separation, positive target amplitudes and finite window constructions were independently checked.','',
        '| Pair coupling g | Required eighth-excitation amplitude q8 | Full spectral-gap lower | Gaussian coercivity c |',
        '|---|---:|---:|---:|']
    for r in s['targets']:
        g=r['g_numerator']/r['g_denominator'];lines.append(f"| {g:g} | {sum(r['positive_ground_amplitudes_intervals'][-1])/2:.12g} | {r['full_spectral_gap_lower']:.8g} | {r['Gaussian_coercivity_lower']:.8g} |")
    lines+=['','For exact representation, the coefficient cost is Theta(Delta^-2). The lower bound applies to every Gaussian in the energy window. Eighteen actual normalized Gaussians give a matching-power upper bound at every sufficiently small Delta; twelve concrete constructions (three g values and four windows) were verified in full Fock space. No optimal leading constant is claimed.','',
        'At the fixed Hilbert error epsilon=q8, dropping the eighth-excitation component yields a 13-Gaussian construction of cost O(Delta^-1). A fourth-excitation functional gives a matching Omega(Delta^-1) lower bound because q4>q8. For every fixed epsilon<q8, the eighth-excitation obstruction still forces Delta^-2. The finite-error approximation is generally unnormalized; this is a representation-cost theorem, not a normalized energy-accuracy claim.','',
        'This is a different number-conserving Hamiltonian and a global Gaussian-dictionary result. It tests the energy-curvature/excitation-order mechanism beyond the FK singlet geometry. The alternative three-cluster FK candidate was prepared as a 4096-dimensional sparse model, but was not selected for an ED-only scale-up.','']
    (args.run/'REPORT.md').write_text('\n'.join(lines))
    print('Verified three physical pairing ground states, global coercivity structure, 12 finite constructions and the exact/finite-error exponent crossover.',flush=True)


if __name__=='__main__':main()
