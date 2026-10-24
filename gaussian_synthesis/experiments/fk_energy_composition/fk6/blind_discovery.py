"""Direction discovery from coherent conditional vectors and a Hamiltonian.

This module accepts no excitation labels, analytic target coordinates or
model-specific certificate. Finite-oracle scores are discovery scores only.
"""
import time
import numpy as np
from scipy.linalg import svd
from scipy.optimize import brentq
import cvxpy as cp
from . import paths
from fk4.tempering import actions
from fk4.variational import optimize_gaussian_energy
from fk.gaussian import sample_gaussian


def hfb_reference(H,seed,starts=12):
    rng=np.random.default_rng(np.random.SeedSequence([seed,621]))
    initial=sample_gaussian(8,starts,rng).kets
    states,report=optimize_gaussian_energy(H,initial,sweeps=160,tolerance=1e-10)
    chosen=int(np.argmin(report['energies']))
    return states[chosen],dict(report,selected_start=chosen,seed=seed)


def generic_window_atoms(H,reference,delta,seed,directions=64):
    """Full-manifold quadratic paths; no analytic mass/hopping arc is supplied."""
    started=time.monotonic();rng=np.random.default_rng(np.random.SeedSequence([seed,622]))
    permutations,factors=actions(8);base=float(np.vdot(reference,H@reference).real)
    atoms=[reference.copy()];records=[dict(kind='numerical_HFB',excess=0.)]
    for j in range(directions):
        planes=rng.choice(len(permutations),16,replace=False)
        angles=rng.normal(size=16);angles/=np.linalg.norm(angles)
        def state(scale):
            g=reference.copy()
            for p,a in zip(planes,angles):
                g=np.cos(scale*a)*g+np.sin(scale*a)*factors[p]*g[permutations[p]]
            return g
        def excess(scale):
            g=state(scale);return float(np.vdot(g,H@g).real-base)
        hi=.01
        while excess(hi)<delta and hi<4:hi*=1.5
        if excess(hi)<delta:continue
        boundary=brentq(lambda x:excess(x)-delta,0,hi,xtol=1e-12)*.999999
        for fraction in [.5,1.]:
            g=state(boundary*fraction);g/=np.linalg.norm(g);atoms.append(g)
            records.append(dict(kind='generic_quadratic_path',path=j,scale=boundary*fraction,excess=float(np.vdot(g,H@g).real-base)))
    return np.array(atoms),dict(records=records,elapsed_seconds=time.monotonic()-started,
        uses_target_wavefunction=False,uses_excitation_directions=False)


def discover(conditional_vectors,low_reference,window_atoms,budget,max_directions=6):
    """SVD of coherent contributions, then a finite-oracle separating direction.

    The target proxy is reconstructed by summing the supplied disjoint bins.
    The only selected subspace comes from these vectors. A Gaussian HFB ray
    removes the trivial low-energy direction before the SVD. Certification
    must happen in a separate, subsequent call with the frozen output.
    """
    started=time.monotonic();vectors=np.asarray(conditional_vectors,complex)
    ref=np.asarray(low_reference,complex);ref=ref/np.linalg.norm(ref)
    target_proxy=vectors.sum(axis=0)
    residuals=vectors.T-ref[:,None]*(ref.conj()@vectors.T)[None,:]
    U,s,_=svd(residuals,full_matrices=False)
    rank=min(max_directions,int(np.sum(s>max(1e-8,s[0]*1e-6))))
    if rank==0:raise ArithmeticError('No nontrivial conditional direction')
    basis=U[:,:rank];projections=window_atoms.conj()@basis
    variable=cp.Variable(rank,complex=True);support=cp.Variable(nonneg=True)
    objective=cp.real((basis.conj().T@target_proxy).conj()@variable)-budget*support
    problem=cp.Problem(cp.Maximize(objective),[cp.norm(variable)<=1,cp.abs(projections@variable)<=support])
    problem.solve(solver='CLARABEL',max_iter=300,tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
    if variable.value is None:raise ArithmeticError('Blind discovery optimization failed')
    witness=basis@variable.value;witness/=np.linalg.norm(witness)
    witness*=np.exp(1j*np.angle(np.vdot(witness,target_proxy)))
    empirical_support=float(np.max(abs(window_atoms.conj()@witness)))
    return witness,dict(conditional_singular_values=s,retained_rank=rank,
        proxy_norm=float(np.linalg.norm(target_proxy)),empirical_support=empirical_support,
        empirical_separation=float(np.vdot(witness,target_proxy).real-budget*empirical_support),
        empirical_support_is_global_upper=False,solver_status=problem.status,
        uses_explicit_target_wavefunction=False,uses_hand_excitation_directions=False,
        elapsed_seconds=time.monotonic()-started)
