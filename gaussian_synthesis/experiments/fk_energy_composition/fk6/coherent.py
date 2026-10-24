"""Energy-conditioned vectors and linear kernels; never Gaussian atom surrogates."""
from itertools import combinations
import numpy as np
from . import paths
from fk4.symmetry import time_reverse
from fk.observables import mass_correlation_wick


def rb_vector(vector, model):
    projected = model.S@(model.S.conj().T@vector)
    return .5*(projected+time_reverse(projected))


def split_vectors(kets, phases, coefficient_mass, energy_excess, windows, model):
    count = len(kets)
    total = coefficient_mass*(phases@kets)/count
    rows, low, high, low_rb, high_rb = [], [], [], [], []
    for delta in windows:
        mask = energy_excess <= delta
        l = coefficient_mass*(phases[mask]@kets[mask])/count
        h = total-l
        low.append(l); high.append(h)
        low_rb.append(rb_vector(l, model)); high_rb.append(rb_vector(h, model))
        rows.append(dict(delta=float(delta),low_count=int(mask.sum()),high_count=int((~mask).sum()),
                         low_particle_ess=int(mask.sum()),high_particle_ess=int((~mask).sum()),
                         low_probability=float(mask.mean()),high_probability=float((~mask).mean()),
                         particle_ess_is_independent_sample_count=False,
                         conditional_sampling_pass=bool(min(mask.sum(),(~mask).sum())>=256),
                         zero_observed_mass_certifies_zero=False))
    return rows, dict(total_raw=total,total_rb=rb_vector(total,model),low_raw=np.array(low),
                      high_raw=np.array(high),low_rb=np.array(low_rb),high_rb=np.array(high_rb))


def coherent_kernels(model,t,low,high):
    """All distinct Majorana 2/4-point kernels of two unnormalized vectors."""
    vectors = np.array([model.embed(low),model.embed(high)])
    perms = np.argmax(abs(model.gammas),axis=2)
    factors = np.take_along_axis(model.gammas,perms[:,:,None],axis=2)[:,:,0]
    def kernel(indices):
        acted = vectors.copy()
        for j in reversed(indices):
            acted = acted[:,perms[j]]*factors[j]
        if len(indices)==2:
            acted *= 1j
        return vectors.conj()@acted.T
    pairs = np.array(list(combinations(range(16),2)))
    quartets = np.array(list(combinations(range(16),4)))
    two = np.array([kernel(p) for p in pairs])
    four = np.array([kernel(p) for p in quartets])
    matrices = dict(norm=vectors.conj()@vectors.T,
                    energy=vectors.conj()@model.full_hamiltonian(t)@vectors.T,
                    mass=vectors.conj()@model.mass_correlation_operator@vectors.T)
    def parts(K):
        return np.array([K[...,0,0].real,K[...,1,1].real,2*K[...,0,1].real])
    summary = {name:parts(K).tolist() for name,K in matrices.items()}
    density = {}
    for name,mask in [('coherent',np.ones((2,2))),('dephased_between_energy_regions',np.eye(2))]:
        norm = float(np.sum(mask*matrices['norm']).real)
        if norm<=0:
            continue
        obs2 = np.einsum('ij,qij->q',mask,two).real/norm
        obs4 = np.einsum('ij,qij->q',mask,four).real/norm
        cov = np.zeros((16,16));cov[pairs[:,0],pairs[:,1]]=obs2;cov-=cov.T
        a,b,c,d=quartets.T
        wick4=-cov[a,b]*cov[c,d]+cov[a,c]*cov[b,d]-cov[a,d]*cov[b,c]
        mass = float(np.sum(mask*matrices['mass']).real/norm)
        density[name]=dict(raw_trace=norm,energy=float(np.sum(mask*matrices['energy']).real/norm),
            mass_correlation=mass,wick_mass_deviation=mass-float(mass_correlation_wick(cov,model.j_matrices)),
            connected_four_point_l2=float(np.linalg.norm(obs4-wick4)))
    arrays=dict(pairs=pairs,quartets=quartets,two_kernels=two,four_kernels=four,
                two_parts=parts(two),four_parts=parts(four),**matrices)
    return dict(unnormalized_parts_low_low_high_high_cross=summary,density_cases=density,
                two_observables=len(pairs),four_observables=len(quartets),
                wick_deviation_is_linear_observable=False),arrays
