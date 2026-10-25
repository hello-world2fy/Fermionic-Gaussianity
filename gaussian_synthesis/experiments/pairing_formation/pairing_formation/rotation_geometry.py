"""Gaussian spin rotations of existing collective pair fields.

These are conventional generator-coordinate configurations. Nonlinear
stereographic shapes do not by themselves imply independent physical fields.
"""
import numpy as np


def spinors(field,amplitude,kind='rotation'):
    w=amplitude*np.asarray(field,complex)
    if kind=='rotation':
        u=np.cos(abs(w));v=w*np.sinc(abs(w)/np.pi)
    elif kind=='Thouless':
        u=1/np.sqrt(1+abs(w)**2);v=w*u
    else:raise ValueError(kind)
    return u,v


def matched_sector_atoms(model,field,amplitude,kind='rotation'):
    """P_m P_N |g> for a normalized matched particle/hole Gaussian.

    The hole spinor is the complex conjugate particle spinor. Independent
    complex coefficients in m amount to coherent one-body gauge orbits.
    Stored sector vectors are not renormalized.
    """
    return two_band_sector_atoms(model,field,np.asarray(field).conj(),amplitude,amplitude,kind)


def two_band_sector_atoms(model,particle_field,hole_field,particle_amplitude,hole_amplitude,kind='rotation'):
    """Two independently chosen normalized Gaussian pair fields, without N rescaling."""
    up,vp=spinors(particle_field,particle_amplitude,kind)
    uh,vh=spinors(hole_field,hole_amplitude,kind);columns=np.zeros((model.dim,model.r+1),complex)
    for m,(subsets,indices) in enumerate(model.blocks):
        p=np.empty(len(subsets),complex);h=np.empty(len(subsets),complex)
        for a,subset in enumerate(subsets):
            chosen=np.zeros(model.r,bool);chosen[list(subset)]=True
            p[a]=np.prod(vp[chosen])*np.prod(up[~chosen])
            h[a]=np.prod(vh[chosen])*np.prod(uh[~chosen])
        columns[indices.ravel(),m]=np.outer(p,h).ravel()
    return columns


def orthogonal_columns(columns,tolerance=1e-9):
    norms=np.linalg.norm(columns,axis=0);columns=columns[:,norms>1e-14]/norms[norms>1e-14]
    U,s,_=np.linalg.svd(columns,full_matrices=False)
    return U[:,s>s[0]*tolerance],s


def profile_directions(count=16,seed=861):
    rng=np.random.default_rng(seed)
    initial=[[1.,0.],[1.,1j],[1.,1.],[1.,.5j]]
    coeff=np.array(initial+[list(rng.normal(size=2)+1j*rng.normal(size=2)) for _ in range(count-4)],complex)
    return coeff/np.linalg.norm(coeff,axis=1)[:,None]
