"""Exact orbital-count variance of a seniority-zero target, in any mode basis.

The measured one-body operator is not projected back to seniority zero:
its broken-pair output is included through 2x2 spin blocks. This makes the
previous arbitrary-orbital projected-Gaussian count bound applicable.
"""
import numpy as np


def reduced_moments(model, state):
    state=np.asarray(state)/np.linalg.norm(state)
    probability=abs(state)**2
    n=probability@model.occ
    joint=model.occ.T@(probability[:,None]*model.occ)
    C=joint-np.outer(n,n)
    pairs=model.pair_density(state)
    return {'occupations':n,'joint_occupations':joint,'covariance':C,'pair_density':pairs}


def orbital_count(moments, projector):
    """m=(c^dagger P c)/2; P may mix levels and spins arbitrarily."""
    n=moments['occupations'];joint=moments['joint_occupations']
    C=moments['covariance'];pairs=moments['pair_density']
    L=len(n);P=np.asarray(projector)
    blocks=P.reshape(L,2,L,2).transpose(0,2,1,3)
    d=np.trace(blocks[np.arange(L),np.arange(L)],axis1=-2,axis2=-1).real
    mean=float(d@n)
    variance=float(d@C@d)
    diagonal=variance;broken=0.
    for i in range(L):
        for j in range(i+1,L):
            B=blocks[i,j]
            broken+=float(np.linalg.norm(B)**2*(n[i]+n[j]-2*joint[i,j])+4*np.real(np.linalg.det(B)*pairs[i,j]))
    variance+=broken
    return {'mean':mean/2,'variance':variance/4,'excess':variance/4-mean/2,
            'paired_variance':diagonal/4,'broken_pair_variance':broken/4,
            'projector_error':float(np.linalg.norm(P@P-P)),
            'hermiticity_error':float(np.linalg.norm(P-P.conj().T))}


def torch_count(moments, projectors):
    """Differentiable batched version; moment tensors already on the device."""
    import torch
    n,joint,C,pairs=(moments[k] for k in ('occupations','joint_occupations','covariance','pair_density'))
    L=len(n)
    blocks=projectors.reshape(-1,L,2,L,2).permute(0,1,3,2,4)
    indices=torch.arange(L,device=projectors.device)
    d=blocks[:,indices,indices].diagonal(dim1=-2,dim2=-1).sum(-1).real
    mean=d@n
    variance=torch.einsum('bi,ij,bj->b',d,C,d)
    norms=(blocks.abs()**2).sum(dim=(-2,-1))
    determinants=blocks[...,0,0]*blocks[...,1,1]-blocks[...,0,1]*blocks[...,1,0]
    pair_term=norms*(n[:,None]+n[None,:]-2*joint)+4*(determinants*pairs).real
    upper=torch.triu(torch.ones((L,L),device=projectors.device,dtype=torch.bool),diagonal=1)
    broken=pair_term[:,upper].sum(dim=1)
    return mean/2,(variance+broken)/4
