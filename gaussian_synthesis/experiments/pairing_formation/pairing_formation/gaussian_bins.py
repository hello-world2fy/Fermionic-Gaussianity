"""Symmetry-exact variance reduction for full Gaussian coherent integrals."""
import numpy as np


def angular_invariant_basis(model):
    """Cyclic rotations, reflection, and particle/hole band exchange.

    The returned columns are normalized orbit indicators. Validity for a
    Hamiltonian must be checked explicitly; this is not an arbitrary-model
    symmetry assumption. A Gaussian orbit average is not one Gaussian atom.
    """
    parents=np.arange(model.dim)
    def find(a):
        while parents[a]!=a:
            parents[a]=parents[parents[a]];a=parents[a]
        return a
    permutations=[]
    for step,reflect,swap in ((1,False,False),(0,True,False),(0,False,True)):
        full=np.empty(model.dim,int)
        for subsets,indices in model.blocks:
            lookup={tuple(s):i for i,s in enumerate(subsets)}
            p=np.array([lookup[tuple(sorted(((-j if reflect else j)+step)%model.r for j in s))] for s in subsets])
            new=indices[np.ix_(p,p)]
            if swap:new=new.T
            full[indices.ravel()]=new.ravel()
            for a,b in zip(indices.ravel(),new.ravel()):parents[find(a)]=find(b)
        permutations.append(full)
    labels=np.array([find(a) for a in range(model.dim)]);_,labels=np.unique(labels,return_inverse=True)
    sizes=np.bincount(labels);Q=np.zeros((model.dim,len(sizes)))
    Q[np.arange(model.dim),labels]=1/np.sqrt(sizes[labels])
    return Q,permutations


def haar_atoms(model,count,device,torch,return_occupations=False,return_covariances=False):
    """Full SO(4L) Haar Gaussian, retaining full-Fock normalization."""
    from .projected_gaussian import torch_pfaffian
    n=2*model.L
    normal=torch.randn((count,2*n,2*n),device=device,dtype=torch.float64)
    O,R=torch.linalg.qr(normal)
    O=O*torch.where(torch.diagonal(R,dim1=1,dim2=2)>=0,1.,-1.)[:,None,:]
    sign,_=torch.linalg.slogdet(O);O[:,:,-1]*=sign[:,None]
    U=.5*(O[:,::2,::2]+O[:,1::2,1::2]+1j*(O[:,1::2,::2]-O[:,::2,1::2]))
    V=.5*(O[:,::2,::2]-O[:,1::2,1::2]+1j*(O[:,1::2,::2]+O[:,::2,1::2]))
    Z=-torch.linalg.solve(U,V);Z=.5*(Z-Z.transpose(1,2))
    vacuum=torch.exp(.5*torch.linalg.slogdet(U)[1])
    idx=torch.tensor([[k for j in range(model.L) if mask&(1<<j) for k in (2*j,2*j+1)] for mask in model.masks],device=device)
    atoms=torch_pfaffian(Z[:,idx[:,:,None],idx[:,None,:]],torch)*vacuum[:,None]
    if return_covariances:
        if return_occupations:raise ValueError('Select one covariance return format')
        rho=V.conj().transpose(1,2)@V
        kappa=U.conj().transpose(1,2)@V
        return atoms,rho,kappa
    if return_occupations:
        # Actual full-Fock Gaussian <n_i>, before any many-body projection.
        # b=U c+V c^dagger annihilates the state, so <c_i^dagger c_j>=(V^dagger V)_ij.
        occupations=(V.abs()**2).sum(dim=1)
        return atoms,occupations
    return atoms


def atom_coordinates(atoms,counts,chi,r,torch):
    probability=atoms.abs()**2;norm=probability.sum(dim=1)
    mean=probability@counts/norm/r
    circulation=((atoms.conj()*(atoms@chi.T)).sum(dim=1).real/norm/r).abs()
    return torch.stack((mean,circulation),dim=1)
