"""Sparse 12-mode FK candidate and an 8-mode number-conserving pairing model."""
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix,diags,eye,kron


def annihilators(n):
    dim=1<<n;result=[]
    for mode in range(n):
        cols=[mask for mask in range(dim) if mask>>mode&1]
        rows=[mask^(1<<mode) for mask in cols]
        values=[(-1)**((mask&((1<<mode)-1)).bit_count()) for mask in cols]
        result.append(csr_matrix((np.array(values,complex),(rows,cols)),shape=(dim,dim)))
    return result


def three_cluster_fk(local_model,t):
    I=eye(16,format='csr');P=diags([(-1)**mask.bit_count() for mask in range(16)],format='csr')
    h=csr_matrix(local_model.local_interaction);g=[csr_matrix(x) for x in local_model.local_gammas]
    H=kron(I,kron(I,h),format='csr')+kron(I,kron(h,I),format='csr')+kron(h,kron(I,I),format='csr')
    gammas=[[kron(I,kron(I,x),format='csr') for x in g],
            [kron(I,kron(x,P),format='csr') for x in g],
            [kron(x,kron(P,P),format='csr') for x in g]]
    for cluster in range(2):
        for a in range(8):H+=.5j*t*(gammas[cluster][a]@gammas[cluster+1][a])
    H.eliminate_zeros()
    return H


def number_conserving_pairing(g):
    operators=annihilators(8);pairs=[operators[2*j+1]@operators[2*j] for j in range(4)]
    eps=np.array([-3,-1,1,3]);levels=np.repeat(eps,2)
    kinetic=np.array([sum(levels[j] for j in range(8) if mask>>j&1) for mask in range(256)])
    V=csr_matrix((256,256),dtype=complex)
    for i in range(4):
        for j in range(4):
            if i!=j:V-=pairs[i].conj().T@pairs[j]
    H=diags(kinetic,format='csr')+g*V
    masks=np.array([sum(3<<(2*i) for i in occupied) for occupied in combinations(range(4),2)])
    H.eliminate_zeros()
    return H,dict(kinetic=kinetic,interaction=V,seniority_zero_masks=masks,reference_mask=15)


def quasiparticle_Fock_transform(reference_mask,n=8):
    """Signed permutation mapping quasiparticle occupations to physical Fock."""
    ops=annihilators(n);create=[ops[j] if reference_mask>>j&1 else ops[j].conj().T for j in range(n)]
    U=np.zeros((1<<n,1<<n),complex);U[reference_mask,0]=1
    for mask in range(1,1<<n):
        j=(mask&-mask).bit_length()-1;U[:,mask]=create[j]@U[:,mask^(1<<j)]
    assert np.array_equal(U.conj().T@U,np.eye(1<<n))
    return U
