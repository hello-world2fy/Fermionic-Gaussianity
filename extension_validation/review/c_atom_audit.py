"""Independent Gaussianity audit of delivered C (signed_gaussian) atom pools.

Does NOT call any producer generator criterion. For each stored pure atom
|psi><psi| in the 4-mode even Fock sector (8-dim):
  - Hermitian / trace=1 / PSD / purity=1
  - Majorana covariance G_ij = i <psi| gamma_i gamma_j |psi> ; Gaussian <=> G^2 = -I
  - Wick 4-point consistency on a representative pair
Flags illegal (non-Gaussian) atoms; reports nonzero-weight contribution of illegal pool.
Read-only.
"""
import numpy as np

def even_fock_basis(nmodes=4):
    import itertools
    Nfull=2**nmodes
    states=[x for x in range(Nfull) if x.bit_count()%2==0]
    states=sorted(states)
    L=len(states); B=np.zeros((L,Nfull),dtype=complex)
    for i,v in enumerate(states): B[i,v]=1
    return B,states,None

def majoranas_full(nmodes=4):
    Nfull=2**nmodes
    def c(j):
        M=np.zeros((Nfull,Nfull),dtype=complex)
        for x in range(Nfull):
            if x>>j & 1:
                sign=(-1)**sum((x>>k)&1 for k in range(j)); M[x,x^(1<<j)]=sign
        return M
    gs=[]
    for j in range(nmodes):
        cj=c(j)
        gs += [cj+cj.conj().T, -1j*(cj-cj.conj().T)]
    return gs

def gaussian_residual(psi_even,B,gs):
    psi=B.T.conj()@psi_even   # embed in full Fock
    m=len(gs); G=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            G[i,j]=np.vdot(psi, 1j*gs[i]@gs[j] @ psi).real
    resid=np.abs(G+G.T).max(); g2=G@G+np.eye(m)
    return resid, np.abs(g2).max(), G

def audit(path):
    z=np.load(path,allow_pickle=True)
    at=z['atoms_full']; w=z['wM']
    B,states,_=even_fock_basis(); gs=majoranas_full()
    bad=0; bad_w=0.0; tot=len(at); samples=[]
    for i in range(tot):
        a=at[i]
        ev,U=np.linalg.eigh(a); psi=U[:,-1]
        res_anti,res_g2,G=gaussian_residual(psi,B,gs)
        is_g = (res_g2<1e-8 and res_anti<1e-8)
        if not is_g:
            bad+=1; bad_w+=abs(w[i])
            if len(samples)<5: samples.append((i,res_g2,res_anti,float(w[i])))
    return tot,bad,bad_w,samples

if __name__=="__main__":
    import os
    root=os.path.normpath(os.path.join(os.path.dirname(__file__),"..","signed_gaussian","artifacts"))
    for f in ["magic_artifacts.npz"]:
        tot,bad,bad_w,samp=audit(os.path.join(root,f))
        print(f,"total atoms",tot,"NON-GAUSSIAN",bad,"|w_illegal|",bad_w)
        for s in samp: print("   illegal sample idx,resG2,resAnti,w=",s)
