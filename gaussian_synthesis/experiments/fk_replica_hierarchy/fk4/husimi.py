"""Independent target-Husimi samples by exact Haar rejection.

For normalized psi and g, acceptance |<g|psi>|^2 is bounded by one.
Accepted rays therefore have density d_n,1 |<g|psi>|^2 against Haar.
No local maximization, temperature tuning, or MCMC burn-in is involved.
"""
import time
import numpy as np
from . import paths
from fk.gaussian import sample_gaussian


def sample_husimi(target,count,rng,*,device=None):
    u=np.asarray(target,complex)
    n=int(np.log2(len(u)))+1
    if 2**(n-1)!=len(u) or count<1 or not np.isfinite(u).all() or np.linalg.norm(u)==0:
        raise ValueError('invalid even-sector target or sample count')
    u=u/np.linalg.norm(u)
    chunks=[]
    retained=0
    proposed=0
    accepted_total=0
    started=time.monotonic()
    while retained<count:
        batch=max(1024,int(np.ceil((count-retained)*len(u)*1.15)))
        if device is None:
            kets=sample_gaussian(n,batch,rng).kets
            probability=abs(kets.conj()@u)**2
            mask=rng.random(batch)<probability
            accepted=kets[mask]
        else:
            import cupy as cp
            from .gpu_haar import sample_haar_device
            with cp.cuda.Device(device):
                kets=sample_haar_device(n,batch,rng)
                probability=cp.asnumpy(cp.abs(kets.conj()@cp.asarray(u))**2)
                mask=rng.random(batch)<probability
                accepted=cp.asnumpy(kets[cp.asarray(mask)])
        if probability.max()>1+1e-12:
            raise ArithmeticError('Husimi rejection envelope violated')
        accepted_total+=len(accepted)
        chunks.append(accepted[:count-retained])
        retained+=len(chunks[-1])
        proposed+=batch
    return np.concatenate(chunks),{'method':'exact_Haar_rejection_Husimi_beta1',
        'elapsed_seconds':time.monotonic()-started,'proposal_count':proposed,
        'accepted_before_truncation':accepted_total,'retained':count,
        'iid_particles':True,'uses_exact_target':True,'device':device,
        'acceptance_envelope':'Cauchy-Schwarz |<g|psi>|^2 <= 1'}
