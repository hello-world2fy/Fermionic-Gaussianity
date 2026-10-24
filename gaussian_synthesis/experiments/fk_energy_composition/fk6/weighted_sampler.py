"""General energy-weighted SMC with reversible Gaussian rotation proposals.

Sampling accuracy is empirical. No weighted Cartan or global support bound
is inferred from a partition estimate. lambda=0 dispatches unchanged to the
existing exact-plane heatbath sampler.
"""
import time
import numpy as np
from scipy.special import logsumexp
from . import paths
from fk4.tempering import next_temperature,systematic_resample,TemperedResult,actions
from fk4.gpu_tempering import heatbath_device,sample_tempered_gpu
from fk4.gpu_haar import sample_haar_device


def weighted_scores_device(kets,witness,hamiltonian_sparse,gaussian_energy,correlation_scale,penalty,block_size=131072):
    """Bounded-width sparse matmuls avoid a reproduced large-RHS CUDA failure.

    In this environment csr.dot(G.T) returned zeros at 1,048,576 columns,
    despite accurate products through 524,288 columns. All chunks stay below
    that regime. Endpoint scores are independently checked on the CPU too.
    """
    import cupy as cp
    results=[]
    for start in range(0,len(kets),block_size):
        block=kets[start:start+block_size]
        hg=hamiltonian_sparse.dot(block.T).T
        excess=(cp.sum(block.conj()*hg,axis=1).real-gaussian_energy)/correlation_scale
        if float(cp.min(excess).get()) < -1e-9:raise ArithmeticError('Gaussian minimum bound violated')
        v=1+penalty*cp.maximum(excess,0)
        overlap=cp.abs(block.conj()@witness)
        results.append(2*(cp.log(cp.maximum(overlap,1e-300))-cp.log(v)))
    return cp.concatenate(results)


def _check_endpoint_scores(host,scores,witness,H,gaussian_energy,correlation_scale,penalty):
    indices=np.array([0,len(host)//2,len(host)-1])
    g=host[indices]
    energies=np.einsum('ni,ij,nj->n',g.conj(),H,g).real
    expected=2*np.log(np.maximum(abs(g.conj()@witness),1e-300)/(1+penalty*np.maximum((energies-gaussian_energy)/correlation_scale,0)))
    if np.max(abs(expected-scores[indices]))>2e-9:
        raise ArithmeticError('GPU weighted scores disagree with independent CPU Hamiltonian evaluation')


def sample_weighted(witness,hamiltonian,gaussian_energy,correlation_scale,penalty,endpoint,count,rng,
                    *,device=0,n=8,ess_fraction=.8,moves=48,planes_per_move=8,
                    angle_scale=.22,callback=None,checkpoints=(),state_callback=None):
    if penalty==0:
        return sample_tempered_gpu(witness,endpoint,count,rng,device=device,n=n,
            ess_fraction=ess_fraction,sweeps=3,callback=callback,checkpoints=checkpoints,state_callback=state_callback)
    if penalty<0 or correlation_scale<=0 or endpoint<0 or moves<1 or count<2:
        raise ValueError('Invalid weighted sampling settings')
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    u=np.asarray(witness,complex)
    H=np.asarray(hamiltonian,complex)
    if H.shape!=(2**(n-1),)*2 or u.shape!=(2**(n-1),):raise ValueError('Dimension mismatch')
    if np.max(abs(H-H.conj().T))>1e-12:raise ValueError('Hermitian Hamiltonian required')
    started=time.monotonic();ladder=[];beta=0.;log_z=0.
    stops=sorted(set(float(x) for x in (*checkpoints,endpoint) if 0<x<=endpoint))
    stop_index=0;ancestors=np.arange(count)
    with cp.cuda.Device(device):
        g=sample_haar_device(n,count,rng);du=cp.asarray(u);dh=csr_matrix(cp.asarray(H))
        def score(kets):
            return weighted_scores_device(kets,du,dh,gaussian_energy,correlation_scale,penalty)
        scores=score(g)
        planes_count=len(actions(n)[0]);block=min(planes_per_move,planes_count)
        while beta<endpoint-1e-12:
            host=cp.asnumpy(scores);new=next_temperature(host,beta,stops[stop_index],ess_fraction)
            increments=(new-beta)*host;norm=float(logsumexp(increments));weights=np.exp(increments-norm)
            log_z+=norm-np.log(count);chosen=systematic_resample(weights,rng)
            g=cp.ascontiguousarray(g[cp.asarray(chosen)]);scores=scores[cp.asarray(chosen)];ancestors=ancestors[chosen]
            accepted=0.
            # The distribution of a uniformly ordered plane subset and symmetric
            # angles equals that of the inverse sequence. Thus the combined
            # Haar-volume-preserving proposal is symmetric before MH acceptance.
            scale=angle_scale/np.sqrt(1+new)
            for _ in range(moves):
                planes=rng.choice(planes_count,block,replace=False)
                angles=rng.normal(scale=scale,size=(count,block))
                proposal=g.copy()
                heatbath_device(proposal,du,0.,rng,n=n,planes=planes,prescribed_angles=angles)
                trial=score(proposal)
                accept=cp.asarray(np.log(rng.random(count)))<new*(trial-scores)
                accepted+=float(cp.mean(accept).get())
                g=cp.where(accept[:,None],proposal,g);scores=cp.where(accept,trial,scores)
            beta=new
            row=dict(beta=beta,log_z=log_z,incremental_ess=float(1/(weights@weights)),
                unique_initial_ancestors=len(np.unique(ancestors)),mh_acceptance=accepted/moves,
                angle_scale=scale,moves=moves,planes_per_move=block,elapsed_seconds=time.monotonic()-started)
            ladder.append(row)
            if callback:callback(row)
            if beta>=stops[stop_index]-1e-12:
                if state_callback:
                    checkpoint_host=cp.asnumpy(g);checkpoint_scores=cp.asnumpy(scores)
                    _check_endpoint_scores(checkpoint_host,checkpoint_scores,u,H,gaussian_energy,correlation_scale,penalty)
                    state_callback(checkpoint_host,beta,log_z,checkpoint_scores,row)
                stop_index+=1
        host=cp.asnumpy(g);log_r=cp.asnumpy(scores)
        _check_endpoint_scores(host,log_r,u,H,gaussian_energy,correlation_scale,penalty)
    norm_error=float(np.max(abs(np.linalg.norm(host,axis=1)-1)))
    if norm_error>1e-9:raise ArithmeticError('Weighted Gaussian chain lost normalization')
    return TemperedResult(host,log_z,beta,log_r,dict(method='weighted_smc_reversible_gaussian_rotation_MH',
        penalty=penalty,count=count,moves_per_temperature=moves,planes_per_move=block,
        max_norm_error=norm_error,ladder=ladder,elapsed_seconds=time.monotonic()-started,
        device=device,iid_particles=False,global_support_certified=False,strictly_certified=False,
        sparse_RHS_chunk_size=131072,independent_CPU_endpoint_scores_checked=True))
