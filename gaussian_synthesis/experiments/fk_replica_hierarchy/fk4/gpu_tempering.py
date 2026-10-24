"""CUDA complex128 implementation of the same exact-plane heat bath.

One warp evolves one Gaussian ket. Philox streams are independent by particle;
all reductions and rotations use float64, with fast math disabled. SMC weights,
temperature selection, and checkpoint semantics remain the CPU reference's.
CuPy RawKernel API: https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.RawKernel.html
"""
from functools import lru_cache
import time

import numpy as np
from scipy.special import logsumexp

from .tempering import (actions, TemperedResult, next_temperature,
                        systematic_resample, log_overlap_squared)
from .gpu_haar import sample_haar_device


CUDA_SOURCE = r'''
#include <curand_kernel.h>
extern "C" __global__ void heatbath(
    double2* kets, const double2* witness, const int* permutations,
    const double2* phases, const int* planes, const int plane_count,
    const int count, const int dim, const double beta,
    const unsigned long long seed, unsigned long long* trials,
    const double* prescribed_angles, const int prescribed) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int particle = blockIdx.x * 4 + warp;
  if (particle >= count) return;
  __shared__ double2 storage[4][128];
  double2* ket = storage[warp];
  for (int j=lane; j<dim; j+=32) ket[j] = kets[(long long)particle*dim+j];
  __syncwarp();
  curandStatePhilox4_32_10_t state;
  if (lane==0) curand_init(seed, particle, 0, &state);
  unsigned long long attempts=0;
  for (int step=0; step<plane_count; ++step) {
    const int p=planes[step];
    double2 transformed[4];
    double ar=0, ai=0, br=0, bi=0;
    for (int j=lane, v=0; j<dim; j+=32, ++v) {
      const double2 x=ket[j], w=witness[j];
      const double2 y=ket[permutations[p*dim+j]], f=phases[p*dim+j];
      const double2 b=make_double2(f.x*y.x-f.y*y.y, f.x*y.y+f.y*y.x);
      transformed[v]=b;
      ar+=w.x*x.x+w.y*x.y; ai+=w.x*x.y-w.y*x.x;
      br+=w.x*b.x+w.y*b.y; bi+=w.x*b.y-w.y*b.x;
    }
    for (int offset=16; offset>0; offset/=2) {
      ar+=__shfl_down_sync(0xffffffff,ar,offset);
      ai+=__shfl_down_sync(0xffffffff,ai,offset);
      br+=__shfl_down_sync(0xffffffff,br,offset);
      bi+=__shfl_down_sync(0xffffffff,bi,offset);
    }
    double c=0, s=0;
    if (lane==0) {
      double angle;
      if (prescribed) {
        angle=prescribed_angles[(long long)particle*plane_count+step];
      } else {
        const double aa=ar*ar+ai*ai, bb=br*br+bi*bi;
        const double cross=ar*br+ai*bi;
        const double center=.5*atan2(2*cross,aa-bb);
        const double spread=hypot(aa-bb,2*cross);
        const double largest=.5*(aa+bb+spread);
        const double smallest=fmax(0.,.5*(aa+bb-spread));
        const double pi=3.14159265358979323846;
        const double anisotropy=largest>0 ? fmax(0.,1-smallest/largest) : 0.;
        if (beta*anisotropy>1.) {
          // On [-pi/2,pi/2], sin^2(x)>=4x^2/pi^2 and
          // (1-a sin^2(x))^beta<=exp(-4 beta a x^2/pi^2).
          // A normal envelope therefore has bounded acceptance cost at high beta.
          const double c=4*beta*anisotropy/(pi*pi);
          while (true) {
            ++attempts;
            angle=curand_normal_double(&state)/sqrt(2*c);
            if (fabs(angle)>pi/2) continue;
            const double sine=sin(angle);
            const double log_accept=beta*log(fmax(1-anisotropy*sine*sine,1e-300))+c*angle*angle;
            if (log(curand_uniform_double(&state))<=log_accept) break;
          }
          if(curand_uniform_double(&state)<.5) angle+=pi;
        } else {
          while (true) {
            angle=(2*curand_uniform_double(&state)-1)*pi;
            ++attempts;
            const double cosine=cos(angle);
            const double ratio=largest>0 ?
                (smallest+(largest-smallest)*cosine*cosine)/largest : 1.;
            if (beta==0 || log(curand_uniform_double(&state)) <=
                beta*log(fmax(ratio,1e-300))) break;
          }
        }
        angle+=center;
      }
      sincos(angle,&s,&c);
    }
    c=__shfl_sync(0xffffffff,c,0);
    s=__shfl_sync(0xffffffff,s,0);
    // All permutation reads must finish before any thread writes the ket.
    __syncwarp();
    for (int j=lane,v=0; j<dim; j+=32,++v) {
      const double2 x=ket[j], b=transformed[v];
      ket[j]=make_double2(c*x.x+s*b.x,c*x.y+s*b.y);
    }
    __syncwarp();
  }
  for (int j=lane; j<dim; j+=32) kets[(long long)particle*dim+j]=ket[j];
  if (lane==0) trials[particle]=attempts;
}
'''


@lru_cache(None)
def _kernel():
    import cupy as cp
    from cupy.cuda import get_cuda_path
    return cp.RawKernel(CUDA_SOURCE, 'heatbath',
                        options=('--std=c++11', '-I'+get_cuda_path()+'/include'))


@lru_cache(None)
def _actions(n, device):
    import cupy as cp
    p, f = actions(n)
    return cp.asarray(p, dtype=cp.int32), cp.asarray(f, dtype=cp.complex128)


def heatbath_device(kets, witness, beta, rng, *, sweeps=1, n=8,
                    planes=None, prescribed_angles=None):
    import cupy as cp
    if not 2 <= n <= 8 or kets.dtype != cp.complex128:
        raise ValueError('CUDA heatbath supports n=2..8, complex128 only')
    permutations, phases = _actions(n, cp.cuda.runtime.getDevice())
    if planes is None:
        planes = np.concatenate([rng.permutation(len(permutations)) for _ in range(sweeps)])
    planes = cp.asarray(planes, dtype=cp.int32)
    seed = rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
    trials = cp.empty(len(kets), dtype=cp.uint64)
    angles = cp.empty(1, cp.float64) if prescribed_angles is None else cp.asarray(prescribed_angles, dtype=cp.float64)
    _kernel()(((len(kets)+3)//4,), (128,),
              (kets, cp.asarray(witness, dtype=cp.complex128), permutations,
               phases, planes, np.int32(len(planes)), np.int32(len(kets)),
               np.int32(kets.shape[1]), np.float64(beta), seed, trials, angles,
               np.int32(prescribed_angles is not None)))
    return {'planes_per_particle': len(planes),
            'angle_trials_mean': float(cp.mean(trials, dtype=cp.float64).get()),
            'angle_proposal': 'uniform_or_proved_truncated_normal_envelope',
            'exact_conditional_updates': True, 'backend': 'cuda_complex128_philox'}


def sample_tempered_gpu(witness, endpoint, count, rng, *, n=8, ess_fraction=.8,
                        sweeps=2, callback=None, checkpoints=(), state_callback=None,
                        log_z_anchors=None, device=0):
    import cupy as cp
    if not 0 < ess_fraction < 1 or count < 2 or endpoint < 0 or sweeps < 1:
        raise ValueError('invalid tempering settings')
    u = np.asarray(witness, complex)
    if u.shape != (2**(n-1),) or not np.isfinite(u).all() or np.linalg.norm(u)==0:
        raise ValueError('expected a finite nonzero even-sector witness')
    started = time.monotonic()
    with cp.cuda.Device(device):
        kets = sample_haar_device(n, count, rng)
        cp.cuda.get_current_stream().synchronize()
        initial_seconds = time.monotonic()-started
        du = cp.asarray(u)
        def overlaps():
            return cp.asnumpy(2*cp.log(cp.maximum(cp.abs(kets.conj()@du), 1e-300)))
        log_r = overlaps()
        ancestors = np.arange(count)
        beta, log_z = 0., 0.
        ladder = []
        anchors = {} if log_z_anchors is None else dict(log_z_anchors)
        stops = sorted(set([float(x) for x in (*checkpoints, *anchors) if 0<x<=endpoint]+[float(endpoint)]))
        stop_index = 0
        while beta < endpoint-1e-12:
            stop = stops[stop_index]
            new_beta = next_temperature(log_r, beta, stop, ess_fraction)
            log_increment = (new_beta-beta)*log_r
            norm = float(logsumexp(log_increment))
            log_z += norm-np.log(count)
            weights = np.exp(log_increment-norm)
            chosen = systematic_resample(weights, rng)
            kets = cp.ascontiguousarray(kets[cp.asarray(chosen)])
            ancestors = ancestors[chosen]
            move = heatbath_device(kets, du, new_beta, rng, sweeps=sweeps, n=n)
            log_r = overlaps()
            beta = new_beta
            row = {'beta': beta, 'log_z': log_z,
                   'incremental_ess': float(1/np.sum(weights**2)),
                   'unique_initial_ancestors': len(np.unique(ancestors)),
                   'log_r_mean': float(log_r.mean()), 'log_r_std': float(log_r.std()),
                   'elapsed_seconds': time.monotonic()-started, **move}
            if beta in anchors:
                row['log_z_before_analytic_anchor'] = log_z
                row['analytic_anchor_log_z'] = float(anchors[beta])
                log_z = float(anchors[beta])
                row['log_z'] = log_z
            ladder.append(row)
            if callback:
                callback(row)
            if beta >= stop-1e-12:
                if state_callback:
                    state_callback(cp.asnumpy(kets), beta, log_z, log_r, row)
                stop_index += 1
        host = cp.asnumpy(kets)
    norm_error = float(np.max(abs(np.linalg.norm(host,axis=1)-1)))
    if norm_error > 1e-9:
        raise ArithmeticError('Gaussian CUDA chain lost normalization')
    diagnostics = {'method': 'adaptive_tempered_smc_exact_plane_heatbath',
                   'backend': 'cuda_complex128_philox', 'device': device,
                   'initial_haar_method': 'recursive_sphere_householder_pin_lift',
                   'count': count, 'sweeps_per_temperature': sweeps,
                   'ess_fraction': ess_fraction, 'ladder': ladder,
                   'max_norm_error': norm_error, 'initial_haar_seconds': initial_seconds,
                   'elapsed_seconds': time.monotonic()-started,
                   'analytic_log_z_anchors': anchors,
                   'iid_particles': False, 'strictly_certified': False}
    return TemperedResult(host, log_z, beta, log_r, diagnostics)
