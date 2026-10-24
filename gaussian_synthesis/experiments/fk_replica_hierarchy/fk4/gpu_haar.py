"""Independent Haar SO(2n) rays by a recursive Householder Pin lift.

At step k, q is uniform on the sphere in coordinates k..2n-1 and
H_k=I-2 vv^T maps e_k to q, v=(q-e_k)/||q-e_k||. Inductively,
H_0 ... H_{2n-2} diag(1,...,1,-1) is Haar on SO(2n): the first
column is uniform and the remaining frame is independently invariant.
There are 2n reflections, so their Clifford product is even and implements
this orthogonal map. It acts on the vacuum in reverse reflection order.
This avoids QR and never approximates Haar by a finite random walk.
"""
from functools import lru_cache
import numpy as np

from . import paths
from fk.gaussian import _masks, _majorana_action


CUDA_SOURCE = r'''
#include <curand_kernel.h>
extern "C" __global__ void haar(double2* output, const int* permutations,
 const double2* phases, const int count, const int dim, const int majoranas,
 const unsigned long long seed) {
  const int lane=threadIdx.x%32, warp=threadIdx.x/32;
  const int particle=blockIdx.x*4+warp;
  if(particle>=count) return;
  __shared__ double2 storage[4][128];
  __shared__ double directions[4][16];
  double2* ket=storage[warp];
  double* v=directions[warp];
  for(int j=lane;j<dim;j+=32) ket[j]=make_double2(j==0?1.:0.,0.);
  __syncwarp();
  curandStatePhilox4_32_10_t state;
  curand_init(seed,(unsigned long long)particle*32+lane,0,&state);
  int parity=0;
  for(int k=majoranas-1;k>=0;--k) {
    double x=0.;
    if(k==majoranas-1) {
      if(lane<majoranas) v[lane]=lane==k?1.:0.;
    } else {
      if(lane>=k && lane<majoranas) x=curand_normal_double(&state);
      double sum=x*x;
      for(int off=16;off>0;off/=2) sum+=__shfl_down_sync(0xffffffff,sum,off);
      sum=__shfl_sync(0xffffffff,sum,0);
      x=x/sqrt(sum)-(lane==k?1.:0.);
      sum=x*x;
      for(int off=16;off>0;off/=2) sum+=__shfl_down_sync(0xffffffff,sum,off);
      sum=__shfl_sync(0xffffffff,sum,0);
      if(lane<majoranas) v[lane]=x/sqrt(sum);
    }
    __syncwarp();
    double2 result[4];
    for(int j=lane,r=0;j<dim;j+=32,++r) {
      double re=0,im=0;
      for(int a=k;a<majoranas;++a) {
        const int idx=(parity*majoranas+a)*dim+j;
        const double2 z=ket[permutations[idx]],f=phases[idx];
        re+=v[a]*(f.x*z.x-f.y*z.y);
        im+=v[a]*(f.x*z.y+f.y*z.x);
      }
      result[r]=make_double2(re,im);
    }
    __syncwarp();
    for(int j=lane,r=0;j<dim;j+=32,++r) ket[j]=result[r];
    __syncwarp();
    parity=1-parity;
  }
  for(int j=lane;j<dim;j+=32) output[(long long)particle*dim+j]=ket[j];
}
'''


@lru_cache(None)
def _kernel():
    import cupy as cp
    from cupy.cuda import get_cuda_path
    return cp.RawKernel(CUDA_SOURCE,'haar',options=('--std=c++11','-I'+get_cuda_path()+'/include'))


@lru_cache(None)
def _pin_actions(n,device):
    import cupy as cp
    permutations,phases=[],[]
    for parity in (0,1):
        source=_masks(n,parity)
        target=_masks(n,1-parity)
        for a in range(2*n):
            destination,factor=_majorana_action(source,a)
            inverse=np.searchsorted(target,destination)
            order=np.argsort(inverse)
            permutations.append(order)
            phases.append(factor[order])
    return cp.asarray(np.array(permutations),dtype=cp.int32),cp.asarray(np.array(phases),dtype=cp.complex128)


def sample_haar_device(n,count,rng):
    import cupy as cp
    if not 2<=n<=8 or count<1:
        raise ValueError('CUDA Haar supports n=2..8 and positive count')
    permutations,phases=_pin_actions(n,cp.cuda.runtime.getDevice())
    result=cp.empty((count,2**(n-1)),cp.complex128)
    seed=rng.integers(0,np.iinfo(np.uint64).max,dtype=np.uint64)
    _kernel()(((count+3)//4,),(128,),
              (result,permutations,phases,np.int32(count),np.int32(2**(n-1)),np.int32(2*n),seed))
    return result
