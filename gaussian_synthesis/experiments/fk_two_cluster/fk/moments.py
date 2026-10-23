"""Sample-average moments in three complex witness coordinates.

The dictionary supplies its own frame constant. No projection of atoms is
renormalized. Real coordinates are [Re(z), Im(z)].
"""
from dataclasses import dataclass
import numpy as np


def pack(z):
    return np.concatenate((np.real(z), np.imag(z)))


def unpack(x):
    x = np.asarray(x, dtype=float)
    return x[:len(x)//2] + 1j*x[len(x)//2:]


@dataclass
class MomentResult:
    q: float
    mass: float
    gradient: np.ndarray
    hessian: np.ndarray | None
    ess: float


class ProjectedMoments:
    def __init__(self, overlaps, k, log_frame_constant):
        self.h = np.asarray(overlaps, dtype=np.complex128)
        if self.h.ndim != 2 or len(self.h) == 0 or not np.isfinite(self.h).all():
            raise ValueError('overlaps must be a finite nonempty matrix')
        if type(k) is not int or k < 1:
            raise ValueError('k must be a positive integer')
        self.k = k
        self.d = float(np.exp(log_frame_constant))
        self.jr = np.concatenate((self.h.real, -self.h.imag), axis=1)
        self.ji = np.concatenate((self.h.imag, self.h.real), axis=1)

    def evaluate(self, z, hessian=True):
        a = np.einsum('ni,i->n', self.h, z, optimize=False)
        r = np.abs(a)**2
        w = r**(self.k-1)
        f = self.d/len(r)
        b = a.real[:,None]*self.jr + a.imag[:,None]*self.ji
        grad = f*np.einsum('n,ni->i', w, b, optimize=False)
        hes = None
        if hessian:
            hes = f*(np.einsum('ni,n,nj->ij', self.jr,w,self.jr,optimize=False)
                     +np.einsum('ni,n,nj->ij',self.ji,w,self.ji,optimize=False))
            if self.k > 1:
                hes += f*2*(self.k-1)*np.einsum('ni,n,nj->ij', b,r**(self.k-2),b,optimize=False)
        mass_weights = self.d*r**(self.k-.5)
        mass = float(np.mean(mass_weights))
        denom = float(np.sum(mass_weights**2))
        ess = float(np.sum(mass_weights)**2/denom) if denom else 0.
        return MomentResult(float(f*np.sum(r**self.k)), mass, grad, hes, ess)

    def objective(self, z, target):
        return self.evaluate(z, hessian=False).q/(2*self.k)-float(np.vdot(target,z).real)


def sample_coefficients(kets, u, k, log_frame_constant):
    a = np.einsum('ni,i->n', kets.conj(), u, optimize=False)
    return np.exp(log_frame_constant)*np.abs(a)**(2*k-2)*a
