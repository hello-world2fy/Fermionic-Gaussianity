"""Hamiltonian variational dynamics of a restricted full-Gaussian k2 map.

The precomputed tensor gives z_a(c)=T_aijk c_i c_j c_k*. This is an actual
coherent integral, not a polynomial fitted to future wavefunctions. The map
is nonholomorphic. Its real symplectic pullback, rather than a holomorphic
complex Gram equation, is used for Hamiltonian TDVP.
"""
import numpy as np
from scipy.integrate import solve_ivp


class CubicManifold:
    def __init__(self,tensor,H,tangent_cutoff=1e-10,symplectic_cutoff=1e-10):
        self.tensor=np.asarray(tensor);self.H=np.asarray(H)
        self.tangent_cutoff=tangent_cutoff;self.symplectic_cutoff=symplectic_cutoff
        self.calls=0;self.min_symplectic=1.;self.max_energy_derivative=0.
        self.ranks=set();self.max_equation_residual=0.

    def raw(self,c):
        return np.einsum('aijk,i,j,k->a',self.tensor,c,c,c.conj(),optimize=True)

    def raw_jacobian(self,c):
        linear=2*np.einsum('aijk,j,k->ai',self.tensor,c,c.conj(),optimize=True)
        anti=np.einsum('aijk,i,j->ak',self.tensor,c,c,optimize=True)
        return np.column_stack((linear+anti,1j*(linear-anti)))

    def state(self,c):
        z=self.raw(c);return z/np.linalg.norm(z)

    def tangent(self,c):
        z=self.raw(c);norm=np.linalg.norm(z);psi=z/norm
        J=self.raw_jacobian(c)/norm
        J-=psi[:,None]*(psi.conj()@J)[None,:]
        return psi,J

    def rhs(self,t,c):
        psi,J=self.tangent(c);n=len(psi)
        U,s,Vh=np.linalg.svd(np.vstack((J.real,J.imag)),full_matrices=False)
        keep=s>self.tangent_cutoff*s[0]
        U=U[:,keep];s=s[keep];Vh=Vh[keep]
        X=U[:n]+1j*U[n:]
        omega=(X.conj().T@X).imag
        force=-(X.conj().T@(self.H@psi)).real
        singular=np.linalg.svd(omega,compute_uv=False)
        minimum=float(singular[-1]);self.min_symplectic=min(self.min_symplectic,minimum)
        if minimum<self.symplectic_cutoff:
            raise RuntimeError(f'Degenerate symplectic tangent at t={t}: {minimum}')
        velocity=np.linalg.solve(omega,force)
        real_rate=Vh.T@(velocity/s)
        k=len(c);dc=real_rate[:k]+1j*real_rate[k:]
        physical_rate=X@velocity
        energy_rate=2*np.vdot(self.H@psi,physical_rate).real
        self.max_energy_derivative=max(self.max_energy_derivative,float(abs(energy_rate)))
        self.max_equation_residual=max(self.max_equation_residual,float(np.linalg.norm(omega@velocity-force)))
        self.calls+=1;self.ranks.add(int(sum(keep)))
        return dc

    def evolve(self,times,initial,rtol=2e-8,atol=2e-10,max_step=.025):
        solution=solve_ivp(self.rhs,(float(times[0]),float(times[-1])),np.asarray(initial,complex),
            t_eval=times,method='DOP853',rtol=rtol,atol=atol,max_step=max_step)
        if not solution.success:raise RuntimeError(solution.message)
        states=np.array([self.state(c) for c in solution.y.T])
        energies=np.einsum('ti,ij,tj->t',states.conj(),self.H,states).real
        report={'calls':self.calls,'physical_tangent_ranks':sorted(self.ranks),
            'minimum_whitened_symplectic_singular':self.min_symplectic,
            'max_instantaneous_energy_derivative':self.max_energy_derivative,
            'max_symplectic_equation_residual':self.max_equation_residual,
            'max_norm_error':float(np.max(abs(np.sum(abs(states)**2,axis=1)-1))),
            'max_energy_drift':float(np.max(abs(energies-energies[0]))),
            'rtol':rtol,'atol':atol,'max_step':max_step,
            'global_phase':'Horizontal projective gauge; compare states modulo global phase.'}
        return states,solution.y.T,report
