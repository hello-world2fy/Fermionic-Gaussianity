"""Eliminate canonical radial scale exactly, retaining the direction equation.

For v with lambda=Re<v|psi>>0 and Im<v|psi>=0, define
u(v)=(lambda/q_k(v))^(1/(2k-1)) v using the exact moment, not an MC
normalization estimate. Then C_k(u(v))=lambda E_pi_k[g/<v|g>],
B(u(v))=lambda/E_P|<g|v>|, and L(v)=lambda/q_k(v)^(1/(2k)).
The first two expressions do not require estimating Z. Residuals validate
the direction equation; the implicit radius is not a certified numeric u.
"""
import numpy as np


def target_overlap(direction,target):
    overlap=np.vdot(direction,target)
    if overlap.real<=0 or abs(overlap.imag)>1e-11*max(1.,abs(overlap)):
        raise ValueError('direction must have positive real target overlap')
    return float(overlap.real)


def normalized_reconstruction(direction,target,kets):
    lam=target_overlap(direction,target)
    a=np.asarray(kets).conj()@direction
    return lam*np.mean(kets/a.conj()[:,None],axis=0)


def coefficient_mass(direction,target,P_kets):
    return target_overlap(direction,target)/float(np.mean(abs(np.asarray(P_kets).conj()@direction)))


def exact_radial_scale(direction,target,q,k):
    return (target_overlap(direction,target)/q)**(1/(2*k-1))
