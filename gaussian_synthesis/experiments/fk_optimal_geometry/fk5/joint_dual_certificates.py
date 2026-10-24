"""Directed Schmidt–magic witnesses with an exact monomial polar matrix.

The Schmidt polar ket is the exact radical combination
S(1,-sqrt(8),sqrt(7)). Its full 16x16 reshape is a signed Gaussian-integer
permutation, hence has operator norm1 exactly. A is the low-bit factor.
Numerical singular vectors are diagnostics only, never certificate inputs.
"""
from fractions import Fraction
import numpy as np
from mpmath import iv
from . import paths
from fk4.certificates import _bounds, _c, exact_basis_action
from fk4.state_enclosure import enclose_ground_state


def exact_schmidt_polar():
    ket=np.zeros(256,complex)
    for mask,sign in [(15,-1),(60,1),(90,-1),(105,1),(150,1),(165,-1),(195,1),(240,-1)]:
        ket[mask]=-1j*sign
    for mask,sign in [(30,1),(45,-1),(75,1),(180,1),(210,-1),(225,1),(120,-1),(135,-1)]:
        ket[mask]=sign
    matrix=ket.reshape(16,16)
    assert np.array_equal(matrix.conj().T@matrix,np.eye(16))
    return ket


def interval_inner(w,target):
    return sum((a*_c(z)[0]+b*_c(z)[1] for (a,b),z in zip(w,target)),iv.mpf(0))


def choose_lambda(S,L):
    # Merely chooses a rational certificate candidate. Optimality of this
    # floating search is unnecessary for any subsequent bound.
    if S<=L:return Fraction(1000000,1)
    x=(np.sqrt(2*S*(S+L))-2*L)/(S+2*L)
    return Fraction(max(4.,2+2/x)).limit_denominator(1000000)


def certify_joint(model,t,upper,lambda_value=None,digits=70):
    target=model.target(t);state=enclose_ground_state(model,t,target,digits=digits)
    iv.dps=digits;error=iv.mpf(state['saved_target_to_exact_ground_state_distance_upper'])
    omega=exact_schmidt_polar()[model.even_masks]
    w0=exact_basis_action(np.array([2.,0.,0.],complex))
    schmidt=interval_inner([_c(x) for x in omega],target)
    magic=interval_inner(w0,target)
    S_interval=_bounds(schmidt+iv.mpf([-1,1])*4*error)
    L0_interval=_bounds(magic+iv.mpf([-1,1])*2*error)
    lam=Fraction(lambda_value) if lambda_value is not None else choose_lambda(float(schmidt.mid),float(magic.mid))
    if lam<4:raise ValueError('lambda>=4 required')
    l=iv.mpf(lam.numerator)/lam.denominator
    envelope=1+l+2/(l-2)
    Wnorm=iv.sqrt(4*l*l+4*l+16)
    numerator=schmidt+l*magic-Wnorm*error
    lower=_bounds(numerator/envelope)[0]
    L=max(1.,lower,L0_interval[0]);U=iv.mpf(float(upper))
    if L>upper:raise ArithmeticError('Primal/dual contradiction')
    mlow=iv.mpf(0)
    if S_interval[0]>upper:mlow=(iv.sqrt(iv.mpf(S_interval[0])/U)-1)**2
    mhigh=min(1.,_bounds(2*(1-iv.mpf(L0_interval[0])/U))[1])
    xstar=2/(l-2);a=(l-2)/2
    variance=(envelope-numerator/U)/a
    tail=[]
    for d in [.1,.2,.3,.4,.5]:
        tail.append({'delta_sqrt_eta':d,'fraction_upper':min(1.,_bounds(variance/iv.mpf(d)**2)[1])})
    floating_full=model.embed(target).reshape(16,16)
    left,sv,right=np.linalg.svd(floating_full)
    polar=left@right
    raw=omega+float(lam)*np.array([complex(float(a.mid),float(b.mid)) for a,b in w0])
    row={'t':float(t),'t_exact_binary64':float(t).hex(),
        'lambda_numerator':lam.numerator,'lambda_denominator':lam.denominator,
        'S_physical_interval':S_interval,'L0_physical_interval':L0_interval,
        'support_upper':_bounds(envelope)[1],'support_interval':_bounds(envelope),
        'joint_physical_overlap_lower':_bounds(numerator)[0],
        'joint_cost_lower':lower,'best_cost_lower':L,'cost_upper':float(upper),
        'relative_gap_upper':_bounds(1-iv.mpf(L)/U)[1],
        'extent_interval':[_bounds(iv.mpf(L)**2)[0],_bounds(U*U)[1]],
        'witness_norm_upper':_bounds(Wnorm)[1],
        'sqrt_eta_star_interval':_bounds(xstar),'eta_star_interval':_bounds(xstar*xstar),
        'all_cost_le_U_mean_eta_interval':[max(0.,_bounds(mlow)[0]),mhigh],
        'all_cost_le_U_mean_squared_sqrt_eta_displacement_upper':_bounds(variance)[1],
        'sqrt_eta_tail_bounds':tail,
        'polar_operator_identity_exact':True,
        'numeric_SVD_polar_error':float(np.linalg.norm(polar-exact_schmidt_polar().reshape(16,16))),
        'numeric_Schmidt_sum':float(sv.sum()),
        'physical_state_enclosure':state,'interval_decimal_digits':digits,
        'global_support_proof':'JOINT_WITNESS_THEORY.md; Gaussian Schmidt and sharp magic bounds; no sampled maximum used',
        'global_optimum_or_contact_classification_claimed':False}
    return row,{'target':target,'omega_S_full':exact_schmidt_polar(),'joint_witness_display':raw}
