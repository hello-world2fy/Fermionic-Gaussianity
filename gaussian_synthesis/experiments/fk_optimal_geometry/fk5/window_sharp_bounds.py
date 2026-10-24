"""Normalized excitation-cap fidelity and spectral-penalty energy certificates."""
from fractions import Fraction
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq,minimize_scalar
from mpmath import iv
from .two_orbit import ground_root_enclosure,interval_fraction
from . import paths
from fk4.certificates import _bounds


def exact_hopping_coordinates(t,energy_interval):
    iv.dps=90
    lo,hi=ground_root_enclosure(t,energy_interval)
    E=iv.mpf([interval_fraction(lo).a,interval_fraction(hi).b]);T=iv.mpf(float(t))
    r1=(E+56)/(iv.sqrt(2)*T);r2=iv.sqrt(14)*T*r1/(E+8)
    z0=1/iv.sqrt(1+r1*r1+r2*r2);z1=r1*z0;z2=r2*z0
    return [z0/4-z1/iv.sqrt(2)+iv.sqrt(7)*z2/4,
            iv.sqrt(14)*z0/4-iv.sqrt(2)*z2/4,
            z0/4+z1/iv.sqrt(2)+iv.sqrt(7)*z2/4]


def penalized_eigenvalue_lower(t,mu1,mu2,steps=90):
    """Exact rational Sylvester minors; the off-diagonal squared is 126."""
    T=Fraction(float(t));d=[-14-4*T,-50+Fraction(float(mu1)),-14+4*T+Fraction(float(mu2))]
    lo=min(d)-24;hi=d[0]
    def positive(x):
        p1=d[0]-x;p2=(d[1]-x)*p1-126;p3=(d[2]-x)*p2-126*p1
        return p1>0 and p2>0 and p3>0
    assert positive(lo) and not positive(hi)
    for _ in range(steps):
        mid=(lo+hi)/2
        if positive(mid):lo=mid
        else:hi=mid
    return lo,hi


def choose_penalties(t,caps):
    EG=-14-4*t;A=3*np.sqrt(14)
    H=np.array([[EG,-A,0],[-A,-50,-A],[0,-A,-14+4*t]])
    candidates=[(0.,0.)]
    for active in [(1,),(2,),(1,2)]:
        r=np.zeros(3)
        if len(active)==2:
            r[1:]=caps
            if np.sum(r*r)>=1:continue
            r[0]=np.sqrt(1-np.sum(r*r))
        else:
            fixed=active[0];free=3-fixed;r[fixed]=caps[fixed-1]
            if r[fixed]>=1:continue
            upper=min(caps[free-1],np.sqrt(1-r[fixed]**2))
            def vector(x):
                v=r.copy();v[free]=x;v[0]=np.sqrt(max(0.,1-np.sum(v[1:]**2)));return v
            def objective(x):v=vector(x);return float(v@H@v)
            optimum=minimize_scalar(objective,bounds=(0,upper),method='bounded',options={'xatol':1e-14})
            x=min([0.,upper,optimum.x],key=objective);r=vector(x)
        if r[0]<=0:continue
        eigen=EG-A*r[1]/r[0]
        mu=[0.,0.]
        for j in active:
            if r[j]>0:mu[j-1]=max(0.,eigen-(H@r)[j]/r[j])
        candidates.append(tuple(mu))
    def bound(mu):return eigh(H+np.diag([0,*mu]),eigvals_only=True)[0]-np.dot(mu,np.array(caps)**2)
    return max(candidates,key=bound)


def certify_window(row,spectrum):
    t=row['t'];iv.dps=90
    D=iv.mpf(row['delta_numerator'])/row['delta_denominator'];B=iv.mpf(row['coefficient_budget'])
    a=4*(iv.mpf(t)-9);X=2*D/(iv.sqrt(a*a+72*D)+a)
    simple=iv.sqrt(iv.mpf(3)/2)*X
    if _bounds(X)[1]<.5:
        cayley=3*X/(iv.sqrt(14)*iv.sqrt(1-2*X))
        h4=cayley if _bounds(cayley)[1]<_bounds(simple)[0] else simple
    else:h4=simple
    cap=[B*h4,B*X*X/4]
    q=exact_hopping_coordinates(t,spectrum['ground_energy_interval'])
    assert all(_bounds(x)[0]>0 for x in q)
    qf=np.array([float(v.mid) for v in q]);caps=np.array([float(v.mid) for v in cap])
    def norm(alpha):return np.sum(np.minimum(np.r_[np.inf,caps],alpha*qf)**2)-1
    alpha=brentq(norm,0,2/qf[0]);r=np.minimum(np.r_[np.inf,caps],alpha*qf)
    lam=np.maximum(0.,qf[1:]-r[1:]/alpha)
    residual=iv.sqrt(q[0]**2+(q[1]-lam[0])**2+(q[2]-lam[1])**2)
    overlap=residual+iv.mpf(float(lam[0]))*cap[0]+iv.mpf(float(lam[1]))*cap[1]
    F=min(1.,_bounds(overlap**2)[1])
    mu=choose_penalties(t,caps)
    elo,ehi=penalized_eigenvalue_lower(t,*mu)
    E=interval_fraction(elo)-iv.mpf(float(mu[0]))*cap[0]**2-iv.mpf(float(mu[1]))*cap[1]**2
    gap=iv.mpf(spectrum['full_spectral_gap_lower'])
    excited=iv.mpf(spectrum['ground_energy_interval'][0])+gap
    assert _bounds(excited)[0]>-14-4*t
    direct_error=max(0.,_bounds(E-iv.mpf(spectrum['ground_energy_interval'][1]))[0])
    fidelity_energy=max(0.,_bounds(gap*(1-iv.mpf(F)))[0])
    return dict(t=t,delta=row['delta'],budget_multiplier=row['budget_multiplier'],coefficient_budget=row['coefficient_budget'],
        excitation_caps_intervals=[_bounds(v) for v in cap],physical_hopping_coordinates_intervals=[_bounds(v) for v in q],
        four_excitation_support_method='minimum of elementary symmetric bound and exact Cayley pairing-operator bound',
        fidelity_test_weights=lam.tolist(),fidelity_upper=F,
        spectral_gap_energy_error_lower=fidelity_energy,
        energy_penalties=list(mu),penalized_singlet_eigenvalue_rational_bracket=[[v.numerator,v.denominator] for v in [elo,ehi]],
        non_ground_full_spectrum_lower=_bounds(excited)[0],
        direct_energy_lower=_bounds(E)[0],direct_energy_error_lower=direct_error,
        best_energy_error_lower=max(direct_error,fidelity_energy,row['ground_energy_error_lower']),
        old_fidelity_upper=row['fidelity_upper'],old_energy_error_lower=row['ground_energy_error_lower'],
        strict_full_window_bounds=True,arbitrary_actual_Gaussian_terms=True,interval_decimal_digits=90)
