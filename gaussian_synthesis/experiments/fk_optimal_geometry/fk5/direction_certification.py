"""Global support brackets and attainable-strength exclusion for fixed directions."""
import numpy as np
from scipy.optimize import minimize
from mpmath import iv
from . import paths
from .window_sharp_bounds import exact_hopping_coordinates
from fk4.certificates import _bounds
from fk4.exact_cartan import dimension


def joint_plus_remainder(z):
    omega=np.array([1.,-np.sqrt(8),np.sqrt(7)])
    w0=np.array([2.,0.,0.])
    def envelope(a,b):
        return 4*a+b/2 if b<=4*a else a+b+2*a*a/(b-2*a)
    def objective(p):return envelope(*p)+np.linalg.norm(z-p[0]*omega-p[1]*w0)
    a=max(0.,-z[1]/np.sqrt(8));b=max(0.,(z[0]-a)/2)
    candidates=[np.array([a,b]),np.array([0.,max(0.,z[0]/2)]),np.array([0.,0.])]
    for start in candidates.copy():
        result=minimize(objective,start,bounds=[(0,None),(0,None)],method='Nelder-Mead',options={'maxiter':500,'xatol':1e-12,'fatol':1e-12})
        if np.all(np.isfinite(result.x)):candidates.append(np.maximum(result.x,0))
    a,b=min(candidates,key=objective);A,B=iv.mpf(float(a)),iv.mpf(float(b))
    h=4*A+B/2 if b<=4*a else A+B+2*A*A/(B-2*A)
    residual=[iv.mpf(float(z[0]))-A-2*B,iv.mpf(float(z[1]))+iv.sqrt(8)*A,iv.mpf(float(z[2]))-iv.sqrt(7)*A]
    return _bounds(h+iv.sqrt(sum(v*v for v in residual)))[1],[float(a),float(b)]


def certify_direction(z,t,k,spectrum,exact4):
    if np.any(np.asarray(z).imag!=0):raise ValueError('Current directed support formulas use the frozen real singlet directions')
    z=np.asarray(z).real;iv.dps=80;y=[iv.mpf(float(v)) for v in z]
    C=(z[0]+z[2]/np.sqrt(7))/2;D=(-z[0]+5*z[2]/np.sqrt(7))/4;linear=-z[1]/np.sqrt(2)
    choices=[]
    for family,constant,quadratic in [('A',C,D),('B',D,C)]:
        parameters=[-1.,0.,1.]
        if quadratic and -1<-linear/(2*quadratic)<1:parameters.append(float(-linear/(2*quadratic)))
        for parameter in parameters:
            x=iv.mpf(parameter)
            projected=([(2-x*x)/4,-x/iv.sqrt(2),(2+5*x*x)/(4*iv.sqrt(7))] if family=='A'
                       else [(2*x*x-1)/4,-x/iv.sqrt(2),(5+2*x*x)/(4*iv.sqrt(7))])
            value=sum(v*w for v,w in zip(y,projected))
            choices.append(dict(family=family,parameter=parameter,signed_overlap_interval=_bounds(value),support_lower=_bounds(abs(value))[0]))
    best=max(choices,key=lambda r:r['support_lower']);hlo=best['support_lower']
    q4,_=exact4.evaluate(z,80);qbounds=_bounds(q4);h4=_bounds(q4**(iv.mpf(1)/8))[1]
    joint,coefficients=joint_plus_remainder(z)
    hup=min(h4,joint)
    q=exact_hopping_coordinates(t,spectrum['ground_energy_interval'])
    yh=[y[0]/4-y[1]/iv.sqrt(2)+iv.sqrt(7)*y[2]/4,
        iv.sqrt(14)*y[0]/4-iv.sqrt(2)*y[2]/4,
        y[0]/4+y[1]/iv.sqrt(2)+iv.sqrt(7)*y[2]/4]
    overlap=sum(v*w for v,w in zip(q,yh));overlap_interval=_bounds(overlap)
    lower=_bounds(overlap/iv.mpf(hup))[0];ceiling=_bounds(overlap/iv.mpf(hlo))[1]
    loglower=max(_bounds(2*k*iv.log(iv.mpf(hlo)))[0],
                 _bounds(iv.log(iv.mpf(dimension(8,k)))+iv.mpf(k)/4*(iv.log(iv.mpf(qbounds[0]))-iv.log(iv.mpf(dimension(8,4)))))[0])
    logupper=_bounds(iv.mpf(k)/4*iv.log(iv.mpf(qbounds[1])))[1]
    return dict(t=t,k=k,frozen_real_z=z.tolist(),physical_target_overlap_interval=overlap_interval,
        support_lower=hlo,support_lower_attaining_atom=best,all_explicit_orbit_candidates=choices,
        q4_support_upper=h4,joint_plus_Hilbert_remainder_support_upper=joint,
        joint_decomposition_coefficients=coefficients,global_support_upper=hup,
        certified_direction_cost_lower=lower,best_possible_direction_cost_upper=ceiling,
        exact_q4_interval=qbounds,strict_high_order_log_q_envelope=[loglower,logupper],
        moment_envelope_method='Cartan block inclusion above; Jensen and actual Gaussian coherent-projector overlap below',
        high_order_point_integral_certified=False,
        support_upper_uses_local_search=False)
