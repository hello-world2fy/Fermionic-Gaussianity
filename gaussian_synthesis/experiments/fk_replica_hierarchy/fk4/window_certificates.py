"""Directed energy-window support envelope for any singlet witness."""
from mpmath import iv
from .exact_cartan import radical_transform
from .certificates import _c,_bounds


def restricted_support(z,t,delta,digits=70):
    if t<9 or delta<=0:raise ValueError('t>=9 and delta>0 required')
    iv.dps=digits
    V=radical_transform();zz=[_c(a) for a in z]
    beta=[(sum((V[j][i]*zz[j][0] for j in range(3)),iv.mpf(0)),
           sum((V[j][i]*zz[j][1] for j in range(3)),iv.mpf(0))) for i in range(3)]
    magnitudes=[iv.sqrt(re**2+im**2) for re,im in beta]
    a=4*(iv.mpf(float(t))-9);d=iv.mpf(float(delta))
    X=2*d/(iv.sqrt(a*a+72*d)+a)
    if _bounds(X)[1]>2:X=iv.mpf(2)
    bound=magnitudes[0]+iv.sqrt(iv.mpf(3)/2)*magnitudes[1]*X+magnitudes[2]*X**2/4
    return {'restricted_support_upper':_bounds(bound)[1],
            'x_cap_interval':_bounds(X),'hopping_coefficient_absolute_intervals':[_bounds(x) for x in magnitudes],
            'bound_formula':'|beta0| + sqrt(3/2)|beta4| X + |beta8| X^2/4',
            'complete_Gaussian_energy_window':True,'interval_decimal_digits':digits}


def necessity(L_lower,h_window_upper,U,digits=70):
    iv.dps=digits
    if U<=0:raise ValueError('positive declared coefficient-cost threshold required')
    if h_window_upper>=1:
        return {'fraction_upper':1.,'nontrivial':False,'overlap_gap_lower':0.}
    gap=1-iv.mpf(float(h_window_upper))
    fraction=(1-iv.mpf(float(L_lower))/iv.mpf(float(U)))/gap
    upper=max(0.,_bounds(fraction)[1])
    return {'fraction_upper':min(1.,upper),'unclipped_fraction_upper':upper,
            'nontrivial':upper<1,'overlap_gap_lower':_bounds(gap)[0]}
