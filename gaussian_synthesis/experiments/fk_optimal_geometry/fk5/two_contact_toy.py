"""Exact Cartan moments for the eight-mode vacuum/full two-contact toy.

The Selberg ratio is evaluated with exact factorials and fractions. Its
Haar/Jacobian derivation is recorded in TWO_CONTACT_TOY.md. Independent
SMC checks use only the elementary k=1 norm anchor.
"""
from fractions import Fraction
from math import comb,factorial
from mpmath import mp


def dimension(k):
    result=Fraction(1)
    for i in range(1,9):
        for j in range(i+1,9):result*=Fraction(k+16-i-j,16-i-j)
    assert result.denominator==1
    return result.numerator


def coefficients(k):
    result=[]
    for l in range(k+1):
        moment=Fraction(1)
        for j in range(4):
            moment*=Fraction(factorial(l+2*j)*factorial(k-l+2*j)*factorial(7+2*j),
                             factorial(k+7+2*j)*factorial(2*j)**2)
        result.append(dimension(k)*comb(k,l)**2*moment)
    assert result[0]==result[-1]==1
    assert result==result[::-1]
    return result


def finite_order(k,target_ratio=Fraction(1,3),digits=90):
    mp.dps=digits
    rational=coefficients(k)
    c=[mp.mpf(x.numerator)/x.denominator for x in rational]
    ratio=mp.mpf(target_ratio.numerator)/target_ratio.denominator
    def components(logr):
        terms=[v*mp.exp(2*l*logr) for l,v in enumerate(c)]
        A=mp.fsum((k-l)*v/k for l,v in enumerate(terms))
        B=mp.fsum(l*v/k for l,v in enumerate(terms))/mp.exp(logr)
        return A,B
    # Monotonicity follows from convexity of the positive moment polynomial;
    # bisection also records the actual sign bracket, rather than trusting Newton.
    lo,hi=mp.log(ratio),mp.mpf(0)
    for _ in range(4*digits):
        mid=(lo+hi)/2;A,B=components(mid)
        if B/A>ratio:hi=mid
        else:lo=mid
    logr=(lo+hi)/2;r=mp.exp(logr);A,B=components(logr)
    norm=mp.sqrt(1+ratio**2);psi0=1/norm;psi1=ratio/norm
    scale=(psi0/A)**(mp.mpf(1)/(2*k-1))
    qbal=sum(rational,Fraction())
    return dict(k=k,balanced_q_exact=[qbal.numerator,qbal.denominator],
        balanced_log_q=float(mp.log(mp.mpf(qbal.numerator)/qbal.denominator)),
        canonical_direction_ratio=mp.nstr(r,digits),
        scaled_log_height_ratio=mp.nstr((2*k-1)*logr,digits),
        asymptotic_scaled_log_height_ratio=mp.nstr(mp.log(ratio),digits),
        canonical_radius=mp.nstr(scale,digits),
        exact_target_cost=mp.nstr(psi0+psi1,digits),
        direct_support_normalized_direction_lower=mp.nstr(psi0+r*psi1,digits),
        canonical_moment_equation_residual=mp.nstr(abs(B/A-ratio),digits),
        log_ratio_bracket=[mp.nstr(lo,digits),mp.nstr(hi,digits)],
        directional_solve_is_interval_certified=False,
        moment_polynomial_exact=True,
        coefficients_rational=[[x.numerator,x.denominator] for x in rational])
