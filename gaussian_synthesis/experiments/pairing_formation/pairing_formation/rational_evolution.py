"""Exact integer Taylor action and a rational real-time remainder bound."""
from fractions import Fraction
from math import factorial


def integer_taylor_action(rows,reference,time,degree):
    """exp(-i time A)e_reference, A an integer Hermitian matrix in row lists.

    Return integer real/imaginary numerators, one common denominator and
    an exact rational upper bound on the Hilbert-norm remainder. The bound
    uses Hermiticity and max absolute row sum; it is not a floating estimate.
    """
    time=Fraction(time);a,b=time.numerator,time.denominator
    real=[int(i==reference) for i in range(len(rows))];imag=[0]*len(rows);denominator=1
    for k in range(degree,0,-1):
        new_real=[a*sum(v*imag[j] for j,v in row) for row in rows]
        new_imag=[-a*sum(v*real[j] for j,v in row) for row in rows]
        new_real[reference]+=k*b*denominator
        real,imag=new_real,new_imag;denominator*=k*b
    row_bound=max(sum(abs(v) for _,v in row) for row in rows)
    x=abs(time)*row_bound
    remainder=x**(degree+1)/factorial(degree+1)
    return real,imag,denominator,remainder
