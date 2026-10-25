import sys
from pathlib import Path
from fractions import Fraction
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.finite_dictionary_cost import FiniteDictionaryCertificate,dyadic,inner,sqrt_upper


class FiniteDictionaryCostTests(unittest.TestCase):
    def test_exact_complex_dot_agrees_with_independent_fractions(self):
        a=np.array([.1+.2j,-.3+.7j]);b=np.array([.8-.9j,.4+.6j]);r,i,d=inner(dyadic(a),dyadic(b))
        real=sum(Fraction(float(x.real))*Fraction(float(y.real))+Fraction(float(x.imag))*Fraction(float(y.imag)) for x,y in zip(a,b))
        imag=sum(Fraction(float(x.real))*Fraction(float(y.imag))-Fraction(float(x.imag))*Fraction(float(y.real)) for x,y in zip(a,b))
        self.assertEqual(Fraction(int(r[0,0]),d),real);self.assertEqual(Fraction(int(i[0,0]),d),imag)

    def test_nearly_parallel_dictionary_with_free_source_direction(self):
        A=np.array([[1.,1.],[1e-6,-1e-6]],complex);B=np.array([[1.],[0.]],complex);state=np.array([3.,.5])
        cert=FiniteDictionaryCertificate(A,B,state).evaluate(np.array([0.,1.]),Fraction(1,10))
        lower=Fraction(int(cert['coefficient_l1_lower']['numerator']),int(cert['coefficient_l1_lower']['denominator']))
        exact=Fraction(2,5)/Fraction(1e-6)
        self.assertLessEqual(lower,exact);self.assertGreater(float(lower),399999.99)

    def test_complex_feasible_coefficients_and_free_space(self):
        rng=np.random.default_rng(919);A=rng.normal(size=(7,9))+1j*rng.normal(size=(7,9))
        B,_=np.linalg.qr(rng.normal(size=(7,2))+1j*rng.normal(size=(7,2)))
        c=rng.normal(size=9)+1j*rng.normal(size=9);b=np.array([2+3j,-1j]);state=A@c+B@b
        w=state-B@(B.conj().T@state);w/=np.linalg.norm(w)
        cert=FiniteDictionaryCertificate(A,B,state).evaluate(w,Fraction(1,10**12))
        self.assertLessEqual(cert['coefficient_l1_lower']['float'],float(np.sum(abs(c))))
        value=Fraction(7,19);bound=sqrt_upper(value)
        self.assertGreaterEqual(bound*bound,value)


if __name__=='__main__':unittest.main()
