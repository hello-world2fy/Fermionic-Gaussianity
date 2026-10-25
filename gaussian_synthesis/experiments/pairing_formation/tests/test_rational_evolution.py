import sys
from pathlib import Path
import unittest
from fractions import Fraction
import numpy as np
from scipy.linalg import expm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.rational_evolution import integer_taylor_action


class RationalEvolutionTests(unittest.TestCase):
    def test_integer_Horner_matches_independent_fraction_series(self):
        rows=[[(0,1),(1,2)],[(0,2),(1,-1)]];t=Fraction(1,3);degree=8
        real,imag,den,_=integer_taylor_action(rows,0,t,degree)
        term_r=[Fraction(1),Fraction(0)];term_i=[Fraction(0),Fraction(0)]
        sum_r=term_r.copy();sum_i=term_i.copy()
        for k in range(1,degree+1):
            nr=[t*sum(v*term_i[j] for j,v in row)/k for row in rows]
            ni=[-t*sum(v*term_r[j] for j,v in row)/k for row in rows]
            term_r,term_i=nr,ni
            sum_r=[a+b for a,b in zip(sum_r,nr)];sum_i=[a+b for a,b in zip(sum_i,ni)]
        self.assertEqual([Fraction(x,den) for x in real],sum_r)
        self.assertEqual([Fraction(x,den) for x in imag],sum_i)

    def test_real_time_remainder_contains_independent_matrix_exponential(self):
        rows=[[(0,1),(1,2)],[(0,2),(1,-1)]];A=np.array([[1,2],[2,-1]])
        for degree in (3,7,11):
            re,im,den,error=integer_taylor_action(rows,0,Fraction(2,5),degree)
            q=np.array([float(Fraction(r,den))+1j*float(Fraction(i,den)) for r,i in zip(re,im)])
            self.assertLess(np.linalg.norm(q-expm(-.4j*A)[:,0]),float(error))


if __name__=='__main__':unittest.main()
