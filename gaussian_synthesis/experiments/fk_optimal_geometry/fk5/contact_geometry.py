"""Joint scalar stability and exact relative-rotation Hessian identities.

The scalar envelope does not imply global quotient-distance stability.
The separate polynomial proof certifies six specific normal eigenvalues.
"""
import numpy as np


def joint_envelope(sqrt_eta,lam):
    x=np.asarray(sqrt_eta)
    if lam<4 or np.any((x<0)|(x>1)):raise ValueError('lambda>=4, sqrt_eta in[0,1]')
    return 1+lam+2*x+(1-lam/2)*x*x


def completed_square(sqrt_eta,lam):
    return 1+lam+2/(lam-2)-(lam-2)/2*(np.asarray(sqrt_eta)-2/(lam-2))**2


def exact_arc_polynomial(model):
    """Gaussian-integer polynomial identity for n=e0; b²=1-a².

    Returns dyadic Fock-vector coefficients of g(a,b). All intermediate
    integers are bounded and checked below 2^45, so binary64 is exact here.
    """
    from .joint_dual_certificates import exact_schmidt_polar
    ids=model.even_masks
    hop=2*model.h_hop[np.ix_(ids,ids)]
    mass=2*(model.q_ops[1,0]-model.q_ops[0,0])[np.ix_(ids,ids)]
    initial=exact_schmidt_polar()[ids]
    current={(0,0):initial};total={}
    for degree in range(1,5):
        following={}
        for (pa,pb),v in current.items():
            for key,value in [((pa+1,pb),hop@v),((pa,pb+1),mass@v)]:
                following[key]=following.get(key,np.zeros(128,complex))+value
        current=following
        for key,value in current.items():
            total[key]=total.get(key,np.zeros(128,complex))+{1:128,2:-16,3:-8,4:1}[degree]*value
    from math import comb
    reduced={}
    for (pa,pb),value in total.items():
        for j in range(pb//2+1):
            key=(pa+2*j,pb%2)
            reduced[key]=reduced.get(key,np.zeros(128,complex))+comb(pb//2,j)*(-1)**j*value
    result={}
    for parity in [0,1]:
        polynomial=[reduced.get((j,parity),np.zeros(128,complex)).copy() for j in range(5)]
        for degree in range(4,1,-1):
            lead=polynomial[degree].copy()
            if lead.any():result[(degree-2,parity)]=lead/6144
            for j,multiplier in enumerate([1,2,1]):polynomial[degree-2+j]-=multiplier*lead
        assert not any(v.any() for v in polynomial)
    for v in reduced.values():
        assert np.array_equal(v.real,np.rint(v.real)) and np.array_equal(v.imag,np.rint(v.imag))
        assert np.max(abs(v))<2**45
    for v in result.values():
        assert np.array_equal(v.real*16,np.rint(v.real*16))
        assert np.array_equal(v.imag*16,np.rint(v.imag*16))
    return result


def exact_relative_rotation_hessian(model):
    """Prove six soft curvatures by exact polynomial, not precision fitting."""
    from fk.conventions import quadratic_operator
    from .two_orbit import scaled_basis
    from math import comb
    g=exact_arc_polynomial(model);ids=model.even_masks;D=scaled_basis()[ids]
    def inner_polynomial(left,right):
        total={}
        for (pa,pb),x in left.items():
            for (qa,qb),y in right.items():
                value=np.vdot(x,y)
                for j in range((pb+qb)//2+1):
                    key=(pa+qa+2*j,(pb+qb)%2)
                    total[key]=total.get(key,0j)+comb((pb+qb)//2,j)*(-1)**j*value
        return {k:v for k,v in total.items() if v!=0}
    assert inner_polynomial(g,g)=={(0,0):1}
    derivatives=[]
    for j in range(1,7):
        q=quadratic_operator(model.j_matrices[0]@model.j_matrices[j],model.local_gammas)
        G=(-.5j*(np.kron(q,np.eye(16))-np.kron(np.eye(16),q)))[np.ix_(ids,ids)]
        v={key:G@value for key,value in g.items()}
        second={key:G@value for key,value in v.items()}
        assert inner_polynomial(g,v)=={}
        assert inner_polynomial(v,v)=={(0,0):1}
        assert all(not np.any(D.conj().T@value) for value in v.values())
        projection={key:D.conj().T@value for key,value in second.items()}
        projection={k:v for k,v in projection.items() if v.any()}
        assert set(projection)=={(0,0),(1,0)}
        assert np.array_equal(projection[(0,0)],[0,0,-4])
        assert np.array_equal(projection[(1,0)],[0,2,0])
        derivatives.append(v)
    for i in range(6):
        for j in range(i):assert inner_polynomial(derivatives[i],derivatives[j])=={}
    # Check the generalized Hessian eigenvector identity against ALL 120
    # Majorana-generator directions, so a Rayleigh quotient is not confused
    # with an eigenvalue. Multiply the witness by 6 to keep exact dyadics.
    from fk4.tempering import actions
    perm,phase=actions(8)
    wC={(0,0):5*D[:,0]+D[:,2]}
    wD={(0,0):-2*D[:,0]+2*D[:,2],(1,0):6*D[:,1]}
    def real(poly):return {k:v.real for k,v in poly.items() if v.real!=0}
    def same(left,right):
        return all(left.get(k,0)==right.get(k,0) for k in set(left)|set(right))
    for j,v in enumerate(derivatives,1):
        q=quadratic_operator(model.j_matrices[0]@model.j_matrices[j],model.local_gammas)
        G=(-.5j*(np.kron(q,np.eye(16))-np.kron(np.eye(16),q)))[np.ix_(ids,ids)]
        for pp,ff in zip(perm,phase):
            acted={key:ff*value[pp] for key,value in g.items()}
            assert not real(inner_polynomial(wC,acted))
            assert not real(inner_polynomial(wD,acted))
            sym={key:G@acted[key]+ff*v[key][pp] for key in g}
            hc={key:-value for key,value in real(inner_polynomial(wC,sym)).items()}
            hd={key:-value for key,value in real(inner_polynomial(wD,sym)).items()}
            metric=real(inner_polynomial(v,acted))
            tc={key:8*value for key,value in metric.items()}
            td={key:16*value for key,value in metric.items()}
            for (pa,pb),value in metric.items():td[(pa+2,pb)]=td.get((pa+2,pb),0)-24*value
            assert same(hc,tc) and same(hd,td)
    return dict(exact_polynomial_identities_passed=True,relative_directions=6,
                metric_exact='I6',singlet_first_derivative_exact_zero=True,
                scaled_singlet_second_derivative='(0,2a,-4)',
                full_generator_eigenvector_identities_checked=720,
                curvature_formula='4*(y2/sqrt7-D*a²)/(C-D*a²), C=(y0+y2/sqrt7)/2, D=(-y0+5*y2/sqrt7)/4',
                stationarity_condition='y1=2*sqrt2*D*a',
                zero_curvature_discriminant='y1²+2*y0*(y2/sqrt7)-10*(y2/sqrt7)²=0')
