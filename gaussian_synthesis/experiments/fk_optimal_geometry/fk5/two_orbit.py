"""Exact signed two-orbit construction and finite8-vertex realization."""
from fractions import Fraction
from math import comb
import numpy as np
from scipy.linalg import hadamard
from mpmath import iv
from . import paths
from .joint_dual_certificates import exact_schmidt_polar
from fk4.certificates import _bounds


def interval_fraction(q):
    q=Fraction(q);return iv.mpf(q.numerator)/q.denominator


def ground_root_enclosure(t,initial_interval,steps=120):
    T=Fraction(float(t));lo,hi=map(Fraction,initial_interval)
    def p(E):
        return (-56-E)*((-14-E)*(-8-E)-14*T*T)-2*T*T*(-8-E)
    flo=p(lo);fhi=p(hi);assert flo*fhi<0
    for _ in range(steps):
        mid=(lo+hi)/2;fm=p(mid)
        if fm==0:return mid,mid
        if fm*flo>0:lo=mid;flo=fm
        else:hi=mid
    return lo,hi


def endpoint_secondary_parameters(t,energy_interval):
    iv.dps=90;lo,hi=ground_root_enclosure(t,energy_interval)
    E=iv.mpf([interval_fraction(lo).a,interval_fraction(hi).b]);T=iv.mpf(float(t))
    r1=(E+56)/(iv.sqrt(2)*T);r2=iv.sqrt(14)*T*r1/(E+8)
    z0=1/iv.sqrt(1+r1*r1+r2*r2);z1=r1*z0;z2=r2*z0
    M0=(5*z0+iv.sqrt(7)*z2)/3
    M1=-iv.sqrt(2)*z1;M2=2*M0-4*z0
    a=(M2+M1)/(M1+M0)
    positive=(M1+M0)**2/(M2+2*M1+M0);negative=positive-M0
    assert _bounds(a)[0]>-1 and _bounds(a)[1]<1
    assert _bounds(positive)[0]>0 and _bounds(negative)[0]>0
    cost=positive+negative
    result={'t':float(t),'ground_energy_rational_bracket':[[lo.numerator,lo.denominator],[hi.numerator,hi.denominator]],
        'physical_singlet_coordinates_intervals':[_bounds(z) for z in [z0,z1,z2]],
        'main_arc_parameter_interval':_bounds(a),'main_eta_interval':_bounds(a*a),
        'positive_orbit_mass_interval':_bounds(positive),'negative_orbit_mass_interval':_bounds(negative),
        'exact_construction_cost_interval':_bounds(cost),'exact_actual_terms':16,'distinct_Gaussian_rays':9,
        'secondary_orbit_is_single_Gaussian_repeated_eight_times':True,
        'no_Fock_repair':True,
        'exact_target_definition':'Unique full ground state; smallest reduced cubic root, z0>0; coefficients and main arc parameter are explicit algebraic functions of that root.',
        'interval_decimal_digits':90}
    return result,float(a.mid),float(positive.mid),float(negative.mid)


def scaled_basis():
    D=np.zeros((256,3),complex)
    for m,s in [(119,1),(120,-1),(135,-1),(136,1)]:D[m,0]=s
    for m,s in [(15,-1),(60,1),(90,-1),(105,1),(150,1),(165,-1),(195,1),(240,-1)]:D[m,1]=1j*s
    for m,s in [(30,2),(45,-2),(75,2),(180,2),(210,-2),(225,2),(119,-1),(120,-1),(135,-1),(136,-1)]:D[m,2]=s
    return D


def arc_projection(a):
    return np.array([.5-a*a/4,-a/np.sqrt(2),(2+5*a*a)/(4*np.sqrt(7))])


def simplex_projector_identity(model):
    """Exact integer coefficient identity for mean P_-(H(a,n))|hop+>.

    H=a Hhop+b sum signs(-QA+QB), b²=(1-a²)/7;8 Hadamard rows.
    Polynomial products use Gaussian integers of magnitude<2^53, making
    every real/imaginary multiplication and sum exact in binary64.
    """
    ids=model.even_masks;hop=2*model.h_hop[np.ix_(ids,ids)]
    signs=hadamard(8)[:,1:]
    masses=np.array([2*(model.q_ops[1,j]-model.q_ops[0,j])[np.ix_(ids,ids)] for j in range(7)])
    W=exact_schmidt_polar()[ids];total={};max_integer=0.
    for sign in signs:
        mass=np.einsum('j,jab->ab',sign,masses)
        assert np.array_equal(mass.real,np.rint(mass.real)) and np.array_equal(mass.imag,np.rint(mass.imag))
        current={(0,0):W}
        for degree in range(1,5):
            following={}
            for (pa,pb),v in current.items():
                for key,acted in [((pa+1,pb),hop@v),((pa,pb+1),mass@v)]:
                    following[key]=following.get(key,np.zeros(128,complex))+acted
            current=following
            multiplier={1:16,2:-4,3:-4,4:1}[degree]*2**(4-degree)
            for key,v in current.items():
                assert np.array_equal(v.real,np.rint(v.real)) and np.array_equal(v.imag,np.rint(v.imag))
                max_integer=max(max_integer,float(np.max(abs(v.real))),float(np.max(abs(v.imag))))
                total[key]=total.get(key,np.zeros(128,complex))+multiplier*v
    reduced={i:np.zeros(128,complex) for i in range(5)}
    for (pa,pb),v in total.items():
        if pb%2:
            assert not v.any();continue
        m=pb//2
        for j in range(m+1):reduced[pa+2*j]+=v*(49//7**m)*comb(m,j)*(-1)**j
    D=scaled_basis()[ids]
    target=[14*D[:,0]+2*D[:,2],-14*D[:,1],-7*D[:,0]+5*D[:,2]]
    expected={i:np.zeros(128,complex) for i in range(5)}
    for i,v in enumerate(target):
        for j,c in enumerate([1,2,1]):expected[i+j]+=43008*c*v
    assert all(np.array_equal(reduced[k],expected[k]) for k in range(5))
    max_integer=max(max_integer,max(float(np.max(abs(x.real))) for x in reduced.values()),max(float(np.max(abs(x.imag))) for x in reduced.values()))
    assert max_integer<2**45
    return {'exact_polynomial_identity_passed':True,'directions':signs.tolist(),
        'common_integer_denominator':384*4*16*8*49,
        'maximum_integer_component':max_integer,
        'claim':'mean8 P_-(H(a,n))gplus = (1+a)^2/4 * S arc_projection(a); identity after b²=(1-a²)/7',
        'actual_8_term_average_is_singlet':True}


def actual_arc_atoms(model,a):
    ids=model.even_masks;hop=model.h_hop[np.ix_(ids,ids)]
    gplus=exact_schmidt_polar()[ids]/4
    masses=np.array([(model.q_ops[1,j]-model.q_ops[0,j])[np.ix_(ids,ids)] for j in range(7)])
    states=[]
    for sign in hadamard(8)[:,1:]:
        H=a*hop+np.sqrt((1-a*a)/7)*np.einsum('j,jab->ab',sign,masses)
        # P_-(H)=(H+2)H(H-2)(H-4)/384 in the even sector.
        v=gplus.copy()
        for shift in [-4,-2,0,2]:v=H@v+shift*v
        states.append(v/(96*(1+a)**2))
    return np.array(states)


def endpoint_secondary_atoms(model,a,positive,negative):
    main=actual_arc_atoms(model,a)
    D=scaled_basis()[model.even_masks];minus=(D[:,0]/2+D[:,1]+D[:,2]/2)/4
    atoms=np.concatenate([main,np.repeat(minus[None,:],8,axis=0)])
    coefficients=np.r_[np.full(8,positive/8),np.full(8,-negative/8)]
    return atoms,coefficients


def endpoint_arc_dual(a):
    """Chebyshev dual for this signed arc family; not globally certified."""
    A=1-2*a*a/(1+a)**2;B=4*a/(1+a)**2;C=-2/(1+a)**2
    return np.array([(5*A-2*C)/3,-np.sqrt(2)*B,np.sqrt(7)*(A+2*C)/3])


def two_orbit_parameters(t,energy_interval):
    iv.dps=90;lo,hi=ground_root_enclosure(t,energy_interval)
    E=iv.mpf([interval_fraction(lo).a,interval_fraction(hi).b]);T=iv.mpf(float(t))
    r1=(E+56)/(iv.sqrt(2)*T);r2=iv.sqrt(14)*T*r1/(E+8)
    z0=1/iv.sqrt(1+r1*r1+r2*r2);z1=r1*z0;z2=r2*z0
    M0=(5*z0+iv.sqrt(7)*z2)/3;M1=-iv.sqrt(2)*z1;M2=2*M0-4*z0
    d=M0-M2;cost=(d+iv.sqrt(d*d+4*M1*M1))/2
    a=M1/cost;negative=(cost-M0)/(1+a*a);positive=cost-negative
    assert _bounds(a)[0]>0 and _bounds(a)[1]<1
    assert _bounds(negative)[0]>0 and _bounds(positive)[0]>0
    return {'t':float(t),'ground_energy_rational_bracket':[[lo.numerator,lo.denominator],[hi.numerator,hi.denominator]],
        'physical_singlet_coordinates_intervals':[_bounds(z) for z in [z0,z1,z2]],
        'main_arc_parameter_interval':_bounds(a),'main_eta_interval':_bounds(a*a),
        'secondary_rotation_cosine_interval':_bounds(-a),'secondary_eta_exact':1,
        'positive_orbit_mass_interval':_bounds(positive),'negative_orbit_mass_interval':_bounds(negative),
        'exact_construction_cost_interval':_bounds(cost),'exact_actual_terms':16,'distinct_Gaussian_rays':16,
        'secondary_orbit_is_single_Gaussian_repeated_eight_times':False,'no_Fock_repair':True,
        'family':'Eight mass/hopping atoms plus eight local-Clifford-rotated maximally-entangled atoms; negative second-orbit coefficient.',
        'exact_target_definition':'Unique full ground state with z0>0; all parameters are algebraic functions of its isolated reduced cubic root.',
        'interval_decimal_digits':90},float(a.mid),float(positive.mid),float(negative.mid)


def rotation_simplex_identity(model):
    ids=model.even_masks;W=exact_schmidt_polar()[ids];D=scaled_basis()[ids]
    Q=np.array([2*model.q_ops[0,j][np.ix_(ids,ids)] for j in range(7)])
    moments={j:np.zeros(128,complex) for j in range(1,5)};largest=0.
    for signs in hadamard(8)[:,1:]:
        N=np.einsum('j,jab->ab',signs,Q);v=W
        for j in range(1,5):
            v=N@v;moments[j]+=v
            assert np.array_equal(v.real,np.rint(v.real)) and np.array_equal(v.imag,np.rint(v.imag))
            largest=max(largest,float(np.max(abs(v.real))),float(np.max(abs(v.imag))))
    assert not moments[1].any() and not moments[3].any()
    A,B=moments[2],moments[4]
    actual=[37632*W-196*A+B,224*A-2*B,-28*A+B]
    expected=[2688*(-7*D[:,0]+5*D[:,2]),2688*(-14*D[:,1]),2688*(14*D[:,0]+2*D[:,2])]
    assert all(np.array_equal(x,y) for x,y in zip(actual,expected))
    assert largest<2**45
    return {'exact_polynomial_identity_passed':True,'common_integer_denominator':150528,
        'maximum_integer_component':largest,
        'claim':'mean8 exp(i theta Q_A(n))gplus=S((2u²-1)/4,-u/sqrt2,(5+2u²)/(4sqrt7)), u=cos(theta)',
        'actual_8_term_average_is_singlet':True}


def actual_rotation_atoms(model,u):
    ids=model.even_masks;gplus=exact_schmidt_polar()[ids]/4
    Q=np.array([model.q_ops[0,j][np.ix_(ids,ids)] for j in range(7)])
    v=np.sqrt(1-u*u)
    coefficients=[1,1j*v*(4-u)/3,(u-1)*(7-u)/6,1j*v*(u-1)/3,(u-1)**2/6]
    states=[]
    for signs in hadamard(8)[:,1:]:
        q=np.einsum('j,jab->ab',signs,Q)/np.sqrt(7);acted=gplus.copy();state=acted.copy()
        for j in range(1,5):acted=q@acted;state+=coefficients[j]*acted
        states.append(state)
    return np.array(states)


def two_orbit_atoms(model,a,positive,negative):
    atoms=np.concatenate([actual_arc_atoms(model,a),actual_rotation_atoms(model,-a)])
    return atoms,np.r_[np.full(8,positive/8),np.full(8,-negative/8)]


def arc_dual(a):
    alpha=1/(1+a*a);beta=2*a/(1+a*a)
    return np.array([7*alpha/3,-np.sqrt(2)*beta,-np.sqrt(7)*alpha/3])
