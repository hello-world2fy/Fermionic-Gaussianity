"""Conservative exact-Gaussian repair of near-Gaussian numerical branches."""
import numpy as np
from mpmath import iv
from .certificates import _bounds
from .distributions import _features
from .tempering import actions


def corrected_cost(kets,coefficients,target,clifford,physical_target_distance=0.):
    g=np.asarray(kets,complex);c=np.asarray(coefficients,complex);target=np.asarray(target,complex)
    assert g.ndim==2 and g.shape[1]==128 and c.shape==(len(g),) and target.shape==(128,)
    assert np.isfinite(g).all() and np.isfinite(c).all() and np.isfinite(target).all()
    assert np.max(abs(c))<1e12 and len(g)<10000000
    norm2=np.sum(g.real*g.real+g.imag*g.imag,axis=1)
    assert norm2.min()>.9 and norm2.max()<1.1
    permutations,phases=actions(8)
    _,purity=_features(np.ascontiguousarray(g),permutations,phases,np.asarray(clifford,float))
    residual=target-c@g
    residual_l1=float(np.sum(abs(residual)))
    coefficient_l1=float(np.sum(abs(c)))
    target_l1=float(np.sum(abs(target)))
    iv.dps=70;u=iv.mpf(2)**-52;tiny=iv.mpf(2)**-1000
    def gamma(m):return m*u/(1-m*u)
    qlo=iv.mpf(float(norm2.min()))/(1+gamma(1024))-tiny
    qhi=iv.mpf(float(norm2.max()))/(1-gamma(1024))+tiny
    cov_error=32*gamma(1024)*qhi+tiny
    Cmax=qhi+cov_error
    reciprocal_error=max(_bounds(abs(1/qlo-1))[1],_bounds(abs(1/qhi-1))[1])
    normalized_cov_error=cov_error/qlo+Cmax*iv.mpf(reciprocal_error)+tiny
    product_error=32*gamma(64)*(16*Cmax*Cmax+1)+tiny
    defect=16*(iv.mpf(float(purity.max()))+product_error+32*Cmax*normalized_cov_error+16*normalized_cov_error**2)
    if _bounds(defect)[1]>=.4:raise ArithmeticError('near-Gaussian repair bound too large')
    distance=iv.sqrt(8*(1-iv.sqrt(1-defect)))
    mass0=iv.mpf(coefficient_l1)/(1-gamma(2*len(c)+32))+tiny
    mass=mass0*iv.sqrt(qhi)
    summed_atom_l1=iv.sqrt(128)*mass
    contraction_error=8*gamma(8*len(c)+32)*summed_atom_l1+tiny
    subtraction_error=8*gamma(8*128+32)*(iv.mpf(target_l1)/(1-gamma(8*128+32))+summed_atom_l1)+tiny
    residual_upper=iv.mpf(residual_l1)/(1-gamma(8*128+32))+contraction_error+subtraction_error
    gaussian_repair=iv.sqrt(128)*distance*mass
    physical_repair=iv.sqrt(128)*iv.mpf(float(physical_target_distance))
    upper=mass+residual_upper+gaussian_repair+physical_repair
    return {'corrected_Gaussian_cost_upper':_bounds(upper)[1],
        'main_coefficient_mass_upper':_bounds(mass)[1],
        'saved_target_residual_l1_upper':_bounds(residual_upper)[1],
        'Gaussian_atom_repair_cost_upper':_bounds(gaussian_repair)[1],
        'physical_target_repair_cost_upper':_bounds(physical_repair)[1],
        'normalized_exact_covariance_defect_upper':_bounds(defect)[1],
        'per_atom_exact_Gaussian_distance_upper':_bounds(distance)[1],
        'main_actual_branch_count':len(g),'fock_repair_branch_count_upper':128,
        'raw_measured_max_purity_defect':float(purity.max()),
        'raw_measured_norm2_range':[float(norm2.min()),float(norm2.max())],
        'raw_measured_residual_l1':residual_l1,
        'raw_measured_coefficient_l1':coefficient_l1,
        'interval_decimal_digits':70,
        'exact_atoms_definition':'Gaussian quasiparticle vacua of exact normalized covariances of saved binary64 kets, phase-aligned with those kets; add exact Fock residual.',
        'original_numeric_kets_not_claimed_exactly_Gaussian':True,
        'arithmetic_assumption':'Standard IEEE-754 dot-product error bounds; u=2^-52, no overflow; absolute subnormal slack 2^-1000.'}
