"""Post-freeze global certificates for arbitrary complex even-sector witnesses.

Discovery does not import this model-specific certificate. All coordinates of
the input binary64 ket, including its component outside the singlet space,
are retained in the mathematical witness being certified.
"""
import numpy as np
from mpmath import iv
from . import paths
from fk.model import build_model
from fk5.two_orbit import scaled_basis, ground_root_enclosure, interval_fraction
from fk5.window_sharp_bounds import exact_hopping_coordinates
from fk4.certificates import _bounds


def _complex(z):
    return iv.mpc(iv.mpf(float(z.real)),iv.mpf(float(z.imag)))


def _square_abs(z):
    return z.real**2+z.imag**2


def _conjugate(z):
    # mpmath's interval mpc.conjugate currently calls the scalar mpf helper.
    return iv.mpc(z.real,-z.imag)


def certify(witness,t,delta,budget,spectrum,steps=128):
    iv.dps=90
    ell=np.asarray(witness,complex)
    if ell.shape!=(128,) or not np.isfinite(ell).all():raise ValueError('Expected finite even-sector ket')
    if t<9 or delta<0 or budget<=0 or steps<1:raise ValueError('Outside certificate domain')
    model=build_model();D=scaled_basis()[model.even_masks]
    exact_ell=[_complex(v) for v in ell]
    norm2=sum(_square_abs(v) for v in exact_ell);norm=iv.sqrt(norm2)
    if _bounds(norm)[0]<=0:raise ValueError('Zero witness')
    z=[sum(_conjugate(_complex(D[i,j]))*exact_ell[i] for i in range(128)) /
       iv.sqrt(int(np.sum(abs(D[:,j])**2))) for j in range(3)]
    residual2=norm2-sum(_square_abs(v) for v in z)
    residual_upper=_bounds(iv.sqrt(iv.mpf(max(0.,_bounds(residual2)[1]))))[1]
    w=[z[0]/4-z[1]/iv.sqrt(2)+iv.sqrt(7)*z[2]/4,
       iv.sqrt(14)*z[0]/4-iv.sqrt(2)*z[2]/4,
       z[0]/4+z[1]/iv.sqrt(2)+iv.sqrt(7)*z[2]/4]
    q=exact_hopping_coordinates(t,spectrum['ground_energy_interval'])
    overlap=abs(sum(_conjugate(v)*a for v,a in zip(w,q)))
    lo,hi=ground_root_enclosure(t,spectrum['ground_energy_interval'])
    energy=iv.mpf([interval_fraction(lo).a,interval_fraction(hi).b])
    a=4*(iv.mpf(float(t))-9);Dlt=iv.mpf(float(delta))
    X=2*Dlt/(iv.sqrt(a*a+72*Dlt)+a) if delta else iv.mpf(0)
    Xupper=min(2.,_bounds(X)[1]);norm_lower,norm_upper=_bounds(norm)
    def h4(x):
        candidates=[1.,_bounds(iv.sqrt(iv.mpf(3)/2)*x)[1]]
        if _bounds(x)[1]<.5:
            candidates.append(_bounds(3*x/(iv.sqrt(14)*iv.sqrt(1-2*x)))[1])
        return iv.mpf(min(candidates))
    cells=[]
    for j in range(steps):
        left=iv.mpf(Xupper)*j/steps;right=iv.mpf(Xupper)*(j+1)/steps
        bound=abs(w[0])*(1-left/2)**2+abs(w[1])*h4(right)+abs(w[2])*right**2/4+iv.mpf(residual_upper)
        cells.append(min(norm_upper,_bounds(bound)[1]))
    support=max(cells)
    # Convert to a unit witness with bounds valid for the original binary64 ket.
    q_lower=max(0.,_bounds(overlap/iv.mpf(norm_upper))[0])
    h_upper=min(1.,_bounds(iv.mpf(support)/iv.mpf(norm_lower))[1])
    masses={}
    for epsilon in [0.,.01]:
        value=0. if h_upper>=1 else max(0.,_bounds(((iv.mpf(q_lower)-iv.mpf(epsilon))/iv.mpf(budget)-iv.mpf(h_upper))/(1-iv.mpf(h_upper)))[0])
        masses[f'{epsilon:g}']=value
    distance=max(0.,_bounds(iv.mpf(q_lower)-iv.mpf(budget)*iv.mpf(h_upper))[0])
    cap=iv.mpf(budget)*iv.mpf(h_upper)
    fidelity=1.
    if _bounds(cap)[1]<q_lower and q_lower<1:
        Q=iv.mpf(q_lower)
        fidelity=min(1.,_bounds((Q*cap+iv.sqrt(1-Q*Q)*iv.sqrt(1-cap*cap))**2)[1])
    return dict(t=float(t),delta=float(delta),coefficient_budget=float(budget),
        witness_binary64_real_imag=np.column_stack((ell.real,ell.imag)).tolist(),
        interval_decimal_digits=90,witness_norm_interval=_bounds(norm),
        outside_singlet_norm_upper=residual_upper,physical_overlap_interval=_bounds(overlap),
        hopping_coordinate_real_intervals=[_bounds(v.real) for v in w],
        hopping_coordinate_imag_intervals=[_bounds(v.imag) for v in w],
        ground_energy_interval=_bounds(energy),X_interval=_bounds(X),X_cover_upper=Xupper,
        partition_steps=steps,partition_support_uppers=cells,
        restricted_support_upper=support,unit_witness_overlap_lower=q_lower,
        unit_witness_restricted_support_upper=h_upper,high_plain_coefficient_mass_lower=masses,
        low_only_hilbert_error_lower=distance,low_only_normalized_fidelity_upper=fidelity,
        strict_global_certificate=True,sampled_maximum_used_as_upper=False,
        full_complex_witness_certified=True,odd_gaussian_overlap_zero=True,
        proof='WEIGHTED_SYNTHESIS.md: Post-freeze restricted support for arbitrary witnesses')
