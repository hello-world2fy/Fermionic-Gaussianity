"""Independent full-space reconstruction and a priori statistical budgets."""
import math
import numpy as np
from .dictionaries import _log_dimension


def log_frame_real(name, s):
    if s < 0:
        raise ValueError('moment order must be nonnegative')
    return _log_dimension(8,s) if name == 'full' else math.log(2)+2*_log_dimension(4,s)


def gaussian_probe_reference(dictionary, k):
    ld = dictionary.log_frame_constant(k)
    return {
        'q':1., 'mass':math.exp(ld-log_frame_real(dictionary.name,k-.5)),
        'q_second_moment':math.exp(2*ld-dictionary.log_frame_constant(2*k)),
        'coefficient_second_moment':math.exp(2*ld-dictionary.log_frame_constant(2*k-1)),
    }


def formal_budget(dictionary,k,u_norm,delta,radius,sample_cap):
    log_v = (2*dictionary.log_frame_constant(k)
             -dictionary.log_frame_constant(2*k-1)+(4*k-2)*math.log(u_norm))
    v = math.exp(log_v)
    required = math.ceil(v/(delta*radius**2))
    return {'second_moment_bound':v,'failure_probability_per_event':delta,
            'statistical_radius_target':radius,'required_samples':required,
            'sample_cap':sample_cap,'status':'ready' if required <= sample_cap else 'validation_budget_exceeded'}


class ReconstructionAccumulator:
    def __init__(self, dictionary, witnesses, orders):
        self.dictionary = dictionary
        self.u = np.asarray(witnesses,complex)
        self.orders = np.asarray(orders,int)
        self.d = np.array([math.exp(dictionary.log_frame_constant(int(k))) for k in orders])
        m = len(orders)
        self.count = 0
        self.csum = np.zeros((m,128),complex)
        self.qsum = np.zeros(m)
        self.bsum = np.zeros(m)
        self.bsqsum = np.zeros(m)
        self.max_b = np.zeros(m)

    def snapshot(self):
        return {'count':np.array(self.count),'witnesses':self.u,'orders':self.orders,
                'c_sum':self.csum,'q_sum':self.qsum,'mass_sum':self.bsum,
                'mass_squared_sum':self.bsqsum,'max_mass_weight':self.max_b}

    def add(self, kets):
        overlap = np.einsum('ni,mi->nm', kets.conj(),self.u,optimize=False)
        absolute = np.abs(overlap)
        coeff = self.d[None,:]*absolute**(2*self.orders[None,:]-2)*overlap
        mass = np.abs(coeff)
        self.csum += np.einsum('nm,ni->mi',coeff,kets,optimize=False)
        self.qsum += np.sum(mass*absolute,axis=0)
        self.bsum += np.sum(mass,axis=0)
        self.bsqsum += np.sum(mass**2,axis=0)
        self.max_b = np.maximum(self.max_b,np.max(mass,axis=0))
        self.count += len(kets)
        return coeff

    def result(self,index,target,S):
        c = self.csum[index]/self.count
        diff = c-target
        projected = np.einsum('ij,i->j',S.conj(),diff,optimize=False)
        outside = diff-np.einsum('ij,j->i',S,projected,optimize=False)
        second = self.bsqsum[index]/self.count
        mass = self.bsum[index]/self.count
        variance = max(0.,second-mass**2)
        rmse = math.sqrt(max(0.,second-float(np.vdot(c,c).real))/self.count)
        return {
            'reconstruction':c,'q_validation':self.qsum[index]/self.count,
            'mass_validation':mass,'mass_standard_error':math.sqrt(variance/max(1,self.count-1)),
            'mass_ess':self.bsum[index]**2/self.bsqsum[index] if second else 0.,
            'max_weight_share':self.max_b[index]/self.bsum[index] if mass else 0.,
            'validation_samples':self.count,
            'sampled_full_residual':float(np.linalg.norm(diff)),
            'validation_singlet_residual':float(np.linalg.norm(projected)),
            'validation_outside_singlet_norm':float(np.linalg.norm(outside)),
            'residual_mc_rmse_estimate':rmse,
        }
