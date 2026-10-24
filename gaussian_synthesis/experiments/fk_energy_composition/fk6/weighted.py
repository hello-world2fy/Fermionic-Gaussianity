"""Weighted quadrature objective, without any global-support certification claim."""
import numpy as np
from . import paths
from fk4.weighted import WeightedMoments


class EnergyWeightedMoments(WeightedMoments):
    def __init__(self, overlaps, log_measure_weights, energies, penalty, k, log_dimension):
        self.energies = np.asarray(energies,float)
        if penalty<0 or np.any(self.energies<0) or not np.isfinite(self.energies).all():
            raise ValueError('Nonnegative finite excess energies and penalty required')
        self.v = 1+float(penalty)*self.energies
        if self.v.shape != (len(overlaps),):
            raise ValueError('One excess energy per atom required')
        self.original_overlaps = np.asarray(overlaps,complex)
        super().__init__(self.original_overlaps/self.v[:,None],log_measure_weights,k,log_dimension)

    def coefficient_costs(self,z):
        from scipy.special import logsumexp
        a=self.original_overlaps@z
        logs=self.log_d+self.log_weights+(2*self.k-1)*np.log(np.maximum(abs(a)/self.v,1e-300))
        bv=float(np.exp(logsumexp(logs)))
        weights=np.exp(logs)/self.v
        return dict(weighted_cost=bv,plain_cost=float(weights.sum()),composition_energy=float(weights@self.energies))
