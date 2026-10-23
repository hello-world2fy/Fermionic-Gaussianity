"""Coherent invariant bins; interpretation is gated by full reconstruction."""
import numpy as np


class CoherentBins:
    def __init__(self, bins=4):
        self.bins = bins
        self.phi_sum = np.zeros((bins*bins,128),complex)
        self.mass_sum = np.zeros(bins*bins)
        self.count = 0

    def add(self,kets,coeff,invariants):
        x = np.minimum((np.clip(invariants['eta'],0,1)*self.bins).astype(int),self.bins-1)
        y = np.minimum((np.clip(invariants['mass_a'],0,1)*self.bins).astype(int),self.bins-1)
        idx = y*self.bins+x
        np.add.at(self.phi_sum,idx,coeff[:,None]*kets)
        self.mass_sum += np.bincount(idx,weights=np.abs(coeff),minlength=self.bins*self.bins)
        self.count += len(kets)

    def result(self,model,target):
        phi = self.phi_sum/self.count
        operator = model.mass_correlation_operator[np.ix_(model.even_masks,model.even_masks)]
        kernel = np.einsum('bi,ij,cj->bc',phi.conj(),operator,phi,optimize=True)
        total = np.sum(phi,axis=0)
        norm2 = float(np.vdot(total,total).real)
        return {'histogram':self.mass_sum.reshape(self.bins,self.bins)/np.sum(self.mass_sum),
                'coherent_vectors':phi,'correlation_kernel':kernel,
                'closure_residual':float(np.linalg.norm(total-target)),
                'reconstructed_norm2':norm2,
                'mass_correlation_unnormalized':float(np.sum(kernel).real),
                'diagonal_contribution':float(np.trace(kernel).real),
                'off_diagonal_contribution':float((np.sum(kernel)-np.trace(kernel)).real)}
