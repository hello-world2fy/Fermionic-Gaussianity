"""Joint kernels of the original collective pair annihilators.

Darkness is defined separately within each band excitation sector. It does
not imply invariance under the full Hamiltonian or a long-lived dark phase.
"""
from math import comb
import numpy as np
from .pairing_channels import annihilator


class PairDarkSectors:
    def __init__(self, model, forms, tolerance=1e-11):
        self.model = model
        self.forms = np.asarray(forms, complex)
        self.bases = []
        self.projectors = []
        self.diagnostics = []
        for m in range(model.r + 1):
            size = comb(model.r, m)
            if m == 0:
                # Exclude the empty band from the excited dark population.
                basis = np.empty((1, 0), complex)
                singular = np.array([])
                residual = 0.
            else:
                operators = [annihilator(model.r, m, f / np.sqrt(model.r)).toarray()
                             for f in self.forms.T]
                stack = np.concatenate(operators, axis=0)
                _, singular, vh = np.linalg.svd(stack, full_matrices=True)
                rank = int(np.sum(singular > tolerance * max(singular[0], 1.)))
                basis = vh[rank:].conj().T
                residual = float(np.linalg.norm(stack @ basis))
            projector = basis @ basis.conj().T
            self.bases.append(basis)
            self.projectors.append(projector)
            self.diagnostics.append({'m': m, 'band_dimension': size,
                'dark_dimension': basis.shape[1], 'annihilator_residual': residual,
                'nonzero_singular_min': float(singular[singular > tolerance].min())
                    if np.any(singular > tolerance) else None,
                'projector_error': float(np.linalg.norm(projector @ projector - projector))})

    def apply(self, state, kind='both'):
        model = self.model
        source = np.asarray(state, complex)
        if source.shape[0] != model.dim:
            raise ValueError('First dimension must be the physical pair-sector dimension')
        result = np.zeros_like(source)
        for m, (_, indices) in enumerate(model.blocks):
            D = self.projectors[m]
            C = source[indices]
            left = np.einsum('ab,bc...->ac...', D, C)
            right = np.einsum('ab...,cb->ac...', C, D)
            both = np.einsum('ab...,cb->ac...', left, D)
            if kind == 'both': value = both
            elif kind == 'either': value = left + right - both
            elif kind == 'particle': value = left
            elif kind == 'hole': value = right
            else: raise ValueError(kind)
            result[indices] = value
        return result

    def evaluate(self, state):
        model = self.model
        result = {'sector_probability': [], 'particle_dark': [], 'hole_dark': [],
                  'both_dark': [], 'either_dark': []}
        for m, (_, indices) in enumerate(model.blocks):
            C = state[indices]
            D = self.projectors[m]
            left = D @ C
            right = C @ D.T
            both = left @ D.T
            p = float(np.linalg.norm(C)**2)
            lp = float(np.linalg.norm(left)**2)
            rp = float(np.linalg.norm(right)**2)
            bp = float(np.linalg.norm(both)**2)
            for key, value in zip(result, (p, lp, rp, bp, lp + rp - bp)):
                result[key].append(value)
        result['total_both_dark'] = sum(result['both_dark'])
        result['total_either_dark'] = sum(result['either_dark'])
        result['multiple_transfer_both_dark'] = sum(result['both_dark'][2:])
        result['multiple_transfer_either_dark'] = sum(result['either_dark'][2:])
        return result
