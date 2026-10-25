"""Coherent full-Gaussian symbols of disjoint pair-transfer monomials.

All even two-copy sectors are retained. Dense matrices still scale
exponentially with L. This is k2 synthesis, not a higher-replica algorithm.
"""
from functools import lru_cache
from itertools import combinations
from math import comb, prod
import numpy as np
from scipy.special import beta
from .pair_symbols import PairSymbol2, masks_of, layout


def odd_factorial(n):
    return prod(range(1, n + 1, 2))


def even_splits(length):
    for size in range(0, length + 1, 2):
        for first in combinations(range(length), size):
            first_set = set(first)
            second = tuple(i for i in range(length) if i not in first_set)
            sign = (-1) ** sum(i > j for i in first for j in second)
            yield first, second, sign * odd_factorial(size - 1) * odd_factorial(length - size - 1)


def physical_masks(pair_masks, L):
    out = np.zeros_like(pair_masks)
    for j in range(L):
        out |= ((pair_masks >> j) & 1) * (3 << (2*j))
    return out


def act_fermion_word(masks, word):
    """Action of an ordered word (mode, creation) on occupation masks."""
    current = np.array(masks, dtype=np.int64, copy=True)
    amplitude = np.ones(len(current))
    for mode, creation in reversed(word):
        occupied = (current >> mode) & 1
        amplitude *= (occupied == (0 if creation else 1))
        amplitude *= (-1.) ** np.bitwise_count(current & ((1 << mode) - 1))
        current ^= 1 << mode
    selected = np.flatnonzero(amplitude)
    return selected, current[selected], amplitude[selected]


def pair_word(creations, annihilations):
    if len(set(creations + annihilations)) != len(creations + annihilations):
        raise ValueError('Every level must be distinct')
    if len(creations) != len(annihilations):
        raise ValueError('Number-conserving pair transfers required')
    word = []
    for j in creations:
        word.extend(((2*j, True), (2*j+1, True)))
    for j in annihilations:
        word.extend(((2*j+1, False), (2*j, False)))
    return tuple(word)


class MultiPairSymbol2(PairSymbol2):
    def prepare(self, u):
        super().prepare(u)
        self._sector_cache = {}
        self._physical = physical_masks(self.model.masks, self.L)

    def sector_matrix(self, broken, left_number):
        """First copy has up singles on broken levels, second has down.

        Reversing the spin on one broken level multiplies the matrix by -1.
        Intact-pair bases exclude every broken level.
        """
        broken = tuple(sorted(broken)); b = len(broken)
        right_number = 2*self.r - b - left_number
        if b % 2:
            raise ValueError('Only even-Fock intermediate sectors')
        if left_number > right_number:
            left, right, matrix = self.sector_matrix(broken, right_number)
            return right, left, matrix.T
        key = (broken, left_number)
        if key in self._sector_cache:
            return self._sector_cache[key]
        if not broken and left_number == self.r:
            return self.model.masks, self.model.masks, self.S
        if not broken and left_number == self.r-1:
            return self.lower, self.upper, self.shifted
        left = masks_of(self.L, left_number, excluded=broken)
        right = masks_of(self.L, right_number, excluded=broken)
        active = sum(1 << j for j in broken)
        a, c, ma, keys, size = layout(left, right, self.L)
        keys |= active << self.L
        size += b
        torch = self.torch
        v = self.tensor(self.u)
        matrix = torch.zeros((len(left), len(right)), device=self.device, dtype=torch.complex128)
        for source in self.source:
            s = source['s']
            number = left_number - self.r + s//2
            if s < b or not 0 <= number <= s-b:
                continue
            chosen = np.flatnonzero(size == s)
            if not len(chosen):
                continue
            chosen = chosen[np.lexsort((ma[chosen], keys[chosen]))]
            width = comb(s-b, number)
            aa = a[chosen].reshape(-1, width)
            cc = c[chosen].reshape(-1, width)
            unique = keys[chosen].reshape(-1, width)[:, 0]
            rows = np.searchsorted(source['keys'], unique)
            assert np.array_equal(source['keys'][rows], unique)
            difference = unique >> self.L
            patterns = np.zeros(len(unique), dtype=np.int64)
            for j in broken:
                patterns |= 1 << np.bitwise_count(difference & ((1 << j)-1)).astype(np.int64)
            inputs = masks_of(s, s//2)
            for pattern in np.unique(patterns):
                selected = np.flatnonzero(patterns == pattern)
                exclusions = tuple(j for j in range(s) if pattern & (1 << j))
                outputs = masks_of(s, number, excluded=exclusions)
                outside = ((1 << s)-1) ^ int(pattern)
                distance = np.bitwise_count((outputs[:, None] ^ inputs[None, :]) & outside) + b//2
                sign = (-1.) ** (b//2 + np.bitwise_count(inputs & pattern))
                K = beta(distance+.5, s-distance+.5)/np.pi * sign[None, :]
                ia = self.tensor(source['a'][rows[selected]].astype(np.int64))
                ib = self.tensor(source['b'][rows[selected]].astype(np.int64))
                oa = self.tensor(aa[selected].astype(np.int64))
                ob = self.tensor(cc[selected].astype(np.int64))
                matrix[oa, ob] = (v[ia]*v[ib]) @ self.tensor(K).to(torch.complex128).T
        result = (left, right, matrix.cpu().numpy())
        self._sector_cache[key] = result
        return result

    @lru_cache(maxsize=2048)
    def inverse_word(self, word):
        dagger = tuple((j, not create) for j, create in reversed(word))
        selected, masks, amplitude = act_fermion_word(self._physical, dagger)
        paired = np.zeros(len(masks), dtype=np.int64)
        first = int(masks[0]) if len(masks) else 0
        broken = tuple(j for j in range(self.L) if ((first >> (2*j)) & 3) in (1, 2))
        down = sum(((first >> (2*j)) & 3) == 2 for j in broken)
        for j in range(self.L):
            paired |= (((masks >> (2*j)) & 3) == 3).astype(np.int64) << j
        number = int(paired[0]).bit_count() if len(paired) else 0
        return selected, paired, amplitude, broken, down, number

    def monomial_symbol(self, creations, annihilations, return_terms=False):
        word = pair_word(tuple(creations), tuple(annihilations))
        m = len(word)//2
        if not m or m > 4:
            raise ValueError('Supports one or two disjoint pair transfers')
        result = np.zeros(self.model.dim, complex)
        sectors = {}
        for first, second, coefficient in even_splits(len(word)):
            A = tuple(word[j] for j in first); B = tuple(word[j] for j in second)
            left_indices, left_pairs, left_sign, broken, down, number = self.inverse_word(A)
            right_indices, right_pairs, right_sign, other_broken, _, _ = self.inverse_word(B)
            if not len(left_indices) or not len(right_indices):
                continue
            assert broken == other_broken
            left, right, matrix = self.sector_matrix(broken, number)
            lo = np.argsort(left); ro = np.argsort(right)
            li = lo[np.searchsorted(left[lo], left_pairs)]
            ri = ro[np.searchsorted(right[ro], right_pairs)]
            assert np.array_equal(left[li], left_pairs) and np.array_equal(right[ri], right_pairs)
            contribution = coefficient * (-1)**down * left_sign * (
                matrix[np.ix_(li, ri)] @ (right_sign*self.u[right_indices]).conj())
            result[left_indices] += contribution
            if return_terms:
                key = (len(first), len(broken))
                if key not in sectors:
                    sectors[key] = np.zeros(self.model.dim, complex)
                sectors[key][left_indices] += contribution
        denominator = prod(2*self.n - 2*j for j in range(m))
        if return_terms:
            return result/denominator, {str(k): v/denominator for k, v in sectors.items()}
        return result/denominator


def full_majorana_symbol(cartan, u, indices, projected=None):
    """Small-Fock exterior-symbol reference from the even-subset coproduct."""
    m = len(indices)//2
    if len(indices) % 2 or len(set(indices)) != len(indices):
        raise ValueError('Distinct even-length Majorana word required')
    S = cartan.project(np.outer(u, u)) if projected is None else projected
    result = np.zeros_like(u, dtype=complex)
    for first, second, coefficient in even_splits(len(indices)):
        X = cartan.majorana_word(tuple(indices[i] for i in first))
        Y = cartan.majorana_word(tuple(indices[i] for i in second))
        result += coefficient * (X @ (S @ (Y.conj().T @ u).conj()))
    return (1j**m) * result / prod(2*cartan.n - 2*j for j in range(m))
