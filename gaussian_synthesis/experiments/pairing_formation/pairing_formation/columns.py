"""Adaptive Gaussian matching pursuit with physical symmetry-orbit atoms.

An atom stored here is P_sector|g>, with |g> normalized in full Fock space.
It represents the coherent average of Gaussian symmetry images, NOT a
Gaussian state normalized after projection. Coefficient l1 mass is therefore
a feasible cost for the represented vector after restoring the symmetry
orbit. No local oracle provides a global Gaussian support certificate.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .projected_gaussian import torch_module, torch_pfaffian


def onsite_atom(model, x):
    a, b = x[:model.L], x[model.L:]
    return np.exp(model.occ@(a+1j*b)-.5*np.logaddexp(0., 2*a).sum())


def onsite_objective(model, witness, x):
    atom = onsite_atom(model, x)
    overlap = np.vdot(witness, atom)
    d = (witness.conj()*atom)@(model.occ-expit(2*x[:model.L]))
    gradient = -2*np.r_[np.real(overlap.conjugate()*d), np.real(1j*overlap.conjugate()*d)]
    return -float(abs(overlap)**2), gradient


def onsite_oracle(model, witness, *, starts=12, seed=1, seeds=(), maxiter=250):
    normalized = witness/np.linalg.norm(witness)
    rng = np.random.default_rng(seed)
    density = abs(normalized)**2@model.occ
    density = np.clip(density, 1e-5, 1-1e-5)
    a = .5*np.log(density/(1-density))
    _, ids = model.blocks[1]
    u, s, vh = np.linalg.svd(normalized[ids], full_matrices=False)
    phase = np.r_[-np.angle(vh[0]), np.angle(u[:, 0])]
    candidates = [np.asarray(x).copy() for x in seeds]
    candidates.extend([np.r_[a, phase], np.r_[a, np.zeros(model.L)]])
    while len(candidates) < starts:
        k = len(candidates)
        candidates.append(np.r_[a+rng.normal(0, .8, model.L),
                                 phase+rng.normal(0, .8 if k % 2 else 2., model.L)])
    best = None
    evaluations = 0
    for x in candidates[:starts]:
        fit = minimize(lambda y: onsite_objective(model, normalized, y), x, jac=True,
                       method='L-BFGS-B', bounds=[(-14., 14.)]*model.L+[(None, None)]*model.L,
                       options={'maxiter': maxiter, 'ftol': 3e-13, 'gtol': 1e-8})
        evaluations += fit.nfev
        if best is None or fit.fun < best.fun:
            best = fit
    atom = onsite_atom(model, best.x)
    return atom, best.x, {'oracle_evaluations': evaluations, 'oracle_starts': starts,
                          'overlap_squared': float(abs(np.vdot(witness, atom))**2),
                          'global_support_certified': False}


def quasiparticle_indices(model):
    groups = []
    for m in range(model.r+1):
        rows = np.flatnonzero(model.m == m)
        ids = []
        for row in rows:
            excitations = np.r_[1-model.occ[row, :model.r], model.occ[row, model.r:]]
            ids.append([a for j in range(model.L) if excitations[j] for a in (2*j, 2*j+1)])
        groups.append((rows, np.array(ids, dtype=int).reshape(len(rows), 4*m)))
    return groups


def onsite_to_qp_Z(model, parameters):
    z = parameters[:model.L]+1j*parameters[model.L:]
    z[:model.r] *= -1
    values = np.exp(z)
    Z = np.zeros((2*model.L, 2*model.L), complex)
    Z[2*np.arange(model.L), 2*np.arange(model.L)+1] = values
    return Z-Z.T


class FullGaussianOracle:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.torch = torch_module()
        torch = self.torch
        torch.set_num_threads(1)
        self.device = device
        self.n = 2*model.L
        self.iu = np.triu_indices(self.n, 1)
        self.aidx = torch.tensor(self.iu[0], device=device)
        self.bidx = torch.tensor(self.iu[1], device=device)
        self.identity = torch.eye(self.n, dtype=torch.complex128, device=device)
        self.groups = [(torch.tensor(rows, device=device), torch.tensor(ids, device=device))
                       for rows, ids in quasiparticle_indices(model)]
        self.calls = 0

    def atom_tensor(self, Z):
        torch = self.torch
        raw = torch.zeros(self.model.dim, device=self.device, dtype=torch.complex128)
        for rows, ids in self.groups:
            values = torch.ones(len(rows), device=self.device, dtype=Z.dtype) if ids.shape[1] == 0 else \
                     torch_pfaffian(Z[ids[:, :, None], ids[:, None, :]], torch)
            raw = raw.index_put((rows,), values)
        _, logdet = torch.linalg.slogdet(self.identity+Z.conj().T@Z)
        return raw*torch.exp(-logdet/4)

    def evaluate(self, parameters, witness, save=False):
        torch = self.torch
        d = len(self.iu[0])
        x = torch.tensor(parameters, dtype=torch.float64, device=self.device, requires_grad=not save)
        values = x[:d]+1j*x[d:]
        Z = torch.zeros_like(self.identity).index_put((self.aidx, self.bidx), values)
        Z = Z-Z.T
        atom = self.atom_tensor(Z)
        if save:
            return atom.detach().cpu().numpy(), Z.detach().cpu().numpy()
        target = torch.as_tensor(witness, dtype=torch.complex128, device=self.device)
        loss = -torch.vdot(target, atom).abs()**2
        loss.backward()
        self.calls += 1
        gradient = x.grad.detach().cpu().numpy()
        return float(loss.detach()), gradient

    def optimize(self, witness, onsite_parameters, *, starts=3, seed=1, maxiter=150):
        target = witness/np.linalg.norm(witness)
        rng = np.random.default_rng(seed)
        base = onsite_to_qp_Z(self.model, onsite_parameters)[self.iu]
        # Near a Fock boundary the reference chart is badly conditioned.
        # Keep the exact onsite candidate and use finite broad charts to probe
        # cross-level directions rather than mistaking enormous coordinates
        # for an exhaustive search around that boundary.
        seed_base = np.minimum(abs(base), 8.)*np.exp(1j*np.angle(base))
        best_atom = onsite_atom(self.model, onsite_parameters)
        best_Z = onsite_to_qp_Z(self.model, onsite_parameters)
        best_value = abs(np.vdot(target, best_atom))**2
        records = []
        before = self.calls
        for k in range(starts):
            noise = (rng.normal(size=len(base))+1j*rng.normal(size=len(base)))/np.sqrt(self.n)
            values = seed_base+(0.03 if k % 2 == 0 else .5)*noise
            result = minimize(lambda x: self.evaluate(x, target), np.r_[values.real, values.imag],
                              jac=True, method='L-BFGS-B',
                              options={'maxiter': maxiter, 'ftol': 3e-12, 'gtol': 2e-8, 'maxcor': 15})
            atom, Z = self.evaluate(result.x, target, save=True)
            value = abs(np.vdot(target, atom))**2
            records.append({'overlap_squared_normalized_witness': float(value),
                            'iterations': int(result.nit), 'success': bool(result.success)})
            if np.isfinite(value) and value > best_value*(1+1e-10)+1e-12:
                best_atom, best_Z, best_value = atom, Z, value
        return best_atom, best_Z, {'full_oracle_evaluations': self.calls-before,
                                  'full_oracle_restarts': records,
                                  'overlap_squared': float(best_value*np.linalg.norm(witness)**2),
                                  'global_support_certified': False}


def fit_coefficients(atoms, target):
    coefficients = np.linalg.lstsq(atoms, target, rcond=1e-11)[0]
    fit = atoms@coefficients
    residual = target-fit
    norm = np.linalg.norm(fit)
    return coefficients, fit, residual, float(norm**2)


def adaptive_columns(model, target, *, count=32, seed=1, device='cpu',
                     full_starts=2, onsite_starts=12, checkpoint=None):
    oracle = FullGaussianOracle(model, device=device) if full_starts else None
    columns, parameters, matrices, history = [], [], [], []
    residual = target.copy()
    start_time = time.monotonic()
    for k in range(count):
        atom, param, onsite_report = onsite_oracle(model, residual, starts=onsite_starts,
                                                  seed=seed+1009*k, seeds=parameters[-2:])
        report = {}
        Z = onsite_to_qp_Z(model, param)
        if oracle:
            atom, Z, report = oracle.optimize(residual, param, starts=full_starts, seed=seed+1009*k)
        # Include all Fock states in the search as exact Gaussian boundary candidates.
        best_fock = int(np.argmax(abs(residual)))
        fock_selected = abs(residual[best_fock]) > abs(np.vdot(residual, atom))*(1+1e-9)
        if fock_selected:
            atom = np.eye(1, model.dim, best_fock, dtype=complex).ravel()
            Z = np.full_like(Z, np.nan)
        columns.append(atom)
        parameters.append(param)
        matrices.append(Z)
        A = np.array(columns).T
        coefficients, fit, residual, weight = fit_coefficients(A, target)
        record = {'columns': k+1, 'fidelity': weight,
                  'residual': float(np.linalg.norm(residual)),
                  'coefficient_mass': float(np.sum(abs(coefficients))),
                  'sector_probability': float(np.vdot(atom, atom).real),
                  'fock_selected': bool(fock_selected),
                  'fock_index': best_fock if fock_selected else None,
                  'elapsed_seconds': time.monotonic()-start_time, **onsite_report, **report}
        history.append(record)
        if checkpoint:
            checkpoint(history, A, np.array(parameters), np.array(matrices), coefficients)
    return history, A, np.array(parameters), np.array(matrices), coefficients
