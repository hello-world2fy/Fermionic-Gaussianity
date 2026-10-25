"""General number-projected Gaussian fits with the FULL number-sector norm.

For an arbitrary antisymmetric n by n matrix Z, the squared norm of its
2r-particle component is e_r(s_1^2,...,s_(n/2)^2), where s occur in equal
Takagi pairs. Only the target overlap uses seniority-zero Pfaffians. The
normalizer includes every broken-pair configuration, avoiding the erroneous
extra seniority projection that would artificially improve fidelity.
"""
import numpy as np
from scipy.optimize import minimize


def torch_module():
    try:
        import torch
    except ImportError:
        import sys
        sys.path.append('/usr/local/lib/python3.11/dist-packages')
        import torch
    return torch


def torch_pfaffian(a, torch):
    answer = torch.ones(a.shape[:-2], device=a.device, dtype=a.dtype)
    for width in range(a.shape[-1], 1, -2):
        pivot_index = a[..., 0, 1:].abs().argmax(dim=-1)+1
        permutation = torch.arange(width, device=a.device).expand(*a.shape[:-2], width).clone()
        permutation[..., 1] = pivot_index
        permutation.scatter_(-1, pivot_index[..., None], 1)
        a = torch.gather(a, -2, permutation[..., :, None].expand_as(a))
        a = torch.gather(a, -1, permutation[..., None, :].expand_as(a))
        pivot = a[..., 0, 1]
        answer = answer*torch.where(pivot_index == 1, 1., -1.)*pivot
        if width > 2:
            first, second = a[..., 0, 2:], a[..., 1, 2:]
            denom = torch.where(pivot.abs() > 1e-100, pivot, torch.ones_like(pivot))
            a = a[..., 2:, 2:]+(second[..., :, None]*first[..., None, :]
                         -first[..., :, None]*second[..., None, :])/denom[..., None, None]
    return answer


def number_norm_squared(Z, pairs, torch):
    eigenvalues = torch.linalg.eigvalsh(Z.conj().T@Z).real
    singular2 = (eigenvalues[::2]+eigenvalues[1::2])*.5
    coeff = [torch.ones((), device=Z.device, dtype=eigenvalues.dtype)]
    coeff.extend(torch.zeros_like(coeff[0]) for _ in range(pairs))
    for value in singular2:
        coeff = [coeff[0]]+[coeff[j]+value*coeff[j-1] for j in range(1, pairs+1)]
    return coeff[pairs]


def fit_general(model, target, onsite_parameters, *, starts=4, seed=2, device='cpu', maxiter=400):
    torch = torch_module()
    torch.set_num_threads(1)
    n = 2*model.L
    iu = np.triu_indices(n, 1)
    aidx = torch.tensor(iu[0], device=device)
    bidx = torch.tensor(iu[1], device=device)
    indices = np.array([[k for j in range(model.L) if mask & (1 << j)
                         for k in (2*j, 2*j+1)] for mask in model.masks])
    indices = torch.tensor(indices, device=device)
    psi = torch.tensor(target, device=device, dtype=torch.complex128)
    d = len(iu[0])
    def evaluate(x, save=False):
        params = torch.tensor(x, device=device, dtype=torch.float64, requires_grad=not save)
        values = params[:d]+1j*params[d:]
        Z = torch.zeros((n, n), device=device, dtype=torch.complex128)
        Z = Z.index_put((aidx, bidx), values)
        Z = Z-Z.T
        # P_N is homogeneous in Z; this scaling improves conditioning.
        Z = Z/torch.linalg.vector_norm(Z)
        raw = torch_pfaffian(Z[indices[:, :, None], indices[:, None, :]], torch)
        norm2 = number_norm_squared(Z, model.r, torch)
        overlap = torch.vdot(psi, raw)
        loss = -(overlap.abs()**2)/norm2
        if save:
            return Z.detach().cpu().numpy(), (raw/norm2.sqrt()).detach().cpu().numpy()
        loss.backward()
        return float(loss.detach()), params.grad.detach().cpu().numpy()
    params = np.asarray(onsite_parameters)
    z = params[:model.L]+1j*params[model.L:]
    physical_z = np.r_[-z[model.r:], z[:model.r]]
    base = np.zeros((n, n), complex)
    base[2*np.arange(model.L), 2*np.arange(model.L)+1] = np.exp(physical_z-physical_z.real.mean())
    base = base-base.T
    base /= np.linalg.norm(base)
    rng = np.random.default_rng(seed)
    results, best = [], None
    for k in range(starts):
        noise = (rng.normal(size=d)+1j*rng.normal(size=d))/np.sqrt(d)
        values = base[iu] + (0 if k == 0 else (.03 if k % 2 else .3))*noise
        x0 = np.r_[values.real, values.imag]
        fit = minimize(evaluate, x0, method='L-BFGS-B', jac=True,
                       options={'ftol': 1e-13, 'gtol': 2e-8, 'maxiter': maxiter, 'maxcor': 20})
        results.append({'fidelity': -float(fit.fun), 'iterations': int(fit.nit),
                        'success': bool(fit.success), 'gradient_norm': float(np.linalg.norm(fit.jac))})
        if best is None or fit.fun < best.fun:
            best = fit
    Z, projection = evaluate(best.x, save=True)
    return {'best_found_fidelity': -float(best.fun), 'starts': results,
            'seniority_zero_probability': float(np.vdot(projection, projection).real),
            'global_optimum_certified': False, 'norm_scope': 'full fixed-physical-number sector'}, Z, projection
