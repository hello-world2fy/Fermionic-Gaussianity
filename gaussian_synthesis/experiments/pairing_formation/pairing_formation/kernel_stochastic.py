"""Fluctuation-seeded spin mean field for a general real symmetric pair kernel.

The exact spin-1/2 onsite identity is used before factorizing different sites.
The Gaussian initialization is SMF-like; discrete initialization is DTWA.
Neither defines a positive full quantum density matrix or a pure-state echo.
"""
import numpy as np
from .stochastic import initial_samples


def numpy_kernel_rhs(state, energies, kernel):
    J = np.asarray(kernel)
    if np.max(abs(J.imag)) > 1e-13:
        raise ValueError('Only real kernels are implemented')
    off = J.real-np.diag(np.diag(J.real))
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    hx, hy = -2*x@off.T, -2*y@off.T
    hz = 2*energies-np.diag(J.real)
    return np.stack((hy*z-hz*y, hz*x-hx*z, hx*y-hy*x), axis=-1)


def upper_pair_symbol(state, matrix, r):
    """Symbol of sum W_ij s_i^+ s_j^- on the upper shell, real symmetric W."""
    x, y, z = state[..., r:, 0], state[..., r:, 1], state[..., r:, 2]
    W = np.asarray(matrix).real
    return np.einsum('...i,ij,...j->...', x, W, x)+np.einsum('...i,ij,...j->...', y, W, y)+(z+.5-x*x-y*y)@np.diag(W)


def torch_kernel_run(model, projectors, *, count, kind, seed, times, dt, device,
                     extra_measure=None, extra_names=()):
    try:
        import torch
    except ImportError:
        import sys
        sys.path.append('/usr/local/lib/python3.11/dist-packages')
        import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    dev = f'cuda:{device}'
    dtype = torch.float64
    if count % 64:
        raise ValueError('Need a sample count divisible by 64 for independent error blocks')
    J = model.kernel
    if np.max(abs(J.imag)) > 1e-13:
        raise ValueError('Only real kernels are implemented')
    state = torch.as_tensor(initial_samples(model.L, count, kind, np.random.default_rng(seed)), device=dev, dtype=dtype)
    off = torch.as_tensor(J.real-np.diag(np.diag(J.real)), device=dev, dtype=dtype)
    hz = torch.as_tensor(2*model.energies-np.diag(J.real), device=dev, dtype=dtype)
    r = model.r
    W = [torch.as_tensor(P, device=dev, dtype=dtype) for P in projectors]
    norm0 = (state*state).sum(dim=-1)
    number0 = state[..., 2].sum(dim=-1)

    def rhs(s):
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        hx, hy = -2*(x@off.T), -2*(y@off.T)
        return torch.stack((hy*z-hz*y, hz*x-hx*z, hx*y-hy*x), dim=-1)

    def energy(s):
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        return (z*hz).sum(dim=-1)-(x*(x@off.T)+y*(y@off.T)).sum(dim=-1)

    energy0 = energy(state)

    def measure(s):
        x, y, z = s[:, r:, 0], s[:, r:, 1], s[:, r:, 2]
        m = z.sum(dim=-1)+r/2
        second = m*m+(.25-z*z).sum(dim=-1)
        obs = [m, second]
        for P in W:
            obs.append((x*(x@P.T)+y*(y@P.T)).sum(dim=-1)+(z+.5-x*x-y*y)@P.diag())
        # The classical second moment is retained separately; it is not the
        # spin-operator counting second moment used for the DTWA comparison.
        obs.append(m*m)
        if extra_measure is not None:
            extra=extra_measure(s,torch)
            if extra.shape!=(count,len(extra_names)):
                raise ValueError('Extra measurement shape does not match its names')
            obs.extend(extra[:,j] for j in range(len(extra_names)))
        return torch.stack(obs, dim=-1).reshape(64, count//64, len(obs)).mean(dim=1).cpu().numpy()

    current = 0.
    rows, raw_blocks = [], []
    names = ['mean', 'count_second_symbol']+[f'projector_{i}' for i in range(len(W))]+['classical_second']+list(extra_names)
    for time in times:
        while current < time-1e-12:
            step = min(dt, time-current)
            k1 = rhs(state); k2 = rhs(state+.5*step*k1)
            k3 = rhs(state+.5*step*k2); k4 = rhs(state+step*k3)
            state = state+step/6*(k1+2*k2+2*k3+k4)
            current += step
        blocks = measure(state)
        raw_blocks.append(blocks)
        avg = blocks.mean(axis=0); se = blocks.std(axis=0, ddof=1)/8
        row = {'time': float(time), **{n:float(a) for n,a in zip(names, avg)},
               **{n+'_se':float(a) for n,a in zip(names, se)}}
        row['variance'] = float(avg[1]-avg[0]**2)
        row['variance_se'] = float(np.std(blocks[:,1]-2*avg[0]*blocks[:,0], ddof=1)/8)
        row['classical_variance'] = float(avg[2+len(W)]-avg[0]**2)
        rows.append(row)
    report = {'method': kind, 'samples': count, 'seed': seed, 'dt': dt, 'device': device,
              'torch': torch.__version__, 'observable_order': names,
              'max_spin_length_squared_drift': float(((state*state).sum(dim=-1)-norm0).abs().max()),
              'max_number_drift': float((state[..., 2].sum(dim=-1)-number0).abs().max()),
              'max_energy_symbol_drift': float((energy(state)-energy0).abs().max()), 'rows': rows}
    return report, np.asarray(raw_blocks)
