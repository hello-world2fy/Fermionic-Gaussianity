"""Hamiltonian-consistent stochastic spin mean field / discrete TWA.

The i=j pair interaction is the exact linear symbol -g(s_z+1/2).
Self transverse fields must therefore be excluded. For second moments,
the on-site operator identity s_z^2=1/4 is imposed in the measurement
symbol; squaring classical total populations alone gives a different
and incorrect estimator of the quantum counting variance.
"""
import numpy as np


def numpy_rhs(state, energies, g):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    hx = -2*g*(x.sum(axis=-1, keepdims=True)-x)
    hy = -2*g*(y.sum(axis=-1, keepdims=True)-y)
    hz = 2*energies-g
    return np.stack((hy*z-hz*y, hz*x-hx*z, hx*y-hy*x), axis=-1)


def initial_samples(L, count, kind, rng):
    state = np.empty((count, L, 3))
    if kind == 'gaussian':
        state[..., :2] = rng.normal(0., .5, (count, L, 2))
    elif kind == 'discrete':
        state[..., :2] = rng.choice([-.5, .5], (count, L, 2))
    else:
        raise ValueError(kind)
    state[:, :L//2, 2] = .5
    state[:, L//2:, 2] = -.5
    return state


def torch_run(model, *, count, kind, seed, times, dt, device):
    try:
        import torch
    except ImportError:
        # This compute image supplies CUDA-enabled torch in the system site.
        import sys
        sys.path.append('/usr/local/lib/python3.11/dist-packages')
        import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    dev = f'cuda:{device}'
    dtype = torch.float64
    initial = initial_samples(model.L, count, kind, np.random.default_rng(seed))
    state = torch.as_tensor(initial, device=dev, dtype=dtype)
    spin_norm0 = (state*state).sum(dim=-1)
    number0 = state[..., 2].sum(dim=-1)
    eps = torch.as_tensor(model.energies, device=dev, dtype=dtype)
    g, r = model.g, model.r
    hz = 2*eps-g
    def rhs(s):
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        hx = -2*g*(x.sum(dim=-1, keepdim=True)-x)
        hy = -2*g*(y.sum(dim=-1, keepdim=True)-y)
        return torch.stack((hy*z-hz*y, hz*x-hx*z, hx*y-hy*x), dim=-1)
    def measure(s):
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        classical_m = z[:, r:].sum(dim=-1)+r/2
        second = classical_m**2+(.25-z[:, r:]**2).sum(dim=-1)
        px, py = x[:, r:].sum(dim=-1), y[:, r:].sum(dim=-1)
        hx, hy = x[:, :r].sum(dim=-1), y[:, :r].sum(dim=-1)
        obs = torch.stack((classical_m, second, (px*hx+py*hy)/r, (py*hx-px*hy)/r), dim=-1)
        blocks = obs.reshape(64, count//64, 4).mean(dim=1)
        return blocks.cpu().numpy()
    rows, raw_blocks = [], []
    current = 0.
    for t in times:
        while current < t-1e-12:
            step = min(dt, t-current)
            k1 = rhs(state)
            k2 = rhs(state+.5*step*k1)
            k3 = rhs(state+.5*step*k2)
            k4 = rhs(state+step*k3)
            state = state+step/6*(k1+2*k2+2*k3+k4)
            current += step
        blocks = measure(state)
        raw_blocks.append(blocks)
        avg = blocks.mean(axis=0)
        se = blocks.std(axis=0, ddof=1)/8
        variance = avg[1]-avg[0]**2
        variance_se = np.std(blocks[:, 1]-2*avg[0]*blocks[:, 0], ddof=1)/8
        rows.append({'time': float(t), 'mean': float(avg[0]), 'mean_se': float(se[0]),
                     'variance': float(variance), 'variance_se': float(variance_se),
                     'cross_real': float(avg[2]), 'cross_imag': float(avg[3]),
                     'cross_real_se': float(se[2]), 'cross_imag_se': float(se[3])})
    report = {'method': kind, 'samples': count, 'seed': seed, 'dt': dt,
              'device': device, 'torch': torch.__version__,
              'max_final_spin_length_squared_drift': float(((state*state).sum(dim=-1)-spin_norm0).abs().max()),
              'max_final_number_drift': float((state[..., 2].sum(dim=-1)-number0).abs().max()), 'rows': rows}
    return report, np.array(raw_blocks)
