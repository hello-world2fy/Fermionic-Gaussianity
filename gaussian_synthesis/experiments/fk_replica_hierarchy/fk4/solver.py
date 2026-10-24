"""Damped Newton with a gradient-based roundoff safeguard near the solution."""
import numpy as np


def solve(backend, target, initial=None, tolerance=1e-11, max_steps=100):
    target = np.asarray(target, complex)
    z = target.copy() if initial is None else np.asarray(initial, complex).copy()
    def pack(v):
        return np.r_[v.real, v.imag]
    def unpack(v):
        return v[:len(target)]+1j*v[len(target):]
    inner = float(np.vdot(target, z).real)
    radial = backend.evaluate(z, hessian=False).q
    if inner > 0 and radial > 0:
        z *= (inner/radial)**(1/(2*backend.k-1))
    history = []
    status = "iterations_exhausted"
    for iteration in range(max_steps):
        value = backend.evaluate(z)
        gradient = value.gradient-pack(target)
        norm = float(np.linalg.norm(gradient))
        objective = value.q/(2*backend.k)-float(np.vdot(target, z).real)
        history.append({"iteration": iteration, "objective": objective, "gradient_norm": norm})
        if norm < tolerance:
            status = "converged"
            break
        eig = np.linalg.eigvalsh(value.hessian)
        damping = max(0., 1e-12*max(1., eig[-1])-eig[0])
        direction = np.linalg.solve(value.hessian+damping*np.eye(len(gradient)), -gradient)
        slope = float(gradient@direction)
        if slope >= 0:
            direction, slope = -gradient, -norm**2
        alpha = 1.
        for _ in range(45):
            candidate = unpack(pack(z)+alpha*direction)
            trial = backend.evaluate(candidate, hessian=False)
            obj_trial = trial.q/(2*backend.k)-float(np.vdot(target, candidate).real)
            grad_trial = float(np.linalg.norm(trial.gradient-pack(target)))
            armijo = obj_trial <= objective+1e-4*alpha*slope
            # Objective differences below rounding resolution must not stall
            # Newton if the independently evaluated gradient improves strongly.
            rounding = abs(obj_trial-objective) <= 16*np.finfo(float).eps*max(1., abs(objective))
            if armijo or (rounding and grad_trial < .8*norm):
                z = candidate
                break
            alpha *= .5
        else:
            status = "line_search_failed"
            break
    final = float(np.linalg.norm(backend.evaluate(z, hessian=False).gradient-pack(target)))
    return {"z": z, "status": status, "converged": final < tolerance,
            "gradient_norm": final, "iterations": len(history), "history": history}

