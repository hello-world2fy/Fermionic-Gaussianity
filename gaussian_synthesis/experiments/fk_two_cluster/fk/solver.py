"""Damped real Newton solve of a fixed empirical convex objective."""
import numpy as np
from .moments import pack, unpack


def solve(backend, target, initial=None, tolerance=1e-8, max_steps=200, budget=None):
    target = np.asarray(target, complex)
    if backend.k == 1:
        return {'z': target.copy(), 'converged': True, 'status': 'analytic_frame_identity',
                'iterations': 0, 'gradient_norm': 0., 'history': []}
    z = target.copy() if initial is None else np.asarray(initial,complex).copy()
    radial = backend.evaluate(z, hessian=False).q
    inner = np.vdot(target,z).real
    if radial > 0 and inner > 0:
        z *= (inner/radial)**(1/(2*backend.k-1))
    history = []
    target_r = pack(target)
    converged = False
    status = 'iterations_exhausted'
    for iteration in range(max_steps):
        if budget: budget.check()
        result = backend.evaluate(z)
        grad = result.gradient-target_r
        gn = float(np.linalg.norm(grad))
        objective = result.q/(2*backend.k)-np.vdot(target,z).real
        history.append({'iteration':iteration,'objective':float(objective),'gradient_norm':gn})
        if gn <= tolerance:
            converged, status = True, 'converged'
            break
        eig = np.linalg.eigvalsh(result.hessian)
        damping = max(0., 1e-10*max(1.,eig[-1])-eig[0])
        direction = np.linalg.solve(result.hessian+damping*np.eye(len(grad)), -grad)
        slope = float(np.dot(grad,direction))
        if slope >= 0:
            direction, slope = -grad, -gn*gn
        step = 1.
        for _ in range(40):
            candidate = unpack(pack(z)+step*direction)
            if backend.objective(candidate,target) <= objective+1e-4*step*slope:
                z = candidate
                break
            step *= .5
        else:
            status = 'line_search_failed'
            break
    final_gradient = float(np.linalg.norm(backend.evaluate(z, hessian=False).gradient-target_r))
    return {'z':z,'converged':converged,'status':status,'iterations':len(history),
            'gradient_norm':final_gradient,'history':history}
