"""Actual window Gaussian dictionaries and normalized budget-constrained solves.

Fidelity uses a convex norm-ball relaxation, then checks normalization and
cost. Energy uses monotone feasible inverse iteration with convex tangent
subproblems; it is a local search, not a global optimality certificate.
"""
import time
import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
from scipy.optimize import brentq
from . import paths
from .two_orbit import actual_arc_atoms
from .joint_dual_certificates import exact_schmidt_polar
from fk4.energy_windows import x_cap
from fk4.tempering import actions
from fk4.distributions import full_distribution


def make_dictionary(model,t,delta,seed=61001,random_directions=32):
    ids=model.even_masks
    h=model.full_hamiltonian(t)[np.ix_(ids,ids)]
    plus=exact_schmidt_polar()[ids]/4
    cap=x_cap(t,delta);states=[plus];metadata=[dict(kind='hopping_vacuum',x=0.)]
    for fraction in [.25,.5,.75,.9999999]:
        x=cap*fraction
        states.extend(actual_arc_atoms(model,1-x))
        metadata.extend([dict(kind='mass_hopping_arc',x=x,vertex=j) for j in range(8)])
    perms,phases=actions(8);generators=np.zeros((120,128,128),complex)
    for i,(p,f) in enumerate(zip(perms,phases)):generators[i,np.arange(128),p]=f
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(t),int(delta*1e6)]))
    EG=-14-4*t
    for i in range(random_directions):
        weights=rng.normal(size=120)
        K=np.einsum('j,jab->ab',weights,generators)
        v=K@plus;speed=np.sqrt(np.vdot(v,v).real-abs(np.vdot(plus,v))**2)
        K/=speed
        ev,U=eigh(-1j*K);coordinates=U.conj().T@plus
        def state(theta):return U@(np.exp(1j*theta*ev)*coordinates)
        def excess(theta):
            g=state(theta);return float(np.vdot(g,h@g).real-EG)
        upper=.002
        while excess(upper)<delta and upper<1:upper*=2
        if excess(upper)<delta:raise ArithmeticError('No small-angle window boundary found')
        theta=brentq(lambda s:excess(s)-delta,0,upper,xtol=1e-14)*(1-1e-7)
        for fraction in [.5,1.]:
            g=state(theta*fraction);g/=np.linalg.norm(g)
            states.append(g);metadata.append(dict(kind='generic_quadratic_rotation',direction=i,angle=theta*fraction))
    atoms=np.array(states)
    energies=np.einsum('ia,ab,ib->i',atoms.conj(),h,atoms).real
    assert np.max(energies-EG)<=delta+1e-10
    features,quality=full_distribution(atoms,model.j_matrices)
    return atoms,dict(seed=seed,count=len(atoms),x_cap=cap,metadata=metadata,
        energy_excess=energies-EG,maximum_energy_excess=float(max(energies-EG)),
        max_purity_error=quality['max_purity_error_all_particles'],features=features['features'])


def normalize_feasible(c,G,budget,margin=1e-7):
    c=np.asarray(c,complex).copy();c/=np.linalg.norm(G@c)
    if np.sum(abs(c))<=budget-margin:return c
    # Interpolate continuously towards the cost-one hopping Gaussian before
    # renormalizing. A final feasible coefficient vector is always returned.
    e=np.zeros_like(c);e[0]=np.exp(1j*np.angle((G@c)[np.argmax(abs(G[:,0]))]/G[np.argmax(abs(G[:,0])),0]))
    lo,hi=0.,1.
    for _ in range(65):
        weight=(lo+hi)/2;v=(1-weight)*c+weight*e
        cost=np.sum(abs(v))/np.linalg.norm(G@v)
        if cost>budget-2*margin:lo=weight
        else:hi=weight
    c=(1-hi)*c+hi*e;c/=np.linalg.norm(G@c)
    if np.sum(abs(c))>budget:raise ArithmeticError('Budget feasibility repair failed')
    return c


def solve_budgets(model,atoms,t,budgets,max_iterations=30):
    G=atoms.T;ids=model.even_masks;h=model.full_hamiltonian(t)[np.ix_(ids,ids)]
    target=model.target(t);evals,V=eigh(h);E0=float(evals[0]);L=(np.sqrt(np.maximum(evals-E0+1.,0))[:,None]*V.conj().T)
    budget=cp.Parameter(nonneg=True);c=cp.Variable(len(atoms),complex=True);phi=G@c
    cost=cp.norm(c,1)
    fidelity=cp.Problem(cp.Maximize(cp.real(target.conj()@phi)),[cp.norm(phi,2)<=1,cost<=budget])
    previous=cp.Parameter(128,complex=True)
    energy=cp.Problem(cp.Minimize(cp.sum_squares(cp.abs(L@phi))),
                      [cp.real(previous.conj()@phi)>=1,cost<=budget])
    options=dict(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=250)
    rows=[];arrays={}
    for index,B in enumerate(budgets):
        budget.value=B-1e-6;started=time.monotonic()
        fidelity.solve(**options)
        if c.value is None:raise ArithmeticError('Fidelity SOCP did not return a solution')
        relaxation_norm=float(np.linalg.norm(G@c.value));relaxation_value=float(fidelity.value)
        cf=normalize_feasible(c.value,G,B);psif=G@cf
        starts=[cf,np.eye(len(atoms),dtype=complex)[0],np.eye(len(atoms),dtype=complex)[min(8,len(atoms)-1)],np.eye(len(atoms),dtype=complex)[-1]]
        candidates=[]
        for start_number,start in enumerate(starts):
            current=normalize_feasible(start,G,B);history=[];old=float(np.vdot(G@current,h@(G@current)).real)
            for iteration in range(max_iterations):
                previous.value=G@current;c.value=current
                energy.solve(warm_start=True,**options)
                if c.value is None:break
                trial=normalize_feasible(c.value,G,B);state=G@trial
                value=float(np.vdot(state,h@state).real)
                history.append(dict(iteration=iteration,energy=value,cost=float(sum(abs(trial))),status=energy.status))
                if value>old+1e-7:break
                improvement=old-value;current=trial;old=value
                if improvement<1e-9:break
            candidates.append((old,current,history,start_number))
        best=min(candidates,key=lambda x:x[0]);ce=best[1];psie=G@ce
        arrays[f'budget{index}_fidelity_coefficients']=cf;arrays[f'budget{index}_energy_coefficients']=ce
        rows.append(dict(t=t,coefficient_budget=B,
            fidelity_optimized=float(abs(np.vdot(target,psif))**2),
            fidelity_solution_cost=float(sum(abs(cf))),fidelity_solution_norm=float(np.linalg.norm(psif)),
            fidelity_solution_energy=float(np.vdot(psif,h@psif).real),
            fidelity_relaxation_overlap=relaxation_value,fidelity_relaxation_norm=relaxation_norm,
            fidelity_convex_solver_status=fidelity.status,
            energy_optimized=best[0],energy_error_optimized=best[0]-E0,
            energy_solution_fidelity=float(abs(np.vdot(target,psie))**2),
            energy_solution_cost=float(sum(abs(ce))),energy_solution_norm=float(np.linalg.norm(psie)),
            energy_selected_start=best[3],energy_multistart_histories=[x[2] for x in candidates],
            energy_global_optimum_claimed=False,
            numerical_variational_results_not_interval_certificates=True,
            elapsed_seconds=time.monotonic()-started))
    return rows,arrays
