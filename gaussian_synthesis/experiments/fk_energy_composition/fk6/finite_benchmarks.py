"""Full-ket finite dictionaries with explicit coefficient budgets and no repair."""
import struct,time,zipfile
import numpy as np
import cvxpy as cp
from . import paths
from fk.gaussian import sample_gaussian
from fk3.discovery import optimize_gaussian_overlap,covariance_from_kets
from fk.observables import mass_correlation_wick

OPTIONS=dict(solver='CLARABEL',max_iter=300,max_threads=1,
             tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)


def take_rows(path,key,indices):
    """Read selected rows without loading a multi-GB uncompressed NPZ member."""
    with zipfile.ZipFile(path) as archive:
        entry=archive.getinfo(key+'.npy')
        if entry.compress_type!=zipfile.ZIP_STORED:
            with np.load(path) as f:return f[key][indices].copy()
    with open(path,'rb') as stream:
        stream.seek(entry.header_offset);header=stream.read(30)
        name,extra=struct.unpack('<HH',header[26:30]);stream.seek(name+extra,1)
        version=np.lib.format.read_magic(stream)
        readers={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
        if version not in readers:raise ValueError(f'Unsupported NPY header {version}')
        shape,fortran,dtype=readers[version](stream)
        offset=stream.tell()
    if dtype.hasobject:raise ValueError('Object arrays are not accepted')
    array=np.memmap(path,mode='r',dtype=dtype,offset=offset,shape=shape,order='F' if fortran else 'C')
    return np.asarray(array[indices]).copy()


def normalize_budget(coefficients,atoms,budget):
    c=np.asarray(coefficients,complex).copy();G=atoms.T;norm=np.linalg.norm(G@c)
    if norm<1e-14:raise ArithmeticError('Zero fitted state')
    c/=norm
    if sum(abs(c))<=budget-1e-7:return c
    # A single one of the supplied Gaussian atoms is a feasible cost-one anchor.
    anchor=int(np.argmax(abs(atoms.conj()@(G@c))))
    e=np.zeros_like(c);e[anchor]=np.exp(1j*np.angle(np.vdot(atoms[anchor],G@c)))
    lo,hi=0.,1.
    for _ in range(65):
        a=(lo+hi)/2;v=(1-a)*c+a*e;n=np.linalg.norm(G@v)
        if n<1e-14 or sum(abs(v))/n>budget-2e-7:lo=a
        else:hi=a
    c=(1-hi)*c+hi*e;c/=np.linalg.norm(G@c)
    if sum(abs(c))>budget:raise ArithmeticError('Budget normalization failed')
    return c


def fit_fidelity(atoms,target,budget):
    started=time.monotonic();G=atoms.T;c=cp.Variable(len(atoms),complex=True)
    problem=cp.Problem(cp.Maximize(cp.real(target.conj()@G@c)),
                       [cp.norm(G@c)<=1,cp.norm(c,1)<=budget-1e-6])
    problem.solve(**OPTIONS)
    if c.value is None:raise ArithmeticError('Finite fidelity solve failed: '+problem.status)
    raw=np.asarray(c.value);relaxation_norm=float(np.linalg.norm(G@raw))
    fitted=normalize_budget(raw,atoms,budget)
    return fitted,dict(elapsed_seconds=time.monotonic()-started,status=problem.status,
        relaxation_norm=relaxation_norm,relaxation_overlap=float(problem.value),
        normalization_adjustment_norm=float(np.linalg.norm(fitted-raw)),
        global_normalized_optimality_claimed=False)


def adaptive_known(target,reference,seed,max_terms=64,budget=None,starts=8,sweeps=30):
    """Greedy full Gaussian pricing; constrained variant refits l1 at each step."""
    started=time.monotonic();rng=np.random.default_rng(np.random.SeedSequence([seed,632]))
    atoms=[reference.copy()];history=[];warm=sample_gaussian(8,starts,rng).kets
    for iteration in range(max_terms-1):
        G=np.array(atoms).T
        if budget is None:
            c=np.linalg.lstsq(G,target,rcond=1e-12)[0];fit_status='unconstrained_least_squares'
        else:
            variable=cp.Variable(len(atoms),complex=True)
            fit=cp.Problem(cp.Minimize(cp.sum_squares(cp.abs(G@variable-target))),[cp.norm(variable,1)<=budget])
            fit.solve(**OPTIONS)
            if variable.value is None:raise ArithmeticError('Constrained pricing fit failed')
            c=np.asarray(variable.value);fit_status=fit.status
        residual=target-G@c
        if np.linalg.norm(residual)<1e-9:break
        fresh=sample_gaussian(8,starts,rng).kets
        candidates,diagnostic=optimize_gaussian_overlap(residual,np.concatenate((warm,fresh)),
            max_sweeps=sweeps,gradient_tolerance=1e-8)
        order=np.argsort(abs(candidates.conj()@residual))[::-1];added=0
        for j in order:
            g=candidates[j]
            if max(abs(np.array(atoms).conj()@g))>=1-1e-9:continue
            atoms.append(g.copy());added+=1
            if added==4 or len(atoms)==max_terms:break
        history.append(dict(iteration=iteration,terms=len(atoms),residual=float(np.linalg.norm(residual)),
            fitted_cost=float(sum(abs(c))),local_pricing_overlap=float(max(abs(candidates.conj()@residual))),
            pricing_seconds=diagnostic['elapsed_seconds'],fit_status=fit_status,added=added))
        warm=candidates[order[:starts]]
        if len(atoms)==max_terms or added==0:break
    return np.array(atoms),dict(method='adaptive_cost_constrained' if budget is not None else 'adaptive_ordinary',
        elapsed_seconds=time.monotonic()-started,history=history,coefficient_budget_during_pricing=budget,
        uses_exact_target=True,full_Hilbert_pricing=True,uses_singlet_projection=False,
        starts_per_new_batch=starts,local_sweeps=sweeps,global_pricing_claimed=False,no_Fock_repair=True)


def evaluate(atoms,coefficients,target,H,model,budget):
    c=np.asarray(coefficients);state=c@atoms;norm=float(np.linalg.norm(state))
    if abs(norm-1)>2e-8:raise ArithmeticError('Evaluation state must be normalized')
    cov=covariance_from_kets(state)[0]
    mass=model.mass_correlation_operator[np.ix_(model.even_masks,model.even_masks)]
    mass_value=float(np.vdot(state,mass@state).real)
    cost=float(sum(abs(c)));overlap=np.vdot(target,state)
    return dict(fidelity=float(abs(overlap)**2),infidelity=max(0.,1-float(abs(overlap)**2)),
        energy=float(np.vdot(state,H@state).real),energy_error=float(np.vdot(state,H@state).real-np.vdot(target,H@target).real),
        hilbert_error_phase_aligned=float(np.linalg.norm(state*np.exp(-1j*np.angle(overlap))-target)),
        coefficient_cost=cost,budget_feasible=cost<=budget+1e-9,coefficient_budget=budget,
        actual_gaussian_terms=len(atoms),state_norm=norm,mass_correlation=mass_value,
        wick_mass_deviation=mass_value-float(mass_correlation_wick(cov,model.j_matrices)),
        no_Fock_repair=True,projected_atoms_counted_as_Gaussian=False)
