#!/usr/bin/env python3
"""E0/E1 FK order gate: pooled training, frozen witness, independent validation.

Each invocation runs one explicit order at declared physical points. Higher
orders are not scheduled automatically when reconstruction has not passed.
"""
from pathlib import Path
import argparse
import hashlib
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import t as student_t
from fk4 import paths
from fk4.io import write_json, source_manifest
from fk4.reference import log_dimension
from fk4.solver import solve
from fk4.tempering import sample_tempered
from fk4.weighted import WeightedMoments
from fk4.symmetry import TimeReversalAveraged, time_reverse
from fk4.distributions import full_distribution
from fk4.cartan_sparse import CartanMoments
from fk4.radial import target_overlap
from fk.model import build_model
from fk2.exact_moments import ExactK2Moments, full_tensor_reconstruction
from fk3.discovery import covariance_from_kets
from fk.observables import atom_invariants


def anchors_for(ref_z, backend, cartan=None):
    anchors = {1.: np.log(np.vdot(ref_z, ref_z).real)-log_dimension(8, 1),
               2.: np.log(backend.evaluate(ref_z, hessian=False).q)-log_dimension(8, 2)}
    for order, moment in (cartan or {}).items():
        anchors[float(order)] = np.log(moment.evaluate(ref_z, hessian=False).q)-log_dimension(8, order)
    return anchors


def _initialize_gpu_worker(queue):
    os.environ['FK_SAMPLER_BACKEND']='cuda'
    os.environ['FK_CUDA_DEVICE']=str(queue.get())


def _map_seed_tasks(function,tasks,devices):
    if not devices:
        return list(map(function,tasks))
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    context=get_context('spawn')
    queue=context.Queue()
    for device in devices:queue.put(device)
    with ProcessPoolExecutor(max_workers=len(devices),mp_context=context,
                             initializer=_initialize_gpu_worker,initargs=(queue,)) as pool:
        return list(pool.map(function,tasks,chunksize=1))


def _training_task(task):
    args,t,key,iteration,seed,ref_z,ref,S,anchors=task
    out=args.output
    rng = np.random.default_rng(np.random.SeedSequence([seed, 401, args.k, int(round(t*10000)), iteration]))
    sampled = sample_tempered(ref, args.k, args.count, rng, sweeps=args.sweeps,
                               log_z_anchors=anchors)
    h = sampled.kets.conj()@S
    log_weights = sampled.log_z-args.k*sampled.log_r-np.log(args.count*len(args.seeds))
    path = f"{key}_train{iteration}_seed{seed}"
    np.savez_compressed(out/f"{path}.npz", overlaps=h, log_measure_weights=log_weights,
                        reference_z=ref_z, reference_witness=ref,
                        final_log_r=sampled.log_r)
    write_json(out/f"{path}.json", sampled.diagnostics)
    print(f"{key} train round={iteration} seed={seed} elapsed={sampled.diagnostics['elapsed_seconds']:.1f}s", flush=True)
    return h,log_weights


def _validation_task(task):
    args,t,key,seed,z,u,target,witness_hash,S,clifford,anchors=task
    out=args.output
    rng = np.random.default_rng(np.random.SeedSequence([seed, 402, args.k, int(round(t*10000))]))
    values = {}
    arrays = {"witness": u, "target": target}
    def snapshot(kets, beta, log_z, log_r, diagnostic):
        if beta not in (args.k-.5, args.k):
            return
        cov = covariance_from_kets(kets[:256])
        purity = float(np.max(abs(cov@cov+np.eye(16))))
        if purity >= 1e-9:
            raise ArithmeticError("Gaussian purity failed")
        if beta == args.k-.5:
            values["B"] = float(np.exp(log_dimension(8, args.k)+log_z))
            values['P_mean_abs_overlap']=float(np.mean(np.exp(.5*log_r)))
            inv = atom_invariants(cov, clifford)
            arrays.update(P_kets=kets[:256].copy(), P_log_r=log_r.copy(), P_covariances=cov)
            arrays.update({"P_"+name: arr for name, arr in inv.items()})
            values["P_purity_error"] = purity
            values["P_summary_count"] = 256
            if args.full_distribution:
                distribution_arrays, distribution_report = full_distribution(kets, clifford)
                arrays["P_all_kets"] = kets.copy()
                arrays.update({"P_all_"+name: value for name, value in distribution_arrays.items()})
                values["P_full_distribution"] = distribution_report
        else:
            q = float(np.exp(log_dimension(8, args.k)+log_z))
            a = kets.conj()@u
            C = q*np.mean(kets/a.conj()[:, None], axis=0)
            C_RB = S@(S.conj().T@C)
            arrays["C_RB_spin7"] = C_RB.copy()
            if args.time_reversal:
                C_RB = .5*(C_RB+time_reverse(C_RB))
            arrays.update(C=C, C_RB=C_RB, gradient_log_r=log_r.copy())
            values.update(q=q, L_point=float(np.vdot(u, target).real/q**(1/(2*args.k))),
                          rho_raw=float(np.linalg.norm(C-target)), rho_RB=float(np.linalg.norm(C_RB-target)),
                          gaussian_purity_error=purity,
                          log_z=log_z)
            if args.radial_elimination:
                factor=target_overlap(u,target)/q
                arrays.update(C_fixed_numeric_u=C.copy(),C_RB_fixed_numeric_u=C_RB.copy())
                arrays['C']=factor*C
                arrays['C_RB']=factor*C_RB
                arrays['C_RB_spin7']=factor*arrays['C_RB_spin7']
                values.update(rho_raw_fixed_numeric_u=values['rho_raw'],rho_RB_fixed_numeric_u=values['rho_RB'],
                              B_fixed_numeric_u=values['B'],B=target_overlap(u,target)/values['P_mean_abs_overlap'],
                              B_radial_partition_ratio_diagnostic=values['B']*factor,
                              rho_raw=float(np.linalg.norm(factor*C-target)),
                              rho_RB=float(np.linalg.norm(factor*C_RB-target)),
                              radial_elimination=True,
                              radial_scale_point=factor**(1/(2*args.k-1)),
                              canonical_radius_numeric_certified=False,
                              reconstruction_definition='C_k(u_eff) with u_eff=(Re<u|target>/exact_q_k(u))^(1/(2k-1))*u; expectation evaluated without Z')
    result = sample_tempered(u, args.k, args.validation_count or args.count, rng,
                             sweeps=args.sweeps, checkpoints=[args.k-.5], state_callback=snapshot,
                             log_z_anchors=anchors)
    values.update(seed=seed, witness_sha256=witness_hash, diagnostics=result.diagnostics,
                  canonical_B_is_certified_upper=False, L_point_is_certified_lower=False)
    save=np.savez if args.uncompressed_distributions else np.savez_compressed
    save(out/f"{key}_validation_seed{seed}.npz", **arrays)
    write_json(out/f"{key}_validation_seed{seed}.json", values)
    print(f"{key} validation seed={seed} rho_RB={values['rho_RB']:.6g} B={values['B']:.6g}", flush=True)
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--t", type=float, nargs="+", default=[0., 11., 20.])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--count", type=int, default=32768)
    p.add_argument("--validation-count", type=int, default=None)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--sweeps", type=int, default=2)
    p.add_argument("--time-reversal", action="store_true", help="Exact T=-P_A K averaging in addition to Spin(7)")
    p.add_argument("--full-distribution", action="store_true", help="Compute invariants and Gaussian purity for every P particle")
    p.add_argument("--seeds", type=int, nargs="+", default=[24101, 24102, 24103, 24104, 24105])
    p.add_argument("--frozen", type=Path, help="Independent validation of an existing frozen witness directory; no training")
    p.add_argument("--cartan-anchors", type=Path, nargs="*", default=[], help="Verified exact low-order Cartan result directories")
    p.add_argument("--initial-from", type=Path, help="Previous-order frozen witnesses used only to initialize training")
    p.add_argument("--initial-order", type=int, default=4)
    p.add_argument("--backend", choices=['cpu','cuda'], default='cpu')
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--radial-elimination", action='store_true', help='Validate C at the direction representative with exact implicit radial normalization; avoids stochastic Z in C and B')
    p.add_argument("--uncompressed-distributions", action='store_true', help='Save large validation NPZ without compression to reduce CPU and verification time')
    p.add_argument("--devices",type=int,nargs="+",help="Run independent seeds concurrently on distinct CUDA devices; preserves seed/task ordering")
    args = p.parse_args()
    if args.devices and (args.backend!="cuda" or len(set(args.devices))!=len(args.devices)):
        raise ValueError("distinct CUDA devices required for parallel seeds")
    os.environ['FK_SAMPLER_BACKEND'] = args.backend
    os.environ['FK_CUDA_DEVICE'] = str(args.device)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else [str(v) for v in value] if key == 'cartan_anchors' else value for key, value in vars(args).items()}
    write_json(out/"config.json", config)
    write_json(out/"manifest.json", source_manifest(out/"source_snapshot"))
    model = build_model()
    exact = ExactK2Moments(model)
    cartan = {}
    import json
    for directory in args.cartan_anchors:
        proof = json.loads((directory/'proof.json').read_text())
        verification = json.loads((directory/'verification.json').read_text())
        if proof['status'] != 'complete' or not verification['artifact_verification_passed']:
            raise ValueError('unverified exact moment anchor')
        order = proof['k']
        cartan[order] = CartanMoments(order, np.load(directory/'moment_matrix.npz')['matrix'])
    np.savez_compressed(out/"model.npz", S=model.S, masks=model.even_masks,
                        reduced_interaction=model.reduced_interaction, reduced_hopping=model.reduced_hopping,
                        exact_k2_matrix=exact.matrix)
    started = time.monotonic()
    summary = {"status": "running", "k": args.k, "rows": [],
               "confidence_intervals_are_empirical": True, "high_order_dual_certified": False,
               "sampling_measure": "full even eight-mode Gaussian orbit"}
    for t in args.t:
        key = f"t{t:g}_k{args.k}"
        target = model.target(t)
        target_z = model.S.conj().T@target
        if args.time_reversal:
            if np.linalg.norm(time_reverse(target)-target) > 1e-12:
                raise ValueError("target is not fixed by T")
            target_z = target_z.real.astype(complex)
        k2_solution = solve(exact, target_z)
        z = k2_solution["z"]
        k2_residual = float(np.linalg.norm(full_tensor_reconstruction(model.S@z)-target))
        if k2_residual > 1e-10:
            raise ArithmeticError("independent deterministic k2 reconstruction failed")
        if args.initial_from and not args.frozen:
            initial = np.load(args.initial_from/f't{t:g}_k{args.initial_order}_frozen.npz')
            if np.linalg.norm(initial['target']-target) > 1e-13:
                raise ValueError('previous-order target mismatch')
            z = initial['z'].copy()
        training = []
        train_started = time.monotonic()
        if args.frozen:
            frozen = np.load(args.frozen/f"{key}_frozen.npz")
            z = frozen["z"].copy()
            if args.time_reversal and np.linalg.norm(z.imag) > 1e-12:
                raise ValueError("cannot apply T averaging to a previously frozen non-real witness")
            if np.linalg.norm(frozen["target"]-target) > 1e-13:
                raise ValueError("frozen target mismatch")
        elif args.k != 2:
            for iteration in range(args.rounds):
                ref_z = z/np.linalg.norm(z)
                ref = model.S@ref_z
                h_all, log_weights_all = [], []
                tasks=[(args,t,key,iteration,seed,ref_z,ref,model.S,anchors_for(ref_z,exact,cartan)) for seed in args.seeds]
                for h,log_weights in _map_seed_tasks(_training_task,tasks,args.devices):
                    h_all.append(h)
                    log_weights_all.append(log_weights)
                backend = WeightedMoments(np.concatenate(h_all), np.concatenate(log_weights_all), args.k,
                                          log_dimension(8, args.k))
                if args.time_reversal:
                    backend = TimeReversalAveraged(backend)
                    z = z.real.astype(complex)
                solution = solve(backend, target_z, initial=z)
                z = solution["z"]
                value = backend.evaluate(z, hessian=False)
                training.append({"round": iteration, "optimizer_status": solution["status"],
                                 "optimizer_converged": solution["converged"],
                                 "empirical_gradient_norm": solution["gradient_norm"],
                                 "importance_ess": value.ess, "q": value.q, "B": value.mass,
                                 "history": solution["history"]})
                if not solution["converged"]:
                    write_json(out/f"{key}_training_failure.json", training)
                    raise ArithmeticError("empirical Newton solve failed")
        u = model.S@z
        witness_hash = hashlib.sha256(u.tobytes()).hexdigest()
        np.savez_compressed(out/f"{key}_frozen.npz", z=z, witness=u, target=target, target_z=target_z)
        write_json(out/f"{key}_training.json", {"rounds": training, "witness_sha256": witness_hash,
                                               "canonical_radial_definition": "u_eff=(Re<u|target>/q_k(u))^(1/(2k-1))*u, with exact q implicit" if args.radial_elimination else "stored numeric u",
                                               "elapsed_seconds": time.monotonic()-train_started,
                                               "frozen_source": str(args.frozen) if args.frozen else None,
                                               "independent_k2_full_residual": k2_residual})
        tasks=[(args,t,key,seed,z,u,target,witness_hash,model.S,model.j_matrices,anchors_for(z,exact,cartan)) for seed in args.seeds]
        validation=list(_map_seed_tasks(_validation_task,tasks,args.devices))
        Bs = np.array([r["B"] for r in validation])
        B_se = float(np.std(Bs, ddof=1)/np.sqrt(len(Bs))) if len(Bs) > 1 else None
        halfwidth = float(student_t.ppf(.975, len(Bs)-1)*B_se) if B_se is not None else None
        Cs = np.array([np.load(out/f"{key}_validation_seed{seed}.npz")["C_RB"] for seed in args.seeds])
        pooled_rho = float(np.linalg.norm(Cs.mean(axis=0)-target))
        row = {"t": t, "k": args.k, "B_mean": float(Bs.mean()), "B_95_percent_halfwidth": halfwidth,
               "L_point_mean": float(np.mean([r["L_point"] for r in validation])),
               "rho_RB_pooled": pooled_rho, "rho_RB_max": max(r["rho_RB"] for r in validation),
               "reconstruction_pass": pooled_rho < 1e-3,
               "all_seeds_reconstruction_pass": all(r["rho_RB"] < 1e-3 for r in validation),
               "mass_repeat_pass": len(Bs) >= 5 and halfwidth/float(Bs.mean()) < .01,
               "physical_analysis_pass": False, "duality_gap_certified": False,
               "warning": "SMC repeat stability does not provide a rigorous N_k upper bound",
               "time_reversal_averaged": args.time_reversal,
               "radial_elimination": args.radial_elimination,
               "all_particle_distribution_saved": args.full_distribution,
               "validation_rows": validation, "training": training, "witness_sha256": witness_hash}
        # Independent exact-formula order-two normalization remains a useful
        # full-Gaussian witness bound, separate from the high-order point L.
        q2 = exact.evaluate(z, hessian=False).q
        row["L_order2_algebraic_normalization"] = float(np.vdot(u, target).real/q2**.25)
        row["order2_bound_arithmetic"] = "algebraic projector evaluated in float64; no directed rounding"
        summary["rows"].append(row)
        write_json(out/"summary.json", summary)
    summary.update(status="complete", elapsed_seconds=time.monotonic()-started,
                   next_order_gate_pass=all(r["all_seeds_reconstruction_pass"] and r["mass_repeat_pass"]
                                            for r in summary["rows"]))
    write_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
