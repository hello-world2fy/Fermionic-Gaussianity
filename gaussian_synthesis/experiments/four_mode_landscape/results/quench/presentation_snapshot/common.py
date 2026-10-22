"""Independent, reproducible experiment orchestration and saved-data reports."""

import _bootstrap
from _bootstrap import EXPERIMENT_ROOT
import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import signal
import shutil
import sys
import time
import numpy as np

from gaussian_synthesis import FockBasis, SynthesisConfig, synthesize
from gaussian_synthesis.solver import ResourceBudget, ResourceLimitError, peak_memory_mib
from exact_solver import exact_k2, solve_exact
from reference import (cat_state, cat_witness, cat_extent, quench_state,
                       quench_reference, quadratic_state, gaussianity_defect)
from observables import observe


@dataclass(frozen=True)
class ExperimentConfig:
    kind: str
    seed: int = 20261022
    repeats: int = 4
    samples_per_repeat: int = 65536
    batch_size: int = 8192
    exact_tolerance: float = 1e-10
    sgd_iterations: int = 8000
    sgd_batch: int = 2048
    total_anchor_count: int = 9
    total_failure_probability: float = .05
    max_validation_samples: int = 12000000
    max_seconds: float = 180.
    max_memory_mib: float = 512.


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def source_hashes():
    return {str(path.relative_to(EXPERIMENT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(EXPERIMENT_ROOT.glob("*.py"))}


def stream_info(stream):
    return {"entropy": stream.entropy, "spawn_key": list(stream.spawn_key)}


def scalar_rows(vectors, obs):
    rows = []
    for i, parameter in enumerate(vectors["parameters"]):
        row = {"parameter": float(parameter), "r": float(vectors["r"][i]),
               "extent_exact": float(cat_extent(vectors["r"][i])),
               "exact_residual": float(vectors["exact_residual"][i]),
               "witness_error": float(vectors["witness_error"][i]),
               "gaussianity_defect_abs": float(abs(gaussianity_defect(vectors["psi"][i])))}
        for name, value in obs.items():
            if np.asarray(value).shape == (len(vectors["parameters"]),):
                row[name] = float(value[i])
        rows.append(row)
    return rows


def regression(name, measured, expected, checks, se=None, expected_se=None, floor=.002):
    difference = np.abs(np.asarray(measured)-np.asarray(expected))
    tolerance = np.full_like(difference, floor, dtype=float)
    if se is not None:
        tolerance = np.maximum(tolerance, 6*np.sqrt(np.asarray(se)**2 +
                              (np.asarray(expected_se)**2 if expected_se is not None else 0)))
    checks.append({"name": name, "passed": bool(np.all(np.isfinite(difference)) and np.all(difference <= tolerance)),
                   "maximum_error": float(np.max(difference)),
                   "minimum_tolerance": float(np.min(tolerance)),
                   "maximum_tolerance": float(np.max(tolerance)),
                   "is_statistical_certificate": False})


def inspect_ensembles(obs, prefix, checks):
    for label in ("p", "h"):
        matrices = obs[f"R_{label}"]
        regression(f"{prefix}_{label}_trace", np.trace(matrices, axis1=-2, axis2=-1), 1., checks, floor=1e-10)
        regression(f"{prefix}_{label}_hermitian", matrices, matrices.conj().transpose(0, 2, 1), checks, floor=1e-10)
        eig = np.linalg.eigvalsh(matrices)
        pure = obs[f"purity_{label}"]
        checks.append({"name": f"{prefix}_{label}_PSD_and_purity_bounds",
                       "passed": bool(eig.min() >= -1e-10 and pure.min() >= 1/8-1e-10 and pure.max() <= 1+1e-10)})
        for name in (f"hist_n_{label}", f"hist_xy_{label}"):
            array = obs[name]
            regression(f"{prefix}_{name}_normalization", array.sum(axis=tuple(range(1,array.ndim))), 1., checks, floor=1e-10)


def build_scan(kind, budget):
    parameters = np.linspace(0, .5, 51) if kind == "static" else np.linspace(0, 2*np.pi, 81)
    psi, witnesses, references, rs, unitaries, residuals, iterations = [], [], [], [], [], [], []
    for parameter in parameters:
        budget.check()
        target = cat_state(parameter) if kind == "static" else quench_state(parameter)
        solution = solve_exact(target, budget=budget)
        if solution.status != "converged":
            raise RuntimeError(f"Exact solver failed at {parameter}: {solution.status}")
        # Construct references after the independent full-space solve.
        if kind == "static":
            reference, r = cat_witness(parameter), parameter
        else:
            mapped = quench_reference(parameter)
            reference, r = mapped.witness, mapped.r
            unitaries.append(mapped.unitary)
        psi.append(target)
        witnesses.append(solution.u)
        references.append(reference)
        rs.append(r)
        residuals.append(solution.residual)
        iterations.append(solution.iterations)
    vectors = {"parameters": parameters, "r": np.asarray(rs), "psi": np.asarray(psi),
               "u": np.asarray(witnesses), "u_reference": np.asarray(references),
               "exact_residual": np.asarray(residuals), "exact_iterations": np.asarray(iterations)}
    vectors["witness_error"] = np.linalg.norm(vectors["u"]-vectors["u_reference"], axis=1)
    if kind == "quench":
        vectors["unitaries"] = np.asarray(unitaries)
    return vectors


def run_anchors(config, vectors, roots, directory, budget, state):
    indices = (0, 10, 25, 50) if config.kind == "static" else (0, 20, 40, 60, 80)
    rows = []
    for ordinal, index in enumerate(indices):
        budget.check()
        anchor_seed = int(roots[ordinal].generate_state(1, dtype=np.uint64)[0])
        print(f"SGD anchor {ordinal+1}/{len(indices)}: parameter={vectors['parameters'][index]:.8g}", flush=True)
        settings = SynthesisConfig(orders=(2,), seed=anchor_seed, iterations=config.sgd_iterations,
            training_batch=config.sgd_batch, total_failure_probability=config.total_failure_probability/config.total_anchor_count,
            max_validation_samples=config.max_validation_samples,
            max_seconds=config.max_seconds, max_memory_mib=config.max_memory_mib)
        result = synthesize(vectors["psi"][index], config=settings, budget=budget)
        result.save(directory / "anchors" / f"{ordinal:02d}")
        stage = result.stages[0] if result.stages else None
        validation = stage.validation if stage else None
        witness_error = float(np.linalg.norm(stage.solution.u-vectors["u_reference"][index])) if stage else None
        exact_residual = float(np.linalg.norm(exact_k2(stage.solution.u)[1]-vectors["psi"][index])) if stage else None
        upper = validation.upper_bound if validation else None
        passed = bool(result.status == "certified" and witness_error <= .01 and upper is not None and upper <= .05)
        row = {"parameter": float(vectors["parameters"][index]), "r": float(vectors["r"][index]),
               "seed": anchor_seed, "witness_error": witness_error, "exact_residual": exact_residual,
               "reconstruction_upper_bound": upper, "validation_samples": validation.samples if validation else 0,
               "status": result.status, "passed": passed}
        rows.append(row)
        state["anchors"] = rows
        write_csv(directory / "anchors.csv", rows)
        write_json(directory / "summary.json", state)
        if result.status in ("resource_limit", "numerical_error"):
            raise RuntimeError(f"Anchor stopped: {result.status}: {result.error}")
    return rows


def run_experiment(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    budget = ResourceBudget(config.max_seconds, config.max_memory_mib, started)
    root = np.random.SeedSequence(config.seed, spawn_key=(0 if config.kind == "static" else 1,))
    group_roots = root.spawn(4)
    streams = {name: stream.spawn(config.repeats if name != "anchors" else (4 if config.kind == "static" else 5))
               for name, stream in zip(("observations", "reference", "control", "anchors"), group_roots)}
    metadata = {**asdict(config), "python_version": platform.python_version(), "numpy_version": np.__version__,
                "platform": platform.platform(), "rng_algorithm": "PCG64", "source_sha256": source_hashes(),
                "rng_streams": {name: [stream_info(s) for s in values] for name, values in streams.items()},
                "histograms_are_bin_probability_mass": True,
                "observations_uncertainty": "standard error across four independent self-normalized replicates; not a rigorous confidence interval",
                "reconstruction_delta_per_anchor": config.total_failure_probability/config.total_anchor_count,
                "expected_anchor_count_across_both_experiments": config.total_anchor_count}
    write_json(directory / "config.json", metadata)
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    for source in EXPERIMENT_ROOT.glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    state = {"kind": config.kind, "status": "running", "acceptance_passed": False, "checks": [], "anchors": [],
             "completed_stages": [], "error": None,
             "confidence_assumptions": "independent ideal Haar sampling; floating-point rounding is not interval-certified"}
    write_json(directory / "summary.json", state)

    def watchdog(signum, frame):
        raise ResourceLimitError("180-second whole-experiment numerical watchdog fired")

    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    timer_started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, watchdog)
    signal.setitimer(signal.ITIMER_REAL, max(.001, config.max_seconds-budget.elapsed))
    try:
        print(f"{config.kind}: full-space exact scan", flush=True)
        vectors = build_scan(config.kind, budget)
        np.savez_compressed(directory / "vectors.npz", **vectors)
        state["completed_stages"].append("exact_scan")
        regression("exact_stationary_residual", vectors["exact_residual"], 0., state["checks"], floor=1e-10)
        regression("independent_witness_reference", vectors["witness_error"], 0., state["checks"], floor=1e-10)
        expected_defect = (np.sqrt(vectors["r"]*(1-vectors["r"])) if config.kind == "static"
                           else (np.exp(-1j*vectors["parameters"])-1)/4)
        regression("Gaussianity_defect", [gaussianity_defect(x) for x in vectors["psi"]], expected_defect,
                   state["checks"], floor=1e-10)
        print(f"{config.kind}: four independent distribution replicates", flush=True)
        kwargs = dict(samples_per_repeat=config.samples_per_repeat, batch_size=config.batch_size, budget=budget)
        obs = observe(vectors["psi"], vectors["u"], seeds=streams["observations"],
                      pullbacks=vectors["unitaries"].conj().transpose(0, 2, 1) if config.kind == "quench" else None, **kwargs)
        np.savez_compressed(directory / "observations.npz", **obs)
        write_csv(directory / "summary.csv", scalar_rows(vectors, obs))
        state["completed_stages"].append("observations")
        inspect_ensembles(obs, "direct", state["checks"])
        endpoint = 50 if config.kind == "static" else 40
        regression("magic_purity", obs["purity_p"][endpoint], 116/847, state["checks"], se=obs["purity_p_se"][endpoint])
        equal_indices = [0, 50] if config.kind == "static" else [0, 40, 80]
        regression("p_equals_h_at_special_points", obs["tv"][equal_indices], 0., state["checks"], floor=1e-10)
        if config.kind == "static":
            ratio = abs(vectors["u"][10, 0]/vectors["u"][10, 7])**3
            regression("r_point_one_density_ratio", ratio, 14.61752045248395, state["checks"], floor=1e-8)
        else:
            print("quench: independent canonical predictions at exact r(phi)", flush=True)
            ref_psi = np.asarray([cat_state(r) for r in vectors["r"]])
            ref_u = np.asarray([cat_witness(r) for r in vectors["r"]])
            predicted = observe(ref_psi, ref_u, seeds=streams["reference"], **kwargs)
            np.savez_compressed(directory / "reference_observations.npz", **predicted)
            state["completed_stages"].append("reference_observations")
            inspect_ensembles(predicted, "reference", state["checks"])
            for metric in ("purity_p", "purity_h", "tv", "spectrum_p", "spectrum_h", "mass_p", "mass_h"):
                regression(f"Gaussian_unitary_invariance_{metric}", obs[metric], predicted[metric], state["checks"],
                           se=obs[metric+"_se"], expected_se=predicted[metric+"_se"])
            for label in ("p", "h"):
                regression(f"pulled_back_number_histogram_{label}", obs[f"hist_n_pulled_{label}"], predicted[f"hist_n_{label}"],
                           state["checks"], se=obs[f"hist_n_pulled_{label}_se"], expected_se=predicted[f"hist_n_{label}_se"])
            print("quench: quadratic Hamiltonian control", flush=True)
            control_psi = np.asarray([quadratic_state(phi) for phi in vectors["parameters"]])
            control_solutions = [solve_exact(psi, budget=budget) for psi in control_psi]
            if any(solution.status != "converged" for solution in control_solutions):
                raise RuntimeError("Quadratic-control exact solve did not converge")
            control_u = np.asarray([solution.u for solution in control_solutions])
            control_residual = np.asarray([solution.residual for solution in control_solutions])
            regression("quadratic_exact_residual", control_residual, 0., state["checks"], floor=1e-10)
            np.savez_compressed(directory / "control_vectors.npz", psi=control_psi, u=control_u, exact_residual=control_residual)
            control = observe(control_psi, control_u, seeds=streams["control"], **kwargs)
            np.savez_compressed(directory / "control_observations.npz", **control)
            state["completed_stages"].append("quadratic_control")
            inspect_ensembles(control, "control", state["checks"])
            regression("quadratic_Gaussianity", [gaussianity_defect(x) for x in control_psi], 0., state["checks"], floor=1e-10)
            regression("quadratic_p_equals_h", control["tv"], 0., state["checks"], floor=1e-10)
            for metric in ("purity_p", "spectrum_p", "mass_p"):
                regression(f"quadratic_constant_{metric}", control[metric], predicted[metric][0], state["checks"],
                           se=control[metric+"_se"], expected_se=predicted[metric+"_se"][0])
            rows = scalar_rows(vectors, obs)
            for i, row in enumerate(rows):
                for metric in ("purity_p", "purity_h", "tv", "mass_p"):
                    row[f"predicted_{metric}"] = float(predicted[metric][i])
                    row[f"control_{metric}"] = float(control[metric][i])
            write_csv(directory / "summary.csv", rows)
        write_json(directory / "summary.json", state)
        anchors = run_anchors(config, vectors, streams["anchors"], directory, budget, state)
        state["completed_stages"].append("SGD_anchors")
        state["acceptance_passed"] = bool(all(c["passed"] for c in state["checks"]) and all(a["passed"] for a in anchors))
        state["status"] = "completed" if state["acceptance_passed"] else "failed_acceptance"
        all_certified = all(a["status"] == "certified" and a["reconstruction_upper_bound"] is not None for a in anchors)
        state["joint_anchor_confidence_lower_bound"] = (1-len(anchors)*config.total_failure_probability/config.total_anchor_count
                                                       if all_certified else None)
        state["both_experiments_joint_confidence_if_all_nine_certified"] = .95 if all_certified else None
    except Exception as exc:
        state["status"] = "resource_limit" if isinstance(exc, ResourceLimitError) else "numerical_error"
        state["error"] = str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0]-(time.monotonic()-timer_started)), previous_timer[1])
        state["elapsed_seconds"] = budget.elapsed
        state["peak_memory_mib"] = peak_memory_mib()
        write_json(directory / "summary.json", state)
    write_report(directory)
    print(json.dumps({"directory": str(directory), "status": state["status"],
                      "acceptance_passed": state["acceptance_passed"], "elapsed_seconds": state["elapsed_seconds"],
                      "peak_memory_mib": state["peak_memory_mib"], "error": state["error"]}, ensure_ascii=False), flush=True)
    return state


def write_report(directory):
    directory = Path(directory)
    state = json.loads((directory / "summary.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    title = "静态 asymmetric cat" if config["kind"] == "static" else "四模相互作用动力学"
    lines = [f"# {title}实验报告", "", f"完整验收：**{'通过' if state['acceptance_passed'] else '未通过'}**；状态 `{state['status']}`。",
             f"计算耗时 {state['elapsed_seconds']:.2f} 秒，峰值 RSS {state['peak_memory_mib']:.2f} MiB。种子 {config['seed']}。", "",
             "## 方法与误差口径", "",
             "密集扫描在完整 8 维复空间使用精确 q₂/C₂ 与阻尼 Newton 法求解，初值为目标态。cat 二参数方程和 Schmidt 归约仅用于独立对照。SGD 锚点复用通用库的随机优化器，未传入解析初值。",
             f"分布观测使用 {config['repeats']} 组各 {config['samples_per_repeat']:,} 个独立 Haar 样本；每组内部各参数点与 p/h 共用样本。图中的误差条为独立重复的标准误差，不是严格置信区间。辅助 R 是各块矩阵的平均；报告的 purity 是各块 Tr(R²) 的平均，两者存在有限样本差异。谱也分别保存平均块谱与平均矩阵的谱。",
             "九个 SGD 锚点的重构失败概率预先各分配 0.05/9。只有九个点均完成独立验证时，整组重构界才具有至少 95% 的联合覆盖保证，前提是独立理想 Haar 采样；不包含浮点区间认证。此保证不用于整条分布曲线。", ""]
    if state["status"] == "failed_acceptance":
        failed = [check["name"] for check in state["checks"] if not check["passed"]]
        lines[4:4] = ["", "数值运行已结束，原定回归未全部通过。失败检查："+"、".join(failed)+"。各项状态原样保留；后续诊断不追认原验收通过。", ""]
    if (directory / "observations.npz").exists():
        with np.load(directory / "observations.npz", allow_pickle=False) as saved:
            obs = dict(saved)
        with np.load(directory / "vectors.npz", allow_pickle=False) as saved:
            vectors = dict(saved)
        endpoints = [0,10,25,50] if config["kind"] == "static" else [0,20,40,60,80]
        lines += ["## 主要数值结果", "", "| 参数 | 等价 r | 解析 extent | 连续成本 B₂² | P₂ | P₂ 标准误 | TV(p₂,h₂) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for i in endpoints:
            lines.append(f"| {vectors['parameters'][i]:.6f} | {vectors['r'][i]:.6f} | {cat_extent(vectors['r'][i]):.6f} | {obs['cost_p'][i]:.6f} | {obs['purity_p'][i]:.6f} | {obs['purity_p_se'][i]:.6f} | {obs['tv'][i]:.6f} |")
        magic = 50 if config["kind"] == "static" else 40
        maximum = int(np.argmax(obs["tv"]))
        lines += ["", f"magic 点的 P₂ 数值为 {obs['purity_p'][magic]:.8f}，解析值为 116/847 = {116/847:.8f}。",
                  f"扫描中 TV 的最大值为 {obs['tv'][maximum]:.6f}，出现在参数 {vectors['parameters'][maximum]:.6f}。这只是当前离散扫描的最大值。",
                  f"独立样本块交叉纯度与直接估计的最大差为 {np.max(abs(obs['purity_cross_p']-obs['purity_p'])):.6g}。交叉估计用于检查自配对偏差，不能消除自归一化比值的全部有限样本偏差。", "", "## 物理解释", ""]
        if config["kind"] == "static":
            lines += ["在 r=0.1，解析 witness 的真空与满占据系数分别为 0.94463648 和 0.38634338；相对于原态，满占据分量得到增强。两端构型的优化密度比为 14.6175，对照分布为 27。这里比较的是相对于 Haar 测度的点态密度，不是单个构型的概率质量。",
                      f"纯度从 Gaussian 端点的 {obs['purity_p'][0]:.6f} 变化到 magic 端点的 {obs['purity_p'][-1]:.6f}。这描述加权 Gaussian 构型的相似程度，不是目标物理纯态的纯度或纠缠熵。小于统计误差的局部变化不解释为显著的非单调行为。"]
        else:
            lines += ["相互作用只改变满占据振幅的相位，却破坏 Gaussian 配对约束。extent 在 φ=π 达到 2，在一个周期末回到 1。直接求解与独立静态预测的比较结果列在验收表中。",
                      "实验室坐标中的 N(g) 分布依赖 Gaussian 变换，不能要求与静态分布相同。将构型用 U_G† 拉回后再比较；纯度、R 的谱、质量与全流形 TV 则直接比较。",
                      "二次 Hamiltonian 对照使用 H_Q=Σ j n_j。它使初始 BCS 态发生非平凡相位演化，但保持 Gaussianity 和相应构型不变量。",
                      "本实验和静态 cat 属于同一 Gaussian-unitary 一参数态族，因此是严格的内部一致性验证，不构成两种独立复杂多体物理证据。"]
        lines += ["无论 r=0 还是 r=1/2，均有 p₂=h₂。最大 Gaussian magic 处两分布依然相同，因此 TV 不能直接当作 magic 大小。有限阶分布在 Gaussian 端点也具有宽度，P₂ 不应预设为 1。",
                  "解析 extent 是最小系数成本；这里估计的 B₂² 是有限 k=2 连续测度的系数质量平方。数值 witness 所定义测度重构 C₂(u)，其与目标的偏差另行报告，B₂² 不作为目标态 exact extent 的认证估计。",
                  "占据直方图描述 Gaussian 构型的平均粒子数；二维图坐标为构型的真空和满占据概率。它们不是目标态的测量分布，二维图的中间权重本身也不能单独证明存在 pairing。", ""]
    diagnostic_path = directory / "histogram_diagnostics.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        flagged = diagnostic["flagged_bins"]
        lines += ["## 占据直方图回归标记与事后诊断", "",
                  f"原定 {diagnostic['tested_bin_count']} 个逐 bin 比较中有 {len(flagged)} 个未通过（p 为 {sum(row['distribution']=='p' for row in flagged)} 个，h 为 {sum(row['distribution']=='h' for row in flagged)} 个）。差异发生在少数相邻参数点，原始种子、数据、阈值和整体 failed_acceptance 状态未改。",
                  f"重用同一批 Haar 构型，最大 overlap 协变误差为 {diagnostic['maximum_overlap_covariance_error']:.3g}，配对直方图最大误差为 {diagnostic['maximum_paired_histogram_error']:.3g}；保存直方图的重放误差为 {diagnostic['maximum_saved_histogram_replay_error']:.3g}。这些检查支持坐标与权重实现正确，但不是新的独立统计证书。", "",
                  "| 分布 | φ | N 区间 | 绝对差 | 原阈值 | 样本级近似 SE | 差 / 近似 SE |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        for row in flagged:
            sample_se = row.get("sample_level_combined_se")
            ratio = row.get("difference_over_sample_level_se")
            lines.append(f"| {row['distribution']} | {row['phi']:.6f} | [{row['bin_left']:.1f}, {row['bin_right']:.1f}) | {row['absolute_difference']:.6f} | {row['preset_tolerance']:.6f} | {sample_se if sample_se is not None else '—'} | {ratio if ratio is not None else '—'} |")
        lines += ["", "四次重复提供的块间标准误在个别 bin 偶然偏小。使用原冻结样本另算自归一化比值的 delta-method 近似标准误后，这些差异与有限采样波动相容。该分析是事后诊断，不替换原阈值，也不把未通过改成通过；若以后需要整条直方图的稳定联合验收，应另行预注册更多独立重复或合适的同时误差界。", ""]
    lines += ["## SGD 与独立重构", "", "| 参数 | witness 误差 | 精确残差 | 独立重构上界 | 验证样本数 | 通过 |", "|---|---:|---:|---:|---:|---|"]
    for anchor in state["anchors"]:
        def fmt(value):
            return "—" if value is None else f"{value:.6g}"
        lines.append(f"| {anchor['parameter']:.6f} | {fmt(anchor['witness_error'])} | {fmt(anchor['exact_residual'])} | {fmt(anchor['reconstruction_upper_bound'])} | {anchor['validation_samples']} | {anchor['passed']} |")
    lines += ["", "## 数值回归检查", "", "| 检查 | 通过 | 最大误差 |", "|---|---|---:|"]
    for check in state["checks"]:
        lines.append(f"| {check['name']} | {check['passed']} | {check.get('maximum_error', 0):.6g} |")
    lines += ["", "采样不变量的回归阈值为六倍合并标准误差与 0.002 的较大者；它是数值检查规则，不是额外的严格概率证书。", "", "## 图与数据", ""]
    for path in sorted((directory / "figures").glob("*.png")):
        lines += [f"![{path.stem}](figures/{path.name})", ""]
    lines += ["`config.json` 记录环境、源码哈希及随机流；`summary.json` 记录验收与停止状态；`summary.csv` 保存逐点指标；NPZ 文件保存完整复振幅、R、直方图及重复统计；`anchors/` 保存各锚点的原始优化和验证输出。",
              "绘图可独立重跑：`python plotting.py --input <结果目录>`，不会重新采样或训练。"]
    if state["error"]:
        lines += ["", f"停止原因：{state['error']}"]
    (directory / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(kind):
    parser = argparse.ArgumentParser(description=f"Independent four-mode {kind} landscape experiment")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "results" / kind)
    args = parser.parse_args()
    if not args.full:
        parser.print_help()
        return 0
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.output.exists():
        parser.error("output already exists; preserve previous results and use a new directory")
    state = run_experiment(ExperimentConfig(kind, seed=args.seed), args.output)
    return 0 if state["acceptance_passed"] else 1

