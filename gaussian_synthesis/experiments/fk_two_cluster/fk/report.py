"""Report recorded FK evidence without upgrading diagnostics to certificates."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics
import numpy as np


def _read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _number(value, digits=6):
    if value is None:
        return "—"
    try:
        number = float(value)
    except (ValueError, TypeError):
        return str(value)
    return f"{number:.{digits}g}" if np.isfinite(number) else "未记录有限值"


def _finite(row, key):
    value = row.get(key)
    return value is not None and isinstance(value, (int, float)) and np.isfinite(value)


def _latest(rows):
    result = {}
    for row in rows:
        key = (row.get("dictionary"), row.get("seed"), row.get("t"), row.get("k"))
        previous = result.get(key)
        if previous is None or row.get("train_samples", 0) > previous.get("train_samples", 0):
            result[key] = row
    return list(result.values())


def _save(figure, directory, name, plt):
    figure.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def _physics(points, directory, plt):
    if not points:
        return False
    rows = sorted(points, key=lambda row: row["t"])
    t = [row["t"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), constrained_layout=True)
    axes[0].plot(t, [row["energy"] for row in rows], label="Exact ground state", color="#183a58", marker=".")
    axes[0].plot(t, [row["gaussian_variational_energy"] for row in rows],
                 label="Single Gaussian", color="#d17b21", linestyle="--", marker=".")
    axes[0].set_ylabel("Energy / V")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, [row["wick_deviation"] for row in rows], color="#7a3e78", marker=".")
    axes[1].set_ylabel("Four-point Wick deviation")
    axes[2].plot(t, [row["entropy"] for row in rows], color="#287e72", marker=".")
    axes[2].set_ylabel("A/B entropy (bits)")
    for axis in axes:
        axis.set_xlabel("t / V")
        axis.grid(alpha=.2)
    fig.suptitle("Physical reference at recorded t points; connecting lines are guides", fontsize=12)
    _save(fig, directory, "physical_reference", plt)
    return True


def _costs(baselines, latest, directory, plt):
    usable = [row for row in baselines if _finite(row, "cost_upper")]
    mass_rows = [row for row in latest if _finite(row, "mass_validation")]
    if not usable and not mass_rows:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    colors = {"full": "#183a58", "product": "#d17b21"}
    for name, color in colors.items():
        rows = sorted((row for row in usable if row["dictionary"] == name), key=lambda row: row["t"])
        if rows:
            axes[0].plot([row["t"] for row in rows], [row["cost_upper"] for row in rows],
                         marker="o", color=color, label=f"{name}: feasible orbit cost upper")
    rows = sorted((row for row in usable if row["dictionary"] == "full"), key=lambda row: row["t"])
    if rows:
        axes[0].plot([row["t"] for row in rows], [row["product_schmidt_lower"] for row in rows],
                     linestyle="--", color="#d17b21", label="Product: Schmidt lower")
        thresholds = sorted({row.get("eta0") for row in rows if row.get("eta0") is not None})
        restriction = f"eta <= {_number(thresholds[0])}" if len(thresholds) == 1 else "restricted eta"
        axes[0].plot([row["t"] for row in rows], [row["restricted_cost_lower"] for row in rows],
                     linestyle=":", color="#7a3e78", label=f"{restriction}: lower bound")
    axes[0].set_title("Numerical bounds on coefficient cost")
    axes[0].set_ylabel("Cost tau (not squared extent)")
    grouped = defaultdict(list)
    for row in mass_rows:
        grouped[(row["dictionary"], row["k"], row["t"])].append(row["mass_validation"])
    orders = sorted({row["k"] for row in mass_rows})
    palette = plt.get_cmap("tab10")
    for name in colors:
        for index, k in enumerate(orders):
            points = sorted((t, values) for (dictionary, order, t), values in grouped.items()
                            if dictionary == name and order == k)
            if not points:
                continue
            t = [point[0] for point in points]
            median = [statistics.median(point[1]) for point in points]
            lower = [min(point[1]) for point in points]
            upper = [max(point[1]) for point in points]
            axes[1].plot(t, median, marker="." if name == "full" else "x",
                         linestyle="-" if name == "full" else "--", color=palette(index),
                         label=f"{name}, k={k}")
            axes[1].fill_between(t, lower, upper, color=palette(index), alpha=.07)
    axes[1].set_title("Finite-k mass diagnostics; largest training N")
    axes[1].set_ylabel("Held-out mass estimate B (not extent)")
    for axis in axes:
        axis.set_xlabel("t / V")
        axis.grid(alpha=.2)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7, loc="best")
    _save(fig, directory, "costs_and_mass_diagnostics", plt)
    return True


def _residuals(rows, config, directory, plt):
    usable = [row for row in rows if row.get("k", 0) >= 2
              and _finite(row, "validation_full_residual")
              and row["validation_full_residual"] > 0
              and row.get("train_samples", 0) > 0]
    if not usable:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    grouped = defaultdict(list)
    for row in usable:
        grouped[(row["dictionary"], row["k"], row["train_samples"])].append(
            row["validation_full_residual"])
    for axis, name in zip(axes, ("full", "product")):
        for k in sorted({key[1] for key in grouped if key[0] == name}):
            points = sorted((n, values) for (dictionary, order, n), values in grouped.items()
                            if dictionary == name and order == k)
            n = [point[0] for point in points]
            axis.plot(n, [statistics.median(point[1]) for point in points], marker="o", label=f"k={k}")
            axis.fill_between(n, [min(point[1]) for point in points], [max(point[1]) for point in points], alpha=.12)
        if config.get("reconstruction_tolerance"):
            axis.axhline(config["reconstruction_tolerance"], color="black", linestyle=":", label="Residual target")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{name} dictionary")
        axis.set_xlabel("Training samples")
        axis.set_ylabel("Full 128-dimensional residual estimate")
        axis.grid(alpha=.2)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    validation_sizes = sorted({row.get("validation_samples", 0) for row in usable})
    validation_note = (f"Held-out validation N = {validation_sizes[0]:,}"
                       if len(validation_sizes) == 1 else f"Held-out validation N varies: {validation_sizes}")
    fig.suptitle(validation_note + "; median / range across t and seeds (not CIs)", fontsize=10)
    _save(fig, directory, "reconstruction_vs_samples", plt)
    return True


def make_report(output_dir):
    """Regenerate REPORT.md and static PNG/PDF figures from saved JSON only."""
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    config = _read(output / "config.json", {})
    exact = _read(output / "exact.json", {})
    baselines = _read(output / "baselines.json", {})
    summary = _read(output / "summary.json", {})
    manifest = _read(output / "manifest.json", {})
    pilot_paths = sorted(output.glob("pilot_*.json"))
    pilots = [(path.name, _read(path, {})) for path in pilot_paths]
    rows = [row for _, pilot in pilots for row in pilot.get("rows", [])]
    latest = _latest(rows)
    high = [row for row in rows if row.get("k", 0) >= 2]
    accepted = [row for row in high if row.get("reconstruction_accepted", False)]
    interpretable = [row for row in accepted if row.get("distribution_interpretation_allowed", False)]
    stage_statuses = Counter(pilot.get("status", "unknown") for _, pilot in pilots)
    statistic_statuses = Counter(str(row.get("statistical_status", "unrecorded")) for row in high)
    physical_points = exact.get("points", [])
    baseline_rows = baselines.get("rows", [])
    separations = [row for row in baseline_rows if row.get("dictionary") == "full"
                   and row.get("numerical_separation", False)]
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    cache = output / ".mplconfig"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    xdg_cache = output / ".cache"
    xdg_cache.mkdir(exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    drawn = {
        "physical_reference": _physics(physical_points, figures, plt),
        "costs_and_mass_diagnostics": _costs(baseline_rows, latest, figures, plt),
        "reconstruction_vs_samples": _residuals(rows, config, figures, plt),
    }
    lines = ["# 双 FK 团簇：首轮实验报告", "",
             "本报告由同目录的原始 JSON 自动生成。精确参考、有限字典可行表示与有限 k 连续合成分别报告；所有计算采用 V=1。", "",
             f"运行状态：`{summary.get('status', '见各阶段状态')}`。精确模型结构核验：`{exact.get('structure', {}).get('verified', '未记录')}`。",
             f"已记录 {len(rows)} 条合成结果，其中 k≥2 为 {len(high)} 条；通过重构门槛 {len(accepted)} 条，允许分布解释 {len(interpretable)} 条。", ""]
    if not interpretable:
        lines += ["**尚无通过分布解释门槛的 k≥2 结果，因此不生成或解释高阶 Gaussian 分布。** 下面的有限 k 系数质量仅作诊断，不证明 canonical 分布收敛，也不证明其解释了四点关联峰。", ""]
    else:
        lines += [f"有 {len(interpretable)} 条 k≥2 结果通过已配置的分布解释门槛；本报告仍以重构与成本证据为主。分布数组保存在对应 NPZ，解释范围不得超出各条记录的统计状态。", ""]
    lines += ["## 配置与执行状态", "",
              f"- 字典：`{config.get('dictionaries', [])}`；随机种子：`{config.get('seeds', [])}`。",
              f"- t/V：`{config.get('t_values', [])}`；k：`{config.get('orders', [])}`；训练样本阶梯：`{config.get('sample_sizes', [])}`。",
              f"- 重构容差：{_number(config.get('reconstruction_tolerance'))}；阶段预算：{_number(config.get('max_seconds'))} 秒；总预算：{_number(config.get('max_total_seconds'))} 秒；内存预算：{_number(config.get('max_memory_mib'))} MiB。",
              f"- 合成阶段状态计数：`{dict(stage_statuses)}`。",
              f"- k≥2 统计状态计数：`{dict(statistic_statuses)}`。", ""]
    lines += ["## 精确物理与单 Gaussian 基线", ""]
    if physical_points:
        peak = exact.get("wick_peak", {})
        lines += [f"完整 Fock 空间基态能量与三维约化的最大差为 {_number(exact.get('structure', {}).get('max_ground_energy_crosscheck_error'))}。",
                  f"已计算网格上的最小能隙为 {_number(min(row['gap'] for row in physical_points))}。Wick 偏差峰搜索返回 t/V={_number(peak.get('t_over_v'))}，|W|={_number(peak.get('absolute_value'))}，优化成功标志为 `{peak.get('success', '未记录')}`。",
                  "这些量来自精确对角化和单 Gaussian 解析基线；不属于连续 Gaussian 合成的产出。", ""]
        if drawn["physical_reference"]:
            lines += ["![精确物理参考](figures/physical_reference.png)", ""]
    else:
        lines += ["缺少 exact.json 中的物理点，无法报告物理参考。", ""]
    lines += ["## 可行表示成本与受限字典比较", "",
              "SOCP 只优化有限的 Spin(7) 轨道字典。轨道平均后的原子保持投影范数，不能重新归一化；系数质量加上 Fock 残差修复给出数值可行上界。求解器的 optimal 状态不证明完整 Gaussian 字典的最优性。", "",
              "| 字典 | t/V | 可行成本上界 | 重构残差 | product Schmidt 下界 | 受限 eta 下界 | 数值分离 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for row in baseline_rows:
        lines.append("| " + " | ".join([row["dictionary"], _number(row["t"]), _number(row.get("cost_upper")),
                      _number(row.get("residual_l2")), _number(row.get("product_schmidt_lower")),
                      _number(row.get("restricted_cost_lower")), "是" if row.get("numerical_separation") else "否"]) + " |")
    lines += ["", f"full 字典出现数值成本分离的 t/V：`{[row['t'] for row in separations]}`。",
              "**成本分离为浮点数值证据，不是区间算术证明。** 未对特征向量、轨道平均和舍入误差作严格包络；该结论也不意味着 full 成本已达到最优。", ""]
    if drawn["costs_and_mass_diagnostics"]:
        lines += ["![成本与有限阶质量诊断](figures/costs_and_mass_diagnostics.png)", "",
                  "右图展示各组最大训练样本量处的独立验证质量估计，曲线为种子中位数，阴影为种子范围；它们不是置信区间。B_k 是有限 k 表示质量，B_k² 不能直接标记为精确 Gaussian extent。未通过重构的 B_k 不能单独作为目标态的可行成本。", ""]
    lines += ["## 连续合成与独立重构", "",
              "优化坐标只有三维；验证仍在完整128维偶宇称空间计算。训练梯度小或投影内残差小均不能代替完整重构检查。独立验证样本估计值与统计证书状态同时保留。", ""]
    if drawn["reconstruction_vs_samples"]:
        validation_sizes = sorted({row.get("validation_samples", 0) for row in high})
        lines += ["![重构残差随训练样本量变化](figures/reconstruction_vs_samples.png)", "",
                  f"横轴为训练样本量，独立验证样本量记录为 `{validation_sizes}`。验证样本量固定时，增加横轴 N 并不降低独立重构积分的 Monte Carlo 噪声。",
                  "图中范围跨越 t 点和随机种子，不能解释为采样置信区间。若部分组提前停止，较大训练 N 的曲线可能仅包含尚在运行的组，不能据此断言整体收敛。", ""]
    lines += ["下表每个 t/k/种子仅保留最大训练样本量记录；质量与残差的范围跨种子汇总。", "",
              "| 字典 | t/V | k | 种子数 | B 中位数 | 全残差范围 | 最小质量 ESS | 重构通过数 |",
              "|---|---:|---:|---:|---:|---|---:|---:|"]
    grouped = defaultdict(list)
    for row in latest:
        grouped[(row["dictionary"], row["t"], row["k"])].append(row)
    for (name, t, k), entries in sorted(grouped.items()):
        masses = [row["mass_validation"] for row in entries if _finite(row, "mass_validation")]
        residuals = [row["validation_full_residual"] for row in entries if _finite(row, "validation_full_residual")]
        ess = [row["mass_ess"] for row in entries if _finite(row, "mass_ess")]
        residual_range = f"{_number(min(residuals))} – {_number(max(residuals))}" if residuals else "—"
        values = [name, _number(t), str(k), str(len(entries)),
                  _number(statistics.median(masses) if masses else None), residual_range,
                  _number(min(ess) if ess else None), str(sum(bool(row.get("reconstruction_accepted")) for row in entries))]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "k=1 的 frame 恒等式可给出解析重构；其通过状态不代表高阶权重集中问题已解决。高 k 的低有效样本数、验证残差或统计预算不足均须在增加物理解释前解决。", "",
              "## 数据与可复现范围", "",
              "- `config.json`：实际配置；`manifest.json`：环境、源码与随机数来源记录。",
              "- `exact.json`、`baselines.json`：物理参考和可行表示；`baseline_vectors.npz`：原子、系数、目标与残差。",
              "- `calibration_*.json`、`pilot_*.json`：采样校准、全部阶段状态与逐条结果。",
              "- `witnesses_*.npz`、`distributions_*.npz`：原始 witness 和分布数据；是否可解释由对应 JSON 的验收标志决定。",
              "- `figures/`：本报告的 PNG 和 PDF。图表直接读取同目录 JSON，没有手工填入实验数值。", "",
              "所有物理关联都需要相干 ket 重构或成分间矩阵元；Gaussian 成分的经典平均不能替代目标态四点关联。", ""]
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(report_path), "figures": [str(figures / f"{name}.png") for name, ok in drawn.items() if ok],
            "pilot_rows": len(rows), "accepted_high_order_rows": len(accepted),
            "interpretable_high_order_rows": len(interpretable), "source_manifest_present": bool(manifest)}

