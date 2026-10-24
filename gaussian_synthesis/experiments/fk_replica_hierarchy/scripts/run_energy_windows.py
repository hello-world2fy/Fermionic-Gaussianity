#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import write_json, source_manifest
from fk4.energy_windows import window_construction, restricted_support_upper
from fk4.certificates import certify_order2_normalization
from fk.model import build_model
from fk2.exact_moments import ExactK2Moments
from fk4.solver import solve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--witness-run", type=Path, action="append", default=[])
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"manifest.json", source_manifest(out/"source_snapshot"))
    deltas = np.logspace(-5, 0, 21)
    ts = [9., 9.1, 11., 20.]
    allowances = [0., 1e-6, 1e-3, 1e-2]
    write_json(out/"config.json", {"t": ts, "deltas": deltas, "error_allowances": allowances,
                                   "witness_runs": [str(v) for v in args.witness_run]})
    model = build_model()
    backend = ExactK2Moments(model)
    rows, witness_rows, fits = [], [], []
    for t in ts:
        target = model.target(t)
        z = model.S.conj().T@target
        for allowance in allowances:
            subset = [window_construction(z, t, delta, error_allowance=allowance) for delta in deltas]
            rows.extend(subset)
            if allowance == 0:
                low = np.array([r["lower_cost"] for r in subset])
                high = np.array([r["upper_cost"] for r in subset])
                fits.append({"t": t, "delta_fit_range": [float(deltas[0]), float(deltas[7])],
                             "lower_log_slope": float(np.polyfit(np.log(deltas[:8]), np.log(low[:8]), 1)[0]),
                             "upper_log_slope": float(np.polyfit(np.log(deltas[:8]), np.log(high[:8]), 1)[0]),
                             "predicted_asymptotic_slope": -1 if t == 9 else -2})
        witnesses = [(2, solve(backend, z)["z"], "deterministic_order2")]
        for root in args.witness_run:
            for file in root.glob(f"t{t:g}_k*_frozen.npz"):
                k = int(file.name.split("_k")[1].split("_")[0])
                witnesses.append((k, np.load(file)["z"], str(file)))
        for k, raw_z, source in witnesses:
            certificate = certify_order2_normalization(raw_z, target)
            safe_z = certificate["z_feasible"]
            L = certificate["L_certified_saved_target"]
            U = float(abs(target).sum())
            for delta in deltas:
                cap = min(1., restricted_support_upper(safe_z, t, delta))
                gap = 1-cap
                fraction = min(1., (1-L/U)/gap) if gap > 0 else 1.
                witness_rows.append({"t": t, "source_order": k, "source": source, "delta": float(delta),
                                     "normalization_order": 2, "L": L, "U_fock": U,
                                     "restricted_overlap_upper_formula": cap, "overlap_gap": gap,
                                     "mass_fraction_upper_numeric": fraction,
                                     "nontrivial_mass_fraction_bound": fraction < 1,
                                     "restricted_formula_evaluation_interval_certified": False,
                                     "is_high_order_normalization_certificate": False})
    summary = {"status": "complete", "rows": rows, "fits": fits, "witness_rows": witness_rows,
               "full_dictionary_window_bounds": True, "strict_interval_certified": False,
               "high_order_witness_analysis_complete": False,
               "maximum_mp_reconstruction_error": max(r["high_precision_reconstruction_error"] for r in rows),
               "all_constructions_within_window": all(r["maximum_component_energy_excess"] < r["delta"] for r in rows),
               "uses_out_of_window_repair": False}
    write_json(out/"summary.json", summary)
    os.environ["MPLCONFIGDIR"] = str(out/".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, t in zip(axes, ts):
        subset = [r for r in rows if r["t"] == t and r["error_allowance"] == 0]
        ax.loglog(deltas, [r["lower_cost"] for r in subset], "o-", markersize=3, label="Full-dictionary lower")
        ax.loglog(deltas, [r["upper_cost"] for r in subset], "s-", markersize=3, label="3-orbit upper")
        ax.set(xlabel="Energy window Δ", title=f"t = {t:g}")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Restricted Gaussian coefficient cost")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out/f"energy_window_scaling.{suffix}", dpi=170)
    plt.close(fig)
    report = ["# 完整 Gaussian 字典的能窗成本障碍", "",
              "用解析不等式给出完整 Gaussian 字典的下界；三个窗内 Gaussian 轨道给出构造上界。系数由显式根式公式以 90 位精度计算，未用窗外 Fock 修补。浮点表格不标记为有向区间认证。", "",
              "| t | 预期指数 | 下界拟合斜率 | 上界拟合斜率 |", "|---:|---:|---:|---:|"]
    for r in fits:
        report.append(f"| {r['t']:g} | {r['predicted_asymptotic_slope']} | {r['lower_log_slope']:.6f} | {r['upper_log_slope']:.6f} |")
    report += ["", "![能窗上下界](energy_window_scaling.png)", "",
               "该指数来自解析上下界，图中拟合作为独立数值检查，不是只对一个构造拟合后便声称完整字典定理。误差容许 0、1e−6、1e−3、1e−2 的全部结果保存在 summary.json。若允许误差消除目标在相反 hopping Gaussian 上的分量，所给 Δ⁻² 下界自然失效；程序保留这一情况。", "",
               "当前 witness 检查使用 order-2 区间归一化及能窗重叠包络。它不能替代方案要求的高阶 witness 比较；未得到非平凡质量限制时明确保存 vacuous 结果。", "",
               "推导见实验根目录 ENERGY_WINDOW_THEORY.md。完整 E3 还包括固定 witness 接触几何及高阶 witness 的能窗排除实验。"]
    (out/"REPORT.md").write_text("\n".join(report)+"\n")
    print(fits)


if __name__ == "__main__":
    main()
