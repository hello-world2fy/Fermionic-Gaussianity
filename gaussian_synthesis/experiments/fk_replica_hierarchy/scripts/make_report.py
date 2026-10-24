#!/usr/bin/env python3
"""Read saved experiment observations; never fabricate or rerun measurements."""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4.reference import magic_reference, gaussian_reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    run = args.run
    data = json.loads((run/"summary.json").read_text())
    os.environ["MPLCONFIGDIR"] = str(run/".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (run/"figures").mkdir(exist_ok=True)
    if "comparisons" in data:
        cases = data["cases"]
        fig, axes = plt.subplots(2, len(cases), figsize=(5*len(cases), 7), squeeze=False)
        lines = ["# E0 高阶解析基准", "", f"运行状态：`{data['status']}`。", "",
                 "使用通用 Gaussian 流形采样后端，未向积分器提供目标态的高阶解析积分。独立种子间 Student-t 区间是经验误差诊断，不是严格覆盖证明。", "",
                 "| 对象 | β | Z 相对误差均值 | 95% 重复半宽 | 1%经验门槛 | Gaussian纯度 |",
                 "|---|---:|---:|---:|---|---|"]
        for r in data["comparisons"]:
            lines.append(f"| {r['case']} | {r['beta']:g} | {r['mean_relative_error']:.4g} | {r['repeat_95_percent_halfwidth']:.4g} | {r['one_percent_empirical_gate']} | {r['all_purity_pass']} |")
        for col, case in enumerate(cases):
            k = np.asarray(data["orders"])
            rows = [r for r in data["rows"] if r["case"] == case]
            q = np.array([[next(o["q"] for o in r["observations"] if o["beta"] == ki) for ki in k] for r in rows])
            b = np.array([[next(o["B"] for o in r["observations"] if o["beta"] == ki-.5) for ki in k] for r in rows])
            refs = [magic_reference(int(ki)) if case == "magic4" else gaussian_reference(int(case[-1]), int(ki)) for ki in k]
            exact_q = np.exp([r["log_q"] for r in refs])
            relative = q/exact_q-1
            axes[0, col].errorbar(k, relative.mean(axis=0), yerr=relative.std(axis=0, ddof=1), marker="o", capsize=3)
            axes[0, col].axhline(0, color="black", linewidth=.7)
            axes[0, col].set(xscale="log", title=case, xlabel="Replica order k", ylabel="q relative error (mean ± seed SD)")
            axes[1, col].plot(k, [r["L"] for r in refs], label="Analytic L")
            axes[1, col].plot(k, [r["B"] for r in refs], label="Analytic canonical B")
            axes[1, col].errorbar(k, b.mean(axis=0), yerr=b.std(axis=0, ddof=1), fmt="o", fillstyle="none", label="Estimated B ± seed SD", capsize=3)
            axes[1, col].axhline(refs[0]["tau"], color="black", linestyle="--", label="Exact atomic cost")
            axes[1, col].set(xscale="log", xlabel="Replica order k", ylabel="Gaussian coefficient cost")
            axes[1, col].legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(run/"figures"/f"analytic_benchmarks.{suffix}", dpi=170)
        plt.close(fig)
        lines += ["", "![解析基准与采样误差](figures/analytic_benchmarks.png)", "",
                  "图中空心点为数值积分估计。纯 Gaussian 的真实最小成本及解析 L 均为 1；canonical B 在有限阶大于 1 是分解本身的成本，不能解释为非 Gaussian 资源。", "",
                  "本报告只覆盖 E0 的解析对象。FK 的 k=4 独立重构、E1–E5 和相应认证仍须逐项完成；此基准运行完成不等于整个方案完成。"]
    else:
        lines = ["# FK 高阶独立验证", "", f"运行状态：`{data['status']}`。下一阶验收：`{data.get('next_order_gate_pass', False)}`。", "",
                 "每个物理点使用多个独立训练种子合并求解，冻结 witness 后从新的随机流验证。表中的 L 和 B 都是采样点估计，未称为认证上下界。", "",
                 "| t | k | L点估计 | B均值 | B重复95%半宽 | 合并RB残差 | 最差种子RB残差 | 全种子重构通过 |",
                 "|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in data["rows"]:
            lines.append(f"| {r['t']:g} | {r['k']} | {r['L_point_mean']:.7g} | {r['B_mean']:.7g} | {r['B_95_percent_halfwidth']:.4g} | {r['rho_RB_pooled']:.4g} | {r['rho_RB_max']:.4g} | {r['all_seeds_reconstruction_pass']} |")
        if data.get('radial_elimination') or any(r.get('radial_elimination') for r in data['rows']):
            lines += ['', '**本轮采用解析径向消元。** 保存的数值u是方向代表；实际被验证的 canonical 向量定义为 u_eff=(Re<u|target>/q_k(u))^(1/(2k−1))u，其中q是精确积分的符号定义。C与B通过归一化分布的期望计算，消除了Z估计噪声。它不把保存的数值u原有重构失败改成通过，也没有认证q或数值径向尺度。L仍是未认证点估计。推导与低阶精确交叉验证见 RADIAL_ELIMINATION.md。', '']
        lines += ["", "RB 表示完整 Spin(7) 轨道平均的重构，未把投影向量作为 Gaussian 原子。每个种子的未平均 full-Fock 残差另外保存在 validation JSON 中。群平均没有解决标量积分的归一化误差。", "",
                  "`L_order2_algebraic_normalization` 使用独立代数 q₂ 归一化，具有 projector 理论依据，但当前计算为浮点数，尚未提供有向舍入包围。", "",
                  "本轮保存的 P 样本是有限阶 canonical 分布的探索性样本。未通过重构、归一化及对偶认证门槛前，不宣称恢复了最优接触集合或证明了物理组成必要性。", "",
                  "后续工作仍包括原方案 E1 的分级扫描以及 E2–E5；失败的验收必须保留。"]
    (run/"REPORT.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
