#!/usr/bin/env python3
"""Explicit full experiment, never discovered as an ordinary unit test."""

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import json
from pathlib import Path
import platform
import signal
import sys
import time

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import numpy as np
from gaussian_synthesis import FockBasis, SynthesisConfig, synthesize
from gaussian_synthesis.solver import ResourceBudget, ResourceLimitError
from _magic_reference import magic_state, magic_q, magic_optimizer, magic_mass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="explicitly run the bounded full acceptance experiment")
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"magic_state")
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if not args.full:
        parser.print_help()
        return 0
    if args.output.exists():
        parser.error("Output already exists; use a new --output directory to preserve earlier validation results")
    config = SynthesisConfig(seed=args.seed)
    budget = ResourceBudget(config.max_seconds, config.max_memory_mib, STARTED)

    def alarm_handler(signum, frame):
        raise ResourceLimitError("180-second numerical-work watchdog fired")

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, max(.001, config.max_seconds-budget.elapsed))
    print("Full experiment: one process, thread limits=1, 180 s / 512 MiB; one final validation per order.", flush=True)

    def progress(k, row):
        print(f"k={k} step={row['iteration']:4d} diagnostic residual={row['diagnostic_residual_estimate']:.5f} "
              f"elapsed={row['elapsed_seconds']:.1f}s", flush=True)

    try:
        basis = FockBasis()
        result = synthesize(magic_state(basis), basis, config, progress=progress, budget=budget)
    finally:
        # After any controlled numerical stop, allow bounded diagnostic output
        # to finish rather than interrupting the report mid-write.
        signal.setitimer(signal.ITIMER_REAL, 0)
    result.save(args.output)
    rows = []
    for stage in result.stages:
        validation = stage.validation
        q_est = validation.q_target_estimate if validation else None
        mass_est = validation.mass_estimate if validation else None
        u_error = float(np.linalg.norm(stage.solution.u-magic_optimizer(stage.k, basis)))
        q_error = abs(q_est/magic_q(stage.k)-1) if q_est is not None else None
        mass_error = abs(mass_est/magic_mass(stage.k)-1) if mass_est is not None else None
        passed = bool(validation and validation.passed and u_error <= .01
                      and q_error is not None and q_error <= .02
                      and mass_error is not None and mass_error <= .02)
        rows.append({
            "k": stage.k, "q_analytic": magic_q(stage.k), "q_estimate": q_est,
            "q_relative_error": q_error, "u_error": u_error,
            "mass_analytic": magic_mass(stage.k), "mass_estimate": mass_est,
            "mass_relative_error": mass_error,
            "residual_estimate": validation.residual_estimate if validation else None,
            "reconstruction_upper_bound": validation.upper_bound if validation else None,
            "validation_samples": validation.samples if validation else 0,
            "certificate_status": validation.status if validation else "not_run",
            "acceptance_passed": passed,
        })
    accepted = len(rows) == len(config.orders) and all(r["acceptance_passed"] for r in rows)
    comparison = {
        "acceptance_passed": accepted,
        "analytic_comparison_is_a_regression_check_not_an_additional_confidence_claim": True,
        "analytic_references_used_only_after_synthesis": True,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "rows": rows,
    }
    (args.output/"comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False)+"\n")
    with (args.output/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["k", "acceptance_passed"])
        writer.writeheader()
        writer.writerows(rows)

    def number(value):
        return "—" if value is None else f"{value:.6g}"

    report = ["# 四模 magic state 运行报告", "",
        f"完整验收：{'通过' if accepted else '未通过'}。重构认证状态：`{result.status}`。",
        f"计算耗时 {result.elapsed_seconds:.2f} 秒；进程峰值 RSS {result.peak_memory_mib:.2f} MiB。",
        f"种子 {config.seed}；NumPy {np.__version__}；模式顺序与随机流记录见 summary.json。", "",
        "| k | q 解析 | q 估计 | u 误差 | B 解析 | B 估计 | 独立残差 | 重构上界 | 样本数 |",
        "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        report.append("| "+" | ".join(str(row[key]) if key in ("k", "validation_samples") else number(row[key])
            for key in ("k", "q_analytic", "q_estimate", "u_error", "mass_analytic", "mass_estimate",
                        "residual_estimate", "reconstruction_upper_bound", "validation_samples"))+" |")
    report += ["", "k=1 的重构使用 C₁(u)=u 精确验证；其样本仅用于系数质量估计。",
        "k=2,4 各使用 δ=0.025 的一次独立验证；两项重构上界同时有效的概率至少 95%。",
        "概率保证基于独立理想 Haar 采样，不包含浮点舍入的区间认证。",
        "q、B 和解析解距离是数值回归检查，未赋予额外置信保证。",
        "每个保存的 u 定义重构 Cₖ(u) 的连续测度；B² 是此测度的系数成本估计，不是目标态 exact extent 的估计。",
        "最优 Gaussian extent ξ(M)=2；有限阶连续测度的 B² 一般高于 2。",
        "", "vectors.npz 保存目标、数值 u 和重构向量；history.csv 保存训练与独立诊断记录。"]
    if result.error:
        report += ["", f"停止原因：{result.error}"]
    (args.output/"report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps({"acceptance_passed": accepted, "status": result.status,
                      "elapsed_seconds": result.elapsed_seconds, "peak_memory_mib": result.peak_memory_mib,
                      "output": str(args.output.resolve()), "rows": rows}, indent=2), flush=True)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
