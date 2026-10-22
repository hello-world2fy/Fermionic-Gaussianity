#!/usr/bin/env python3
"""Bounded magic-state extent benchmark; run explicitly with --full."""

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from gaussian_synthesis import ExtentConfig, FockBasis, estimate_extent
from gaussian_synthesis.solver import ResourceBudget, ResourceLimitError
from _magic_reference import magic_state, magic_q, magic_optimizer


def number(value):
    return "—" if value is None else f"{value:.7g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the bounded full extent benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "magic_extent")
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if not args.full:
        parser.print_help()
        return 0
    if args.output.exists():
        parser.error("Output already exists; use a new --output directory to preserve earlier validation results")

    config = ExtentConfig(seed=args.seed)
    budget = ResourceBudget(config.max_seconds, config.max_memory_mib, STARTED)

    def alarm_handler(signum, frame):
        raise ResourceLimitError("180-second numerical-work watchdog fired")

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, max(.001, config.max_seconds - budget.elapsed))
    print("Extent benchmark: one process, thread limits=1, 180 s / 512 MiB; one held-out validation per order.",
          flush=True)

    def progress(k, row):
        print(f"k={k} step={row['iteration']:4d} diagnostic residual={row['diagnostic_residual_estimate']:.5f} "
              f"elapsed={row['elapsed_seconds']:.1f}s", flush=True)

    basis = FockBasis()
    psi = magic_state(basis)
    try:
        result = estimate_extent(psi, basis, config, progress=progress, budget=budget)
    finally:
        # Keep the numerical work bounded; permit small reports after a controlled stop.
        signal.setitimer(signal.ITIMER_REAL, 0)
    result.save(args.output)

    # All analytic optimization and extent references are evaluated only now.
    rows = []
    for stage in result.stages:
        validation, certificate = stage.validation, stage.certificate
        exact_t_squared = magic_q(stage.k) ** (-1 / stage.k)
        q_estimate = validation.q_estimate if validation else None
        overlap = abs(np.vdot(stage.solution.u, psi))
        point = None
        if q_estimate is not None and q_estimate > 0 and overlap > 0:
            point = math.exp(2 * math.log(overlap) - math.log(q_estimate) / stage.k)
        relative_error = abs(point / exact_t_squared - 1) if point is not None else None
        u_error = float(np.linalg.norm(stage.solution.u - magic_optimizer(stage.k, basis)))
        t_lower_squared = certificate.t_lower ** 2 if certificate else None
        t_upper_squared = certificate.t_upper ** 2 if certificate else None
        # Small comparison tolerance addresses ordinary arithmetic only; it is
        # not an additional confidence claim or floating-point enclosure proof.
        contains_t = bool(certificate and t_lower_squared - 1e-12 <= exact_t_squared
                          <= t_upper_squared + 1e-12)
        passed = bool(stage.applied and certificate and contains_t and u_error <= .01
                      and relative_error is not None and relative_error <= .02)
        rows.append({
            "k": stage.k,
            "t_squared_analytic": exact_t_squared,
            "witness_quotient_point_estimate": point,
            "point_relative_error": relative_error,
            "u_error": u_error,
            "q_u_estimate": q_estimate,
            "q_u_lower": validation.q_lower if validation else None,
            "q_u_upper": validation.q_upper if validation else None,
            "residual_upper_bound": validation.residual_bound if validation else None,
            "t_squared_lower": t_lower_squared,
            "t_squared_upper": t_upper_squared,
            "finite_order_gamma": certificate.gamma if certificate else None,
            "extent_lower": certificate.lower_bound if certificate else None,
            "extent_upper": certificate.upper_bound if certificate else None,
            "validation_samples": validation.samples if validation else 0,
            "validation_status": validation.status if validation else "not_run",
            "certificate_applied": stage.applied,
            "dual_interval_contains_analytic_value": contains_t,
            "regression_passed": passed,
        })

    contains_extent = bool(result.interval_certified
                           and result.lower_bound - 1e-12 <= 2 <= result.upper_bound + 1e-12)
    regression_passed = bool(len(rows) == len(config.orders) and contains_extent
                             and all(row["regression_passed"] for row in rows))
    comparison = {
        "regression_passed": regression_passed,
        "extent_analytic": 2.,
        "extent_interval_contains_analytic_value": contains_extent,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "confidence": result.confidence,
        "interval_certified": result.interval_certified,
        "target_reached": result.target_reached,
        "relative_tolerance": config.relative_tolerance,
        "stop_reason": result.stop_reason,
        "error": result.error,
        "analytic_references_used_only_after_estimation": True,
        "regression_is_not_an_additional_confidence_claim": True,
        "point_estimate_is_not_a_certified_bound": True,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "rows": rows,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["k", "regression_passed"])
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# 四模 magic state Gaussian extent 运行报告", "",
        f"解析回归检查：{'通过' if regression_passed else '未通过'}。",
        f"extent 区间：[{number(result.lower_bound)}, {number(result.upper_bound)}]；"
        f"区间证书：{'有效' if result.interval_certified else '未完成'}。",
        f"目标相对间隙 {config.relative_tolerance:.1%}：{'达到' if result.target_reached else '未达到'}；"
        f"停止原因：`{result.stop_reason}`。",
        f"联合置信度：{number(result.confidence)}；计算耗时 {result.elapsed_seconds:.2f} 秒；"
        f"进程峰值 RSS {result.peak_memory_mib:.2f} MiB。",
        f"种子 {config.seed}；真实 Gaussian extent ξ(M)=2。", "",
        "| k | t² 解析 | witness 商点估计 | 相对误差 | u 误差 | t² 下界 | t² 上界 | 验证样本数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        keys = ("k", "t_squared_analytic", "witness_quotient_point_estimate", "point_relative_error",
                "u_error", "t_squared_lower", "t_squared_upper", "validation_samples")
        report.append("| " + " | ".join(str(row[key]) if key in ("k", "validation_samples")
                                             else number(row[key]) for key in keys) + " |")
    report += ["",
        "解析公式仅用于运行后的回归检查，没有用于训练、选取 witness 或决定验证样本量。",
        "witness 商的点估计以随机 q 估计作分母，本身不是可信下界；可信下界使用 q 的上置信界。",
        "k=1 的矩和梯度精确计算；默认 k=2,4 各分配 δ=0.025，且各自只进行一次独立固定样本量验证。",
        "q 与 C 的误差界来自同一个验证事件，不重复消耗失败概率；多阶结果通过 union bound 联合覆盖。",
        "最终 extent 区间还与 Fock 基础界 [||ψ||₂², ||ψ||₁²] 相交。",
        "区间证书依赖独立理想 Haar 采样的统计假设，不包含浮点舍入的区间认证。",
        "回归通过与达到目标精度分别报告。低阶对偶值本身低于 ξ(M)=2，默认 k=1,2,4 不预期达到 10% 间隙。",
        "解析比较仅是数值回归检查；一次抽样中解析值落入区间不能证明统计覆盖率。", "",
        "summary.json 保存配置、随机流和证书；intervals.csv 保存逐阶区间；vectors.npz 保存目标与数值向量；"
        "history.csv 保存优化记录；comparison.json/csv 保存本次解析对照。",
    ]
    if result.error:
        report += ["", f"诊断：{result.error}"]
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "regression_passed": regression_passed,
        "interval_certified": result.interval_certified,
        "target_reached": result.target_reached,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_memory_mib": result.peak_memory_mib,
        "output": str(args.output.resolve()),
    }, indent=2, allow_nan=False), flush=True)
    # A valid but wide interval is expected for the default low-order benchmark.
    # Exit status reports regression success; target_reached is separate above.
    return 0 if regression_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
