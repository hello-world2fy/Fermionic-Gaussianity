# Fermionic Gaussianity

四模费米子 Gaussian 连续分解与 extent 估计的数值实现：在完整的 8 维复向量空间优化有限阶凸目标，输出连续系数测度或 extent 上下界，并使用独立 Haar 样本验证数值误差。

代码位于 [`gaussian_synthesis/`](gaussian_synthesis/)。第一版使用 NumPy 和标准库 unittest，支持四模偶 parity，以及阶数 `k=1,2,4`。

## 快速开始

```bash
git clone https://github.com/hello-world2fy/Fermionic-Gaussianity.git
cd Fermionic-Gaussianity/gaussian_synthesis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python run_tests.py
python examples/inspect_measure.py --k 4
```

完整实验需显式运行，输出目录必须是新目录：

```bash
python examples/four_mode_magic.py --full --output results/my_run
python examples/four_mode_extent.py --full --output results/my_extent
```

完整流程默认单进程运行，设置数值库线程数为 1；限制 180 秒计算时间、512 MiB 进程内存预算及每阶 1200 万验证样本。样本流式累加，日常单元测试不自动运行完整实验。资源监控入口面向 macOS / Linux。

## 已保存的四模 magic state 实验

首次分解完整验收通过，计算耗时 7.74 秒，峰值 RSS 61.98 MiB（具体耗时取决于运行环境）。加入 extent 功能后的 40 项单元测试全部通过，总耗时 1.14 秒，峰值 RSS 55.19 MiB。

| 阶数 | 与解析最优向量的距离 | 独立重构误差上界 |
|---|---:|---:|
| 2 | 0.000985 | 0.028010 |
| 4 | 0.000672 | 0.028019 |

两项重构上界的联合覆盖保证至少为 95%，前提为独立理想 Haar 采样；不包含浮点舍入的区间认证。连续测度重构的是数值 `C_k(u)`，其系数成本不等于目标态的 exact Gaussian extent。

- [模块、接口、资源限制与误差界推导](gaussian_synthesis/README.md)
- [完整验收报告](gaussian_synthesis/results/magic_state/report.md)
- [解析对照数据](gaussian_synthesis/results/magic_state/comparison.csv)
- [单元测试记录](gaussian_synthesis/results/unit_tests.log)

## Gaussian extent 估计

`estimate_extent(psi, basis, ExtentConfig(...))` 复用优化核心，为近似解构造有限阶对偶范数和 extent 的上下界。每阶独立验证一次，联合分配失败概率；返回 `interval_certified`、`target_reached` 和 `stop_reason`，区分区间有效性、目标精度与停止原因。

首次四模 extent 解析对照通过：默认 `k=1,2,4` 得到至少 95% 联合置信度的区间 **[1.168044, 2.000000]**，包含解析值 2。计算耗时 7.77 秒，峰值 RSS 62.52 MiB。默认低阶结果未达到 10% 相对间隙，报告记录 `target_reached=false`、`stop_reason=orders_exhausted`。不自动扩大阶数或突破采样预算。

- [extent 使用方法与证书推导](gaussian_synthesis/README.md#从波函数估计-gaussian-extent)
- [extent 完整运行报告](gaussian_synthesis/results/magic_extent/report.md)
- [extent 解析对照数据](gaussian_synthesis/results/magic_extent/comparison.csv)
- [本次单元测试结果](gaussian_synthesis/results/unit_tests_extent.json)

## 四模构型分布实验

静态 asymmetric cat 与四模相互作用动力学的代码、独立测试及保存结果位于
[`gaussian_synthesis/experiments/four_mode_landscape/`](gaussian_synthesis/experiments/four_mode_landscape/)。
实验单向调用通用库，不修改主算法；完整运行与绘图均通过独立入口显式执行。

51 个静态点、81 个动力学点和九个 SGD 独立重构验收已完成，附有 PNG/PDF 图及 CSV/NPZ 数据。
静态完整验收通过；动力学保留 `failed_acceptance`：求解、重构和不变量检查通过，
但 7/6480 个占据直方图区间比较超过预设阈值。事后诊断与原始结果一起保存，不追认通过。

- [结果与分析](gaussian_synthesis/experiments/four_mode_landscape/RESULTS.md)
- [实验结构、数学约定与复现方法](gaussian_synthesis/experiments/four_mode_landscape/README.md)

仓库包含实现代码和可复现实验结果；研究论文原稿、算法 PDF、压缩包及本机缓存不随代码分发。
