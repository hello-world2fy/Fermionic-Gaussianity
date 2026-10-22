# Fermionic Gaussianity

四模费米子 Gaussian 连续分解的数值实现：在完整的 8 维复向量空间优化有限阶凸目标，输出连续系数测度，并使用独立 Haar 样本验证重构误差。

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
```

完整流程默认单进程运行，设置数值库线程数为 1；限制 180 秒计算时间、512 MiB 进程内存预算及每阶 1200 万验证样本。样本流式累加，日常单元测试不自动运行完整实验。资源监控入口面向 macOS / Linux。

## 已保存的四模 magic state 实验

首次完整验收通过，计算耗时 7.74 秒，峰值 RSS 61.98 MiB（具体耗时取决于运行环境）。20 项单元测试全部通过。

| 阶数 | 与解析最优向量的距离 | 独立重构误差上界 |
|---|---:|---:|
| 2 | 0.000985 | 0.028010 |
| 4 | 0.000672 | 0.028019 |

两项重构上界的联合覆盖保证至少为 95%，前提为独立理想 Haar 采样；不包含浮点舍入的区间认证。连续测度重构的是数值 `C_k(u)`，其系数成本不等于目标态的 exact Gaussian extent。

- [模块、接口、资源限制与误差界推导](gaussian_synthesis/README.md)
- [完整验收报告](gaussian_synthesis/results/magic_state/report.md)
- [解析对照数据](gaussian_synthesis/results/magic_state/comparison.csv)
- [单元测试记录](gaussian_synthesis/results/unit_tests.log)

仓库包含实现代码和可复现实验结果；研究论文原稿、算法 PDF、压缩包及本机缓存未包含在本次上传中。
