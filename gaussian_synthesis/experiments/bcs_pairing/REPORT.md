# L=4 BCS 基线验收报告

日期：2026-10-23。最终运行：[l4_baseline_20261023_02](results/l4_baseline_20261023_02/)。状态：**passed**。

第一批通用空间接口、BCS 基线及小规模验证已完成。结果支持一个明确区分：平均场给出的特定轨迹可以有明显误差，但每时刻重新优化一个 onsite PBCS 构型仍可获得高 fidelity。本批没有实现一般跨能级 projected Gaussian、通用多模 Haar 或 Gaussian extent，不据此声称必须使用多个一般 Gaussian 构型。

## 实现与兼容性

通用 `src` 新增 `FermionSpace`、`OccupationSector`、`NumberSector`、平方 fidelity 和相位对齐距离。空间元数据不枚举完整 Fock 空间；固定数目基底在分配前检查容量；数目元数据及坐标不可变；投影/嵌入不自动归一化。

旧 `FockBasis` 的构造、排序及四模 Gaussian/synthesis/extent 接口保留。新工具区分 `L=4` 对应的 8 个物理模式、6 维 pair 坐标、70 维完整固定数目 sector 和 128 维完整偶宇称空间。现有四模后端不会因此自动支持八模合成。

BCS Hamiltonian、Dicke 初态、解析态、精确演化、两种平均场、PBCS 拟合、minor 证书、物理观测量和运行脚本均保存在本实验目录。

## 验收

| 验证集合 | 结果 |
|---|---:|
| 主库测试：40 项原有测试 + 9 项新接口测试 | 49/49 通过 |
| BCS 结构测试 | 15/15 通过 |
| 保存结果的离线核验，含哈希检查 | 35/35 通过 |
| 缺失 CSV 行的负测试 | 正确拒绝 |
| 篡改 fidelity 的负测试 | 标量重算及哈希检查均拒绝 |
| 当前数值源码与保存快照 | 一致 |

测试使用 `-W error::RuntimeWarning`。原始 stdout/stderr、每项结果与命令保存在 [validation/](results/l4_baseline_20261023_02/validation/)，汇总为 [summary.json](results/l4_baseline_20261023_02/validation/summary.json)。

独立检查包括：逐位费米算符对照 pair Hamiltonian；矩阵指数对照谱演化；线性时间 ODE 对照 log-time `1/t` 积分；完整量子 product-state 能量变分对照 TDVP；PBCS 实/虚参数方向的梯度差分；初态、G=0、守恒量、相位控制及拟合下界/minor 上界的一致性。

## 恒耦合淬火

参数 `L=4,m=2,W=1,G=0.6`，即 `lambda=LG/W=2.4`，均匀能级与强配对极限 Dicke 初态。输出 81 个时间点；6 个预定时刻做 PBCS 拟合。

| t | 投影 TD-BCS | 投影 product TDVP | 可实现 onsite PBCS fidelity | 最优 onsite fidelity 上界 |
|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 1 | 0.990963 | 0.977427 | 0.994004 | 0.997683 |
| 2 | 0.925363 | 0.726851 | 0.986619 | 0.995950 |
| 5 | 0.781664 | 0.546302 | 0.995414 | 0.998494 |
| 10 | 0.822819 | 0.853425 | 0.992139 | 0.996769 |
| 20 | 0.810766 | 0.757064 | 0.990875 | 0.997441 |

因此在 `t=20`，最佳 onsite PBCS fidelity 被夹在约 `[0.990875,0.997441]` 内。最优 onsite infidelity 介于约 `0.002559` 和 `0.009125`；非凸优化没有消除这段不确定性。

在全部输出时间上，投影 TD-BCS 的最低 fidelity 为 `0.736007`（t=9）；product TDVP 为 `0.539294`（t=4.75）。保留有限 L 修正并不保证投影轨迹的 fidelity 提高，二者不能混称同一种基线。

## 解析渐近态与相位对照

完成 11 个 `(eta,alpha)` 点。以 eta=0.8 为例：

| alpha | 可实现 onsite fidelity | onsite 上界 |
|---:|---:|---:|
| 0 | 1.000000000 | 1.000000000 |
| 0.25 | 0.999934925 | 0.999967811 |
| 0.50 | 0.999740573 | 0.999871755 |
| 0.75 | 0.999419543 | 0.999713352 |
| 1.00 | 0.998976124 | 0.999495111 |

alpha=0 的去相位对照由 onsite PBCS 精确表示；alpha=1 的额外相位产生可检测的 onsite 表示误差，但仍小于约 0.11%。不能称作严重的单构型失效。

所有 alpha 控制的 Fock 概率最大差为 `2.22e-16`。eta=0.2、2、8 的完整相位目标得到的可实现 fidelity 分别约为 `0.999282`、`0.999990` 和 `0.999999999989`。此有限体系的误差不随 eta 单调；eta=8 的 minor 上界在 float64 中舍入为 1，不以显示精度推断严格等号。

## `1/t` 公式的动态交叉验证

eta=0.8，从 Dicke 态在 t0 启动，比较数值态与同一末时刻的解析渐近态：

| 末时刻 T | t0=1e-4 的 infidelity | t0=1e-5 的 infidelity |
|---:|---:|---:|
| 20 | 5.10861e-4 | 5.08736e-4 |
| 40 | 1.04994e-4 | 1.05124e-4 |
| 80 | 9.84551e-6 | 1.00131e-5 |
| 160 | 7.12745e-6 | 6.79903e-6 |

这验证了实现与渐近公式的接近趋势；T=160 仍是有限时间。两个 t0 结果的最大相位对齐距离为 `6.93e-5`；收紧积分容差的距离变化为 `1.09e-9`，二者分别衡量初时刻截断与积分误差。

数值态最大范数漂移 `3.80e-10`，两种平均场能量漂移不超过 `4.83e-10`。G=0 的平方 fidelity 误差不超过 `2.22e-16`。所有阈值在运行前由 [配置](configs/l4_pilot.json) 固定。

## 复现与资源

17 次拟合共 272 个起点，均报告局部优化收敛；这不构成全局最优证书。每次拟合的参数、seed 和状态全部保存。

本次数值阶段计时约 `0.54 s`、峰值 RSS `73.41 MiB`，不含环境安装、源码快照、单元测试和绘图；它只是本机小体系记录，不作为可扩展性结论。运行采用 Python 3.12.14、NumPy 2.5.4、SciPy 1.18.1，完整依赖见 [requirements-freeze.txt](results/l4_baseline_20261023_02/requirements-freeze.txt)。

- [原始状态与观测量](results/l4_baseline_20261023_02/states.npz)
- [淬火结果](results/l4_baseline_20261023_02/baselines.csv) 与 [渐近结构结果](results/l4_baseline_20261023_02/asymptotic.csv)
- [数值验收](results/l4_baseline_20261023_02/checks.json)、[源码与数据清单](results/l4_baseline_20261023_02/manifest.json)
- [PNG 结果图](results/l4_baseline_20261023_02/figures/overview.png) 与 [SVG 结果图](results/l4_baseline_20261023_02/figures/overview.svg)

上界采用数学上成立的 minor 推导和 float64 数值计算，不含浮点舍入的区间认证。本次也没有 Gaussian extent 的统计置信区间。

初次开发运行 `_01` 原样保留；最终 `_02` 纳入 NumberSector 元数据不可变性修正及新增测试，参数与数值阈值未改变。以 `_02` 的源码快照和日志作为交付依据。

下一批按规划实现一般 projected Gaussian 的表示、完整数目投影分母和拟合，再决定是否进入通用 Haar 与有限阶合成。
