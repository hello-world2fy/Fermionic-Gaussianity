# 四模 Gaussian 连续合成

这是 `Gaussian_Convex_Synthesis_Algorithm.pdf` 的第一版数值实现。目标是四模偶 parity 的一般态；随附完整实验使用

\[
M=(|0000\rangle+|1111\rangle)/\sqrt2,\qquad k=1,2,4.
\]

数值优化在完整的 **8 维复向量空间**中进行。`src/` 不导入 magic state 解析解；解析表达式仅在 `examples/_magic_reference.py` 中用于测试和运行后的对照。默认运行已通过一次正式独立验收，见 [运行报告](results/magic_state/report.md)。

## 运行

运行需要 Python 3.10+ 和 NumPy。已在 Python 3.12.14 / NumPy 2.3.5 / macOS arm64 上验证；不需要 SciPy、pytest 或 GPU。

从仓库根目录进入代码项目，使用自己的 Python 环境：

```bash
cd gaussian_synthesis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python run_tests.py
python examples/inspect_measure.py --k 4
```

单元测试串行运行，独立于完整验收。完整流程必须显式使用 `--full`；已有结果目录不会被覆盖。要另行运行一个实验，应指定新的输出目录，例如：

```bash
python examples/four_mode_magic.py --full --output results/my_run
```

默认种子为 `20261022`；可使用 `--seed` 显式改变。每次运行都有自己的结果与证书，不能根据多次尝试选择通过的报告后仍声称具有单次 95% 保证。仅查看已有结果使用 `inspect_measure.py`，它不会重新训练或采样。

已有 NumPy 的环境也可以直接运行上述脚本，无需安装项目。示例和测试会在导入 NumPy 前将 `VECLIB_MAXIMUM_THREADS`、`OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS` 设为 1。直接使用库时也应在导入 NumPy 前设置这些值。资源监控入口使用 `resource` 和 POSIX 定时信号，面向 macOS / Linux。

## 模块与数据流

| 模块 | 职责 |
|---|---|
| `basis.py` | occupation 顺序、parity、一般态输入检查 |
| `gaussian.py` | Pfaffian、pairing chart、完整复振幅、四模 spin/Fock 映射 |
| `theory.py` | Weyl 维数、Gamma、搜索半径、验证二阶矩上界 |
| `sampling.py` | 四模偶 parity Gaussian rays 的独立 Haar 采样 |
| `moments.py` | 稳定计算 q、C、系数质量及其二阶矩，流式累加 |
| `objective.py` | J 和实 Hilbert 空间梯度 C−psi |
| `solver.py` | 投影 SGD、continuation、理论基线、资源监控 |
| `measure.py` | 连续系数密度、相位对齐态和估计的概率密度 |
| `validation.py` | 冻结候选后的单次独立重构验证 |
| `api.py` | 配置、互相独立的随机流、结果保存 |

bit j 对应模式 j，Fock ket 的产生算符按模式编号升序排列。四模偶 parity 的数组顺序为 bitmask `[0,3,5,6,9,10,12,15]`。输入态不会被自动归一化；优化变量 u 的尺度也不会被归一化。内积使用 `np.vdot(v,u)`，即第一个变量共轭。

基础 Fock 容器允许 2–12 模，防止误建巨大 dense 基底；第一版 GaussianState、Haar sampler 和 synthesis 明确只支持四模偶 parity。默认及可配置阶数是 `(1,2,4)` 的递增子集。不构造任何 replica Hilbert 空间或四拷贝投影算符。

### Gaussian 采样约定

设 J 的非零互补配对为 `(0,7,+1)`、`(1,6,-1)`、`(2,5,+1)`、`(3,4,-1)`，则 Gaussian quadric 条件为

\[
g^TJg=2(g_0g_{1234}-g_{12}g_{34}+g_{13}g_{24}-g_{14}g_{23})=0.
\]

代码使用酉矩阵 U，使 `U.T @ J @ U = I` 且 `U e0 = M`。先取均匀实正交二标架 r,s，再令 `g=U(r+i s)/sqrt(2)`。实正交变换的不变测度由 triality 对应到四模 Gaussian rays 的不变测度。两个标准正态实向量经顺序正交化产生此二标架。采样器不使用随机 pairing matrix 代替 Haar 测度，也不要求 vacuum 振幅非零。

### 优化

\[
J_k(u)=\frac{d_{4,k}}{2k}\mathbb E_\mu |\langle g|u\rangle|^{2k}
       -\operatorname{Re}\langle u|\psi\rangle,\qquad \nabla J_k=C_k(u)-\psi.
\]

每步使用新 Haar 样本并投影到理论搜索球。默认 `k=1` 使用精确解；后续阶使用上一阶数值解初始化。每阶训练 8000 步、批量 2048，步长 `0.1*(1+t/200)**(-0.6)/(2*k-1)`，返回后半程 post-update 迭代的平均值。

`solver_mode="theory"` 提供零初值、文档常步长和全部 pre-update 迭代平均。它与默认 continuation 的步长及初始化不同；默认模式不继承那一条形式化迭代次数保证。

训练、诊断、最终验证由 `SeedSequence` 派生的三个独立 PCG64 流生成，spawn keys 记录在 `summary.json`。诊断样本不参与更新或候选选择。诊断使用较少样本，所以其梯度噪声通常高于最终验证，不应把诊断残差当作真实残差。

## 连续分解如何使用

主要输出是一个隐式、可逐点求值的连续测度：

\[
\frac{d\Lambda_{k,u}}{d\mu}(g)=d_{4,k}|\langle g|u\rangle|^{2k-1},\qquad
\widetilde g_u=\frac{\langle g|u\rangle}{|\langle g|u\rangle|}g.
\]

该测度的积分等于 **C_k(u)**。数值 u 对目标 psi 的重构精度由独立验证给出；不能把近似 u 的系数成本直接当作 psi 的 exact Gaussian extent。零重叠处密度为零，对齐态任选原相位。

在包可导入的环境中，可从结果恢复连续测度：

```python
import numpy as np
from gaussian_synthesis import FockBasis, GaussianState, GaussianCoefficientMeasure

basis = FockBasis(4, "even")
with np.load("results/magic_state/vectors.npz", allow_pickle=False) as saved:
    measure = GaussianCoefficientMeasure(basis, 4, saved["u_k4"])
vacuum = GaussianState(basis, basis.ket(0))
density = measure.density(vacuum)
aligned_ket = measure.aligned_state(vacuum).to_fock()
```

`density` 是相对于连续 Haar 测度的密度，不是单个点的概率。`probability_density_estimate(g, mass_estimate=B)` 要求显式提供正的质量估计，函数名也标明其归一化是估计的。`estimate_mass` 返回估计值及标准误差；这个标准误差不等同于严格置信区间。

公共入口 `synthesize(psi, basis, SynthesisConfig(...))` 返回每阶的 u、measure、validation、history 和随机流信息。`result.save(directory)` 保存：

- `summary.json`：配置、理论常数、置信界、资源使用和随机流；
- `vectors.npz`：psi、各阶 u 和独立估计的重构向量；
- `history.csv`：训练目标值与独立诊断记录。

magic 示例额外生成 `comparison.json`、`comparison.csv`、`report.md`，其中解析比较只在 `synthesize` 完成后进行。

## 独立误差证书及其证明

冻结 u 后，令

\[
Y(g)=d_{n,k}|\langle g|u\rangle|^{2k-2}\langle g|u\rangle g.
\]

由 coherent frame 恒等式和正交投影的收缩性，取 p=2k−1，有

\[
\mathbb E\|Y\|^2
=d_{n,k}^2\frac{q_p(u)}{d_{n,p}}
\le\frac{d_{n,k}^2\|u\|^{4k-2}}{d_{n,2k-1}}
=:V_k(u).
\]

固定 u 与预先决定的样本数 N，独立样本均值满足

\[
\mathbb E\|\widehat C-C\|^2
=\frac{\mathbb E\|Y\|^2-\|C\|^2}{N}\le V_k(u)/N.
\]

对平方误差使用 Markov 不等式，再用三角不等式，可得以至少 1−δ 的概率

\[
\|\psi-C_k(u)\|\le\|\psi-\widehat C\|+\sqrt{V_k(u)/(N\delta)}.
\]

它比 PDF 第 6 节式 (32) 的统计余量缩小 `sqrt(d_(n,2k-1))` 倍。原 PDF 的余量也保存在结果中，便于对照。此推导不使用 magic state 的解析解，适用于任意冻结的 u。

默认 k=2,4 各分配 δ=0.025。两项结论通过 union bound 得到联合至少 95% 的覆盖保证。N 由冻结 u 的范数和预定统计余量 0.025 决定，不依赖验证样本。k=1 使用 C₁(u)=u 精确验证；其额外样本只用于估计系数质量。验证使用的概率前提是理想独立 Haar 采样；代码的 PRNG 和浮点运算不是区间算术或密码学随机性认证。

单次验收不重试、不根据验证结果更换 u，也不把中途中止后的样本当作完整固定 N 证书。q、B 和解析向量距离的阈值是数值回归标准，没有额外赋予 95% 置信含义。

## 资源限制与状态

单元测试限制为 30 秒 / 256 MiB；完整数值流程限制为 180 秒 / 512 MiB。命令行入口有 wall-clock watchdog，库在每次迭代和积分批次间检查时间及进程峰值 RSS。内存预算是监控上限；批量和样本上限同时防止在两次检查间分配巨大的数组。

单批最多 8192 个态；训练步数最多 8000；每阶正式验证最多 1200 万样本。样本流式累加，总样本数不会转化为同等规模的内存占用。不启动进程池、GPU 或大模式数测试，不自动运行论文审计脚本。

验证超预算时，在抽样之前返回 `sample_budget_exceeded`。时间或内存预算触发 `resource_limit`，保存已完成阶段和停止原因；数值溢出返回 `numerical_error`。已完成验证但残差上界超阈值返回 `not_certified`。所有阶段通过后才给出整体 `certified`。受控停止后允许写完少量诊断文件，避免报告在写入中途被截断。

## 解析验收基准

\[
q_k(M)=2^{-k}\frac{(k+3)(k+4)(k+5)}{60},\qquad
u_k=q_k(M)^{-1/(2k-1)}M,
\]

\[
B_k=\sqrt2\frac{(k+1)(k+2)(k+3)}{(k+1/2)(k+3/2)(k+5/2)}.
\]

完整实验要求：每阶 u 距离解析解不超过 0.01，q 和 B 的相对误差不超过 2%，正式重构上界不超过 0.05。Gaussian extent ξ(M)=2，而有限阶连续测度的 B² 一般大于 2。第一版不输出有限项离散分解，也不实现高阶重要性采样。

理论来源：研究工作区中的算法 PDF `Gaussian_Convex_Synthesis_Algorithm.pdf`，以及 `paper_v2/sections/dual_core.tex` 的 coherent frame、合成和四模解析小节。这些论文文件保留在本地，未包含在本次代码上传中；本说明已列出实现及验证所用的主要公式。
