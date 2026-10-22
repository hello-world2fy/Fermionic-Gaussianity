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
| `validation.py` | 冻结候选后的独立重构验证与 extent 矩/残差区间 |
| `certificates.py` | 将矩区间和残差上界转换为对偶范数与 extent 上下界 |
| `extent.py` | extent 配置、逐阶优化与验证、停止规则及结果保存 |
| `api.py` | 对外导出 synthesis/extent 接口，管理原分解流程 |

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

## 从波函数估计 Gaussian extent

新增 `estimate_extent(psi, basis, ExtentConfig(...))`，实现论文 Theorem 2 的有限阶对偶估计，并对近似优化解加入数值误差证书。它接收同一 Fock 基底中的一般四模偶 parity 复振幅向量，不要求输入态是 magic state，也不使用解析最优方向。当前版本继续只允许 `(1,2,4)` 的递增子集，不实现一般模式数或高阶矩确定性递推。

在导入 NumPy 前设置上文的线程限制后，可以调用：

```python
import numpy as np
from gaussian_synthesis import ExtentConfig, FockBasis, estimate_extent

basis = FockBasis(4, "even")
psi = (basis.ket(0) + basis.ket(15)) / np.sqrt(2)
config = ExtentConfig(
    orders=(1, 2, 4),
    relative_tolerance=0.1,
    total_failure_probability=0.05,
    seed=20261022,
)
result = estimate_extent(psi, basis, config)
print(result.lower_bound, result.upper_bound)
print(result.interval_certified, result.target_reached, result.stop_reason)
result.save("results/my_extent")
```

`interval_certified` 表示结果具有所报告的区间证书；`target_reached` 表示区间还满足 `upper_bound / lower_bound <= 1 + relative_tolerance`。这两个状态分别记录。计算资源耗尽或配置阶数用尽时，返回已经得到的区间及停止原因；不会为了通过验收自动追加验证或突破资源预算。

任何有效输入都先具有确定性的 Fock 基础区间。即使首个随机验证因预算限制未执行，这个基础区间仍然有效，此时置信度为 1。应用随机证书后，置信度使用预先分配的完整失败概率预算，不能根据提前停止后实际执行的阶数缩减该预算。`stages[*].applied` 标明哪些证书进入最终区间；`sample_budget_exceeded`、`resource_limit`、`numerical_error` 或 `inconsistent_bounds` 会停止计算并保留此前区间。

### 有限阶对偶问题

令

\[
N_k(w)=q_k(w)^{1/(2k)},\qquad
t_k(\psi)=N_k^*(\psi)=\sup_{w\ne0}
\frac{|\langle w|\psi\rangle|}{N_k(w)}.
\]

论文的有限阶控制给出

\[
t_k(\psi)^2\le\xi_G(\psi)
\le\Gamma_{n,k}\,t_k(\psi)^2,
\qquad \Gamma_{n,k}=d_{n,k}^{1/k}.
\]

若计算全部阶数，`Xi_K = max(t_k**2 for k in 1,...,K)`；默认只计算稀疏阶数 `(1,2,4)`，不能称为完整 `Xi_4`。每个已计算阶数仍提供有效的单阶证书。

优化器复用前述凸目标 J，在完整复向量空间中求解。精确最优点满足 Cₖ(u*)=ψ，并且 `t_k**2 = q_k(u*)**((2*k-1)/k)`。近似解不直接使用这个等式作证书，也不把近似 witness 的商乘以 Gamma 当作上界。

### 近似解的 extent 证书

冻结 u 后，独立验证给出 C 的估计与半径

\[
\|\widehat C-C_k(u)\|\le e_C,
\qquad \rho=\|\psi-\widehat C\|+e_C.
\]

由 Euler 恒等式 \(q_k(u)=\operatorname{Re}\langle u|C_k(u)\rangle\)，同一事件还给出

\[
q_k(u)\in
\left[\operatorname{Re}\langle u|\widehat C\rangle
      -\|u\|e_C,\quad
      \operatorname{Re}\langle u|\widehat C\rangle
      +\|u\|e_C\right].
\]

该区间与确定性界 `d_(n,k)*||u||**(2*k)/D**k <= q_k(u) <= ||u||**(2*k)` 相交，得到 `[q_-, q_+]`。q 和 C 使用同一个置信事件，不额外分配失败概率。这里验证的是优化向量的 `q_k(u)`，与原分解报告中的目标态 `q_k(psi)` 不同。

记 `D=2**(n-1)`、`c_k=sqrt(D)*d_(n,k)**(-1/(2*k))`、`a=(2*k-1)/(2*k)`。frame 恒等式和 Jensen 不等式给出 `N_k(v) >= ||v||/c_k`，所以 `N_k^*(e) <= c_k*||e||`。同时范数梯度恒等式给出 `N_k^*(C_k(u)) = q_k(u)**a`。于是

\[
\ell_k=\max\left\{
\frac{|\langle u|\psi\rangle|}{q_+^{1/(2k)}},
q_-^a-c_k\rho,\ 0\right\},\qquad
T_k=q_+^a+c_k\rho
\]

满足 `ell_k <= t_k(psi) <= T_k`，最终得到

\[
\ell_k^2\le\xi_G(\psi)\le\Gamma_{n,k}T_k^2.
\]

以上公式针对非零 witness；u=0 时单独使用 qₖ(0)=0 和 Cₖ(0)=0，避免除零。这是为论文有限阶定理补充的近似解证书。`certificates.py` 只进行确定性的数学转换，`validation.py` 负责独立采样和统计保证。合并各阶区间时，还加入 Fock 基础界 `||psi||_2**2 <= xi_G(psi) <= ||psi||_1**2`。单个 Fock ket 是 Gaussian 态，因此这个上界适用于所有输入振幅。

默认两个非平凡阶数分别分配 δ=0.025；每阶验证样本数由冻结 u 和余量 0.025 在看到验证数据之前确定，单阶最多 1200 万样本。k=1 的 q 和 C 精确计算，不采样。训练、诊断和验证使用独立随机数流。与重构证书一样，联合至少 95% 的保证是统计保证，依赖理想独立 Haar 采样，不包含浮点舍入的严格区间认证。

### 独立的 extent 完整实验

从 `gaussian_synthesis/` 目录执行：

```bash
python run_tests.py
python examples/four_mode_extent.py --full --output results/my_magic_extent
```

示例必须显式使用 `--full`，并拒绝覆盖已有输出目录。它使用与分解实验相同的 180 秒 / 512 MiB 限制和单进程线程设置；不会由日常单元测试自动运行。

`result.save(directory)` 保存 `summary.json`、`history.csv`、`intervals.csv` 和 `vectors.npz`。完整实验额外保存 `comparison.json`、`comparison.csv` 和中文 `report.md`，所有解析比较在 `estimate_extent` 返回后进行。

四模 magic state 的解析对偶值是

\[
t_k(M)^2=q_k(M)^{-1/k}
=2\left[\frac{(k+3)(k+4)(k+5)}{60}\right]^{-1/k}.
\]

| k | 解析 t² | 真实 extent |
|---|---|---|
| 1 | 1.000000 | 2 |
| 2 | 1.069045 | 2 |
| 4 | 1.174790 | 2 |

完整实验检查：数值 u 距离解析最优解不超过 0.01，witness 商点估计距离解析 t² 的相对偏差不超过 2%，每阶 t² 区间包含解析值，最终 extent 区间包含 2。点估计使用随机 q 估计作分母，本身没有上下界保证；其相对误差仅作回归检查。

默认低阶实验预期得到有效但仍较宽的 extent 区间，**不预期达到 10% 相对间隙**。脚本退出码表示上述回归检查是否通过，`target_reached` 单独报告精度目标是否达到。准确重构 Cₖ(u)≈ψ 不等于有限阶 t² 已接近 extent；高阶矩计算需要后续工作，不能靠自动扩大采样量绕过预算。

已保存的一次完整运行：[报告](results/magic_extent/report.md)、[解析对照](results/magic_extent/comparison.csv)。默认种子得到区间 `[1.1680441895, 2.0000000000]`，联合置信度至少 95%；解析回归通过，10% 目标未达到。计算耗时 7.77 秒，峰值 RSS 62.52 MiB。新增功能后的 40 个日常测试全部通过，总耗时 1.14 秒，峰值 RSS 55.19 MiB，见 [测试结果](results/unit_tests_extent.json)。

本地论文来源：[Theorem 2 正文](../paper_v2/main.tex) 和 [对偶范数补充材料](../paper_v2/sections/dual_core.tex)。论文未随代码仓库分发，这些链接仅在原研究工作区有效；运行、安装与测试不读取论文文件。
