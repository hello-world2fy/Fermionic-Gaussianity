# BCS 配对动力学与 Gaussian 分解：实验设计与实现规划

日期：2026-10-23。状态：第一批通用空间接口和 BCS 基线已实现，L=4 验证见 [REPORT.md](REPORT.md)。一般 PG、多模 Haar 与 Gaussian 合成仍属于后续计划。

本文锁定数学约定、比较对象、数据流和验收条件。先完成小规模精确参考与强基线，再判断通用 Gaussian 合成是否具有足够的精度与计算可行性。有限阶结果称为“有限阶 Gaussian 连续合成”，只有给出足够窄的有效 extent 区间时，才使用“近最优”措辞。

## 1. 已核对的输入与现有实现

- 用户提供了实验设想文本；其中提到的 `bcs_pairing_starter.py` 没有包含在附件中，也未在当前工作区找到。因此 `0.99315`、`0.8604`、`0.9898` 是待复现的参考数值，不作为已验证结果或强制验收值。
- 解析态已对照 Zabalo 等人 [arXiv:2208.07320v2，第 V 节式 (42)–(43)](https://arxiv.org/html/2208.07320v2#S5)；正式发表信息为 [Phys. Rev. B 106, 104513](https://doi.org/10.1103/PhysRevB.106.104513)。文中能级数 `N` 对应本项目 `L`。
- 已检查当前库的 `basis.py`、`gaussian.py`、`sampling.py`、`moments.py`、`api.py`、`extent.py`、`certificates.py`，以及本地 `paper_v2/sections/dual_core.tex` 的 frame、dual 和 constructive synthesis 推导。
- 当前 `GaussianState`、Haar sampler、公开 synthesis/extent API 只支持 **n=4 个费米模式**；允许的阶数为 `(1,2,4)`。`FockBasis` 虽可创建 `n<=12` 的 dense 容器，但这不等于算法支持这些模式数。`moments.py` 和证书入口还硬编码了完整偶宇称向量长度。
- 本实验 **L=4 个配对能级对应 n=8 个费米模式**。不能将配对赝自旋直接当作原库的费米模式，也不能将 `n=4` 的 triality sampler 延用到 `n=8`。
- 本目录隔离保存 BCS 实验及结果；通用能力按第 8 节进入 src。其他探索目录不属于本实验。

## 2. 要回答的问题与允许的结论

按以下顺序回答，避免后一步结论先行：

1. 解析长时间态中的相关相位能否被正确重建，并与直接小体系演化一致？
2. TD-BCS 的误差有多少可以通过重新优化一个 onsite PBCS 构型消除？
3. onsite PBCS 仍解释不了的误差，在允许任意配对矩阵的单个 number-projected Gaussian 后还剩多少？
4. 有限阶连续合成能否在独立验证下重构目标，给出有用的 coefficient cost 与 extent 区间？
5. 在相同占据概率、相同粒子数投影成本的对照下，相关相位怎样改变这些结果？

多起点优化得到的是可实现 fidelity 下界，不能证明单构型的全局失效。minor 证书只针对 onsite PBCS。一般 projected Gaussian 的局部优化失败，不构成必须使用多个一般构型的证明。

分析某个有限 k 的系数分布，也不构成“所有最优分解都必须如此”的结论。后者需要对偶接触集合或其他独立必要条件，是后续理论任务。

## 3. 数学与数值约定

### 3.1 模式、基底和单位

- 配对能级 `j=0,...,L-1`；物理模式 `(j↑,j↓)=(2j,2j+1)`。
- Fock 产生算符按模式编号升序；`s_j^+=c†_(2j)c†_(2j+1)`，`s_j^-=c_(2j+1)c_(2j)`。
- 电子数 `N_f=2m`；pair bitmask 的置位数为 `m`，对应完整 Fock bitmask 中相应的两个模式同时置位。
- pair bitmask 按整数升序排列；所有保存向量必须同时保存这份顺序。
- 取 `ħ=1`，默认带宽 `W=1`，均匀非简并能级 `eps=linspace(-0.5,0.5,L)`。时间报告 `Wt`。
- 内积第一项共轭；所有 fidelity 为**平方重叠**，比较前双方必须归一化。另行报告相位对齐后的 Hilbert 距离。
- 区分三种空间，不以同一个 `dimension` 变量代替：

| L | n=2L | m | D_pair=C(L,m) | D_N=C(2L,2m) | D_even=2^(2L-1) |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 1 | 2 | 6 | 8 |
| 4 | 8 | 2 | 6 | 70 | 128 |
| 6 | 12 | 3 | 20 | 924 | 2048 |
| 8 | 16 | 4 | 70 | 12870 | 32768 |
| 10 | 20 | 5 | 252 | 184756 | 524288 |
| 12 | 24 | 6 | 924 | 2704156 | 8388608 |

`L=2,m=1` 是连接现有四模库的校准点；有非平凡四能级 minor 的首个半填充点是 `L=4,m=2`。

### 3.2 Hamiltonian 与初态

以用户给定的赝自旋 Hamiltonian 为实现定义：

\[
H(t)=H_0-g(t)K,\quad H_0=2\sum_j\epsilon_j s_j^z,\quad K=\sum_{i,j}s_i^+s_j^-.
\]

在 pair basis：

\[
H_{SS}=2\sum_{j\in S}\epsilon_j-\sum_j\epsilon_j-g(t)m,
\qquad H_{S-j+i,S}=-g(t)\quad(j\in S,\ i\notin S).
\]

`i=j` 项不能无意漏掉；固定 m 时它是 `-g(t)m I`，可显式剥离为全局相位，但需在 metadata 中记录。费米矩阵版本的单粒子项使用 `sum eps_j(n_up+n_down)`，从而与上述 `2 eps_j s_j^z` 一致到常数；避免引入额外因子 2。

强吸引极限的固定数目初态为均匀 Dicke 态，且

\[
\psi_0=D_{L,m},\qquad K\psi_0=m(L-m+1)\psi_0.
\]

这提供独立于时间积分器的 Hamiltonian 校验。

两条物理路线分别记录：

- 可解模型：`g(t)=1/(nu t)=eta/(L t)`，尺寸比较固定 `eta`、填充和带宽。
- 恒耦合淬火：先复现 `L=8,m=4,G=0.3,W=1`；尺寸比较另设 `lambda=L G/W=2.4`，不要把固定 G 与固定 lambda 混在同一尺寸趋势里。

### 3.3 解析长时间态与相关相位对照

能级先严格升序排列。设 `r_j=j+1` 为能量排名，`ell_ij=log(|eps_i-eps_j|/E_ref)`，默认 `E_ref=W`。原文解析态可按以下形式实现：

\[
A_S(t)=\mathcal Z^{-1/2}
\exp\!\left[-\frac\pi\nu\sum_{j\in S}r_j\right]
\exp\!\left[-2it\sum_{j\in S}\epsilon_j
-\frac i\nu\sum_{j\in S}\sum_{a\ne j}\ell_{ja}
+\frac{2i}\nu\sum_{i<j\in S}\ell_{ij}\right],
\]

\[
\mathcal Z=\sum_{|S|=m}\exp\!\left[-\frac{2\pi}\nu\sum_{j\in S}r_j\right].
\]

排名从 0 或 1 开始仅差固定 m 下的公共归一化因子。`E_ref` 的改变也只产生公共相位。使用 log-sum-exp 归一化；拒绝简并能级，不能对 `log(0)` 任意加小量。

结构实验统一剥离自由演化和单能级相位，构造

\[
\widetilde A_S(\alpha)=\mathcal Z^{-1/2}e^{-(\pi/\nu)\sum_{j\in S}r_j}
\exp\!\left[\frac{2i\alpha}\nu\sum_{i<j\in S}\ell_{ij}\right].
\]

`alpha=1` 为去除这些 Gaussian 相位后的精确渐近态，`alpha=0` 为同模长 onsite PBCS 对照。中间 alpha 是人工相位插值，不是物理时间轨迹。

所有 alpha 的 Fock 概率及占据数对角观测量完全相同。去除的相位由 Gaussian unitary 实现，因此 extent 不变；在比较非对角观测量时必须恢复相同参考系或同步变换算符。

不要把渐近公式当作任意有限 t 的精确态。小规模交叉验证从 `t0>0` 出发，分别考察 `t0→0`、末时刻增大和积分容差收紧的误差。早期可用 `tau=log(t/t0)` 改写为

\[
i\partial_\tau\psi=[tH_0-K/\nu]\psi,
\]

必要时再去除初态的已知奇异全局相位。不能直接在 `t=0` 调用 `1/t`。

### 3.4 基线的精确定义

记 `P_N` 为完整电子数投影，`P_pair` 为同时固定电子数及每能级局域偶宇称的投影。

| 名称 | 定义及输出 | 结论范围 |
|---|---|---|
| P-TDBCS | 演化 onsite BCS spinor 后，以 `P_N` 投影并归一化，与精确态比较 | 指定平均场轨迹的误差 |
| best-found onsite PBCS | 优化 `A_S∝prod_(j∈S) z_j` | 最佳 onsite fidelity 的可实现下界 |
| best-found PG | 优化任意纯偶 Gaussian 的 `P_N g / norm(P_N g)` | 最佳一般 projected Gaussian fidelity 的可实现下界 |
| unprojected Gaussian | 直接使用规范化完整 Gaussian atom | Gaussian extent 字典；与前两类投影态不同 |

一般 PG 的目标函数必须是

\[
F(g)=\frac{|\langle\psi|g\rangle|^2}{\langle g|P_N|g\rangle}.
\]

分子只需目标 pair amplitudes；分母须包含固定 `N_f=2m` 下**全部** Fock 构型，包括 singly occupied levels。若将分母改为 `||P_pair g||²`，研究对象会变为额外施加 seniority 投影的另一类态，不能仍称作上述 PG。

实现采用反对称配对矩阵的 Pfaffian 振幅或 Bogoliubov 描述。`L=4` 先显式计算全部 `D_N=70` 个振幅作为分母参考。投影概率另行保存；小概率、chart 奇异、收敛失败必须报告，不能用归一化裁剪掩盖。

PBCS 使用复 spinor/多 chart 处理零分量；不能仅用有界 `log z` 排除边界解。PG 使用多 chart 或 Gaussian 流形更新，并纳入 PBCS 候选和 Fock 极限作为初值。固定随机种子、多起点及预算，保存所有起点的值和停止原因。将已知较好候选保留在候选集合中，保证实测 `F_PG_best >= F_onsite_best >= F_TD`，允许浮点容差；这仍不是全局优化证书。

有限 L 需要区分两种平均场约定：

1. 与原文对应的常规 TD-BCS：`dS_j/dt=2(-Re Delta,+Im Delta,eps_j)×S_j`，`Delta=g sum S_j^-`。
2. 作为更强有限体系对照的 product-state TDVP：由完整量子 H 的 product expectation 推导，保留 `i=j` 的有限 L 修正。

后者可用 `kappa_j=u_j* v_j`、`Delta_j=g sum_(a!=j) kappa_a`，单点 spinor Hamiltonian

\[
h_j=\begin{pmatrix}-\epsilon_j&-\Delta_j^*\\-\Delta_j&\epsilon_j-g\end{pmatrix}
\]

来实现。两个 RHS 分开命名和输出，不将差异误记为数值错误。初始化 `u_j=sqrt(1-m/L)`、`v_j=sqrt(m/L)`，投影后与 Dicke 初态完全一致；它不是在固定数目的态上直接施加非线性平均场方程。

## 4. 无需非凸优化的 onsite PBCS 证书

对四个不同能级和不相交的 `|R|=m-2`：

\[
M=A_{Rij}A_{Rkl}-A_{Rik}A_{Rjl}.
\]

PBCS 恒有 `M=0`。实现直接计算差，而不以可能极小的振幅作除数。解析态的交比用于另一条独立测试：

\[
\frac{A_{Rij}A_{Rkl}}{A_{Rik}A_{Rjl}}
=\exp\!\left[\frac{2i}\nu
\log\left|\frac{(\epsilon_i-\epsilon_j)(\epsilon_k-\epsilon_l)}
{(\epsilon_i-\epsilon_k)(\epsilon_j-\epsilon_l)}\right|\right].
\]

**本规划中的代数推导：** 对四个不同构型 `a=Rij,b=Rkl,c=Rik,d=Rjl`，取规范化二副本向量

\[
w=(|a,b\rangle+|b,a\rangle-|c,d\rangle-|d,c\rangle)/2.
\]

`<w|psi⊗psi>=M`，而对任意规范化 onsite PBCS `phi`，`<w|phi⊗phi>=0`。所以由正交投影收缩性：

\[
|M|^2\le1-|\langle\phi|\psi\rangle|^4,
\qquad F_{\rm onsite}^{\max}\le\sqrt{1-|M|^2}.
\]

多个 minor 可通过它们生成的子空间 `W` 加强：先对约束向量去重、正交化，算 `q_W=||P_W psi⊗psi||²`，得到 `F_onsite^max<=sqrt(1-q_W)`。不能直接累加不正交 minor 的平方。

因此第一阶段即可给出 `F_found <= F_onsite^max <= F_minor_upper`。这个界可能很松；是否足够有物理区分度需实测。它不约束任意跨能级 PG。对 `m=0,1,L-1,L` 不使用此四能级判据；这些是 onsite 表示能力的边界校准点。

“一般 PG 也不够”的严格上界若成为论文核心，另立任务寻找适用该族的代数或全局优化证书。不能用 Gaussian extent、onsite minor 或参数维数计数替代该特定态的证明。

附件的维数计数可作为宽松上界使用：`L=10` 时完整 Gaussian ray 流形复维数为 190，而目标 pair 射影空间维数为 251。固定非零粒子数投影后，配对矩阵整体缩放不改变投影 ray，故该投影像的维数上界还可减至 189。无论采用哪一个上界，都只能排除一般目标的可表示性，不能决定本实验特定目标的 fidelity。

## 5. 接入有限阶 Gaussian 合成的正确方式

### 5.1 对称性缩小优化变量，不改变积分字典

令 `P=P_pair`，`u=P u`，完整规范化 Gaussian atom 为 `g`，只保存其子空间坐标 `g_P=P g`。`g_P` 一般**不规范化且不是 Gaussian 态**。

\[
a(g)=\langle g_P|u\rangle,\quad
q_k(u)=d_{2L,k}\,\mathbb E_{\rm Haar}|a(g)|^{2k},
\]

\[
J_k(u)=q_k(u)/(2k)-\operatorname{Re}\langle u|\psi\rangle,
\quad
C_{k,P}(u)=d_{2L,k}\mathbb E[|a|^{2k-2}a\,g_P].
\]

`grad J=C_(k,P)-psi`。禁止将 `g_P` 除以自身范数再送入 moments；禁止用 `D_pair` 替代 `d_(2L,k)` 或完整 frame 维度。

对固定粒子数使用 `exp(i phi (N_f-2m))`，结合各能级局域宇称，形成逐点保持目标的 Gaussian 对称群。Haar 不变性使 J 不变；严格凸性保证唯一 minimizer 落在共同不变子空间。C 的等变性还保证精确积分 `C_k(u)` 没有子空间外分量。

有限 Monte Carlo 的未对称化重构可能仍有子空间外泄漏。处理办法二选一，并记录采用哪一种：

- `L=4` 直接重构全部 128 维，检查完整 residual 与 leakage；
- 对每个 atom 连同系数实施解析的数目/局域宇称群平均，使每个估计项严格落在目标子空间。

后一方法仍有由完整 Gaussian atoms 构成的展开，群平均不会增加 l1 系数成本。但 `P g` 本身不能按单个 Gaussian atom 计数；若输出显式有限和，必须把轨道展开的实际 atom 数或隐式轨道成本报清楚。群平均后的独立样本单元是一个完整轨道，不是轨道内相关的各分量。

### 5.2 通用 Haar sampler 是独立里程碑

计划路线：标准正态实矩阵 QR 并正确修正对角符号，生成 Haar `SO(2n)`；将参考真空协方差变换为 `Gamma=R Gamma_0 R^T`；用相位一致的 Bogoliubov/Pfaffian 表示求目标 sector amplitudes。

必须控制 `det R=+1`，并定义 Majorana、协方差和配对矩阵符号。随机复配对矩阵不是 Haar Gaussian sampler；仅取随机 Slater 或 onsite BCS 也不等价。

单 chart 在近零真空振幅处会病态。建立替代 Fock reference/chart 或稳定转换；不得通过丢弃病态样本改变采样分布。Gaussian ket 的公共相位不影响 moments，但各 Fock 振幅之间的相对符号必须正确。

采样分布的正确性由群不变构造与数学推导保证；统计矩检验只能发现实现错误，不能单独证明 Haar。先在 n=4 与旧 sampler 对照，再在 n=8 验证。

### 5.3 低阶的能力边界与资源预检

Weyl 维数及 `Gamma=d^(1/k)` 已用有理数乘积核算：

| n | k | d_(n,k) | Gamma_(n,k) |
|---:|---:|---:|---:|
| 8 | 1 | 128 | 128 |
| 8 | 2 | 6435 | 80.21845 |
| 8 | 3 | 183040 | 56.77825 |
| 8 | 4 | 3476330 | 43.17976 |
| 12 | 4 | 77266057400 | 527.22645 |

因此 `L=4,k<=4` 是算法验证起点，不能预设它已接近最优；`L=6` 更不能仅靠低阶理论因子承诺精度。`k=1` 使用 `C_1(u)=u`、`u_1=psi` 作为精确单元校准。

先计时单批采样、Pfaffian、moment 和梯度，再据冻结 u 的范数与

\[
V_k(u)=\frac{d_{n,k}^2\|u\|^{4k-2}}{d_{n,2k-1}}
\]

估计正式验证样本数。子空间投影/群平均不会放大此二阶矩上界。若所需样本超预算，记录 `sample_budget_exceeded`，不自动升阶、加样本或放大体系。

### 5.4 训练、验证和 extent 的报告口径

训练、诊断/选择、最终验证采用独立 RNG 流。候选及验证样本数在读取最终验证数据前冻结；验证失败后更换候选属于新实验，需要新预算与新的失败概率分配。

同时输出：`q_k`、`||u||`、目标值、独立 residual 上界、系数质量 `B_k`、有效 extent 上下界、训练/验证时间、样本数、峰值内存和停止原因。

有限数值 u 的质量

\[
B_k=d_{n,k}\mathbb E|\langle g|u\rangle|^{2k-1}
\]

属于 `C_k(u)` 的分解，不能直接标为 `xi_G(psi)`。若已验证 `B_k<=B_up`、`||psi-C_k(u)||<=epsilon` 且 residual 在 pair sector，则用 Fock Gaussian atoms 修补可得

\[
\xi_G(\psi)\le(B_{\rm up}+\sqrt{D_{\rm pair}}\epsilon)^2.
\]

也保留恒真的 Fock 上界 `(sum_S |A_S|)^2`，以及经复核适用于子空间存储的 dual certificate。证书内部使用的完整维度不能只因存储降维而替换。

整个预定正式验收集合统一分配总失败概率 `delta_total=0.05`；不是每个点各取 95% 后声称全图 95%。统计保证明确依赖理想独立 Haar 抽样；不声称包含浮点区间认证。

## 6. 分阶段实验流程

| 阶段 | 规模与任务 | 产物 | 进入下一阶段的条件 |
|---|---|---|---|
| A：约定与参考 | L=2、4；构造 pair H、Dicke 初态、解析态、去相位对照 | conventions、解析检查、精确态文件 | 符号、维度、归一化、Hamiltonian 独立构造一致 |
| B：基线与结构 | 先 L=4，再 L=6、8；精确淬火、两种 MF、PBCS、minor | fidelity 曲线、minor 上界、占据/关联/熵 | G=0、初态、容差收敛全部通过；非凸结果标识完整 |
| C：一般 PG | 先 L=4，含完整 D_N 分母；预算允许才 L=6 | PG fidelity 下界、投影概率、多起点记录 | 与 onsite 嵌套一致、完整与紧凑分母一致、chart 测试通过 |
| D：通用 Gaussian 后端 | n=4 校准后 n=8；群采样、Pfaffian、sector moments | sampler 与 gradient 验证、资源预检 | Haar 构造核对及结构测试通过；验证预算可承担 |
| E：有限阶合成 | L=4；先 k=1,2，通过后 k=3,4 | 连续测度、独立重构、cost 与 extent 区间 | 通过正式验收或诚实记录预算/精度失败 |
| F：物理结构分析 | E 中合格的少量目标与对照 | 加权相对配对结构、跨 k/seed 稳健性 | 区分分解特征与目标态观测量，不做全体最优分解推断 |
| G：有限升级 | L=6、少量点；L=8 合成暂缓 | 扩展可行性报告 | L=4 有稳定结论且资源模型支持，才执行 |

所有阶段是独立命令，不因导入模块或运行单元测试触发完整实验。先 A/B/C，再 D/E；D/E 的采样难度不阻塞前面物理基线的价值判断。

### 首轮固定矩阵

1. **渐近结构：** `L=4,6,8`、半填充、`eta={0.2,0.8,2,8}`；先均匀能级。`eta=0.8` 对齐附件预检，其他参数覆盖较快与较慢过程；不预设误差随 eta 单调。每点比较 `alpha=0,1`，仅在 `eta=0.8` 加 `alpha=0.25,0.5,0.75`。
2. **谱形控制：** 首批稳定后，仅 `L=4,eta=0.8` 加一组固定种子的非均匀、非简并能级；保存全部 eps 数组。
3. **渐近式动态交叉验证：** `L=4,eta=0.8`，计划 `t0 W={1e-4,1e-5}`、`TW={20,40,80}`，允许到预设上限 `TW=160`；分别检查初时刻误差和晚时刻逼近。未收敛则记录，不把某个 T 强行称为无穷远。
4. **恒耦合淬火：** 首先 `L=4` 做 `G=0` 控制和一个相互作用点；再 `L=8,m=4,G=0.3`、`Wt∈[0,20]` 的 81 个输出时间。全部时间输出 TD 基线；PBCS/PG 与合成先用预先固定的 `Wt={0,1,2,5,10,20}`，不根据最终验证结果挑点。
5. **首批合成目标：** 仅 `L=4` 的 Dicke 初态、`eta=0.8` 的 `alpha=0,1`、固定 `lambda=2.4` 淬火的 `Wt=2`。共 4 个目标；先全部 k=1,2，只有预检允许才增加 k=3,4。

这些是分阶段上限，不是首个命令一次跑完的任务集合。若基线差异已小于数值误差或所有合成证书都很宽，报告负结果并停止升级。

### 默认计算预算

- 单进程；BLAS 线程设为 1；不启用 GPU/进程池。
- 结构单元测试：目标 60 秒、512 MiB；不含正式采样实验。
- 每个拟合任务先 16 个固定起点，最多 32 个、总计 60 秒；预算耗尽也保存已有候选。
- 首次 n=8 sampler 预检至多 10,000 个样本，batch 128；检验统计阈值据样本数和解析矩确定，不能要求所有 MC 误差达到机器精度。
- 首个合成目标每阶训练最多 20 万样本，正式验证最多 100 万样本；总时间 180 秒/阶，内存 512 MiB。实际正式 N 由预定证书规则决定；超过上限时停止而不是降低证书标准。
- 整批合成 wall-clock 上限 15 分钟；保存部分进度和未执行项。资源限制只给出停止边界，不保证在这些预算内收敛。

## 7. 观测量、控制与物理解释

每个参考目标保存：完整 pair 振幅、概率、`<n_j>`（pair 占据）、`<n_i n_j>`、`<s_i^+s_j^->`、固定能级二分的 von Neumann entropy，以及与明确参考态的平方重叠。半区熵用 pair 量子比特嵌入构造 reduced density matrix；数目守恒引起的分块不能丢弃。

固定电子数的目标有 `<c_down c_up>=0`。不能直接把它与未投影平均场的非零序参量作数值误差比较；使用数目守恒配对关联，或明确标为对称破缺参考。

四个必须并列的资源控制：

- 单个 Fock Gaussian 态，`xi_G=1`。
- 均匀 Dicke 初态，来自 Gaussian BCS 的数目投影。半填充上界 `2^L/C(L,m)`；`0<m<L` 时一般填充可用 `1/[C(L,m)p^m(1-p)^(L-m)]`、`p=m/L`。这是上界，不是未经证明的等号；空态/满态的 extent 为 1。
- 同模长 `alpha=0` 的 PBCS 对照；不期待其 extent 等于 1。
- `alpha=1` 的目标，以及最佳拟合的 onsite/PG 基线。

相关相位可能提高或降低某个态的 extent；不预设单调性。只有目标下界高于对照上界时，才能在相应联合置信事件上断言 extent 增加。单独比较两个带误差的点估计不够。

对于 Gaussian 成分，定义 `Delta_j(g)=<g|c_(2j+1)c_(2j)|g>`，分析 `Delta_i Delta_j*`、异常协方差模长和相对结构。相位分析须避开零幅度，或以幅度加权相关量替代直接角度直方图。

明确采用**系数质量分布** `dLambda/B ∝ |a|^(2k-1)dmu`，或 **tilted 分布** `pi ∝ |a|^(2k)dmu`；二者不能混用，原始 Haar 直方图也不是合成分布。保存有效样本量、权重集中度和跨 seed/k 的变化。

Gaussian 分解是相干叠加。将 constituent observables 按正权重平均，一般不等于目标态的观测量；物理观测量从重构波函数或含交叉项的表达式计算。分布图只解释该分解的构成。

## 8. 计划代码结构与接口

### 8.1 划分原则与本次调整

本节按后续要求重新划分，替代上一版“把新后端先全部放进实验目录”的安排。**确定属于通用 Gaussian 能力的实现直接进入 `src/gaussian_synthesis/`，BCS 的物理模型、假设和运行策略进入 `experiments/bcs_pairing/`。** 通用实现以小批次加入并经过库级测试，不先复制一份实验版再维护两套。

判定标准是：更换 Hamiltonian、目标态来源或实验参数后，这段代码及其数学前提是否仍然成立。通用不等于必须支持所有模式数和所有对称性；第一版只开放已经验证的能力。

约束如下：

- `src` 不读取 BCS 配置，不认识 `eta`、`G`、Dicke 初态、原文公式或“这张图应怎样解释”。
- `src` 可以接受一般的模式集合、粒子数、occupation masks、目标振幅及资源预算。
- `experiments` 负责产生这些输入并解释输出；不能重写 Pfaffian、Haar、moments 或 extent 证书。
- 维持现有扁平包结构；暂不引入插件注册系统、通用动力学框架或任意 projector 框架。
- 以下为整体架构；第一批实际落地清单与后续边界见 README 和 REPORT，不能将所有规划模块视为已经实现。

### 8.2 放入 src 的通用能力

路径基准为 `gaussian_synthesis/src/gaussian_synthesis/`。

| 模块 | 处理方式 | 通用职责及边界 |
|---|---|---|
| `basis.py` | 保留并小幅扩展 | 费米 bit 顺序、现有 `FockBasis`；增加不枚举 Fock 空间的 `FermionSpace` 元数据，区分物理空间与存储坐标 |
| `sectors.py` | 新增 | 一般 occupation 子空间、固定电子数子空间、restrict/embed、坐标一致性检查；不定义 BCS 的配对规则 |
| `gaussian.py` | 保留 | 现有四模 `GaussianState`、triality 映射及唯一一份 Pfaffian 实现；保留旧入口行为 |
| `bogoliubov.py` | 新增 | 通用纯偶 Gaussian 表示、规范化、chart 切换、指定 Fock 振幅、normal/anomalous covariance；完整 dense 导出仅作有预算的小规模验证 |
| `sampling.py` | 扩展 | 保留 `HaarFourModeSampler`；增加通用 SO/Bogoliubov Haar sampler，输出带空间与测度信息的样本 |
| `projection.py` | 新增 | Gaussian 的数目分布、完整固定电子数投影概率及投影振幅；`NumberProjectedGaussian` 明确不是 Gaussian atom |
| `fitting.py` | 新增 | 模型无关的单个一般 projected Gaussian fidelity 拟合、chart/初值/预算管理、`FitResult`；不含 BCS 专属初值和物理扫描 |
| `symmetry.py` | 新增 | 支持明确数学保证的粒子数与指定模式块宇称约束、带目标 character 的群平均、与坐标子空间的一致性验证 |
| `theory.py` | 保留复用 | Weyl 维数、搜索半径、理论误差因子、二阶矩界；参数始终为实际费米模式数 n |
| `moments.py` | 扩展 | 流式 q/C/B 计算，支持未再归一化的子空间振幅；不推断模型或目标对称性 |
| `objective.py` | 扩展 | 与坐标空间一致的凸目标及梯度；保留 k=1 精确路径 |
| `solver.py` | 扩展 | 与模型无关的有界优化、continuation、诊断；采样器由上层提供 |
| `measure.py` | 扩展 | 连续系数测度、相位对齐、质量/tilted 权重；保留完整 Gaussian 字典语义 |
| `validation.py` | 扩展 | 冻结候选后的独立验证、样本预算、残差范围、统计区间；不选择物理参数点 |
| `certificates.py` | 扩展 | 从有效矩/残差界得到 extent 界，及有明确 Gaussian basis 前提的 residual repair；不能因向量变短而替换理论维数 |
| `api.py`、`extent.py` | 扩展 | 统一 synthesis/extent 入口、能力检查、独立随机流、结果保存；默认调用维持四模兼容 |
| `runtime.py` | 按需从 solver 提取 | 一份共享的 `ResourceBudget` 和资源异常，供拟合与合成使用；旧导入可保留兼容别名 |
| `metrics.py` | 新增小模块 | 平方 fidelity、相位对齐距离、范数检查等模型无关状态比较；不包括 BCS 观测量 |

通用 PG 拟合放进 `src`，因为其输入仅是目标态、电子数和一般 Gaussian 族，其他固定粒子数模型也能使用。**拟合一般 PG 是通用能力；比较哪些 BCS 基线、选择哪些初值和时间点是实验策略。**

协方差到 normal/anomalous correlations 的转换属于通用 Gaussian 表示。具体哪些模式组成 `Delta_j`、分析哪些相位差，以及输出哪种物理图，属于实验。

### 8.3 留在 BCS 实验目录的代码

路径基准为 `gaussian_synthesis/experiments/bcs_pairing/`。

```text
bcs_pairing/
  PLAN.md
  README.md
  METHOD.md
  requirements.txt                    # 实验依赖；运行时另记录准确版本
  configs/
    smoke.json
    asymptotic_pilot.json
    quench_pilot.json
    synthesis_pilot.json
  bcs_pairing/                         # 实验 Python 包
    __init__.py
    conventions.py                    # L/m/eta/G/单位/能级约定
    sectors.py                        # 构造 BCS pair sector 和对应约束
    model.py                          # reduced BCS H0/K、Dicke 初态
    asymptotic.py                     # 论文解析态、相位剥离、alpha 控制
    exact.py                          # 本模型的谱演化与奇异 1/t 演化
    mean_field.py                     # 常规 TD-BCS、有限 L product TDVP
    pbcs.py                           # onsite PBCS 参数化与拟合
    minors.py                         # onsite minor、交比、二副本上界
    baselines.py                      # 编排 TD/PBCS/通用 PG 比较与初值
    observables.py                    # 占据、配对关联、能级分区熵
    synthesis_run.py                  # 将目标/约束/预算交给通用 API
    component_analysis.py            # 相对配对结构及物理解释
    provenance.py                     # 物理配置、代码/数据 hash、批次状态
  scripts/
    run_baselines.py
    run_synthesis.py
    analyze_components.py
    plot_results.py
    verify_results.py
  tests/
    test_pair_sector_model.py
    test_asymptotic.py
    test_exact_mean_field.py
    test_pbcs_minors.py
    test_baseline_integration.py
    test_synthesis_integration.py
  results/<unique_run_id>/
    manifest.json
    states.npz
    baselines.csv
    optimization_starts.json
    synthesis/                        # 通用 SynthesisResult 保存内容
    timings.json
    checks.json
    report.md
    figures/
```

实验 `sectors.py` 只构造 `n=2L`、`N_f=2m`、模式块 `[(0,1),(2,3),...]` 及 pair masks，调用通用 `OccupationSector`/`SymmetryReduction`。它不重新实现 restrict/embed 或群平均。

实验 `synthesis_run.py` 只做参数映射、任务编排与批次置信预算分配，不重新实现 SGD、moments 或证书。实验 `baselines.py` 调用 `src` 的 PG 拟合；上一版实验 `projected_gaussian.py` 不再保留同名完整实现。

`pbcs.py` 和 `minors.py` 首轮留在实验中：它们服务当前选定的 onsite 配对基线及其判据。若后续另一个实验也需要任意配对图案的 PBCS 或一般多项式约束证书，再根据实际需求提取；本轮不为单一用例建立大而泛的 ansatz/约束框架。

`exact.py` 调用 NumPy/SciPy 的成熟演化工具，不把一个 `eigh`/ODE 包装器先扩成通用动力学库。每种模型自己的守恒量、初值奇异性和末时刻收敛仍由实验检查。

### 8.4 最重要的接口约束

**空间与坐标。** `FermionSpace(n, parity)` 只保存物理模式和解析的 `ambient_dimension`，不枚举 `2^n` 个 masks。`OccupationSector(space,masks)` 保存所选正交 Fock 坐标，提供 `coordinate_dimension`；`NumberSector(space,N_f)` 表示完整固定数目 sector。BCS pair sector 与 NumberSector 永远是不同对象。大空间的 dense embed 需要显式预算，不能为了拿到元数据先创建旧 `FockBasis`。

**Gaussian 表示与投影。** 通用 `BogoliubovGaussian` 保留完整规范化 Gaussian 的含义；`amplitudes(sector)` 返回原态在该子空间的振幅，不自动重归一化。数目投影的概率始终针对完整 NumberSector 计算，即使目标只在 pair sector 存储。`NumberProjectedGaussian` 只用于 PG 基线，不允许传入 extent 的 Gaussian atom 接口。

**样本。** 内部 `GaussianBatch` 携带 `space`、`sector`、`reference_measure`、`amplitudes` 和采样来源。moments 的维数常数读取 `space.n`；数组长度由 `sector.coordinate_dimension` 校验。未明确为 Haar 的批次不能直接走 Haar 证书路径。旧四模 ndarray 调用通过兼容适配进入同一计算内核，不维护两份 q/C/B 公式。

**对称性与证书范围。** 任意 Fock 子空间本身不足以证明 C 没有外部泄漏。首版 `SymmetryReduction` 只支持已经推导并测试的数目和模式块宇称构造，不接受一个用户手填的 `symmetry_verified=True` 来跳过检查。完整重构保证必须来自完整空间验证，或与目标、坐标严格匹配的有效对称性降维。其余结果明确标为 `projected_only`，不能出具完整态重构/extent 证书。

**拟合输出。** `FitResult` 保存 `best_found_fidelity`、候选参数、投影概率、restart 历史、预算与停止原因，`certified_upper_bound` 默认空值。实验层负责保留 PBCS 候选、比较嵌套基线，不把局部最优值改名为全局最大值。

**API 兼容。** 保留 `synthesize(psi,basis,config)` 和 `estimate_extent(psi,basis,config)` 的现有调用，逐步增加对 sector 坐标及显式 `reduction` 参数的支持。旧四模默认继续使用原 sampler。新后端先开放 n=8、k=1,2，k=3,4 通过新增测试后开放；配置中请求尚未支持的能力时显式失败，不静默删阶或回退成 onsite 采样。整数 k 的数学合法性与后端实际支持范围分开检查。

**结果与置信预算。** 通用结果保存 u、重构、B、extent 界、残差范围、RNG spawn keys、资源与 schema_version。实验 manifest 引用这些结果并加上 eps/eta/G/t/reference frame、输入和源码 hash。整批总 delta 由实验分配至各目标，库再在该目标内部按阶数/统计量分配；不能每个目标各用默认 0.05 后宣称整批 95%。

**依赖方向。** `scripts → 实验包 → gaussian_synthesis 公共接口 → 通用数学内核`。`src` 不反向 import `experiments`，库级测试也不以实验数据或 BCS 模块作为前提。实验通过项目安装导入库，不复制源码或依赖模块加载顺序。

### 8.5 测试和依赖也按职责分开

库级测试放在 `gaussian_synthesis/tests/`：sector 映射、通用 Gaussian 表示、Pfaffian、数目投影分母、Haar/frame、moments/梯度、有效对称性降维、非法降维拒绝、PG 拟合约定、证书和旧 API 回归。测试用随机/解析构造的小态，不导入 BCS 模型。通用采样器正式验收如需较长运行，可在 `gaussian_synthesis/examples/validate_general_sampler.py` 提供显式入口，不能藏进日常单元测试。

实验级测试放在本目录 `tests/`：pair masks 映射、BCS H 符号、Dicke 初态、原文解析态、两种平均场、相位对照、onsite minor、物理观测量以及调用通用库的端到端小样例。

保留基础库 NumPy 依赖。通用拟合若使用 SciPy，规划为 `optimize` 可选依赖并仅在调用拟合时加载；导入基础合成库不强制安装 SciPy。BCS 实验声明所需的优化/ODE/绘图依赖，Matplotlib 不进入核心运行依赖。固定种子、独立验证及资源限制的实现复用通用库；实验只配置具体预算和组合实验的停止策略。

### 8.6 本次相对于上一版的模块迁移表

| 上一版放在实验中的能力 | 新归属 |
|---|---|
| `gaussian_backend.py` | src 的 `bogoliubov.py`，复用 `gaussian.py` 的 Pfaffian |
| `haar.py` | 扩展 src 的 `sampling.py` |
| `projected_gaussian.py` | src 的 `projection.py` + `fitting.py`；BCS 编排进 `baselines.py` |
| `symmetry.py` | 通用机制进 src；具体 pair 块配置进实验 `sectors.py` |
| `moments.py`、`synthesis.py`、`certificates.py` | 扩展现有 src 模块；实验只保留 `synthesis_run.py` |
| `observables.py` 中的通用 fidelity/距离 | src 的 `metrics.py`；物理关联与能级分区仍在实验 |
| `provenance.py` 中的库结果/RNG 信息 | 复用 src 结果；实验仅补充物理元数据与批次清单 |

## 9. 验收清单与停止条件

| 检查 | 方法 | 计划标准 |
|---|---|---|
| 基底/费米符号 | pair H 对照独立完整 Fock 算符构造；L=2,4 | 矩阵最大差 <=1e-12 |
| 初态 | 归一化、固定数目、K 本征方程、t=0 投影态一致 | residual <=1e-12 |
| 非相互作用 | 精确相位解、谱演化、P-TDBCS 三方对照 | `1-F <=1e-10` |
| 有限 L 平均场 | product TDVP RHS 对照完整 product-state 能量的变分导数；两类 MF 各自守恒量 | 导数差分相对误差 <=1e-6；分别验证常 G 下各自能量和平均数目守恒 |
| 常 H 参考 | `eigh` 与独立 `expm_multiply`/ODE 小点对照 | 相位对齐态差 <=1e-9 |
| 时间积分 | 范数、能量（仅常 H）、总数及容差收紧 | 收紧后关键 fidelity 变化 <=1e-7；单独报告初/末时截断误差 |
| 解析相位 | 两种公式实现、改变 E_ref、自由相位剥离 | 归一化态/概率关系 <=1e-12；近零量用绝对误差 |
| minor | 随机 PBCS、解析交比、相位对照；聚合约束正交化 | PBCS minor <=1e-12；上界不得低于可实现 fidelity 超过数值容差 |
| PG 分母 | L=4 完整数目 sector 枚举与紧凑公式对照；含泄漏例子 | 相对/绝对混合误差 <=1e-10；不得误用 pair 分母 |
| Gaussian 表示 | Pfaffian 平方与 determinant、完整小态范数、纯协方差、边界 chart | 稳定测试点 <=1e-10；病态点单列条件数与状态 |
| Haar/frame | n=4 与旧 sampler 对照；一阶 frame；固定 Gaussian 的高阶 overlap moment | 采用预先分配显著性及解析二阶矩；`E abs(overlap(g,g0))^(2k)=1/d_(n,k)` |
| 梯度 | 固定样本实/虚方向中心差分；独立 Monte Carlo frame 检验分开 | 适当差分步长区间内相对误差 <=1e-6 |
| 对称性 | 小体系完整梯度与 sector 梯度、精确群平均对照 | 群平均后 leakage <=1e-10；未平均 MC 泄漏单列 |
| 合成 | 冻结候选、独立样本、预定置信预算 | 首轮 residual 上界 <=0.05 才标记 reconstruction_accepted |
| extent 精度 | 同时有效上下界，另检查 Fock/控制态界 | `(upper-lower)/upper<=0.1` 才标记 extent_target_reached |
| 复现 | manifest、代码 hash、数组 hash、单独 verify | 已保存输出能独立重算关键标量；失败项全部保留 |

上述阈值是实施前的计划标准；不能在看到正式验证失败后修改阈值并追认通过。ODE 的奇异初时刻误差、渐近误差、拟合误差、采样误差、有限 k 误差分别报告。

允许状态包括：`passed`、`best_found_only`、`failed_acceptance`、`not_certified`、`sample_budget_exceeded`、`resource_limit`、`numerical_error`、`not_run`。重构通过与 extent 区间够窄是两个独立状态。

第一轮成功不要求预定物理观点成立：如果最好的单构型仍十分准确，或低阶合成没有提供有用增益，应保留这个结果。若只看到 TD-BCS 失败而重优化 PBCS 成功，应明确结论限于平均场轨迹；若 PG 优化尚无全局证书，不声称排除了所有单个一般 PG。

## 10. 推荐实施顺序

按以下可独立审阅的批次实施；不先重构整个 src：

1. **坐标基础与第一份物理基线：** src 增加最小的 space/sector 和状态比较能力；实验实现 pair sector、model/asymptotic/exact/mean_field/pbcs/minors。通过各自测试后，以 L=4 生成 baseline report。此时不改变旧 synthesis 行为。
2. **一般 PG 基线：** src 增加 Bogoliubov 振幅、完整数目投影和通用 PG 拟合；实验只添加基线编排。先验证 L=4 的 70 维数目投影分母，再评估扩大模式数。
3. **通用采样与 moment 内核：** src 增加 n=8 Haar 和有效对称性降维，统一新旧 moments 实现；验证完整 128 维与 6 维 pair 坐标路径的一致性及旧四模回归。
4. **API 与证书接入：** 扩展 src 的 solver/measure/validation/certificates/api/extent，先 k=1,2。实验通过显式入口运行少量目标，并按第 6 节资源门槛决定是否加 k=3,4。
5. **物理解读：** 实验添加 component analysis、图表与报告。达到前述升级条件后再考虑 L=6。

这样可在投入通用多模采样前，先确认目标态、有限体系平均场约定、可实现 fidelity 与严格 onsite 上界。第一轮结果将决定是否有充分理由进入更大 L、更多 k 或有关所有近最优分解的理论工作。
