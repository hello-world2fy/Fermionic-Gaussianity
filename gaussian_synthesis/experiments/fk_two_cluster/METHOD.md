# FK 两团簇实验：数学约定与证据口径

本实验固定相互作用系数 `V=1`，输入参数 `t` 即 `t/V`。模型参考解、Gaussian 采样、有限阶连续合成和成本上界分别验证。本文说明公式与实现的关系，不将一次双精度计算提升为区间算术证明。

## 1. Fock、Majorana 与 covariance

物理系统含 8 个复费米模式、16 个 Majorana。整数占据掩码的第 `j` 位对应模式 `j`，A 团簇为低 4 位，B 团簇为高 4 位。因此全空间向量的索引为 `16*b+a`，张量积使用 `np.kron(state_B, state_A)`。even 坐标按满足偶数占据的掩码递增排列，维数为 128；完整 ED 空间维数为 256。

约定

\[
\gamma_{2j}=c_j+c_j^\dagger,\qquad
\gamma_{2j+1}=i(c_j^\dagger-c_j),\qquad
\{\gamma_a,\gamma_b\}=2\delta_{ab}.
\]

协方差采用

\[
\Gamma_{ab}=\frac{i}{2}\langle[\gamma_a,\gamma_b]\rangle.
\]

真空满足 `Gamma[2*j,2*j+1]=-1`。纯 Gaussian covariance 满足 `Gamma.T=-Gamma`、`Gamma@Gamma=-I`。一般非 Gaussian 纯态不满足后一条件；程序不把它当作精确目标态的恒等式。

全局偶宇称与局域偶宇称不同。全局偶态包含 `even_A⊗even_B` 和 `odd_A⊗odd_B` 两个正交扇区。跨团簇 hopping 同时翻转两侧局域宇称。完整字典使用 global-even Gaussian；乘积字典必须包含 EE 与 OO。单个奇宇称占据 Fock 态也是合法的纯 Gaussian 态。

## 2. 相互作用、三维目标与对称性

`conventions.clifford()` 通过八元数虚单位的左乘表示给出七个实反对称矩阵，直接检查

\[
J_\alpha J_\beta+J_\beta J_\alpha=-2\delta_{\alpha\beta}I_8.
\]

定义

\[
Q_{x,\alpha}=\frac{i}{4}\gamma_x^T J_\alpha\gamma_x,\qquad
H(t)=-\sum_{x,\alpha}Q_{x,\alpha}^2
     +\frac{it}{2}\sum_{a=1}^8\gamma_{A,a}\gamma_{B,a}.
\]

局域相互作用的三个谱值为 `-28,-7,-4`，重数分别为 `1,8,7`。从局域唯一最低态的乘积出发，hopping 生成三维 Krylov 空间；两项 Hamiltonian 分别在该空间闭合。程序由原始矩阵构造正交基 `S`，再比较所得矩阵与

\[
H_{\rm singlet}(t)=
\begin{pmatrix}
-56&\sqrt2t&0\\
\sqrt2t&-14&\sqrt{14}t\\
0&\sqrt{14}t&-8
\end{pmatrix}.
\]

Krylov 闭合本身不足以证明所有对称 witness 都在此三维空间。因此 `symmetry.py` 额外构造 21 个对角 Spin(7) 生成元：先以 `J_alpha J_beta` 构造局域二次生成元，再将 A、B 的生成元相加。在完整 even 128 维空间计算它们的公共 kernel，检查维数恰为 3、kernel projector 与 `S S†` 一致，并检查生成元作用在 `S` 上的残差。输出包括 kernel 阈值、最小正 Gram 本征值和 projector 误差。

这是可重复的双精度公共 kernel 检查。没有使用有向舍入或精确有理秩证明，因此报告名称明确保留 `numerical`。

对各个实际扫描点，还将三维最低态能量与完整 256 维 ED 的最低本征值比较；能隙取完整空间最低两个本征值之差。三维矩阵的第二本征值不能代替完整体系激发能隙。

## 3. 质量、跨团簇关联与 Wick 偏差

将 covariance 分块为

\[
\Gamma=\begin{pmatrix}A&C\\-C^T&B\end{pmatrix},\qquad
m_{A,\alpha}=\frac18\langle J_\alpha,A\rangle_F,
\qquad \eta=\frac{\|C\|_F^2}{8}.
\]

`mass_a`、`mass_b` 表示质量向量的长度平方，`mass_dot` 表示两向量内积。对于纯 Gaussian 原子，Clifford 正交性与 Bessel 不等式给出

\[
\|m_A\|^2+\eta\le1,\qquad \|m_B\|^2+\eta\le1.
\]

这里局域约化态允许是混合 Gaussian 态；不使用仅适用于四模纯 even 态的 magic-overlap 公式。

真实四点关联由完整算符直接计算：

\[
C_Q=\sum_\alpha\langle Q_{A,\alpha}Q_{B,\alpha}\rangle.
\]

一般 covariance 的 Wick 预测为

\[
C_Q^{\rm Wick}=4m_A\cdot m_B
-\frac18\sum_{\alpha,a,b,c,d}
(J_\alpha)_{ab}(J_\alpha)_{cd}C_{ac}C_{bd}.
\]

该式保留局域均值与交换收缩，不预设目标态为 singlet。先用它计算 `wick_deviation=C_Q-C_Q_Wick`，再独立检查本模型对称目标上的简式 `C_Q_Wick=-7 eta`。随机 full/product Gaussian 原子则直接比较显式四点算符期望与一般 Wick 公式，检查 Gaussian 原子的偏差为零。

局域 Gaussian Wick 能量为

\[
\langle-\sum_\alpha Q_{A,\alpha}^2\rangle
=-7+\frac38\|A\|_F^2-12\|m_A\|^2.
\]

完整 Gaussian 能量还包括 B 侧和 `t*Tr(C)/2`。互补界可由
`A=aJ_1, B=-aJ_1, C=-sI, a²+s²=1` 饱和，得到

\[
E_G(t)=\begin{cases}-32-2t^2/9,&0\le t\le9,\\-14-4t,&t\ge9.\end{cases}
\]

单 Gaussian 的结构变化与完整有限体系的非简并、有限能隙并不矛盾。实验只将前者称作 Gaussian 变分临界结构。

## 4. 两种字典、参考测度与 frame 常数

完整字典从 `SO(16)/U(8)` 的 global-even Gaussian 轨道 Haar 采样。实现用实 Gaussian 随机矩阵 QR、对角符号修正与行列式修正生成 Haar `SO(16)`，再通过 Givens 旋转的 Spin lift 得到带相位的 Fock ket。不能以任意分布的反对称 BCS 参数代替 Haar。

对于 `n` 模固定宇称 Gaussian 轨道，定义

\[
d_{n,k}=\prod_{1\le i<j\le n}\frac{2n-i-j+k}{2n-i-j}.
\]

full 字典使用 `D_full(k)=d_(8,k)`。

product 字典先用独立等概率 Bernoulli 选择 EE 或 OO，再在该扇区独立采样两个四模 Haar Gaussian。对每个局域宇称，k 次复制的 moment operator 是 `P_(4,k)/d_(4,k)`。因此全局乘积测度的 moment operator 为

\[
\int(|g\rangle\langle g|)^{\otimes k}d\mu_{\rm prod}
=\frac{P_{EE,k}+P_{OO,k}}{2d_{4,k}^2}.
\]

两个 projector 的支撑正交，由此得到

\[
D_{\rm prod}(k)=2d_{4,k}^2.
\]

特别地 `D_prod(1)=2*8²=128`，与完整 global-even 空间的一阶 frame identity 一致。遗漏因子 2 或仅采 EE 都会改变问题。product 的 OO 样本由一个局域 Majorana 将 even Haar 原子映到 odd 轨道；采样的协方差和 Fock ket 使用同一变换。

## 5. 三维 witness 与有限阶合成

对不变目标，精确 Haar 目标函数在对角 Spin(7) 下不变。`k>=1` 时其严格凸性保证最优 witness 唯一，故 witness 必须属于全部不变空间，可写为 `u=S z`，`z∈C³`。

记录每个完整 Gaussian 原子的三个重叠 `h_j(g)=<g|s_j>`，令

\[
a(g)=\sum_jh_j(g)z_j,\quad
q_k(z)=D(k)\mathbb E|a|^{2k},\quad
J_k(z)=q_k(z)/(2k)-\operatorname{Re}\langle Sz|\psi\rangle.
\]

训练在六个实坐标上求经验目标的驻点。投影 `S†g` 不做归一化；归一化将改变积分、frame 与字典成本。字典仍然是完整八模或指定乘积 Gaussian 轨道，不是三维 Gaussian 子流形。

完整重构向量与系数质量为

\[
C_k(u)=D(k)\mathbb E\big[|a|^{2k-2}a\,g\big],\qquad
B_k(u)=D(k)\mathbb E|a|^{2k-1}.
\]

只有精确满足 `C_k(u)=psi` 时，这个 canonical 分解直接表示目标态。经验三维梯度很小，只说明投影经验目标优化成功，不能推出独立完整重构准确。

`k=1` 单独采用精确 frame 恒等式：`u_1=psi`、`C_1=psi`。如果同时计算随机一阶重构，它用于采样诊断；不能把其 Monte Carlo 波动解释成解析一阶恒等式失败。其余阶数使用独立全 even 128 维重构验收。

系数质量分布是 `dP∝|a|^(2k-1)dmu`。它不同于 `|a|^(2k)` 的 tilted 分布。相干重构还包含 `a/|a|` 的相位。真实四点关联不能用原子四点期望的正权平均代替，因为不同原子间的相干矩阵元不可省略。

## 6. 有限 k 的解析成本偏差与采样噪声

即使目标本身是 normalized Gaussian `g0`，精确最优 witness 为 `u_k=g0`、`q_k=1`，canonical 成本仍不等于真正最小成本 1。

令 `X=|<g|g0>|²∈[0,1]`。整数矩由 coherent frame 恒等式给出 `E X^k=1/d_(n,k)`。该矩序列等于独立 `Beta(a,1)` 变量乘积的矩，其中 `a=2n-i-j`。紧区间矩序列唯一，因此对所有实 `s>=0`，合法延拓为 `E X^s=1/d_(n,s)`。由此

\[
B_k(g_0)=\frac{d_{8,k}}{d_{8,k-1/2}}.
\]

| k | 完整字典 Gaussian 端点 canonical B |
| --- | ---: |
| 1 | 9.66394858 |
| 2 | 6.50414490 |
| 3 | 5.03020352 |
| 4 | 4.17277425 |
| 6 | 3.21723304 |
| 8 | 2.69899866 |

首次达到 `B<=1.2` 的整数阶数是 71。这是有限阶方法本身的偏差，不能通过提高样本量消除。低阶 pilot 不承担得到近最优 canonical 成本的承诺。对于固定局域扇区的 product Gaussian probe，则有 `E X^s=1/(2d_(4,s)²)`，用于 product 字典的相应校准。

对 Gaussian full probe，设 `Y=d_(8,k)|a|^(2k-2)a g`，则独立 N 样本平均满足精确的均方误差关系

\[
\mathbb E\|\widehat C-g_0\|^2
=\frac{d_{8,k}^2/d_{8,2k-1}-1}{N}.
\]

`q` 的标准误为

\[
\operatorname{SE}(\widehat q)
=\sqrt{\frac{d_{8,k}^2/d_{8,2k}-1}{N}}.
\]

在 `N=65536`，随机 `C` 的 RMS 误差在 `k=1,2,3,4` 分别约为 `0.04402,0.05862,0.10255,0.19697`；`k=2` 的 q 标准误约为 `0.01290`。因此 `1e-2` 独立重构目标可能在固定预算内失败，即使采样器与优化器均正确。

质量权重的 population ESS 比例可由相同矩公式计算。高 k 权重集中会使普通 Haar importance weighting 极其低效；低阶失败后不得自动升阶或无限扩样本。

## 7. 采样校准、开发验收与统计保证

这三类判断使用不同字段、阈值和结论：

1. **预先定义的 sampler calibration。** 固定 probe、种子、N 和阶数。标量解析矩以 `|estimate-truth|<=5*SE_exact` 检查；完整随机重构以 `||C_hat-C_exact||<=6*RMSE_exact` 检查。它们判断偏差是否与已知采样波动相容，不承诺 `1e-2` 精度。
2. **开发重构验收。** 冻结 witness 后，用与训练独立的样本检查完整 even 128 维残差。`k>=2` 的经验残差须不超过 `1e-2` 才标记 `reconstruction_accepted`。这是开发精度门槛，不是概率保证。另报 singlet 投影残差与 singlet 外分量，检查平方和等于完整残差平方。
3. **正式统计置信保证。** 需要独立验证和事先分配的失败概率，并将统计半径加到经验残差上。配置中的总失败概率 `0.05`、统计半径目标 `0.005` 不会因出现在配置里而自动成立。样本上限内未达到半径，就报告未通过。

`5 SE` 与 `6 RMSE` 是预注册诊断倍数，不能自动解释成多重比较后的 95% 联合置信保证。前者没有假设高阶权重近似正态，后者也不是某个分量的六倍标准差。它们不能被拿来认证 Gaussian extent。

在独立 Haar 假设下，一种保守重构半径来自

\[
\mathbb E\|Y\|^2\le
\frac{D(k)^2\|u\|^{4k-2}}{D(2k-1)},\quad
e_C:=\sqrt{\frac{D(k)^2\|u\|^{4k-2}}
{D(2k-1)N\delta}}.
\]

于是 `P(||C_hat-C_k(u)||<=e_C)>=1-delta`。若使用此类公式，应保存公式、`delta`、分配规则和样本数；预算不足时不能只保留较小的经验误差。反复查看同一验证样本并继续优化会破坏冻结 witness 的独立性要求。

需要区分：模型与约定错误、有限 k 偏差、有限训练样本偏差、优化残差、独立验证的 Monte Carlo 波动、目标本征态误差及浮点误差。`optimizer_converged=true` 不等于其余误差都通过验收。

## 8. 不依赖高阶 canonical 的合法成本基线

令 `G=Spin(7)` 对角作用、`U(h)` 为其 Gaussian 单位表示，`Pi` 是全部不变子空间的正交投影。紧群 Haar 平均给出

\[
\Pi=\int_G U(h)dh.
\]

对于 normalized Gaussian `g`，`p=Pi g` 通常不是 Gaussian，且通常未归一化；但

\[
p=\int_G U(h)g\,dh
\]

是系数总质量为 1 的合法连续 Gaussian 表示。因此若三维 SOCP 找到

\[
\psi\simeq\sum_jc_j\Pi g_j,
\]

就得到实际 Gaussian 轨道平均的表示，成本不超过 `sum|c_j|`，而不是将非 Gaussian 投影原子冒称为免费原子。对 product 字典也成立，因为对角作用分别在 A、B 内部，是局域 Gaussian 变换，保持乘积结构和各自宇称。

保存 normalized 原始 `g_j`、未归一化 `Pi g_j`、`S`、系数和完整重构残差，可以明确重建这个连续表示。有限候选池的最优值只是全字典成本上界，不是全局最优 extent 的数值解。不能将投影原子归一化后仍按单位 Gaussian 成本计价。

## 9. 残差修补与受限字典分离

任意 occupation Fock ket 都是 Gaussian。若已知表示 `psi_hat` 的系数成本 B，且完整 even 残差为 `r=psi-psi_hat`，则

\[
\tau_G(\psi)\le U=B+\|r\|_{1,\rm Fock}
\le B+\sqrt{128}\|r\|_2,\qquad \xi_G(\psi)\le U^2.
\]

对乘积字典，even Fock ket 也属于 EE 或 OO，因此同样适用。若数值目标距真正目标的二范数有上界 `epsilon_state`，再加 `sqrt(128)*epsilon_state`。

对四对 Gaussian Schmidt 参数 `s_i=sin(2theta_i)`，有

\[
\eta=\tfrac14\sum_i s_i^2,\qquad
L(g)=\prod_i\sqrt{1+s_i}\le(1+\sqrt\eta)^2.
\]

核范数三角不等式给出受限字典 `eta<=eta0` 的下界

\[
\tau_{\eta\le\eta_0}(\psi)\ge
\frac{L(\psi)}{(1+\sqrt{\eta_0})^2},\qquad
L(\psi)=\operatorname{Tr}\sqrt{\rho_A}.
\]

有目标误差时，分子改为 `(L_numeric-4*epsilon_state)_+`。完整字典修补上界低于受限字典下界时，说明低跨团簇关联字典不能达到这份完整字典表示的成本；无需声称上界已全局最优。

精确输入下这些不等式成立。当前双精度本征态、投影矩阵、原子、SOCP 系数及残差尚未作区间包围；小的 `numerical_margin` 仅是数值缓冲。因此输出只允许 `numerical_evidence`，保留 `strict_interval_certified=false` 和 `statistically_certified=false`。真正的严格计算证明还需对这些输入作有向舍入、区间包围或精确代数重建。

## 10. 可解释的物理结论边界

canonical 分布变化、近最优成本和对所有低成本表示都成立的必要性是三个不同结论。低阶分布的峰移动不能单独证明真实最优支持发生转变；最佳分解也可能不唯一。受限字典分离只说明某些低关联限制代价过高，并不自动证明双峰、临界点或解释真实 Wick 偏差峰。精确物理曲线与成本证据均应独立报告，无须等待分布叙事成功。
