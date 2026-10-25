# 连通两次转移与径向反馈的对照定义

本轮扩展的是已有变分 coupled-cluster 思路，不声称发明新的传播原则。
所有模型由相同费米海开始，用 Hermitian Dirac–Frenkel 方程传播。
低秩高阶 CC 激发已有研究，例如 [Lesiuk](https://arxiv.org/abs/2205.02369)；
时间依赖变分框架见 [Kvaal 等](https://arxiv.org/abs/2410.24192)。

## 冻结什么、允许什么运动

设 $K_1(T)=\sum_{ph}T_{ph}s_p^+s_h^-$。
固定二次转移矩阵 $D_\alpha(P,H)$ 定义 $K_{2,\alpha}$，使
$K_{2,\alpha}|F\rangle$ 在 m=2 扇区等于该矩阵。
所有这些激发算符彼此交换且幂零。

\[
|q\rangle\propto e^{\beta_{\hat m}}
\exp[K_1(T)+\sum_\alpha\lambda_\alpha K_{2,\alpha}]|F\rangle,
\qquad\beta_0=0.
\]

T 是完整 $r\times r$ 复矩阵；每个 m 有独立复径向参数；
各 $\lambda_\alpha$ 也可运动。只有 $D_\alpha$ 在训练 t=1 冻结。
初值 T、$\lambda$、$\beta$ 全零，所有模型物理初态精确相同。
因此此处不会重复第五轮不同拟合初态造成的误差抵消。

原始波函数按次数满足 Euler 递推

\[
m\psi_m=K_1\psi_{m-1}
+\sum_\alpha 2\lambda_\alpha K_{2,\alpha}\psi_{m-2}.
\]

更一般的 order-a 固定簇在递推中乘 a。
解析 Jacobian 由对应激发作用得到；径向导数为 $\partial_{\beta_m}\psi=\Pi_m\psi$。
因此未截断的切空间包含每个计数投影算符作用，DF 方程满足所有计数扇区的 Ehrenfest 方程。
数值 metric 截断可能破坏这个等式，所以在实际轨迹上另检验。
计算使用完整 $\binom{2r}{r}$ 配对空间，不是可扩展成本结论。

## 三种两次转移方向

在参考重叠非零的图中，定义

\[
T_1=\psi_1/\psi_0,\qquad
C_2=\psi_2/\psi_0-\operatorname{perm}_2(T_1).
\]

比较直接训练态的 $C_2$、k=2 合成 witness 的同一张量，以及普通 pair-CC
训练预测态上的 Hamiltonian 导数 $\dot C_2$。第三者不需要训练目标精确态。
witness 的系数不等同于 canonical 分布采样原子。

全部方法先共享一个来自 leading Riccati 通道的径向 m=2 方向。
从剩余张量的 SVD 取 q=1、2、4 个方向，再对共享方向和先前方向正交化。
增加 q 是在既有条件上的结构诊断，不被称作独立的新参数预测。

若共享形状为 rank-one $ab^\dagger$，
仅减去该矩阵仍保留了 $a\,\delta b^\dagger+\delta a\,b^\dagger$ 等单侧形状变化。
真正去掉这些切向变化需计算

\[
(I-aa^\dagger)C_2(I-bb^\dagger).
\]

训练时直接态、witness、Hamiltonian 张量中这部分的平方范数占比分别为
0.1795%、0.1910%、0.0542%。不能把整个 $C_2$ 都解释为新的独立带间关联方向。

## 任意轨道计数的独立实现

对成对目标态，记 $n_i=\langle x_i\rangle$、$C=\operatorname{Cov}(x)$、

\[
R_{ij}=\langle s_i^+s_j^-\rangle,\quad q_{ij}=\langle x_ix_j\rangle.
\]

任意 $2L$ 物理模式投影 P 的块为 $P_{ij}\in\mathbb C^{2\times2}$，
$d_i=\operatorname{Tr}P_{ii}$。对 $A=c^\dagger Pc$ 有

\[
\langle A\rangle=d^Tn,\qquad
\operatorname{Var}(A)=d^TCd+
\sum_{i<j}\left[\|P_{ij}\|_F^2(n_i+n_j-2q_{ij})
+4\operatorname{Re}(\det(P_{ij})R_{ij})\right].
\]

后半项是完整的破对输出及其干涉。独立完整 Fock 作用检验该式，
并对复 QR 参数化的 Torch 版本做数值方向导数检查。
八块 GPU 上共 512 个局部起点没有给所测晚时目标找到旧计数界的新违反；
它不证明全部测量子空间不存在违反。加强后的加权判据另见
[WEIGHTED_GAUSSIAN_BOUND.md](WEIGHTED_GAUSSIAN_BOUND.md)。

## 数值图与误差控制

径向坐标在 Fermi 初态退化，且 T 的整体缩放与径向参数有精确规范冗余。
Gram 矩阵会平方条件数；可改为直接切矩阵 SVD，保留相同平方奇异值阈值。
误差控制可使用归一化物理波函数的切向误差（含相位和范数），
并额外检查坐标到态的有限位移非线性余项，避免用空扇区坐标误差主导步长。
这只是数值积分的误差度量；必须用更严格容差、不同求解器和小规模精确解复核。
原始坐标误差的两组加严尝试因过慢终止，日志和终止原因保留，不计为通过的复核。
