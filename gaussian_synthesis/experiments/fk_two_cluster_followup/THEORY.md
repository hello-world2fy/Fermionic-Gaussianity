# 有限阶精确矩、群平均与成本结构

本轮继承原实验的 Fock 位序、Majorana、Clifford 和 Spin(7) singlet 基 `S`。以下“精确矩”指由精确代数公式计算 Haar 积分，不表示双精度舍入已得到区间包围。原 pilot 的采样失败结果保持不变。

## 1. q_k 的有限矩阵并不自动提供精确积分

令 `u=S z`、`h_j=<g|s_j>`，则 `q_k(z)=d_(8,k) E|h·z|^(2k)` 是三变量复齐次多项式，双次数为 `(k,k)`。它可写成 `Sym^k(C³)` 上的 Hermitian PSD 矩阵，尺寸 `binomial(k+2,2)`：对于 `k=1,2,3,4` 为 `3,6,10,15`。

这说明最多只需保存 15×15 的矩阵，但不说明其中积分系数容易精确计算。对 Monte Carlo 值作插值、拟合或有理数猜测仍然包含采样误差。本轮只实现已推导出确定性 projector 的完整字典 `k=2`；`k=3,4` 暂不声称解决。

## 2. Spin(16) 对称平方的 Gaussian projector

八模 even Fock 空间是 Spin(16) 的 128 维半旋量表示 `S_+`。对称平方分解为

\[
\operatorname{Sym}^2(S_+)=\mathbf1\oplus\Lambda^4(\mathbb R^{16})
\oplus\Lambda^8_+(\mathbb R^{16}),
\]

维数为 `1+1820+6435=8256`。Gaussian coherent state `g⊗g` 位于最高权 Cartan 分量 `Lambda^8_+`，其维数等于 `d_(8,2)=6435`。Haar 不变性与 Schur 引理给出

\[
d_{8,2}\int(|g\rangle\langle g|)^{\otimes2}d\mu(g)=P_G.
\]

取 Hermitian 生成元 `T_ab=(i/2) gamma_a gamma_b`。半旋量的二次 Casimir 为 30；张量两份的总 Casimir

\[
\mathcal C=\sum_{a<b}(T_{ab}\otimes I+I\otimes T_{ab})^2
\]

在上述三个分量上的本征值分别为 `0,48,64`，因此

\[
P_G=\frac{\mathcal C(\mathcal C-48I)}{1024}.
\]

无需构造 16384×16384 的矩阵。定义

\[
\Lambda=\sum_{a=1}^{16}\gamma_a\otimes\gamma_a.
\]

它将 even⊗even 映到 odd⊗odd，反向作用为其伴随。由 CAR，

\[
\Lambda^2=256I-4\mathcal C.
\]

故在对称 even-even 张量上

\[
\boxed{P_G=(I-\Lambda^2/64)(I-\Lambda^2/256)
=I-\tfrac5{256}\Lambda^2+\tfrac1{16384}\Lambda^4.}
\]

此多项式只在对称张量子空间上作为 Gaussian projector 使用。两份是普通复制 Hilbert 空间，不能在这里额外加入空间分割的 graded-tensor 符号。

Majorana 在 Fock 基中是带相位的置换，因而 `Lambda` 的作用是 16 次行列置换及相位乘法。对 `Sym²(span S)` 的六个基向量，每个只需保存 128×128 的数组；几次这样的数组远低于 512 MiB，不构造巨型复制算符。

## 3. 6×6 moment、梯度与重构

采用正交对称张量基

\[
V_{ii}=s_i\otimes s_i,\qquad
V_{ij}=(s_i\otimes s_j+s_j\otimes s_i)/\sqrt2\quad(i<j),
\]

和 normalized Veronese 坐标

\[
v(z)=(z_0^2,\sqrt2z_0z_1,\sqrt2z_0z_2,z_1^2,
\sqrt2z_1z_2,z_2^2)^T.
\]

程序计算 `A=V† P_G V`，从而

\[
q_2(z)=v(z)^\dagger A v(z).
\]

`matrix` 保存已乘 `d_(8,2)` 的 A；`haar_moment_matrix=A/6435` 保存未放大的 Haar moment。不能再次乘 6435。

在原模型固定的 S 相位约定下，计算还给出闭式

\[
A=\frac1{128}\begin{pmatrix}
99&0&0&1&0&3\\
0&100&0&0&4\sqrt7&0\\
0&0&96&4\sqrt{14}&0&0\\
1&0&4\sqrt{14}&99&0&25\\
0&4\sqrt7&0&0&124&0\\
3&0&0&25&0&99
\end{pmatrix}.
\]

这不是从 Monte Carlo 结果拟合或有理数猜测得到的。`integer_fk_matrix_proof()` 用显式 Fock 支撑构造根式基底：`s0=N0/2`、`s1=i*N1/sqrt(8)`、`s2=N2/(2sqrt(7))`，其中 N 全为整数向量。对应六个对称 tensor 各自抽出常数

\[
(1/4,\ i/8,\ 1/(4\sqrt{14}),\ -1/8,\ i/(8\sqrt7),\ 1/28),
\]

剩下整数 tensor。`Lambda` 在复制空间只是带正负号的整数置换；所以

\[
16384P_G=16384I-320\Lambda^2+\Lambda^4
\]

的作用与六维 Gram 完全用 int64 算术计算，并逐项比较一个固定整数矩阵。中间量保守界低于 `2^40`，不存在 int64 溢出。只在恢复根式和比较浮点模型时使用平方根。这样可区分“明示根式基底上的精确代数恒等式”与“该基底和浮点构造 S 的数值匹配”。两者都保存可复核数组。

对六个实坐标 `x=(Re z,Im z)`，若 `J=partial v/partial x`、`K=partial²v/partial x²`，则原目标 `q_2/4` 的梯度、Hessian 为

\[
\nabla_x(q_2/4)=\tfrac12\operatorname{Re}(J^\dagger Av),
\]

\[
[\nabla_x^2(q_2/4)]_{rs}=\tfrac12\operatorname{Re}
[(J^\dagger A J)_{rs}+\sum_a\overline{K_{a,rs}}(Av)_a].
\]

这与原 Newton solver 的接口和尺度一致。把实梯度前后两半合成复三维向量得到 `S†C_2(u)`；因为完整目标与 witness 对称，`C_2(u)` 本身在 singlet 空间，完整 128 维重构是 `S C_reduced`。

可复核内容包括：Majorana 置换与独立 dense 算符相等、`Lambda P_G V≈0`、`P_G²V≈P_GV`、moment Hermiticity/PSD、Gaussian probe 的 `q_2=1` 和 `C_2(g)=g`、梯度 Hessian 的有限差分、原 solver 的完整重构残差。

此外，求解后的重构还在完整 tensor 空间独立计算：

\[
C_2(u)_i=\sum_j[P_G(u\otimes u)]_{ij}\overline{u_j}.
\]

这一步不插入 S 投影。输出比较它与三维梯度嵌入及精确目标，因而完整重构误差不是由预先删掉 singlet 外分量人为设为零。

## 4. 精确整数矩没有自动解决 B_k

`B_2=d_(8,2) E|<g|u>|³` 是 overlap 平方的半整数矩，不由这一个 degree-four moment 矩阵唯一决定。因此后端的 `mass`、`ess` 明确返回 `None`。不能把 `q_2` 的某个幂冒充 canonical B，也不能因此宣布最优 Gaussian extent 已确定。

如需成本，可独立求积或估计半整数矩，或者构造该半整数函数在有界区间上的多项式上下包络，再用更多经过验证的整数矩积分。后者仍需要新的误差控制。

已知 Gaussian 端点的低阶 canonical 成本远高于真实成本：八模 `k=2` 约为 6.504，`k=8` 约为 2.699。修复积分误差不会消除有限阶偏差；因此本轮精确 k=2 的价值是区分积分、优化与方法偏差，不是通过盲目提高 k 来追逐低成本。

## 5. 群平均 / Rao–Blackwell 的合法性

令 `Pi=S S†=∫Spin(7) U(h)dh`。对 singlet witness，`a(g)=<g|u>` 在对角群轨道上不变。因此原随机重构样本 `Y(g)=d|a|^(2k-2)a g` 可作条件平均

\[
Y_{RB}(g)=\int Y(U(h)g)dh=\Pi Y(g).
\]

期望保持不变，方差不增，并精确去除 singlet 外的方向噪声。这是对完整 Gaussian 群轨道的解析积分，不是把投影后的非 Gaussian 向量当作新 Gaussian 原子。

合法 RB 的完整 128 维估计为 `Pi C_hat`，其残差数值上等于原来保存的 singlet 残差。旧数据可做只读后验诊断，但不可把它改写成原始方法预注册验收已通过。新方法需单独标记、比较并保存独立验证。

标量 q、B 以及群不变量 `mass_a,mass_b,mass_dot,eta` 不随群轨道方向变化，所以群平均不会自动降低这些量的稀有重叠方差，也不自动解决低 ESS。精确 k=2 作为参考可以直接区分 RB 后剩余积分误差与经验 witness 的训练偏差。

## 6. 有限轨道与列生成的证据边界

每个未归一化 `Pi g_j` 都有单位成本的真实 Gaussian 轨道平均表示。有限 SOCP 给出可行表示成本；加完整 Fock 残差修补后是合法成本上界。连续优化某个结构族的参数只会改善这个上界，不能自动证明该结构族已覆盖全部最优 Gaussian 原子。

有限字典的对偶向量 w 只满足已加入原子的约束。要把它变为完整 Gaussian 字典的 lower certificate，需要证明

\[
\sup_{g\in\mathcal G_8}|\langle Sw|g\rangle|\le1.
\]

多初值局部搜索发现的最大 overlap 是该 supremum 的下估计，不能当作全局上界来缩放 witness。若有可靠上界 `M_up`，则 `w/M_up` 才给出完整字典可行对偶。列生成能提供越来越好的候选上下信息，但全局最优性仍需全局 oracle、解析 support function 或可靠的区间上界。

## 7. product 成本的解析线索：仅限物理实坐标基态

原数据提示物理基态 `psi=z0*s0+z1*s1+z2*s2`（取 `z0,z2>=0`，z1 为实）满足

\[
\tau_{prod}(\psi)\stackrel{?}=
\sqrt8|z_1|+\max\{2z_0,z_0+\sqrt7z_2\}.
\]

五个已计算点的有限字典上界与右边在约 `4e-10` 内一致；这不是任意复三维态的公式。相对复相位可能改变最优成本，因此不得未经证明推广。

候选证明分三步：局域宇称分支正交使其成本相加；s1 的八项 product Fock 展开达到其 Schmidt 下界；其余二维分支结合 Schmidt 下界与局域 magic 的最大 Gaussian fidelity 1/2，给出 `max(2z0,z0+sqrt7*z2)`。上界可以用保存基底中已出现的 twirled product Fock 投影极点构造，而不需要随机候选。

在三维基态方程中，分支交点 `sqrt7*z2=z0` 给出 `E=-64`，再由特征方程得到 `t/V=10`。这是表示成本公式的可能折点，不是物理有限体系能隙闭合，也不是 `t/V=9` 的 Gaussian 变分临界点。本轮优先实现 k=2 reference；此线索仍需逐步完成下界和构造的正式证明后才可报告为定理。

当前具体 Clifford 表示的局域 magic 在 odd 扇区：s0 与 s2 属于 OO，s1 属于 EE。用另一个局域 Majorana 约定可以交换名称，但不能省掉任一乘积宇称分支。
