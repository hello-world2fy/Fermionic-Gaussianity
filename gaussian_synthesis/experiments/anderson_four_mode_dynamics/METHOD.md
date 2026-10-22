# 连续 Gaussian 测度如何接入演化

## 模型与闭环

模式顺序为 `(d_up, bath_up, d_down, bath_down)`。一个自旋简并 impurity 轨道加一个自旋简并 bath 轨道，共 4 个 fermionic modes。取 ħ=1，杂化幅度 v=0.6，bath 能量 0：

\[
H_G= v\sum_\sigma(d_\sigma^\dagger b_\sigma+\mathrm{h.c.})
 +\epsilon'_d(n_{d\uparrow}+n_{d\downarrow}),\qquad
H=H_G+U(n_{d\uparrow}-1/2)(n_{d\downarrow}-1/2).
\]

主实验 ε′d=0、U=4。初态为 U=0、每自旋占据一个单粒子基态的 Slater determinant。固定中心化 Hamiltonian 中的 ε′d，瞬时开启 U。换成未中心化形式，impurity 能级是 εd=ε′d−U/2；因此主实验保持粒子空穴对称，不是固定裸 εd 的 quench。非对称对照取 ε′d=0.4。

用偶宇称 Fock 空间的 8 个振幅保存 Gaussian 原子，态的物理子空间 N↑=N↓=1 是 4 维。这里选择稠密振幅是为了可独立核对四模原型；没有声称已经获得随模式数多项式扩展的实现。

每个 Strang 步为

\[
S_{\Delta t}=e^{-iH_G\Delta t/2}
\left[\cos(U\Delta t/4)I-i\sin(U\Delta t/4)Z_0Z_2\right]
e^{-iH_G\Delta t/2}.
\]

I 和 Z₀Z₂ 保持 Gaussianity；每个非平凡时间步将分支数翻倍。每 4 步把**当前近似 Gaussian sum** 合成为 target，调用原 decomposition，离散化后替换当前分解，再继续传播。精确轨迹在整个计时和传播结束后才构造，没有作为压缩目标或反馈校正。

## 原算法和实际调用

完整原接口路径为 `gaussian_synthesis.api.synthesize(target, BASIS, config)`，本轮固定 k=2、800 次 SGD、训练 batch=256；保留原接口的独立验证。较轻版本直接调用原 `projected_sgd`，256 次、batch=128，使用上一 witness 经自由演化的结果作 warm start。

四模确定性版本调用仓库原有 `four_mode_landscape.exact_solver.solve_exact(target)`，在全部 16 个实分量上求解原来的 k=2 凸目标：

\[
q_2(u)=\|u\|^4-\frac{|u^TJ u|^2}{8},\qquad
f_\psi(u)=\frac14q_2(u)-\operatorname{Re}\langle u,\psi\rangle,
\]

\[
C_2(u)=\nabla_{\mathbb R}f_\psi(u)+\psi
=\|u\|^2u-\frac18(u^TJu)J\bar u.
\]

J 是原实现的 Gaussian quadric 矩阵，Gaussian 原子满足 gᵀJg=0、‖g‖=1。停止条件是 ‖C₂(u)−target‖≤10⁻¹⁰。这个求解器没有 Anderson 专用的解析态输入。

另保留 `api_polished` 和 `sgd_polished`：实际先跑原 SGD，以它输出的 witness 为初值，调用原 `exact_k2` 与 `exact_hessian` 做 Newton 残差校正（≤10⁻¹¹），然后离散化。原 API 的训练和独立验证没有从该版本的计时中删除。这样可以验证：低精度 SGD 后直接两节点压缩的失败确实来自求解残差。

## 路径一：真正从连续测度采样

原 `GaussianCoefficientMeasure` 在 k=2 时给出

\[
d\nu_u(g)=35|\langle g,u\rangle|^3d\mu(g),\quad
\widetilde g=\frac{\langle g,u\rangle}{|\langle g,u\rangle|}g,
\quad \int \widetilde g\,d\nu_u(g)=C_2(u).
\]

用原 Haar sampler 提议，接受概率为 (|⟨g,u⟩|/‖u‖)³。因此接受点精确服从 νu/νu(1) 的分布，不需要预先知道归一化常数。每次接受 128 个原子；独立调用原质量估计器，使用 1024 个 Haar 样本估计总质量 B，构造 (B̂/128)∑g̃j。所有拒绝提议、接受点、质量估计和运行时间均记录。

直接使用该 Monte Carlo 和式会积累误差。改进版以这些**实际采到的原子**为字典，OMP 选择并做复系数最小二乘重拟合，最多保留 8 个；目标仍是当前近似演化态。在 8 维空间，8 个合适的字典原子可以张成目标，因而重拟合能去掉采样和 SGD 的重构误差。这说明精度主要来自重拟合；凸优化是否改善字典质量，要另外对照，不能仅凭恢复精度判断。

两个消融对照分别是均匀 Haar 字典，以及直接取 u=target 的重叠加权采样。后者完全不解凸问题，但保留相同的采样、质量估计和重拟合流程。

对称性版本先把每个采样原子投影到 N↑=N↓=1，再用其 2×2 系数矩阵的 SVD 分成至多 2 个 Slater 原子。**投影后的单个 ket 未必 Gaussian**；分裂步骤不可省略。只在这一步使用逐原子的 Schmidt 分解；另有独立的目标态 Schmidt 基线。

## 路径二：两节点的一阶向量矩压缩

这是四模几何带来的确定性改进：保持连续测度代表的 ket，但不保持整个测度或其总质量，也不是抽取两个 Monte Carlo 样本。

令原代码的变换 W 满足 WᵀJW=I。将 z=W†u 乘以一个整体相位，使其实部和虚部正交，写作

\[
u=e^{i\phi}W(ar+ibs),\qquad r,s\in\mathbb R^8,
\quad r^Tr=s^Ts=1,\quad r^Ts=0,\quad a,b\ge0.
\]

则

\[
C_2(u)=e^{i\phi}W(Ar+iBs),\quad
A=\frac{a(7a^2+9b^2)}8,\quad B=\frac{b(9a^2+7b^2)}8.
\]

构造两个正交、归一化 Gaussian 原子

\[
g_\pm=W(r\pm is)/\sqrt2,
\qquad
c_\pm=e^{i\phi}(A\pm B)/\sqrt2,
\]

便有 c₊g₊+c₋g₋=C₂(u)。实 witness 的退化情形任选一个与 r 正交的 s；Gaussian witness 的一个系数为零，自动删去该原子。实现从**原求解器输出的 witness** 构造 r、s 和 C₂(u)，没有调用 target 的 Schmidt 分解。

这是对一阶向量矩的有限原子替换，通常改变总系数质量；不能宣称准确重现原连续 νu 的所有统计量。求解残差会被原样保留。因此 `SGD → 两节点` 在本轮失败，`SGD → Newton 校正 → 两节点` 才通过精度标准。

为避免把四模几何的收益误归于凸求解，还实现了 `direct_quadric`：直接对 u=target 使用相同几何构造，再在两个原子上拟合 target，无凸求解。它同样能表示四模目标，也比完整凸求解快。两节点结果不能外推为更高模式数的 Gaussian rank 定理。

## 误差与系数成本

未在循环内强制归一化，防止掩盖幅值误差。每次压缩记录

\[
\delta_j=\|\psi_j^{\rm before}-\psi_j^{\rm after}\|_2.
\]

Strang 每步是 unitary，故累计压缩误差满足

\[
\|\widetilde\psi(t)-\psi_{\rm Strang}(t)\|_2\le\sum_{j\le t}\delta_j
\]

（数值检查允许舍入误差）。它与 Trotter 误差分开记录。观测量和 infidelity 用归一化后的当前态计算，范数本身也保存。

报告的系数成本是实际离散分解的

\[
B_{\rm disc}^2=\frac{(\sum_j|c_j|)^2}{\|\sum_jc_jg_j\|^2}.
\]

它是所表示的归一化态的 Gaussian extent 上界，**没有把它称为求出的最优 extent**，也没有把连续测度质量、Monte Carlo 质量估计和重拟合后的系数成本混为一谈。

原 API 的 `certified` 只对应单次调用给定的误差容限（这里 0.1）及失效概率。多次调用没有设置整条轨迹的联合置信预算；最终精度来自四模完整向量的独立数值核对，不声称整段轨迹获得了原 API 的 95% 严格认证。
