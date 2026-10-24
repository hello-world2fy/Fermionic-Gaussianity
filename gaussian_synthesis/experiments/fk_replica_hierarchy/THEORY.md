# 本轮积分与证据口径

## Gaussian 流形上的精确单平面条件更新

对归一化 Gaussian ket g，Majorana 双线性 G=γₐγᵦ 满足 G²=−I，故 exp(θG)g=(cos θ)g+(sin θ)Gg。记 a=⟨u|g⟩，b=⟨u|Gg⟩，则 overlap 平方是实二阶型：

\[
r(\theta)=|a|^2\cos^2\theta+|b|^2\sin^2\theta
 +2\operatorname{Re}(\bar a b)\sin\theta\cos\theta.
\]

Haar 测度在每个该单参数子群的圆轨道上的条件测度均匀。对上述 2×2 实对称矩阵作解析对角化，得到非负 λ₋≤λ₊ 和旋转中心 θ₀。对均匀角 α，以概率

\[
[(\lambda_-+(\lambda_+-\lambda_-)\cos^2\alpha)/\lambda_+]^\beta
\]

接受，并使用 θ=θ₀+α。该拒绝采样给出精确的 r(θ)^β 条件分布，适用于整数和半整数等任意 β≥0。它保持目标测度，但有限次数的复合更新不等于已经从全流形平衡分布独立采样。程序随机排列全部 Majorana 平面并重复完整 sweeps，记录祖先数与独立种子结果。

## 温度归一化和重构

在相邻温度上估计 Z_(β+δ)/Z_β=E_πβ exp(δ log r)，用 log-sum-exp 计算，再系统重采样并作条件更新。下一温度由相邻重加权 ESS 确定。粒子的 ESS 不是独立样本数；自适应粒子系统的点估计不提供严格统计包围。

对 FK singlet witness，可用已证明的 k=1 frame identity 和 k=2 Cartan projector 提供 Z₁、Z₂ 锚点。锚点之前的随机估计及其偏差仍保存。高阶归一化不能因此视为已知。解析基准特意不向采样器提供这些锚点，用来独立检查从 Haar 起点的归一化。

若 a(g)=⟨g|u⟩，则

\[
C_k(u)=d_{8,k}Z_k(u)\,E_{\pi_{k,u}}[g/\overline{a(g)}],
\quad B_k=d_{8,k}Z_{k-1/2}(u).
\]

对固定 proposal 的矩积分使用其相对于 Haar 的密度倒数，优化器不构造 Sym^k(C³) 的大矩阵。合并多个独立训练种子时保留每批自己的 Z 和 proposal 密度；优化完冻结 witness，以完全独立的随机流验证。

## 额外反幺正对称性

在当前 Fock 规范中，T=−P_A K（局域 A 宇称乘以 occupation 复共轭）固定三个 singlet 基矢，因而 T(Sz)=S\bar z。局域宇称与复共轭都保持 Gaussian 集合和 Haar 测度。物理目标的 singlet 坐标为实，严格凸目标的唯一极小值也可取实。`--time-reversal` 将经验积分与其 T 像成对平均，使优化和验证保留这一精确对称性。该方法是所有 Gaussian 原子和其 T 像的平均，没有限制接触构型。两项成对样本不是独立样本，不增加 ESS；也不能降低标量归一化误差。未作 T 平均的结果保留作对照。

## 与高阶点估计独立的全字典证书

`fk4/certificates.py` 用 70 位有向区间算术计算前轮经过整数恒等式验证的 6×6 q₂ 矩阵。对任意 z，取 w=S z/N₂(Sz)，再留出舍入余量，并重新区间验证 q₂(w)≤1。因此完整 Gaussian 字典满足 h_G(w)≤1，得到对保存目标向量的可行对偶下界。这一证书使用的是 order 2 的归一化；不能把它标成更强的高阶 L_k 证书。

S 是前轮整数证明中明示的根式基底；证书的精确定义是该基底乘以保存的 binary64 坐标。用于展示的 full-Fock 浮点向量仅为近似。目标范围是保存的 binary64 Fock 振幅所精确定义的向量；若要把结论转为数学上精确 Hamiltonian 基态的认证，还需独立验证的本征态误差上界。

## 解析基准与外部依据

- 四模 magic 矩、B 和质量集中式由原方案及主库解析基准给出。通用积分器不调用这些公式。
- 纯 Gaussian 的 q_k=1、L_k=τ=1；有限阶 canonical B=d_(n,k)/d_(n,k−1/2)>1。后者不能被误判为 non-Gaussian extent。
- t=0 的两个 magic 团簇满足 τ=2、ξ=4。其 extent 乘法性的适用范围可参照 [Reardon-Smith, 2024](https://arxiv.org/abs/2407.20934)：任意纯态与四模宇称本征态的 tensor product。实际本地约定中 magic 在 odd 局域宇称扇区，不能错误地只保留 EE 字典。
- Gaussian/FLO 资源量与相位敏感分解背景：[Reardon-Smith, Oszmaniec, Korzekwa, Quantum 8, 1549 (2024)](https://arxiv.org/abs/2307.12702)。

## 尚须完成的理论工作

高阶积分误差的可靠包围、接触集合的全局支持函数、受限 eta/能窗 overlap 上界，以及固定 witness 的接触 Hessian 和几何维数仍需进一步推导和实验。此文件不宣称这些结论已经成立。
