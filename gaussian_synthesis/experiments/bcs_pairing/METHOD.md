# 第一批实现的约定与验证

## 空间接口

`FermionSpace(n,parity)` 只保存模式信息和解析的完整 parity 维度，不枚举基底。`OccupationSector` 保持传入的唯一 bitmask 顺序，存储维数为 `coordinate_dimension`，完整空间维数为 `ambient_dimension`。`NumberSector` 在枚举前计算二项式维数并检查容量，包含该电子数下所有构型。

`embed` 需要显式 `max_dimension`；`restrict` 和 `embed_into` 均保留原振幅的尺度。两种 parity 的完整基底按整数 mask 升序排列，每两个相邻整数中恰有一个属于指定 parity，因此其坐标索引为 `mask >> 1`。这与旧 `FockBasis` 的排序逐项测试一致。

`OccupationSector` 只是坐标选择，不宣称是合法的 Gaussian 字典或对称性证书。本批没有把它接入旧四模 synthesis 入口。核心包新增部分只依赖 NumPy。

## BCS Hamiltonian 与动力学

模式为 `(j↑,j↓)=(2j,2j+1)`，Fock 产生算符按模式编号升序。`H=H0-gK`，`H0=2 sum eps_j s_j^z`，`K=sum_(i,j) s_i^+s_j^-`。固定 m 下保留对角项 `-gm I`。Hamiltonian 通过独立的逐位费米产生/湮灭运算验证，包含反对易符号。

常耦合演化采用 Hermitian 谱分解，并与独立 matrix exponential 对照。`1/t` 演化在 `tau=log(t/t0)` 下使用 DOP853；减去 `K` 在 Dicke 态上的本征值 `m(L-m+1)I`，只消除已知的奇异公共相位。输出与真实态只差这个公共相位，fidelity 和数目守恒观测量不受影响。原文渐近公式不是有限时间精确态，末时刻误差与初时刻截断误差单列报告。

渐近态使用原文 [arXiv:2208.07320v2 式 (42)-(43)](https://arxiv.org/html/2208.07320v2#S5)。程序拒绝能级简并；对数的能量参考单位只改变固定 m 态的公共相位。数值结构扫描移除自由演化及单能级相位，以 alpha 调节剩余 Jastrow 相位。alpha 是人工结构参数，不是物理时间。

## 两种平均场约定

每个 spinor 为 `(u_j,v_j)`，`kappa_j=u_j* v_j`。

- 常规 TD-BCS：`h_j=[[-eps_j,-Delta*],[-Delta,eps_j]]`，`Delta=g sum kappa_j`。
- product TDVP：`Delta_j=g sum_(a!=j) kappa_a`，`h_j=[[-eps_j,-Delta_j*],[-Delta_j,eps_j-g]]`。

后者保留有限 L 的自作用修正，并以完整 product-state 量子能量的切向变分导数交叉检查。两类方法分别检查自身守恒能量、每个 spinor 范数和平均总粒子对数。product TDVP 不保证比常规 TD-BCS 的投影态 fidelity 更高。

默认 MF 积分精度 `rtol=1e-11,atol=1e-13`，与 `1e-12,1e-14` 对照。保存积分原始 spinor，不通过每步重新归一化掩盖漂移。比较态由显式固定 m 投影后归一化得到；投影概率为零时拒绝输入。

## onsite PBCS 与上界

使用 `u_j=cos(theta_j)`、`v_j=sin(theta_j) exp(i phi_j)`，`theta∈[0,pi/2]`；最后一个 phi 置零。投影 amplitudes 是每个能级取 u 或 v 的乘积，因此端点不需要除以 u。共同实缩放带来的冗余不强行用有界 log-z 消除。显式纳入所有 Fock 边界候选，以免零投影概率点妨碍优化。

L-BFGS-B 使用包含归一化导数的解析实梯度，已逐个幅度/相位坐标进行中心差分检验。多个起点与候选保证所报告 fidelity 可实现；迭代收敛不表示全局最优。

四能级 onsite minor `M=A_a A_b-A_c A_d` 对 PBCS 恒为零。令

\[
w=(|a,b\rangle+|b,a\rangle-|c,d\rangle-|d,c\rangle)/2,
\]

则 `<w|psi⊗psi>=M` 且 `<w|phi⊗phi>=0`。对所有约束行经 SVD 正交化得到其张成空间 W，计算

\[
q_W=\|P_W\psi^{\otimes2}\|^2,\qquad F_{\mathrm{onsite}}^{\max}\le\sqrt{1-q_W}.
\]

不能把非正交 minor 的平方直接相加。上界针对 onsite PBCS，包括其连续边界；不适用于任意跨能级 projected Gaussian。这里是数学界的 float64 实现，不包含舍入误差的区间算术认证。

## 观测量与验收

输出 pair 占据 `<n_j>`、`<n_i n_j>`、`<s_i^+s_j^->` 及按完整能级分区的自然对数 von Neumann entropy。熵通过 pair-occupation 张量的 Schmidt 奇异值计算，保留所有粒子数分块。

验收标准在运行前保存在 JSON 配置中，包括守恒量、G=0 极限、积分容差收紧、相位对照、PBCS 嵌套关系、minor 上界、独立矩阵指数及晚时刻近似。保存的 CSV 可由 NPZ 波函数重算。主库测试完全不依赖 BCS 实验或保存数据。
