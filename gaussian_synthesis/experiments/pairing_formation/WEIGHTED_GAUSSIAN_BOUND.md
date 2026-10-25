# 固定粒子数的加权涨落界

本轮得到一个比第六轮 `Var(m) ≤ mean(m)` 更强的解析约束。
它适用于任意轨道的纯数目投影 Gaussian；不是只对 onsite PBCS 的数值拟合结果。
推导使用已有的对称多项式凹性。其在配对计量学文献中的最优性和发现优先权尚未建立。

## 一般单体算符

设物理总粒子数为 $N=2M>0$，

\[
|\phi\rangle=P_{2M}|g\rangle/\|P_{2M}|g\rangle\|,
\qquad A=c^\dagger Kc,\quad K=K^\dagger.
\]

记一体密度矩阵为 $\gamma$，使 $\langle A\rangle=\operatorname{Tr}(\gamma K)$。则

\[
\boxed{\operatorname{Var}_\phi(A)
\le 2\operatorname{Tr}(\gamma K^2)
-\frac{[\operatorname{Tr}(\gamma K)]^2}{M}.}
\tag{1}
\]

K 可以有正负本征值，不必是投影矩阵。右侧和方差一样，在 K 加常数倍单位矩阵时不变。
Gaussian 只作总粒子数投影；额外的 seniority、自旋或空间多体投影不在此命题范围内。
边界 Gaussian 由连续极限包含。

在 Gaussian 自身的规范成对基中，固定 M 对的占据概率为
$p(S)=\prod_{i\in S}z_i/e_M(z)$，其中 $z_i>0$。令 $n_i=\langle x_i\rangle$，

\[
C_{ij}=\operatorname{Cov}(x_i,x_j),\qquad W=\operatorname{diag}(z).
\]

已有的 Marcus–Lopes 凹性给出 $e_M(z)^{1/M}$ 凹，因而

\[
\nabla^2\log e_M+\frac1M(\nabla\log e_M)(\nabla\log e_M)^T\preceq0,
\qquad W(\nabla^2\log e_M)W=C-\operatorname{diag}(n).
\]

所以得到矩阵不等式

\[
\boxed{C\preceq\operatorname{diag}(n)-nn^T/M.}\tag{2}
\]

这里的凹性是已知数学结果，见 [Sra，Sec. 2 及其中 Marcus–Lopes 文献](https://arxiv.org/html/1803.10141v1)。
式（2）比逐项负协方差更强，允许对任意实权重使用。

将 K 按规范对写为 $2\times2$ 块，$d_i=\operatorname{Tr}K_{ii}\in\mathbb R$。
完全成对部分的方差由式（2）满足

\[
\operatorname{Var}(\textstyle\sum_i d_i x_i)
\le\sum_i n_i d_i^2-(\sum_i n_i d_i)^2/M.
\]

任意轨道混合还产生破对分量，不能投影掉。不同无序对 ${i,j}$ 的破对输出彼此正交。
保留正反向振幅干涉，Cauchy–Schwarz 给出其范数平方至多

\[
2\|K_{ij}\|_F^2(n_i+n_j).
\]

利用 $d_i^2\le2\operatorname{Tr}(K_{ii}^2)$，加总后为

\[
\operatorname{Var}(A)
\le2\sum_i n_i\operatorname{Tr}[(K^2)_{ii}]
-(\sum_i n_i d_i)^2/M,
\]

这就是式（1）。测量 K 在规范基中完全一般，因此候选 Gaussian 不必在物理能级基保持成对。

## 计数、能量分辨权重与相干证书

若 K=P 是单粒子投影，$m=A/2$，式（1）化为

\[
\operatorname{Var}_\phi(m)\le\mu(1-\mu/M),\quad\mu=\langle m\rangle.
\tag{3}
\]

在本实验的成对目标态中，取实权重 $w_i$，

\[
X=\tfrac12\sum_i w_i(N_{i\uparrow}+N_{i\downarrow}),\quad
Y=\tfrac12\sum_i w_i^2(N_{i\uparrow}+N_{i\downarrow}).
\]

目标态上 X 为加权对占据，纯 PG 必须满足

\[
\operatorname{Var}(X)\le B=\langle Y\rangle-\langle X\rangle^2/M.
\tag{4}
\]

因此目标态的超额为 $w^T\mathcal E w$，其中

\[
\mathcal E=C-\operatorname{diag}(n)+nn^T/M.
\tag{5}
\]

最大本征值给出单位欧氏范数能级权重的最优超额。这是一个已知目标态上的观测量选择，
不是 Gaussian 重叠的全局优化，也不是未见参数预测。
权重整体平移不改变超额；常数权重对应守恒总数的零模。

对纯 PG 的经典混合，普通方差不再受式（4）约束。
但 QFI 凸性与 B 对密度矩阵的凹性给出

\[
F_Q(\rho,X)\le4B(\rho),\qquad
K(0)-K(\theta)\le\theta^2B(\rho),
\quad K(\theta)=\operatorname{Tr}(\rho e^{-i\theta X}\rho e^{i\theta X}).
\tag{6}
\]

第二步沿用第六轮已说明的 sub-QFI 下界，不是新的两副本估计原理。
混态需要 K(0)=纯度；不能把它设置为 1。纯态时才用计数特征函数模平方。

## 有限保真度界

定义 $a=1-1/M$，$f(\rho)=\langle X^2-Y\rangle-a\langle X\rangle^2$。
若纯目标 $\psi$ 有 $\delta=f(\psi)>0$，任意纯 PG $\phi$ 有 $f(\phi)\le0$。
令 $D=\sqrt{1-|\langle\psi|\phi\rangle|^2}$，则

\[
\delta\le R_BD+aR_X^2D^2,
\quad R_B=\operatorname{range}(X^2-Y-2a\mu_\psi X),
\quad R_X=\operatorname{range}(X).
\]

因为任意 Hermitian O 有 $|\operatorname{Tr}O(\rho_\psi-\rho_\phi)|\le\operatorname{range}(O)D$。
取正根得到 $D\ge2\delta/[R_B+\sqrt{R_B^2+4aR_X^2\delta}]$。
实现枚举全部 $N_i=0,1,2$、$\sum_iN_i=2M$ 的谱值；保留单占据配置，
没有以目标态的 seniority 限制候选态。小规模枚举不是多项式算法。

## 本轮结果的实际强度

原 $L=12,G=2,t=2.15$ 脉冲的正本征值为 $2.51910\times10^{-5}$。
独立稠密对角化在固定权重上复现该数值，物理态差异 $2.25\times10^{-14}$ 以下。
它得到非平凡但非常弱的全一般 PG 保真度界：

\[
1-F_{\rm PG}\ge1.1820\times10^{-10}.
\]

上述数值是解析不等式的 float64 评价；下一节另给固定有理权重的严格包围。
两者都不足以证明实际逼近必须有大误差。
该权重在读取脉冲目标后选择；早期 t=1 的最优权重在脉冲处没有违反。
相干回波的最大已测试正信号只有 $4.40\times10^{-9}$，不宣称现有测量资源可用。

等强角向模型 $L=14,t=2$ 的上壳层计数虽然方差小于均值，
却超过式（3）约 0.1754；相应谱权重的保真度排除界较强，但仍不紧。
这扩大了认证范围，没有证明 canonical 分布带来独有的物理预测。

独立记录：`results/weighted_count_bound_02/`。包含 800 个任意 Hermitian 单体算符的完整
Fock 作用检查、100 个旋转 Pfaffian 态检查、七个已知条件的事后评价。
`_01` 保留了脚本参数错误导致的未完成首跑，不计为额外实验。

## 固定有理权重的精确算术证书

随后将权重固定为

\[
w=(-12,-18,-36,-110,-367,-727,120,118,147,216,299,370)/1000.
\]

该选择仍是事后选择，但此后的证明不依赖浮点本征向量。
本模型 $\epsilon_j=(2j+1-12)/12$、$G=2$，配对 Hamiltonian 可精确写为

\[
H=A/6,\quad A_{SS}=\sum_{j\in S}(2j+1-12)-6,\quad
A_{S',S}=-1
\]

（非对角元只在一次允许的配对移动时非零）。A 是整数 Hermitian 矩阵，
最大绝对行和为 78，且 t=43/20。因此演化是 $e^{-i(43/120)A}|F\rangle$。
用整数 Horner 递推精确保存 140 阶 Taylor 多项式的实、虚分子和共同分母。
Hermitian 实时间演化的积分余项给出严格范数界

\[
\varepsilon\le\frac{(78\cdot43/120)^{141}}{141!}
<4.587\times10^{-40}.
\]

这里不需要 $e^{\|Ht\|}$ 因子，因为余项中的实时间传播是酉算符。
对于 Taylor 向量 q 与精确归一化态 $\psi$，

\[
|\langle q|O|q\rangle-\langle\psi|O|\psi\rangle|
\le\|O\|(2\varepsilon+\varepsilon^2).
\]

对 $O=X$ 与 $X^2-Y$ 使用完整 fixed-N 谱的精确有理范数，
将均值误差传播到 $f=\langle X^2-Y\rangle-a\langle X\rangle^2$。
所有比较均使用无界整数和 `Fraction`，不依赖显示用的十进制舍入。
得到严格正区间，其较松的向外包围为

\[
2.4998763496\times10^{-5}<f(\psi)<2.4998763497\times10^{-5}.
\]

再用均值区间控制谱宽，并以有理二分给正根的下界，得到

\[
\boxed{1-F_{\rm PG}>1.10249\times10^{-10}.}
\]

这个界的理论范围是全部归一化总数投影纯 Gaussian，数值包围针对上面明确的有理模型参数。
它严格补上原首脉冲处的非零排除，但仍然非常弱。
独立 float64 精确演化与有理 Taylor 向量的态差异为 $1.93\times10^{-14}$，
只作为实现交叉检查；严格性来自整数计算和余项界。
原始精确分数、Taylor 向量及参数见 `results/rational_pulse_certificate_01/`。
其他时间点和本轮局部拟合并未因此变成严格认证。
