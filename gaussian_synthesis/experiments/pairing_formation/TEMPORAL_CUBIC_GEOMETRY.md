# 时间历史的完整 Gaussian 三次合成流形

本笔记定义第十四轮的非线性传播及受限流形的距离不等式。
结果见 [REPORT_TEMPORAL_CUBIC.md](REPORT_TEMPORAL_CUBIC.md)。
方法测试和解析障碍都不替代用户要求的重要新物理与真正未见条件预测。

## 1. 时间历史如何进入实际相干积分

在 L=14、G=4、等强双通道模型中，只传播到八个训练时刻
0.30、0.45、0.60、0.75、0.90、1.05、1.20、1.35。
对每个训练态求完整 even Gaussian 字典的二阶合成解 $C_2(u_i)=\psi_i$。
没有在训练阶段读取之后的参考轨迹。

令 U 的列为冻结的 witness 坐标空间，取

$$
z(c)=C_2(Uc),\qquad
\psi(c)=\frac{z(c)}{\|z(c)\|}.
$$

若 $a_i(g)=\langle g,U_i\rangle$，则

$$
z(c)=d_2\int
\left|\sum_i c_i a_i(g)\right|^2
\left(\sum_j c_j a_j(g)\right)g\,d\mu(g).
$$

这保留不同时间方向之间的全部三次相干交叉项。
它是完整 Gaussian 相干积分，不是把 witness 当成原子，也不是对未来态拟合的多项式。
归一化后的态仍有 canonical k=2 合成解
$Uc/\|C_2(Uc)\|^{1/3}$。

直接预计算张量

$$
z_a(c)=\sum_{ijk}T_{aijk}c_ic_jc_k^*,\qquad T_{aijk}=T_{ajik}.
$$

张量由完整 Cartan 投影器的有限收缩得到，输出只压缩到精确的 170 维对称空间。
所有 Gaussian 原子仍来自 28 个物理费米模式的完整字典。
随机复 c 的张量结果与原 GPU 合成器独立比较。

## 2. 共同信息与对照

三种主要三次流形都有十个复坐标，并共同包含 F 及最终训练点的 canonical witness。
因此都能严格表示同一个费米海和同一个 t=1.35 训练态。

canonical 历史空间额外使用早期 witnesses 和最终物理态；
直接历史空间使用同样时刻的物理态。
后者应准确称为“终点 canonical witness 加普通历史”，不能称为完全未优化字典。
Hamiltonian 对照则用同一公共部分加普通 Krylov 方向。

另比较十维普通线性“历史加 Hamiltonian 响应”空间及十维普通线性 Krylov 空间。
所有主要模型的物理射影切空间均有 18 个实自由度。
这匹配局部参数数目，不代表匹配预计算成本、全局表达能力或运行时间。

删除实验只保留 F 和终点 witness 两个复方向，
所以删去历史结构后两个初态仍相同。
所有模型分别从精确 F、精确 t=1.35 态开始；未来传播不再拟合。
这些是已查看物理条件的先导，尚非真正未见条件。

## 3. 非全纯映射须使用实辛形式

由于 z 同时依赖 c 和 c*，不能直接套用全纯参数的复 Gram 方程。
令 θ 为 c 的实部和虚部，取水平切向量

$$
J_\alpha=
\frac{(1-|\psi\rangle\langle\psi|)\partial_\alpha z}{\|z\|}.
$$

本轮采用实 Lagrangian／Hamiltonian 变分方程

$$
\operatorname{Im}(J^\dagger J)\dot\theta
=-\operatorname{Re}(J^\dagger H\psi).
$$

先对 J 的实嵌入做 SVD，移除归一化和整体相位的规范零模，
再在白化的物理切空间中解辛方程。若出现辛退化则停止，不静默伪逆。
这与非 Kähler 流形上的 McLachlan 最小残差原则一般不同；
相关几何框架已有系统讨论，见
[Hackl 等（2020）](https://arxiv.org/abs/2004.01015)。
本轮并不声称发明新的传播原理。

在精确算术下，反对称辛形式给出 $\dot E=0$。
测试分别覆盖完整射影空间的 Schrödinger 解、
真正非 Kähler 例子的辛方程与能量导数、非全纯 Jacobian 及规范零方向。
本轮轨迹的最小白化辛奇异值超过 0.9959，没有接近退化。

能量守恒不保证所有计数扇区的 Ehrenfest 方程自动成立。
报告中的电流是实际 H 的扇区净电流期望；
另把变分轨迹的计数导数与该电流作比较，保留连续性偏差。

## 4. 固定数目、无未配对电子子空间中的解析强单调性

以下结论限定于该配对振幅空间 S 上的完整 Gaussian k=2 映射。
它不是任意 full-Fock 向量的统一常数，也不是一般 projected Gaussian 的保真度界。

两副本 Cartan 投影器压缩到 S⊗S 后，按交集和对称差分块。
一个块有 s 个不同占据能级，s 为偶数，其核为

$$
K_s(x,y)=
\frac{\mathrm B(d(x,y)+1/2,s-d(x,y)+1/2)}{\pi},
\quad d(x,y)=|x\triangle y|.
$$

交换两副本对应取补集。对交换对称子空间，每个非平凡补集轨道合并为一维，
块矩阵的对角元为 $2k_s(0)$，所有非对角元非负，
行和仍为

$$
R_s=\sum_{j=0}^{s/2}\binom{s/2}{j}^2 k_s(2j).
$$

其中精确有理数

$$
k_s(d)=\frac{(2d)!\,[2(s-d)]!}
 {4^s d!(s-d)!s!}.
$$

Gershgorin 不等式于是给出交换对称块的下界
$4k_s(0)-R_s$；s=0 的下界为 1。
L=14 的所有块均为正，最小值是

$$
\boxed{\mu=\frac{1301961}{8388608}>0.1552.}
$$

这些常数由整数与 Fraction 运算计算，无浮点特征值假设。

设 $A_u(v)=d_2\int |\langle g,u\rangle|^2\langle g,v\rangle g\,d\mu$。
对称张量的范数给出

$$
\langle v,A_uv\rangle
\ge\frac{\mu}{2}
\bigl(\|u\|^2\|v\|^2+|\langle u,v\rangle|^2\bigr).
$$

又由积分逐点不等式

$$
\operatorname{Re}\langle v,DC_2(u)[v]\rangle
\ge\langle v,A_uv\rangle,
$$

沿连接 u 和 v 的线段积分，得到

$$
\operatorname{Re}\langle u-v,C_2(u)-C_2(v)\rangle
\ge\frac{\mu}{2}\|u-v\|^2
\int_0^1\|(1-t)u+tv\|^2dt.
$$

而积分等于
$\|u\|^2/4+\|v+u/2\|^2/3$，因而

$$
\boxed{\|C_2(u)-C_2(v)\|
\ge\frac{\mu}{8}\|u\|^2\|u-v\|.}
$$

这也直接给出此有限配对空间中 C2 的单射性。

## 5. 对冻结非线性流形的含义

设目标 ψ 归一化，合成残差
$\epsilon=\|\psi-C_2(u)\|$，冻结 witness 线性空间为 W。
任意归一化的 $C_2(v)$，v∈W，都可通过正实缩放 v 表示；
整体相位也保持 v∈W。因此，相位对齐后的距离满足

$$
\|\psi-\psi_W\|
\ge
\delta=
\max\!\left(0,
\frac{\mu}{8}\|u\|^2\,\operatorname{dist}(u,W)-\epsilon\right).
$$

在非平凡范围 $0\le\delta\le\sqrt2$，

$$
F(\psi,\psi_W)\le(1-\delta^2/2)^2.
$$

这是一条关于整个冻结非线性族的解析不等式，不依赖局部优化能否找到最佳 c。
本轮的 u、距离和残差仍来自 float64，
所以表中的代入值不是区间认证，也不是精确物理演化的认证保真度上界。
晚期 witness 仅在冻结预测评价之后计算，不用于修改预测结构。

局部多起点拟合另报告可行保真度，只能说明已经找到的近似质量。
解析不等式及其数值代入都不能证明较大的局部拟合误差不可避免，
更不能把这个受限流形障碍当成新物理或完整 Gaussian 表示障碍。
