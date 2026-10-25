# 正常费米海配对实验：约定与可审计推导

这里 `L=2r` 表示配对能级数，物理费米模式数是 `2L`，电子总数是 `L`。
能谱默认 `epsilon_j = 2(j+1/2)/L - 1`，`g=G/L`，时间单位取带宽半宽的倒数。
初态是下半带满、上半带空。所有局部优化给出可行下界，不是最优保真度上界。

## 1. 精确 Hamiltonian 与有限尺寸线性响应

使用 `H=2 sum epsilon_j s_j^z - g sum_ij s_i^+ s_j^-`。
固定 r 个电子对后，对角自项是常数 `-gr`，但线性化前不能丢掉它在不同准粒子扇区的作用。
在正常态的粒子、空穴玻色表示中，二次矩阵为

```
A_p = diag(2 epsilon_p) - g J
A_h = diag(-2 epsilon_h + 2g) - g J
B   = -g J
L_B = [[A_p, B], [-B^dagger, -A_h^T]].
```

这保证单粒子对、单空穴对的对角能量分别为 `2 epsilon_p-g`、`-2 epsilon_h+g`。
真实一次转移的能量差仍是 `2 epsilon_p-2 epsilon_h`。
`i Tdot=B+A_p T+T A_h^T+T B^dagger T` 给出二次玻色压缩态的配对矩阵。
将它代入费米 `exp(sum T_ph s_p^+s_h^-)|F>` 是有限费米 lift，不是精确完整演化。
相应振幅是 T 的子矩阵 permanent；代码使用跨转移次数的子集递推。

检验包括：稀疏指数与独立 dense 矩阵指数、短时 Fano 系数、
`d<m>/dt=-2gr Im <sum_ph s_p^+s_h^-/r>`、守恒量和简并能谱极限。
发现当前 SciPy 1.17.1 的非零起点、极小步距 interval-expm 路径可失去归一化；
代码先传播到起点，再以零起点传播差分区间，回归测试覆盖了这一问题。

## 2. 将径向自由度和条件纠缠分开

固定转移 m 后，把波函数看作矩阵 `Psi_m[P,H]`，P、H 分别是上带电子对和下带空穴对的集合。
设其奇异值为 `s_(m,a)`。任何每个 m 块均为粒子—空穴可分离的态都满足

```
fidelity <= sum_m s_(m,1)^2.
```

证明：每块最佳秩一近似保留最大奇异值，再对块权重应用 Cauchy–Schwarz。
单个集体配对方向、任意复径向系数的态包含在这个类中，所以它也受此上界约束。
这个界不约束一般数目投影 Gaussian；数值 SVD 也没有进行区间舍入认证。
单方向拟合的误差可能来自带内波函数不能分解成相同轨道权重的乘积，
不能把“单方向拟合不好”直接解释成粒子—空穴条件纠缠很大。

标准 onsite PBCS 的计数满足 `Var(m)<=mean(m)`。
若目标超出量为 `Delta=Var_p(m)-mean_p(m)>0`，则任何该类 q 与它的
计数总变差距离至少是 `Delta/(3r^2-r)`：分别界定 `m(m-1)` 与均值平方的变化即可。
由测量收缩性，这也是纯态 trace distance 的下界，给出保守的保真度上界。
这只是 onsite 类的证书，而且有限误差界相当弱。

## 3. 二阶 replica 的完整 Gaussian 字典积分

采用物理 Majorana 的普通双副本张量积
`Lambda=sum_a gamma_a tensor gamma_a`。其零空间投影在固定偶 parity 中等于
二阶 Cartan 投影 `Q_2=d_(n,2) integral |g,g><g,g| dmu(g)`，
`d_(n,2)=binomial(2n,n)/2`，`n=2L`。
Gaussian-symmetric 子空间与零空间的联系见
[de Melo–Cwikliński–Terhal, Appendix A](https://ir.cwi.nl/pub/21177/21177B.pdf)。

两个 Fock 字符串相同占据的模式上 Lambda 为零。
在 d 个不同占据的物理模式上，Lambda 化为 `2 sum_i (-1)^(i-1) X_i`。
这些 X 彼此对易，因此零频 Fourier 系数给出投影矩阵元。
对 paired 字符串，设它们相差 s 个配对能级，另一次副本配置分配翻转 j 个配对能级，则

```
<A,B|Q_2|C,D> = Beta(j+1/2, s-j+1/2)/pi,
```

前提是每个物理模式在两副本中的总占据相同。两种费米符号在成对翻转时相消。
这是**先做完整投影、再读取 paired 矩阵元**，没有把中间态限制为 seniority zero。
显式完整双副本 Fourier 投影、独立 SO(2n) Haar 四次矩和四模 magic 嵌入分别核对了实现。

目标受总数 U(1) 以及每个能级自旋占据差 U(1) 不变群固定。
Haar 目标凸且严格凸，其唯一 witness 同样受该群固定；可以在目标扇区求 witness，
Gaussian 字典仍是完整字典。`q_2(u)=<u,u|Q_2|u,u>`，`C_2` 由一次收缩得到。
优化 `q_2(u)/4-Re<u|psi>`，梯度为 `C_2(u)-psi`。

输出残差是确定性双精度误差，不是区间算术证书。
`|<u|psi>|^2/sqrt(q_2(u))` 是有限阶对偶给出的 extent 下界的数值计算值，
不能误称为最优 extent。低阶 canonical 测度的成本还可能很高；
精确计算其合成向量，不意味着已经找到紧凑或最优的 Gaussian 构型分布。

## 4. 一般数目投影 Gaussian 的正确归一化

任意复反对称 Z 定义 Gaussian chart `exp(c^dagger Z c^dagger/2)|0>`。
其固定 `2r` 粒子分量的范数平方是 `e_r(s_1^2,...,s_L^2)`，s 是 Z 的成对 Takagi 奇异值。
目标重叠只需 paired 配置上的 Pfaffian，但分母必须计入所有 broken-pair 配置。
`projected_gaussian.py` 用完整范数；测试独立枚举所有固定粒子数配置核查，并核查自动微分。
有限 chart 和有限次局部重启不提供全局最佳投影 Gaussian 的上界。

## 5. 随机平均场与计数测量

spin Hamiltonian 的符号自项是 `-g(z+1/2)`。
经典轨迹采用 `h_x=-2g sum_(j!=i)x_j`、`h_y=-2g sum_(j!=i)y_j`、`h_z=2epsilon_i-g`。
初态 x,y 使用独立方差 1/4 的 Gaussian 或独立 ±1/2 离散样本，z 为确定的 ±1/2。
后者对应自旋离散 TWA 的一种常用取样方案。

测量 `m^2` 时先使用量子算符恒等式 `(s_i^z)^2=1/4`，估计量是
`m_classical^2 + sum_upper(1/4-z_i^2)`。直接把经典 m 的方差当量子方差并不等价。
这类一阶截断方法仍可能漏掉短时关联；例如本初态和此处独立取样下，
计数方差的领先系数为精确值的 `(r+1)/(2r)`。
其原因包括同一初始全空/全满带内的经典随机横向电流。
因此击败这些计数估计量不代表新物理，也不代表击败所有更高阶 TWA/BBGKY 方法。

随机 TDHFB 的既有背景见 [Lacroix et al.](https://arxiv.org/abs/1303.0748)，
自旋离散相空间方法见 [Schachenmayer et al.](https://arxiv.org/abs/1408.4441)。
本实现的有限自项和计数符号按当前 Hamiltonian 明确推导，没有直接照搬不同约定的方程。

## 6. Gaussian 对称轨道成本与响应训练

第二轮保存的列 `a=P_sector g` 不重新归一化，g 的完整 Fock 范数为 1。
投影由总粒子数和每个能级 `n_up-n_down` 的紧群平均实现；每个群元素均为 Gaussian 幺正。
固定总数的投影包含相应 Fourier 相位，系数绝对值的积分为 1。
因此 `sum c_a P_sector g_a` 的 Gaussian 合成质量至多 `sum |c_a|`，
但列数 R 并非 R 个未经投影的 Gaussian 态。余量用正交 Fock 原子展开，
给出 `B <= sum |c_a| + ||psi-Ac||_1` 的可行上界。
有限字典 SOCP 的对偶只约束保存列和 Fock 原子，不自动约束完整 Gaussian 流形。

二阶 canonical 合成 `C_2` 对 u 是含共轭的三次映射。
对真实时间路径求导得到 `DC_2(u)[du/dt]=dpsi/dt`，其导数是实线性算子，
不是复线性矩阵。使用中心化 Hamiltonian 是选定整体相位规范：
`dpsi/dt=-i(H-<H>)psi`。响应空间提取只用训练时刻的 u、psi 和 H。
它并不等价于从 canonical 系数测度抽取实际 Gaussian 原子。

## 7. 转移数分辨的相位电流及单方向反例

定义 `A=sum_(p,h) s_p^+ s_h^- / r`，它使转移数加一。
令 `C_m=<psi_(m+1)|A|psi_m>`、`J_m=-2gr Im C_m`，则
`dot p_m=J_(m-1)-J_m`，边界电流为零。
由离散分部求和得到 `dot mean=sum J_m` 和
`dot variance=sum (2m+1)J_m - 2 mean sum J_m`。
集体自旋升降算子的范数给出
`|C_m| <= (m+1)(r-m)/r sqrt(p_m p_(m+1))`。
连续性等式与直接 Schrodinger 导数独立核对。

对于实 Hamiltonian，态 psi 的复共轭使所有 C_m 复共轭，保持所有占据概率。
在均值极值处 `Im sum C_m=0`，所以
`rho=(|psi><psi|+|psi*><psi*|)/2` 保持 p(m)、能量和总复四点函数，
却令每个 J_m 和方差初始变化率为零。
这是一般量子态的对照构造，复共轭本身不是通用物理量子通道，不能误称为可直接实施的退相干操作。
它证明所保留的少数观测量没有闭合未来涨落动力学，而不证明分布独有的新信息。

两条简并能带 `epsilon_h=-1/2, epsilon_p=1/2` 的动力学恰好限制于均匀 Dicke 配对基 `|m>`。
此时仅有每带一个集体方向，精确 Hamiltonian 为

```
H_mm = -r + 2m - g [r + 2m(r-m)]
H_(m+1,m) = -g (r-m)(m+1).
```

该 r+1 维三对角模型同样产生首脉冲处的反向分区电流。
因此这种现象不能单独认证新独立配对通道。独立解析 lift 与完整模型演化均通过测试。

## 8. 条件轨道的变分传播与 Jastrow 联系

在固定转移数 m 的占据配置 s 上，取实特征 F_j(s) 为上带对占据和下带空穴占据。
定义 `q_m(s)=exp[F(s) z_m]/||exp[F z_m]||`，`psi_m=c_m q_m`。
共同轨道模型约束所有 z_m 相等；条件模型分别传播 z_m。
令 `mu_m=<F>_(q_m)`、`S_m=Cov_(q_m)(F,F)`，以及
`h_m=<q_m|(H psi)_m>`、`b_m=<q_m|(F-mu_m)(H psi)_m>`。
条件轨道的水平切向方程是

```
zdot_m = -i S_m^+ b_m / c_m
cdot_m = -i h_m - i c_m mu_m Im(zdot_m).
```

第二项补偿归一化指数坐标的 Berry 相位，使真正的状态导数满足 Dirac–Frenkel 投影。
共同轨道时，改用 `S=sum_m |c_m|^2 S_m`、`b=sum_m conj(c_m) b_m`，
`zdot=-i S^+ b`；各 c_m 保留对应的相位补偿。
每带固定占据造成协方差的两个零模；使用相对本征值截断的 Moore–Penrose 逆去除规范冗余。
稀有扇区或轨道零点可能产生 chart 条件数问题，因此保存守恒量、局部投影残差，并做容差/截断复核。

此模型在纯单体 Hamiltonian 下封闭，已与精确演化逐振幅核对。
一般 Hermitian Hamiltonian 的连续 TDVP 保持范数和能量；实际积分误差另行记录，
不能把局部投影残差混同为时间离散误差。
目前实现显式访问固定粒子数 Hilbert 空间，有限坐标数不意味着已经实现多项式复杂度。

在所有相关坐标非零的 chart 中，可选择一个固定参考轨道 z0，并写成
`psi(s)=exp[alpha(m)+F(s)(z_m-z0)] psi_reference(s)`，其中 alpha 吸收 c_m 和扇区范数。
因此这是集体密度相关 Jastrow 的一个有结构子族。
`z_m=z0+m z1` 给出密度乘积型修正；任意离散 z_m 则可以由有限阶 m 多项式插值。
这说明条件轨道的数值收益不能独自认证 canonical 方法的新颖性。

## 9. 自由径向 pair-CC 的强化对照

令各配置的 CC 原始振幅为 `r_s(T)=perm(T[P,H])`，每个 m 额外允许独立复系数。
对固定 T 消去最优径向系数后，保真度目标为
`sum_m |<psi_m|r_m(T)>|^2 / ||r_m(T)||^2`。
各块目标的伴随导数作为同一个永久式递推的反向初始梯度，然后跨所有 m 累加。
rank-1 时 `perm(T[P,H])=m! product p_P product h_H`，m! 被自由径向系数吸收，
因此应与共同轨道任意径向拟合相同；独立实现验证了这个等价性和复梯度。
