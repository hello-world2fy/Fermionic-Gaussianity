# 非线性配对场、同维数对照与有限列成本

本文定义第十一轮的机制检查。结果见 [REPORT_ROTATION_GEOMETRY.md](REPORT_ROTATION_GEOMETRY.md)。
所有物理条件此前都已查看；冻结计算文件后再读取参考态，只保证这次实现没有用未来态选择基，
不使这些条件重新成为未见预测。新增旋转构型由常规源场构造，不是 canonical 测度抽样。

## 1. 两个已有源场可以产生更多 Thouless 谐波

沿用两条平带、等强双通道的角向模型，$L=2r$，
$f_j=\sqrt2(c_1\cos\phi_j+c_2\sin\phi_j)$，$\phi_j=2\pi j/r$。
对费米海的粒子或空穴准粒子，取归一化 onsite Gaussian 自旋子

$$
u_j=\cos(A\lvert f_j\rvert),\qquad
v_j=A f_j\,\operatorname{sinc}(A\lvert f_j\rvert/\pi).
$$

这里采用 NumPy 的 $\operatorname{sinc}(x)=\sin(\pi x)/(\pi x)$。
于是 $\lvert u_j\rvert^2+\lvert v_j\rvert^2=1$。
在 $u_j\ne0$ 的局部图内，Thouless 坐标为

$$
z_j=\frac{v_j}{u_j}
=A f_j+\frac{A^3}{3}\lvert f_j\rvert^2f_j+O(A^5).
$$

即使 $f$ 只有两个已知的线性源方向，$z$ 也可包含三次及更高谐波。
对常模长圆周场，非线性只改变整体幅度，不增加形状方向。
因此，Thouless 坐标出现高谐波，本身不证明增加了独立物理配对场。
该运动学展开也不等于 [PAULI_CHANNELS.md](PAULI_CHANNELS.md) 中真实动力学的三次源系数。

对照的线性 Thouless 参数化取
$u_j=(1+A^2\lvert f_j\rvert^2)^{-1/2}$、$v_j=A f_j u_j$；
它落在两个源轨道的既有 Gaussian 张成空间中。
测试独立比较自旋子和原 onsite Gaussian 振幅，并核查上述三次 Fourier 系数。

## 2. 扇区列、对称性与同初态传播

粒子场和空穴场分别记为 $f_p,f_h$。对每个转移数 $m$，

$$
p_m(S)=\prod_{j\in S}v_{p,j}\prod_{j\notin S}u_{p,j},\qquad
h_m(T)=\prod_{j\in T}v_{h,j}\prod_{j\notin T}u_{h,j},
\qquad C_m(S,T)=p_m(S)h_m(T).
$$

每列是完整 Fock 归一化 Gaussian 的 $P_mP_N$ 投影，投影后不再归一化。
匹配场 $f_h=f_p^*$、相同实幅度时 $C_m=p_mp_m^\dagger$。
任意复扇区系数可以通过一体规范旋转的相干叠加表示；
独立扇区列及其 SVD 组合本身一般不是单 Gaussian。

同时循环、反射和粒子／空穴互换的对称子空间以等距矩阵 $Q$ 表示。
其维数在 $L=14,18$ 分别为 170、1530。所有保存列先变成 $Q^\dagger C_m$。
两源的常规多项式空间是共享基底 $B$，维数分别为 17、31，并含精确费米海。
对标准正交基 $V$，传播为

$$
c(t)=e^{-itV^\dagger Q^\dagger H QV}V^\dagger Q^\dagger F,\qquad
\psi_V(t)=QV c(t).
$$

删除旋转列时保留同一 $F$ 和 $B$。所有比较都从同一精确费米海开始。
相干诊断使用实际 Hamiltonian 下各计数扇区的净概率流：

$$
I_m(\psi)=2\,\operatorname{Im}\langle P_m\psi,H\psi\rangle.
$$

它等于完整动力学的 $\dot p_m$；对不与 $P_m$ 对易的近似投影空间，
不额外假定它等于投影传播得到的 $\dot p_m$。
报告的电流误差为 $\sum_m\lvert I_m(\psi_V)-I_m(\psi_{\rm exact})\rvert$。
循环算符二阶矩使用 $\|\chi QVc\|^2$，保留 $\chi$ 将态移出对称子空间的分量。

主集合为 16 个单位复二维系数方向，每方向四个幅度 0.25、0.6、1、1.4。
前四方向固定，余下由种子 861 产生。各扇区可有任意复叠加系数。
匹配、相对旋转 $\pi/4$、独立粒子／空穴系数三种规则分别保存。
独立规则的前四个确定性方向相同，之后的随机方向独立。
普通筛选先等范数化非零列，再以相对奇异值阈值 $10^{-9}$ 截断。

同维数比较固定保留整个 $B$，仅对投掉 $B$ 后的等范数列取主 SVD 方向，
总维数固定为 105、195。Hamiltonian 对照从同一 $B$ 出发，
反复施加 $Q^\dagger HQ$、双重正交化并选取新块，直到同一维数。
该控制不读取精确训练态或未来态；相同维数不等于已匹配全部计算成本。

## 3. 局部 Gaussian 模式的含义

另在五个已查看的源点，分别最大化完整归一化 Gaussian 与直接态或 k=2 witness 的重叠。
完整 witness 的密度是重叠模长的单调函数，故这些搜索可以定位密度的局部峰。
每源 32 次 onsite 局部初始化；八组一般反对称配对矩阵检查每组四次初始化。
一般搜索从所找到的 onsite 构型附近出发。

这些结果不是完整密度的质量积分，也不是 canonical 测度的独立样本。
初始化落入某峰的频率不是峰的概率；局部最优不认证全 Gaussian 支持函数。
直接态和 witness 的射线重叠、它们各自最强 Gaussian 峰的重叠、
以及投影／对称化后构型的重叠，是三个不同量。

## 4. 对保存有限矩阵的精确系数质量下界

本节严格对象只有保存的 binary64 列矩阵 $A$、共享基 $B$ 和保存目标向量 $\psi$。
它们被解释成精确二进有理数。目标是匹配 16 方向空间在 $t=2$ 的预测态，
不是精确多体参考态。$A$ 保留投影列原来的范数；$B$ 的任意系数免费。
考虑所有满足

$$
\|\psi-Ac-Bb\|_2\le\varepsilon
$$

的复系数，求 $\|c\|_1$ 的下界。
这允许比固定 Gaussian 系数规则更自由的表示，但仍只讨论这个给定有限矩阵。

令 $G=B^\dagger B$，以精确行和构造

$$
\rho=\max_i\sum_j
\bigl(\lvert\operatorname{Re}(G-I)_{ij}\rvert+
\lvert\operatorname{Im}(G-I)_{ij}\rvert\bigr)<1.
$$

对任意保存的见证向量 $w$，定义
$\widetilde w=(I-BG^{-1}B^\dagger)w$，则 $\widetilde w^\dagger B=0$。
无需计算有理矩阵逆，因为

$$
\|\widetilde w-w\|_2\le\delta,\qquad
\delta=\frac{1+\rho}{1-\rho}
\sum_j\bigl(\lvert\operatorname{Re}(B^\dagger w)_j\rvert+
\lvert\operatorname{Im}(B^\dagger w)_j\rvert\bigr).
$$

这里使用 $\|B\|_2\le\sqrt{1+\rho}\le1+\rho$ 和
$\|G^{-1}\|_2\le(1-\rho)^{-1}$。
于是只要下式分母为正，

$$
\|c\|_1\ge
\frac{
\max\{0,\operatorname{Re}\langle w,\psi\rangle-\delta\|\psi\|_2
-\varepsilon(\|w\|_2+\delta)\}
}{
\max_j\lvert\langle w,A_j\rangle\rvert+\delta\max_j\|A_j\|_2
}.
$$

所有内积用 Python 整数精确计算，平方根以整数平方根及 $2^{-128}$ 网格向上包络。
JSON 保存完整分子／分母；其中十进制显示值仅为摘要。
审计从保存输入独立重算八个证书并逐项比较。

**这不是理想 Gaussian 构型生成的区间证明，也不是连续旋转流形或完整 Gaussian extent 的下界。**
原自旋子、$Q$ 和 SVD 均用浮点计算；不能把精确解释保存矩阵，转写成对其理想解析前身的认证。
其他幅度、其他构型或一般 Gaussian 字典可能便宜得多。
该计算只检查当前有限列模型内部的系数抵消，不能替代用户要求的物理预测。

## 5. 对方法来源和新颖性的限制

配对态的相干生成坐标叠加已有先例：
[Hizawa、Hagino 与 Yoshida（2021）](https://arxiv.org/abs/2012.11123)
联合粒子数规范角与不同平均粒子数的 BCS 态；
[Li 等（2024）](https://arxiv.org/abs/2309.12564)
在含动态 BCS 配对的 TD-GCM 中同时演化生成态和权重。
这些工作不是本模型的数值验证，但足以说明“多个配对 Gaussian 相干叠加”本身不构成新发现。
本轮不声称穷尽查新、算法优势、可扩展多项式成本或新相。

实现见 [rotation_geometry.py](pairing_formation/rotation_geometry.py)、
[finite_dictionary_cost.py](pairing_formation/finite_dictionary_cost.py)。
完整冻结与核查记录见 [audit.json](results/assessment_rotation_geometry_01/audit.json)。
