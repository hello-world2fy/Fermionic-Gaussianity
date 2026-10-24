# 稀有低能区域：独立提议与误差包络

M1 首批在 20 个独立 SMC 重复中均未命中四个能窗。
因此不能用零计数当作零贡献。这里为条件积分建立另一个具有已知归一化的提议。

固定任意 Gaussian 真空；八模偶 Gaussian 的四个 canonical pair 概率记作 p_i。
在 Thouless 坐标 Z 中，轨道 Haar 密度正比于 det(I+ZZ†)^(-7) dZ。
理由是 Gaussian 群的分式线性作用在反对称切空间上诱导 wedge² 的 Jacobian，
det(wedge² M)=det(M)^7；其模方给出这个不变体积因子。
复反对称 Takagi 分解 Z=U diag(z_i J2) U^T 后，角度为 U(8) Haar，
径向 Jacobian 为 product_{i<j}(y_i-y_j)^4 product_i dy_i，y_i=|z_i|²。
在两个二维块之间，四个复切向方向给出四次 Vandermonde；单块的极坐标
给出 dy_i。令 p_i=y_i/(1+y_i)，Jacobian、Vandermonde 分母与
product(1+y_i)^(-14) 正好抵消，故 p 的联合密度正比于 Vandermonde(p)^4。

在 sum p_i<=s<=1 内，这个密度是 24 次齐次多项式，四维体积缩放为 s^28。
将该多项式完全展开，用有理数分别积分立方体与单纯形，得到精确体积

    mu{sum p_i<=s}=s^28/23178480。

实现不依赖数值积分计算这个归一化常数。也以展开多项式独立计算矩，
检验条件 sampler 的径向分布。

可用 beta=4、m=4、a=7 的 Laguerre 双对角矩阵采样形状：对角自由度
14,10,6,2，次对角自由度 12,8,4，取 BB^T 的本征值除以其总和。
另取半径 s U^(1/28)，乘以该形状得到条件 p，再采样独立 U(8) 角度。
这个双对角模型依据 [Dumitriu–Edelman, Theorem 3.4](https://arxiv.org/pdf/math-ph/0206043)。
它产生独立的实际 Gaussian ket，而不是对称投影向量。

FK 的既有全局 coercivity 定理给出低能窗内 sum p_i=<N>/2<=2 X_Delta。
所有首轮窗口满足 2X<1，故以上分布完整覆盖低能事件。
提议区域还包含能量高于阈值的 Gaussian，必须保留真实 Hamiltonian 指示函数。
对 canonical 冻结 witness u，低能向量为

    d_k V_s E_proposal[1_window g <g,u>|<g,u>|^(2k-2)]。

低能绝对系数质量为 d_k V_s E[1_window |<g,u>|^(2k-1)]。
使用原始能窗指示函数、真实相位和全部 Hilbert 分量；不得用上包络代替数值积分值。

若有效命中太少，独立提议允许用二项分布的精确单侧置信上界约束命中概率，
再乘全局被积函数上界，控制整个低能向量的范数。
这种概率置信陈述与确定性数学证书分别记录；它不认证条件均值或尾部形状。
需要重要尾部细节时，仍必须提高命中数或改进条件提议。
