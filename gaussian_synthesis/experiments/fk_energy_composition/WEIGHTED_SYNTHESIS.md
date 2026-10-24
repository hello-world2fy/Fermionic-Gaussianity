# 加权连续合成的数学基础

设规范化紧 Gaussian 轨道的 Haar 概率测度为 mu，v(g)>0 连续且相位不变。
固定 k>=1，q(u)=d integral |<g,u>/v(g)|^(2k) dmu。
轨道张成目标 Hilbert 空间，故 q 在实化空间严格凸；有限维紧性还给出
q(u)>=c||u||^(2k)，c>0。因此 q/(2k)-Re<u,psi> 强制增长且具有唯一极小点。
这里严格凸不表示其 Hessian 在所有点具有统一正下界。

令 a=<g,u>，求实梯度得到

    C(u)=d integral g a |a|^(2k-2) / v^(2k) dmu = psi。

定义 Z_P=integral (|a|/v)^(2k-1)dmu，Bv=d Z_P，
P 的密度为 (|a|/v)^(2k-1)/Z_P，则

    psi=Bv E_P[(a/|a|)g/v]。
    Bplain=Bv E_P[1/v]，A=Bv E_P[e/v]，Bv=Bplain+lambda A。

普通系数概率密度为 (1/v)dP/E_P[1/v]，不是 P。
若采样 pi 的密度与 (|a|/v)^(2k) 成比例，方向 w 的半径可隐式消去：
设 s=Re<w,psi>>0、Im<w,psi>=0，则 u=(s/q(w))^(1/(2k-1))w，且

    C(u)=s E_pi[g/conj(<g,w>)]。
    Bv=s/E_P[|<g,w>|/v]。

以上只消去精确归一化常数，不使有限样本估计变成精确等式。
优化方向和独立验证流必须分开。

tau_lambda 是所有可行精确分解的 B+lambda A 的下确界，故非递减且凹。
紧轨道、严格正权以及有限维原子范数对偶给出

    tau_lambda=max Re<u,psi> subject to sup_g |<g,u>|/v(g)<=1。

存在最优分解时，其 A 是 tau 在该点的一个超梯度。
若 tau 可微则所有最优分解的 A 等于导数；不可微点不指定唯一导数。
任一全局认证支持上界 h_v(u)<=h 给出 L=Re<u,psi>/h。
由 B<=U 得 A>=[L-U]_+/lambda；若允许 Hilbert 误差 epsilon，
把 L 换成 [Re<u,psi>-||u||epsilon]_+/h。
Monte Carlo 极值、经验矩和局部最优点不能提供这个全局上界。

零惩罚严格恢复原 q、P、C 与成本。lambda>0 时没有沿用 Cartan
projector 的 h_v<=q^(1/(2k))；需要单独证明的全局支撑包络。

任意归一化见证 ell、窗内界 h<1、目标幅值 q=|<ell,psi>|，
成本 B<=U 且误差 epsilon 的分解满足高能普通系数比例

    p_high >= [(q-epsilon)/U-h]_+/(1-h)。

若使用非单位 ell，应把误差项改为 ||ell||epsilon，窗外界改为
一个已认证全局支持上界。该三角不等式本身不证明分布方法的独立增益。

## FK 加权支持函数的单独全局认证

令 x=<N>/4，0<=x<=2。此前证明的能量包络为
E(g)-EG>=f_t(x)=18x²+4(t-9)x。
四个 canonical pair 概率之和为 2x；AM–GM 给出
|<e0,g>|<=h0(x)=(1-x/2)²，|<e8,g>|<=h8(x)=x²/4。
使用此前已证明的 h4(x)=min(1,sqrt(3/2)x)，在 x<1/2 时还取
3x/(sqrt14 sqrt(1-2x)) 的最小值。h4 与 h8 非递减，h0 非递增。

对任意冻结 singlet witness w=sum w_r e_r，每个区间 [a,b] 上

    |<w,g>|/v(g)
    <= min(||w||, |w0|h0(a)+|w4|h4(b)+|w8|h8(b))
       /[1+lambda f_t(a)/(EG-E0)]。

覆盖 [0,2] 的有理网格及有向算术因此给出全局支持上界。
这既没有将局部最大值当作上界，也没有沿用加权 Cartan 恒等式。
所有奇宇称 Gaussian 与 even witness 正交，因此同一界覆盖它们。
网格仅影响上界的松紧；有限网格本身已经覆盖完整 Gaussian 流形。

认证器采用冻结坐标在精确代数 singlet 基底中的向量定义，并记录其与
采样使用的 binary64 ket 的差。方向优化本身不必是精确极小点，
全局 witness 证书仍有效。这个认证器也必须同等开放给强基线；
由它得到一个有效证书不自动通过 G3。

## Post-freeze restricted support for arbitrary witnesses

自动发现与认证是两个程序。发现阶段只接收带相位条件向量、Hamiltonian、
数值 HFB 参考态和有限个真实低能 Gaussian；不接收 e4/e8 或解析支持界。
冻结后，认证器读取完整的 128 维复 binary64 ket ell，视这些坐标为精确二进制有理数。
令 P 为精确 singlet 正交投影，w=P ell=sum w_r e_r，d=||(1-P)ell||。
使用精确 Gaussian-integer 矩阵 D 的归一化列计算 P，并用有向算术上界 d；
不将数值舍入产生的 singlet 外分量默认为零。

在 t>=9、Delta>=0 时，低能 Gaussian 必须满足

    0 <= x <= X = 2 Delta / (sqrt(16(t-9)^2+72 Delta)+4(t-9)).

Delta=0 时取 X=0。对 [0,min(X,2)] 的完整覆盖区间 [a,b]，

    |<ell,g>| <= min(||ell||, |w0|h0(a)+|w4|h4(b)+|w8|h8(b)+d).

取所有区间上界的最大值即得低能受限支持上界。奇宇称的 overlap 为零。
算法使用 X 的外向上界覆盖，不依赖低能数值 oracle 是否找到全局最大值。
此处认证的是原始复 ket，不是事后改成实 singlet 的另一条 witness。

令单位化后 q=|<ell,psi>|/||ell||，h 为单位化受限支持上界。
给定总普通系数成本不超过 B、Hilbert 误差不超过 epsilon 的表示，有

    p_high >= [((q-epsilon)/B-h)/(1-h)]_+    (h<1).

完全只用低能原子的表示至少有 max(q-Bh,0) 的 Hilbert 误差。
若候选态本身归一化，b=Bh<q 时还满足

    fidelity <= (q b + sqrt(1-q^2) sqrt(1-b^2))^2.

证明只需分别沿单位 witness 及其正交补分解候选态，并应用 Cauchy–Schwarz。
计算采用 q 的下界和 h 的上界；在 b<q 区域上式随 q 减小、b 增大而增大。
独立核验器使用 130 位谱分解、全 ket 投影和每个覆盖区间检查所有数值后果。
解析全原子包络的证明仍来自既有 FK 理论；不能把认证阶段使用这些理论
说成发现程序从样本重新证明了它们。

## 条件向量发现与直接子空间搜索的可替代条件

令 r 为 HFB 单位参考态，C 为各相干能量箱的贡献向量矩阵，
V=range((I-rr†)C)，psi_hat=C 1。当前发现程序实际求解

    max_{ell in V, ||ell||<=1} Re<ell,psi_hat>
        - B max_{g in finite_low_oracle}|<ell,g>|.

一旦 V 已确定，目标函数不再读取各能量箱的标签或分别的权重。
如果由 H、r 生成的 Krylov 子空间在移去 r 后恰好也是 V，直接对偶搜索
可以在不读取连续分布的情况下求解同一个优化问题。已知态模式允许它读取 psi。
对任意可行单位 witness，替换 psi_hat 为 psi 引起的目标函数差不超过
||psi_hat-psi||，两个最优值的差也不超过这个量。

这是一个带明确条件的可替代性结论，不是对所有分布算法的不可能性结论。
FK 本轮的两维剩余子空间需用保存的数据检查；若该条件成立，恢复这一子空间
和得到有效证书本身都不足以证明 G1 的新增信息或 G3 的独立增益。
为了公平，直接对偶基线同时运行完整 HFB Krylov 空间及移去参考态的版本。
后者在第一批基线后加入，报告会保留时间关系，不将它伪称为预注册实现。
