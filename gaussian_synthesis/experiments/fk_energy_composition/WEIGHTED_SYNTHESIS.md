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
