# 能窗成本：完整 Gaussian 下界与窗内构造上界

取 t≥9，E_G=−14−4t。对任意八模 pure Gaussian，定义 s=−Tr Γ_AB/8、x=1−s。纯度关系给出局域 covariance 平方范数之和 16(1−η)；Clifford 质量子空间的正交投影给出 m_A²+m_B²≤2(1−η)，而 Cauchy–Schwarz 给出 η≥s²。因此

\[
E(g)\ge-32+18s^2-4ts,
\quad E(g)-E_G\ge18x^2+4(t-9)x.
\]

能窗 E≤E_G+Δ 强制

\[
0\le x\le X_\Delta=
\frac{2\Delta}{\sqrt{16(t-9)^2+72\Delta}+4(t-9)}.
\]

相对于 hopping Gaussian 真空，H_hop=N−4，故 ⟨N⟩=4x。任意偶 Gaussian 在该真空的 canonical pairing 分解含四个独立 pair 概率 p_i，Σp_i=2x。相反 hopping Gaussian 是全占据态 |8⟩；AM–GM 给出

\[
|\langle8|g\rangle|=\sqrt{p_1p_2p_3p_4}
\le (\textstyle\sum p_i/4)^2=x^2/4.
\]

因此允许 Hilbert 重构误差 ε 时，完整受限字典满足

\[
\tau_\Delta^{(\varepsilon)}(\psi)
\ge \frac{4\max(|\langle8|\psi\rangle|-\varepsilon,0)}{X_\Delta^2}.
\]

当该分量非零且 ε 较小时，t=9 的 X∼√Δ 导致 Δ⁻¹；t>9 的 X∼Δ 导致 Δ⁻²。该推导没有假设 Gaussian 位于人工弧族。

## 上界的实际 Gaussian 构造

令 V 的三列为 reduced hopping 的 −4、0、+4 本征矢：

\[
V=\begin{pmatrix}
1/4&\sqrt{14}/4&1/4\\
-1/\sqrt2&0&1/\sqrt2\\
\sqrt7/4&-\sqrt2/4&\sqrt7/4
\end{pmatrix}.
\]

质量/hopping Gaussian 弧的 Spin(7) 平均在此基底中的坐标为

\[
p(x)=((1-x/2)^2,\ \frac{3}{2\sqrt{14}}(2x-x^2),\ x^2/4)^T.
\]

这些 Gaussian 的能量正好为 E_G+18x²+4(t−9)x。选择 x=0、X/2、X，X=0.99X_Δ，即为三个严格处于窗内的连续 Gaussian 轨道；对称平均中的所有实际 Gaussian 具有相同能量。

给定 ψ 在 V 基底的坐标 (a,b,c)，设 M₂=4c、M₁=√14 b/3+2c、M₀=a+√14 b/3+c。三个系数为

\[
d_0=M_0-3M_1/X+2M_2/X^2,\quad
d_1=4M_1/X-4M_2/X^2,\quad
d_2=2M_2/X^2-M_1/X.
\]

直接代入验证 Σd_j p(x_j)=(a,b,c)，成本 U=Σ|d_j|=O(X⁻²)。因此在 c≠0 时，与上述完整字典下界一起证明要求的 Θ 发散阶。对于允许误差 ε，可先沿 |8⟩方向移除至多 ε 的分量，再用同一窗内构造表示余下向量。没有窗外修补。

计算以 90 位数字保存系数，原因是小能窗中的巨大系数相消会使简单 binary64 ket 求和失去精度；这不是目标分解在数学上失效。显式公式定义构造，数值残差用于实现验证。当前未将该高精度数值计算标成有向区间算术证书。

## 有限 witness 的受限 overlap 包络

对 w 在 V 基底的系数 (w₀,w₄,w₈)，pair 概率的第二对称多项式给出 P(N=4)≤Σ_{i<j}p_i p_j≤3(Σp_i)²/8=3x²/2。因此

\[
\sup_{E(g)\le E_G+\Delta}|\langle w|g\rangle|
\le |w_0|+\sqrt{3/2}|w_4|X_\Delta+|w_8|X_\Delta^2/4.
\]

只有该上界严格小于已认证的全局归一化值时，才得到正 overlap gap。若质量分数上界≥1，应明确记录为无信息；不能把有限阶的近饱和峰解释为结构必要性。
