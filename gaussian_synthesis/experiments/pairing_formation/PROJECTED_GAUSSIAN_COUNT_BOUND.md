# 任意轨道数目投影 Gaussian 的计数界与相位回波证书

这里给出第六轮的解析推导。它补上此前只对 onsite PBCS 证明计数界的范围限制。
**不要求候选态在测量能级基中保持成对占据，也不对候选态额外作 seniority 投影。**
随机与独立 Pfaffian 核查支持实现，但不是普遍命题的证明；证明是下面的逐步不等式。
新颖性及与现有计量学界的完整关系尚未查清，不在此主张发现优先权。

## 命题与适用范围

设 \(|\Phi\rangle\) 是任意纯偶宇称费米 Gaussian 态，
\(P_{2M}|\Phi\rangle\ne0\)，且

\[
|\phi\rangle=\frac{P_{2M}|\Phi\rangle}{\|P_{2M}|\Phi\rangle\|}.
\]

对任意单粒子子空间的正交投影矩阵 \(P\)，定义
\(\hat A=c^\dagger P c\)，即该子空间中的物理费米子数。则

\[
\boxed{\operatorname{Var}_{\phi}(\hat A)\le2\langle\hat A\rangle_{\phi}.}
\tag{1}
\]

对子空间补集同样成立。候选态可有任意复配对矩阵、跨能级或跨自旋配对，
允许零/满占据的边界极限。总粒子数投影后再进行任意单粒子幺正变换仍在适用范围内。
不包含额外的自旋、空间、seniority 或其他多体投影，也不包含任意不同 geminal 的乘积。

本实验 \(m=N_A/2\)，因此

\[
\boxed{\operatorname{Var}_{\phi}(m)\le\langle m\rangle_{\phi}.}
\tag{2}
\]

一般候选态的 \(m\) 可以取半整数；目标配对 Hamiltonian 的纯态才仅有整数支持。
式（2）首先是纯态族限制，不能直接施加到该族的经典混合的普通方差上。

## 1. 规范成对基中的负协方差

纯 Gaussian 的 Bloch–Messiah 标准形式及其数目投影可写为

\[
|\phi\rangle=Z^{-1/2}\sum_{S\subseteq\{1,\ldots,K\},\,|S|=M}
\left(\prod_{k\in S}z_k\right)
\prod_{k\in S}a_{k1}^\dagger a_{k2}^\dagger|0\rangle.
\tag{3}
\]

这是任意轨道的 AGP；规范基通常与能量测量基不同。
标准形式和数目守恒 BCS 的关系可见
[Kraus、Wolf、Cirac、Giedke，Sec. II](https://arxiv.org/abs/0810.4772)。

令 \(x_k\in\{0,1\}\) 表示第 k 个规范对是否占据，\(w_k=|z_k|^2\)。
概率为 \(\Pr(S)=\prod_{k\in S}w_k/e_M(w)\)。
删除 i、j 后其余权重的基本对称多项式记作 \(E_q\)，有

\[
\operatorname{Cov}(x_i,x_j)
=\frac{w_iw_j}{Z^2}
\left(E_{M-2}E_M-E_{M-1}^2\right)\le0.
\tag{4}
\]

最后一步是非负变量的 Newton 不等式。约定 \(E_0=1\)、范围外 \(E_q=0\)，
包含端点粒子数。零权重及确定占据可取极限处理。

## 2. 必须保留任意轨道混合引入的破对部分

将单粒子投影 P 转到式（3）的规范基，并按两个模式一组写成 \(2\times2\) 块 \(P_{ij}\)。
它仍满足 \(P=P^\dagger=P^2\)。令

\[
d_i=\operatorname{Tr}P_{ii}\ge0,\qquad n_i=\langle x_i\rangle.
\]

在规范配对子空间中，\(\hat A\) 的带内块只产生
\(D=\sum_i d_i x_i\)。带间块 \(K_{ij}\) 将一个粒子从规范对 i 移到 j 或反向，
产生恰好 i、j 两个单占据规范对。

不同无序对 \(\{i,j\}\) 的结果相互正交，也与 D 的完全配对结果正交。因此

\[
\operatorname{Var}(\hat A)=\operatorname{Var}(D)
+\sum_{i<j}\|K_{ij}\phi\|^2.
\tag{5}
\]

由式（4）及 \(d_i\ge0\)，

\[
\operatorname{Var}(D)\le\sum_i d_i^2 n_i(1-n_i).
\tag{6}
\]

再记 \(t_{ij}=\Pr(x_i=0,x_j=1)\)。从这部分态转移一个粒子的平方范数为
\(t_{ij}\|P_{ij}\|_F^2\)；反向部分为 \(t_{ji}\|P_{ij}\|_F^2\)。
两者会干涉，**不能删掉交叉项**。Cauchy–Schwarz 给出

\[
\begin{aligned}
\|K_{ij}\phi\|^2
&\le\|P_{ij}\|_F^2(\sqrt{t_{ij}}+\sqrt{t_{ji}})^2\\
&\le2\|P_{ij}\|_F^2(t_{ij}+t_{ji})\\
&\le2\|P_{ij}\|_F^2(n_i+n_j).
\end{aligned}
\tag{7}
\]

式（7）显式包含任意轨道的破对振幅及其最大相长干涉。
这正是此前仅使用 onsite 负相关证明时缺少的一步。

## 3. 用单粒子投影的几何性质封闭上界

由于 \(P^2=P\)，

\[
\sum_{j\ne i}\|P_{ij}\|_F^2=d_i-\operatorname{Tr}(P_{ii}^2).
\tag{8}
\]

又因为 \(P_{ii}\) 是二维 Hermitian 矩阵，
\(d_i^2\le2\operatorname{Tr}(P_{ii}^2)\)。综合（5）—（8），

\[
\begin{aligned}
\operatorname{Var}(\hat A)
&\le\sum_i d_i^2n_i(1-n_i)
 +2\sum_i n_i\big[d_i-\operatorname{Tr}(P_{ii}^2)\big]\\
&\le2\sum_i n_i d_i-\sum_i(n_i d_i)^2\\
&\le2\langle\hat A\rangle.
\end{aligned}
\tag{9}
\]

整个论证只在候选 Gaussian 自身的规范基中使用成对概率，而测量投影 P 保持一般形式。
因此不是把 onsite 假设偷偷施加到物理测量基。
有限非零参数的命题通过态的连续性延伸到投影态族的闭包。

## 4. 从纯态计数界到排除经典混合的 QFI 界

设 \(\mathcal C_{\rm PG}\) 为同一固定总粒子数下，归一化纯数目投影 Gaussian 态的凸包。
对 \(\rho=\sum_a p_a|\phi_a\rangle\langle\phi_a|\)，QFI 的凸性及纯态公式给出

\[
F_Q(\rho,m)\le\sum_a p_a\,4\operatorname{Var}_{\phi_a}(m)
\le4\operatorname{Tr}(\rho m).
\tag{10}
\]

QFI 与方差凸屋顶的关系是已知结果，参见
[Sixia Yu](https://arxiv.org/abs/1302.5311)；这里使用标准纯态归一化 \(F_Q=4\operatorname{Var}\)。

所以，足够强的**相干**相位响应可以排除整个凸包。
普通计数方差没有这一结论：不同自由态的均值波动会使混合态超 Poisson。
例如，与目标相同 Fock 概率的对角混合态由 Slater 行列式组成，
可以具有相同的超 Poisson 计数，但对 m 的 QFI 为零。

式（10）不是声称配对态不能增强计量精度。
[Kraus 等](https://arxiv.org/abs/0810.4772) 已讨论配对相对未配对态的 Ramsey 增益，
并区分数目耦合与更复杂的 pair-interaction 耦合。
这里比较的是整个数目投影 Gaussian 凸包，且上界按被计数子空间的平均占据给出；
不适用于任意多体相位生成元。与已有最优界的穷尽比较仍待完成。

## 5. 不需要零角度外推的两副本回波

定义

\[
\rho_\theta=e^{-i\theta m}\rho e^{i\theta m},\qquad
K(\theta)=\operatorname{Tr}(\rho\rho_\theta).
\]

在 m 本征基中，

\[
K(0)-K(\theta)
=\sum_{ij}|\rho_{ij}|^2[1-\cos(\theta(m_i-m_j))].
\tag{11}
\]

利用 \(1-\cos x\le x^2/2\)，以及标准 Hilbert–Schmidt 下界

\[
\mathcal I_{\rm sub}=2\|[m,\rho]\|_F^2\le F_Q(\rho,m),
\]

得到对任意 \(\theta\) 的自由态限制

\[
\boxed{K(0)-K(\theta)\le\theta^2\langle m\rangle
\quad(\rho\in\mathcal C_{\rm PG}).}
\tag{12}
\]

下界可直接从 QFI 谱公式证明：本征值满足 \(\lambda_i+\lambda_j\le1\)（i≠j），
所以去掉分母后只会降低 QFI。它属于已知 sub-QFI 工具，参见
[Cerezo、Sone、Beckey、Coles](https://arxiv.org/abs/2101.10144)。
本轮不将 sub-QFI 或两副本估计本身作为新方法。

两个重叠可用独立两副本 SWAP 期望估计，均值 m 由另一次计数估计。
无需把有限角度曲率假装成精确 QFI。
混态必须实测或估计 \(K(0)=\operatorname{Tr}\rho^2\)，不能错误地设为 1。
纯态时才可将 \(K(\theta)\) 写成计数特征函数的模平方。

若以概率 ε 完全随机化集体数目相位，
\(\rho_\epsilon=(1-\epsilon)|\psi\rangle\langle\psi|+\epsilon\mathcal D_m(|\psi\rangle\langle\psi|)\)，
所有 \(p(m)\) 不变，但

\[
K_\epsilon(0)-K_\epsilon(\theta)
=(1-\epsilon)^2[1-|\langle e^{-i\theta m}\rangle_\psi|^2].
\tag{13}
\]

这提供可明确删除的相干结构。完全数目去相干不一定使整个态成为自由态；
它只是消除此特定见证对跨 m 相干的灵敏度。

## 6. 独立于局部拟合的有限保真度障碍

对一个已知纯目标态，记
\(\mu=\langle m\rangle_\psi\)、\(\delta=\operatorname{Var}_\psi(m)-\mu>0\)，且 \(0\le m\le r\)。
令

\[
O_\mu=(m-\mu)^2-m,\qquad
R_\mu=\max_{x\in[0,r]}[(x-\mu)^2-x]
-\min_{x\in[0,r]}[(x-\mu)^2-x].
\]

对任意归一化纯投影 Gaussian 候选 \(\phi\)，令
\(D=\sqrt{1-|\langle\psi|\phi\rangle|^2}\)。
有 \(|\langle m\rangle_\phi-\mu|\le rD\)，且

\[
\langle O_\mu\rangle_\phi
\le(\langle m\rangle_\phi-\mu)^2\le r^2D^2.
\]

由有界可观测量的迹距离界，
\(\delta\le R_\mu D+r^2D^2\)。于是

\[
\boxed{F_{\rm PG}^{\max}(\psi)\le1-\eta^2,\qquad
\eta=\frac{2\delta}{R_\mu+\sqrt{R_\mu^2+4r^2\delta}}.}
\tag{14}
\]

连续区间涵盖一般候选的半整数计数。
对纯目标，\(\langle\psi|\sigma|\psi\rangle\) 对 σ 线性，因此相同上界也适用于
\(\sigma\in\mathcal C_{\rm PG}\)。这不意味着仅凭未知混态的计数就能排除凸包。
界通常较松，但它是覆盖整个态族的有限误差障碍，不依赖优化器是否找到最优点。

## 核查和当前证据等级

`projected_count_bound.py` 独立构造完整固定物理粒子数 Fock 空间的单体算符作用，
并用任意轨道旋转后的 Pfaffian 态交叉验证。随机检查覆盖 6、8、12、16 个物理模式，
多个粒子数、子空间秩、复相位和权重尺度，共 3300 例；33 例另用全 Pfaffian 计算。
测试还包含经典混合、QFI/sub-QFI、有限角度不等式以及相同计数的相干/去相干反例。

本轮报告的普遍性来自上述解析证明；具体数值证书使用 float64，未作区间算术认证。
合成测量遵守预设模型的 IID 假设，不是实验数据。
数目投影 Bogoliubov 态的完整分布及取样本身已有研究，例如
[Bernard 等，EPJA 2026](https://link.springer.com/article/10.1140/epja/s10050-026-01878-5)。
本推导没有证明 canonical 分布比直接态、RPA 或其他强方法更能预测这些见证，
也没有证明低成本 Gaussian 表示在非线性长时区间仍可闭合。
