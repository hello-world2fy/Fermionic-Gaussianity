# 固定转移数内的纠缠、相干源与预测观测量

本笔记对应第五轮实现 `interband.py` 和 `multiorbit.py`。它解释可检验的结构，
不声称这些代数恒等式或多构型变分原理是新发现。数值结论见 [REPORT_INTERBAND.md](REPORT_INTERBAND.md)。

## 1. 计数涨落之外，还有固定计数内部的纠缠

以上方粒子能级为 A、下方空穴能级为 B，在固定总粒子数子空间写

\[
|\psi\rangle=\sum_{m=0}^r\sum_{P,H:\,|P|=|H|=m}
 A_m(P,H)|P\rangle_A|H\rangle_B,
\qquad p_m=\|A_m\|_F^2.
\]

设 \(A_m\) 的奇异值为 \(s_{m\alpha}\)，并令
\(q_{m\alpha}=s_{m\alpha}^2/p_m\)。纯态的能带纠缠满足

\[
S(\rho_A)=H(p)+E_{\rm fixed},\qquad
E_{\rm fixed}=\sum_m p_m H(q_m).
\]

只允许局域粒子数守恒操作时，后者具有固定局域粒子数条件下的操作意义；
这与 [Wiseman–Vaccaro 的粒子纠缠框架](https://arxiv.org/abs/quant-ph/0210002) 一致。
此处的能带分区并不自动对应空间分离的实验双方。

任意径向系数、任意随 \(m\) 改变的单对轨道族仍满足
\(A_m=c_m a_m b_m^T\)，因此 \(E_{\rm fixed}=0\)。
它可以有很宽的 \(p_m\)，也可以保留不同 \(m\) 的量子相干，却仍没有固定计数内部的纠缠。
这给出一种比“增加径向自由度”更明确的结构区分。

最优的、完全不限制带内波函数的条件秩一近似满足

\[
F_{\rm rank1}^{\max}=\sum_m s_{m1}^2.
\]

它是条件 onsite 轨道族的上界，不是一般 projected Gaussian 的上界。
一般跨能级配对、多个 Gaussian 的相干叠加以及经典混合必须分别判断。

## 2. 一个可固定在训练时刻的局域滤波观测量

取 \(m=1\)，在 A、B 各选两个能级，得到未归一化的 \(2\times2\) 振幅矩阵 \(M\)。
对纯态，成功概率和条件态为

\[
p_f=\operatorname{Tr}MM^\dagger,\qquad
\rho_f=|M\rangle\!\rangle\langle\!\langle M|/p_f.
\]

条件 negativity 为

\[
\mathcal N(\rho_f)=\frac{|\det M|}{p_f},\qquad
p_f\mathcal N=|\det M|.
\]

因此训练时可用 \(|\det M|\) 选择滤波器，同时兼顾事件权重。
混态则直接计算 \(\rho_f^{T_B}\) 的负特征值；纯态行列式公式不可直接用于混态。
在本轮训练态、归一化 canonical witness 和预设带边滤波器中，三种选择恰好相同：
每个能带的第一个与最后一个能级。这是一个零差异结果，不能称为 canonical 找到了特殊滤波器。

Pauli 系数
\(t_{ab}=\operatorname{Tr}(\rho_f\sigma_a\otimes\sigma_b)\)
给出完整两逻辑比特层析。逻辑操作在每个能带的固定局域数目子空间内进行，
要求能实现能级分辨的相干操作；它们并非都等于裸占据计数。
层析得到的最大 CHSH 数值只用于诊断，未实现无漏洞 Bell 实验。

预测评价同时使用两个预选量：

\[
|\mathcal N_{\rm pred}-\mathcal N_{\rm exact}|,
\qquad D_{\rm unnorm}=\tfrac12\|p_f^{\rm pred}\rho_f^{\rm pred}
 -p_f^{\rm exact}\rho_f^{\rm exact}\|_1.
\]

后者保留事件概率信息。只报告归一化后的大纠缠会掩盖稀有事件代价。

## 3. 两条相邻计数扇区的源与相位干涉

将 Hamiltonian 写成
\(H=H_A+H_B+V_++V_-\)，其中
\(V_+=-(G/L)S_A^+S_B^+\)，\(V_-=V_+^\dagger\)；B 使用空穴记号。
在一个每块均为秩一的状态 \(A_m=c_m u_m v_m^\dagger\) 处，
投影到该块秩一流形的法向空间：

\[
\mathcal P_m^\perp C=(I-u_mu_m^\dagger)C(I-v_mv_m^\dagger).
\]

带内 Hamiltonian 的法向投影为零。来自 \(m-1\) 和 \(m+1\) 的两项分别为
\(N_m^-\)、\(N_m^+\)，每项至多秩一，所以

\[
\operatorname{rank}(N_m^-+N_m^+)\le2.
\]

对每个块施加 \(e^{-i\alpha m}\) 不改变 \(p_m\) 或任一带的约化态。
法向源的平方范数却变为

\[
\Gamma_m(\alpha)=\|N_m^-\|_F^2+\|N_m^+\|_F^2
 +2\Re\!\left[e^{-2i\alpha}\langle N_m^-,N_m^+\rangle_F\right].
\]

这解释了为什么同样的初始计数和局域态不能决定后续的条件纠缠。
对无纠缠的秩一块，经过无穷小演化后的次级 Schmidt 权重以
\(dt^2\|\mathcal P_m^\perp H\psi\|_F^2\) 开始，已用有限差分核查。

代码在真实态的**保持每个 \(p_m\) 的最优条件 Schmidt 秩一替代态**上评价此源。
它不是在原精确纠缠态上计算熵产生率，也不是证明真实态只有两个纠缠通道。
这里的带内 \(u_m,v_m\) 可以是任意多体向量，通常比 onsite 成对乘积更宽。
有限时间积累、不同块的反馈和带内运动仍能产生更多方向；秩二瞬时源不构成闭合定理。

## 4. 两个轨道构型的变分传播

本轮的可压缩族为

\[
A_m(P,H)=\sum_{a=1}^2 c_{ma}\,
 \frac{\exp[F_m(P,H)\cdot z_{ma}]}{
 \sqrt{\sum_{P',H'}|\exp[F_m(P',H')\cdot z_{ma}]|^2}}.
\]

每列是固定 \(m\) 的 onsite 粒子/空穴乘积，两个列之间保留复相干系数。
它们是投影 Gaussian 对称轨道所产生的向量，不是仅两个纯 Gaussian 原子。
每个块的两列也不被强制正交，且一般不等于两个 Schmidt 向量。

先分别用较早的目标态或 canonical 合成 witness 提取 \(z\)，
再用同一个真实训练态确定 \(c\)。未来只用新 Hamiltonian 的矩阵元做 TDVP：
将 \(-iH\psi\) 正交投影到物理切空间，列归一化后作 SVD 伪逆，
并包含轨道规范产生的 Berry 补偿。
测试覆盖物理态导数、范数/能量瞬时守恒、秩一极限及无相互作用极限。

在 L=12 一般点，数值复切空间维数为 108；L=16 为 208。
这不与静态 408 维 RPA 空间构成等成本比较：非线性积分每步需重新构造切空间。
目前仍显式枚举 \(\binom{L}{L/2}\) 个配对配置，未建立多项式算法或 Gaussian extent 界。
多构型时间变分和含配对的生成坐标方法已有先例，例如
[Li 等的 TD-GCM](https://arxiv.org/abs/2309.12564)。
这里需要验证的是提取结构的预测价值，而不是把多构型方法本身作为创新。

## 5. 保留边缘态的删除控制

取每块精确 Schmidt 分解，并对 A 实施局域 Kraus 操作

\[
K_\alpha=\bigoplus_m|u_{m\alpha}\rangle\langle u_{m\alpha}|,
\qquad \mathcal D_A(\rho)=\sum_\alpha K_\alpha\rho K_\alpha^\dagger.
\]

包括全部 Schmidt 基后 \(\sum_\alpha K_\alpha^\dagger K_\alpha=I\)。
它保持两边约化态与所有 \(p_m\)，并使每个固定 \(m\) 块可分。
但也改变一部分不同 \(m\) 之间的相干。因此后续计数变化不能全部归因于
“仅删除了块内纠缠”。这是依赖精确态的离线控制，未声称容易实验实现。

## 6. 由预测选择线性纠缠见证，再做合成测量

根据预测 \(\rho_{\rm pred}^{T_B}\) 的最小特征向量 \(v\)，冻结

\[
W=(|v\rangle\langle v|)^{T_B}.
\]

任意可分态满足 \(\operatorname{Tr}(W\rho)\ge0\)。这是
[Peres 部分转置判据](https://arxiv.org/abs/quant-ph/9604005) 的直接应用。
这里的线性算符 \(W\) 与 Gaussian 合成使用的向量 witness 是不同对象。

展开 \(W=\sum_{ab}w_{ab}\sigma_a\otimes\sigma_b\)，去掉恒等项，令
\(L_1=\sum_{(a,b)\ne(0,0)}|w_{ab}|\)。按 \(|w_{ab}|/L_1\) 随机选择测量，
带符号的加权结果落在 \([-L_1,L_1]\)。若成功筛选到 \(n\) 个独立样本，
单侧 Hoeffding 半径为

\[
\varepsilon=L_1\sqrt{2\log(1/\delta)/n}.
\]

81 个预先冻结的见证使用 \(\delta=0.01/81\)，并以 union bound 控制总失败概率。
随机成功样本数可条件化处理。该结论要求独立制备、指定测量模型与已校准操作；
本轮仅用精确模型产生合成测量数据，没有实验数据。

单轨道预测的部分转置有简并零空间。数值算法任意选出的零空间向量可能碰巧
检测到真实态的纠缠，不能据此说这个预测模型预言了非零纠缠。
归档保留原始测量结果，并另报告预测最小特征值小于 \(-10^{-8}\) 的适用样本。

## 7. 为什么有限时间散射已经解释一部分增强

忽略高阶反馈，从正常态出发的最低阶单转移振幅含有

\[
I_\Delta(t)=\frac{1-e^{-i\Delta t}}{i\Delta},\qquad
\Delta_{ph}=2(\epsilon_p-\epsilon_h).
\]

训练到 \(t_0\) 后做集体相位旋转及 \(G_0\to G_1\)，再传播 \(\tau\)：

\[
A^{(1)}_{ph}(\tau)=\frac{i}{L}\left[
G_0e^{-i\alpha-i\Delta_{ph}\tau}I_{\Delta_{ph}}(t_0)
+G_1 I_{\Delta_{ph}}(\tau)\right].
\]

不同散射时间的自由传播积累能量依赖相位，可使原本瞬时可分的源形成非可分矩阵。
相互作用反转改变两段振幅的干涉。因此“反转后条件纠缠增大”本身并不证明
新非线性机制或 canonical 独有信息。在 G=2 的强淬火中，裸一阶概率甚至可超过 1；
归一化后的对照仅用于解释机制，不带受控微扰误差保证。

配对模型的能量空间纠缠及额外纠缠谱带也已有直接研究：
[Rodríguez-Laguna、Ibáñez Berganza、Sierra](https://arxiv.org/abs/1307.8363)。
该工作针对基态；它不等于本轮的动态反转任务，但足以排除“发现额外纠缠谱带本身就是新物理”的表述。
