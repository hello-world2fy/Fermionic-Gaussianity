# 已认证端点witness的跨参数基线

必须把高阶方向与已知解析witness公平比较。`w0=2M_A⊗M_B` 的尖锐全Gaussian界

`|<w0|g>| ≤ 1−eta(g)/2`

只依赖Gaussian原子，不依赖待表示目标。因此它也可用于任意非零t基态。令 `L0(t)=Re<w0|psi_t>`，并取已认证独立构造成本上界U。对任意精确表示、任意B≤U，三角不等式给出

`L0(t) ≤ B [1−E_Lambda eta/2]`，

从而 `E_Lambda eta ≤ 2[1−L0(t)/U]`，以及 `Pr_Lambda(eta≥eta0) ≤ 2[1−L0(t)/U]/eta0`（均与1取小）。L0使用保存目标的精确二进制幅度及物理基态误差包围，未用未认证的高阶积分。

这条基线在部分参数比精确q4归一化的高阶方向更强；不能省略它后宣称高阶witness具有独特的必要性证明能力。

## 能窗中的质量比例

ENERGY_WINDOW_THEORY.md 已给出 `x=1+<H_hop>/4≤X_Delta` 和 `eta≥(1−x)²`（后者直接来自交叉协方差矩阵的Cauchy–Schwarz）。若X_Delta<1，所有窗内Gaussian满足 `eta≥(1−X_Delta)²`，于是

`Pr_Lambda(E≤E_G+Delta) ≤ 2[1−L0(t)/U]/(1−X_Delta)²`。

等价地，w0在该能窗上的支持上界是 `1−(1−X_Delta)²/2`。这不是额外的高k积分结果，而是已知尖锐端点支持界与全字典能量不等式的直接组合。它约束的是所有成本≤U的精确分解，不只是采样分布。

当X_Delta≤1时，该受限支持公式本身也是尖锐的：ENERGY_WINDOW_THEORY.md中的质量/hopping弧在x=X_Delta处恰有能量E_G+Delta、eta=(1−X_Delta)²。用该文的精确投影坐标p(x)与w0的hopping坐标 `(1/2,sqrt(14)/2,1/2)` 收缩，得到 `1/2+x−x²/2=1−(1−x)²/2`，达到支持上界。数值表使用有向舍入的保守包围；这不声称目标态的质量比例不等式也必然可饱和。

## Gaussian extent

全局系数成本下界可取 `max(1,L0(t),L4_certified(t))`；**eta或能窗比例式中仍须使用同一个w0的L0，不能擅自换成其他witness的更大L**。平方这些严格成本界才得到Gaussian extent `xi=tau²` 的界。
