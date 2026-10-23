# FK 两团簇实验：有界 pilot 与验收计划

本轮首先建立可审计的小规模实验包：复现模型、验证八模 Haar/乘积采样器、测量有限阶求解与独立重构的实际表现，并尝试不依赖高阶 canonical 的成本分离。所有代码与结果留在此实验目录，不修改核心库，不进行大规模参数扫描。数学约定和证据限制见 [METHOD.md](METHOD.md)。

## 1. 预先固定的问题与资源

物理扫描固定为 `t/V = 0,5,9,11,20`；有限阶主 pilot 使用 `k=1,2,3,4`。两个字典分别为 full global-even Gaussian 与同时包含 EE/OO 的 product Gaussian。配置列出三个种子 `20261023,20261024,20261025` 和样本量 `4096,16384,65536`，用于检查跨种子差异和样本量趋势。

这些配置定义最大允许实验网格，不承诺耗尽笛卡尔积。先做预检和有界运行，预算达到后保存已完成行、未运行行和停止原因；不以资源耗尽为由省略失败结果。

资源限制如下：

| 项目 | 限制或目标 |
| --- | --- |
| 并行度 | 单进程，BLAS/OMP/MKL/VECLIB 单线程 |
| 采样 batch | 256 |
| 单次训练或独立验证样本上限 | 65,536 |
| Newton 最大迭代 | 200 |
| 经验投影梯度容差 | `1e-8` |
| 单个运行块时间上限 | 180 秒 |
| 本轮总运行预算 | 1,800 秒 |
| 进程内存上限 | 512 MiB |
| 开发重构精度 | 独立完整残差 `<=1e-2` |
| 正式统计半径目标 | `<=0.005`，达到与否单独报告 |
| 正式置信失败概率总预算 | `0.05`，须预先明确分配 |

本轮不自动提高 `k`、不将样本 cap 放大、不启动大规模精细扫描。若 cap 内无法达到精度，保留失败及噪声诊断，后续方法变更另立配置和结果目录。

## 2. 文件边界与数据流

| 文件或模块 | 职责 |
| --- | --- |
| `fk/conventions.py` | little-endian Fock、Majorana、Clifford、基础收缩 |
| `fk/model.py` | 完整 Hamiltonian、局域 magic、Krylov 基、目标态 |
| `fk/symmetry.py` | 全 even 空间的 Spin(7) common-kernel 检查 |
| `fk/exact.py`, `fk/observables.py` | 完整 ED、一般 Wick、Schmidt、熵、物理曲线 |
| `fk/gaussian.py` | Haar SO 采样、Spin lift、相位一致的 ket/covariance |
| `fk/dictionaries.py` | full/product 测度、EE/OO 分支与 frame 常数 |
| `fk/moments.py` | 三维 witness 的 q、B、梯度、Hessian、质量 ESS |
| `fk/solver.py` | 固定经验凸目标的带阻尼 Newton |
| `fk/baselines.py` | 显式或群平均候选字典的有限成本上界 |
| `fk/certificates.py` | Fock 残差修补、受限字典下界与证据标记 |
| `fk/runtime.py` | 时间/内存约束、不覆盖结果、可复现元数据 |
| `configs/` | 固定实验参数与预算 |
| `scripts/` | 用户可调用的阶段运行入口 |
| `tests/` | 数学结构、约定、独立实现一致性的快速回归 |
| `results/` | 按运行名隔离的指标、数组、日志及源码摘要 |

训练尽量只保存三维重叠、原子不变量及必要元数据。完整 ket 的批量大小受限；独立验证流式累加 128 维重构。用于成本基线的原始 normalized Gaussian atoms、投影和系数必须保存，确保可以复核该具体表示。

## 3. 阶段 A：模型与精确物理基准

执行以下确定性或双精度结构检查：

- CAR、Clifford 反对易、Hermiticity、真空 covariance 符号与 A/B 位序一致。
- Krylov 基正交；相互作用与 hopping 分别闭合；三维矩阵与给定解析式一致。
- 21 个对角 Spin(7) 生成元的完整 even common kernel 维数为 3，投影匹配 Krylov 空间。
- 每个扫描点均用完整 256 维最低谱验证目标能量；完整谱差定义能隙。
- 精确态输出 `energy,gap,entropy,schmidt_l1,mass_correlation,wick_prediction,wick_deviation,eta`。
- Wick 先按一般 covariance 公式计算，再比较 singlet 简式；峰值搜索使用小规模三维目标而非增加物理系统尺寸。

结构残差默认应在 `1e-9` 以下；重要已知端点和参考值写入回归容差。若模型或对称性失败，停止合成与成本叙事，先修复约定。不能只用三维 ED 自证三维基态正确。

## 4. 阶段 B：Gaussian 后端与预注册采样校准

先验证每个原子的 norm、宇称、由 ket 直接计算的 covariance、`Gamma²=-I`。product 原子必须覆盖 EE/OO，且跨团簇 covariance 为零。

随后选固定 Gaussian probes，比较一阶 frame、高阶 overlap 矩和完整重构。full 使用 `d_(8,k)`，product 使用 `2*d_(4,k)²`。原子能量和四点关联还应与显式 Hamiltonian/四点算符计算相符；随机原子检查两侧 `mass+eta<=1` 及 Schmidt–eta 界。

在运行前固定以下诊断阈值：

- 标量解析矩误差不超过 **5 倍解析标准误**。
- 完整随机重构误差不超过 **6 倍解析 RMSE**。
- 所有结构恒等式按独立的浮点容差验收。

`5 SE / 6 RMSE` 说明所见波动与准确 Haar 采样相容，不是正式联合置信水平。不要根据观察到的误差把倍数放宽。若随机诊断未通过，记录具体 probe、种子、N、解析值、估计值和阈值，先调查采样、归一化或异常波动；不把该批数据用于分布解释。

同时输出严格 Gaussian 目标的有限阶成本解析值。尤其 `k=8` 时 canonical B 仍约为 2.699，不能因为目标 Gaussian 而要求低阶 B 接近 1。该校准防止把方法偏差误诊成优化器错误。

## 5. 阶段 C：固定预算的有限阶 pilot

采样器结构与解析矩诊断通过后，允许运行固定预算 pilot，即使解析噪声预估已经说明 `1e-2` 可能达不到。此阶段的目标是测量可达精度、稳定性、权重集中和运行成本，而不是保证重构通过。

这一区分尤其重要：在 `N=65536,k=2`，Gaussian 端点的随机完整重构 RMS 误差约为 0.0586。若 sampler 在 `6*RMSE` 内通过，但实际残差大于 0.01，这同时支持“采样实现与理论相容”和“当前预算不足以满足开发精度”。二者不矛盾。

每次求解冻结训练样本，以 `z∈C³` 优化固定经验目标；记录最终返回 z 上重算的梯度，不用更新前的 history 值代替。跨 t 或 k 的 warm start 只改善优化初值，不能共用训练样本充当独立验证。

`k=1` 使用解析 frame 恒等式；其随机估计只作为后端诊断，不把约 `0.044` 的 Haar 重构噪声称为一阶解析重构失败。`k>=2` 冻结最终 witness 后使用独立验证样本。

完整重构必须分解报告：

\[
\rho_{\rm full}=\|\psi-\widehat C\|,
\quad \rho_{\rm singlet}=\|S^\dagger(\psi-\widehat C)\|,
\quad \rho_\perp=\|(I-SS^\dagger)\widehat C\|.
\]

检查 `rho_full²=rho_singlet²+rho_perp²`。有限样本破坏精确群对称性；三维拟合误差小、singlet 外误差大是必须保留的诊断现象。

## 6. 结果字段与门槛

每个 `(dictionary,t,k,seed,N)` 至少记录：

| 类别 | 必要字段 |
| --- | --- |
| 复现 | 配置、种子流、train/validation N、源码摘要、环境依赖 |
| 优化 | status、收敛标志、迭代数、最终目标、最终经验投影梯度范数 |
| 有限阶量 | q_train/q_validation、B_train/B_validation、witness norm |
| 采样 | mass ESS、最大质量权重占比、采样与求解耗时、峰值内存 |
| 重构 | full/singlet/outside 三种残差、MC RMS 诊断、独立性说明 |
| 验收 | 阈值、reconstruction_accepted、统计半径与统计验收状态 |
| 停止 | 预算是否耗尽、失败原因、未尝试的后续阶段 |

经验残差不超过 `1e-2` 才给出开发重构通过；若正式统计半径未通过，则只能称“经验精度通过，未统计认证”。若成本或残差的置信声明跨多个阶数、参数和候选，必须在观察数据前分配总失败概率；重复验证不能悄然消耗无限次试验机会。

以下状态必须分开：

- `sampler_calibration_passed`：固定诊断门槛内与解析参考相容。
- `optimizer_converged`：经验三维优化达到梯度门槛。
- `reconstruction_accepted`：独立完整重构达到开发精度。
- `statistically_certified`：独立样本和明确置信半径支持声明。
- `strict_interval_certified`：浮点及目标误差获得严格包围。

任何前一项都不自动推出后一项。若精度失败，仍保存固定 pilot 结果，但停止该 witness 的 Gaussian 分布物理解读和正式 canonical 成本认证，不继续提高样本上限以追求通过。

## 7. 阶段 D：独立成本基线与受限字典比较

成本基线独立于 canonical 的大 k 收敛。候选池可以包含结构化 Gaussian、随机 Gaussian、已知端点以及它们的 Spin(7) 轨道平均。三维群平均投影的合法性见 METHOD；所有投影原子保持未归一化。

对各个固定 t，解有限候选上的复系数最小 `l1` 问题，保留 normalized 原始 Gaussian、投影、系数与完整残差。修补上界为

\[
U=\sum_j|c_j|+\|\psi-\sum_jc_j\Pi g_j\|_{1,\rm Fock}.
\]

在指定 `eta0=0.25` 上，与

\[
L(\psi)/(1+\sqrt{\eta_0})^2
\]

比较。若数值出现正裕量，报告有限池具体表示给出的成本分离证据，并保留 solver 状态和残差。此上界不需要宣称找到了 true extent。

开发重构精度失败不阻止这个独立基线：基线使用自己的显式表示、残差修补与证据标签，而不是把失败的 canonical 分解当作精确表示。反之，基线成功也不能证明 canonical 分布已准确。

当前采用双精度计算与数值缓冲，因此默认 `numerical_evidence`，`strict_interval_certified=false`、`statistically_certified=false`。本文不预填最终分离数值；以具体运行结果、保存原子和系数为准。

## 8. 分布解释与后续扩大实验的入口

只有相应 witness 的独立完整重构达到开发精度，才分析质量系数分布的 `(mass_a,mass_b,mass_dot,eta)`。先比较跨种子、跨 N、跨低阶结果，再决定是否有稳定的构型重组证据。ESS 过低或权重被极少数样本支配时，保留诊断并停止物理解读。

必要性界可以证明低关联字典的成本限制，但不会自动证明双峰、支持集合转变或 Wick 峰的成因。若要讨论相干解释，必须保留原子间矩阵元，而不能仅平均每个原子的 Wick 结果。

只有以下结果值得开启下一轮独立工作：

1. 低阶 pilot 已清楚识别方差瓶颈，能够提出可独立校准的改进采样器。
2. 显式或群平均基线给出稳定成本分离，值得做区间或代数认证。
3. 重构、ESS、跨种子与跨阶数均支持稳定分布结构，值得在 `5<=t/V<=20` 中加密。

本轮完成条件是：模型与后端有回归验证、固定预算 pilot 如实保存成功和失败、成本基线可复现、证据层级清晰。不以获得某个预想的分布形状或有利数值作为完成条件。
