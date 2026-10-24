# FK replica hierarchy

本目录执行 [原始方案](EXPERIMENT_SPEC.txt)，依次推进 E0–E5。之前三个实验包及其结果只读；新代码、失败诊断、独立验证、科学图和报告均保存在此处。

本轮 E0–E5 实验已完成，入口为[总报告](REPORT.md)和[完成审计](results/complete_plan_audit_01/audit.json)。55个扫描点、48项GPU解析回归、10650个最终预测已独立复核，40项测试通过。完成实验不表示已证明非零t的全局最优性或高阶算法优越性；证据范围见报告。

使用仓库根目录 `.venv/bin/python`（Python 3.11，依赖冻结于 `requirements-lock.txt`）。不要复用旧报告中 macOS 的 Python 路径或旧轮次的预算上限。

## 完成标准与证据

| 阶段 | 必需工作 | 完成证据 | 当前状态 |
|---|---|---|---|
| E0 | 四模 M / Gaussian 到 k=128；t=0；修复 k=4 独立验证 | 解析误差、Gaussian 纯度、5 种子独立重构与积分 | 完成；全部48项GPU解析回归通过经验门槛，精确k4完整态独立验证通过 |
| E1 | 方案分级 t/k 扫描，L、B、独立 U、残差、P、Figure 1 | 原始数组、训练/验证分离、误差及认证标记 | 完成55点；t0至128，t20至64，六个迁移参数点均至32；全部通过经验门槛 |
| E2 | 列生成、全字典对偶上界、接触轨道、受限组成必要性 | 可行 Gaussian 展开及有证明来源的 support 上界 | 45次列生成已核验；相邻精确Gaussian修补给严格U，Cartan分块给保守高阶Nk界；非零t最优性未证 |
| E3 | 固定 witness 的几何/宽度/Hessian、能窗及误差依赖 | 固定向量哈希、有限阶拟合与退化检验、受限界 | 完成336个能窗构造、1008条高阶窗证书与84条更强解析参考；t0精确Hessian导出B=2+22/k+O(k^-2) |
| E4 | 全 Fock 相干核、Wick 峰及受限组成的观测量限制 | H/Q/四点矩阵、相干与非相干对照 | 完成四阶六点对照，全部完整Fock核、Wick收缩及同原子非相干混合独立复核 |
| E5 | 冻结字典的非均匀 FK 预测、6 类基线、实际 Gaussian 项数与总成本 | 预测前冻结候选；新态 ED 仅最终验证 | 完成10650个最终预测；高阶改善有限字典，但未胜过物理先验或列生成的32项结果 |

第二模型和 k=128 的 FK 扩展为条件目标，不能用它们替代 E0–E5。实验失败保留失败状态；失败的数值估计不提供论文级物理认证。完成审计必须逐项回到原始方案。

## 数值口径

- 自适应温度 SMC 使用已知 Haar 初始测度、对数权重和在 Gaussian 流形上保持目标测度的更新。逐温度保存归一化、ESS、祖先数及实际更新时间。
- 整数 k 的优化积分与半整数 k−1/2 的系数分布分开计算。Spin(7) 平均只去除重构向量的非 singlet 方差，不改变标量积分质量。
- 至少 5 个独立种子；训练、开发、最终验证随机流独立。样本数逐级提升至 2^15、2^17、2^19，是否通过取决于残差与误差。
- 可信物理分析门槛：完整表示重构残差 < 1e−3；B 的独立重复相对不确定度约 1%；Gaussian 纯度误差 < 1e−9；多初始化一致。
- SMC 点估计、ESS、跨种子稳定性不是严格统计包围。只有有可靠 N_k 或 h_G 上界的 witness 才标记为对偶证书。实际有限表示含完整 Fock 修补成本。
- Gaussian 端点的精确 L=τ=1，但有限 k 的 canonical B 通常 >1；不能把 B>1 误报为虚假的 non-Gaussian extent。

## 执行记录

- 起始提交 `a480bab`。现有 k=2 代数后端与 k=4 失败结果已检查。
- 当前机器 Python 3.11；使用独立虚拟环境。用户随后授权停止全部 H20 占用任务，已通过 `/home/tiger/gpu_hold/STOP` 正常退出占卡程序，释放8张96GB H20。CPU affinity允许174个逻辑核；旧的19来自OMP环境设置。
- CUDA后端使用CuPy13.6、complex128、逐粒子Philox随机流。初始Haar样本采用可证明的球面Householder/Clifford构造；高温度角度拒绝采样使用有解析包络的正态提议。CPU后端保留。独立旋转、纯度、Haar矩、解析温度分布测试通过；相同131072粒子k=8基准，CPU50.8秒，GPU15.2秒。见 `results/gpu_backend_benchmark_02/`。
- 新后端为自适应温度 SMC + 精确单平面条件更新。数学说明见 [THEORY.md](THEORY.md)。
- 旧 Newton 在此环境 t=5 的梯度停在约 2e−9；新实验中的 Newton 加入浮点舍入尺度的梯度下降检查，通过完整张量回归。旧代码保持不变。
- `results/e0_n32768_01/REPORT.md`：已运行到 k=128 的三个解析对象、5 个种子；尚非全部通过 1% 经验门槛。
- `results/fk_k4_development_01/REPORT.md`：k=4、t=0/11/20 的训练和独立验证；保留未通过的残差。
- `results/fk_k4_development_01_order2_certificates/`：独立 order-2 区间归一化证书。范围是保存的目标向量，不是高阶点估计的认证。

## 当前主要结果

- [解析对象高阶回归](results/e0_n131072_01/REPORT.md)：3 个对象、5 个独立种子、k≤128，全部 48 个温度/对象组合通过 1% 经验门槛。
- [精确矩推导](CARTAN_THEORY.md)：k=2、3、4 的整数 Casimir 投影；全部模素数结果独立重放。k=4 完整态残差区间上界 <9e−13，精确物理基态误差转移 <1.4e−14。
- [k=2 扫描](results/fk_k2_exact_n32768_01/REPORT.md)和[k=4 扫描](results/fk_k4_exact_n32768_01/REPORT.md)：全部 9 个参数点、各5种子；完整 P 原子、相位、纯度与成本重复均已核验。
- [最终Figure 1](results/hierarchy_final_01/REPORT.md)：完整55点、经验不确定度、严格独立U与分块下界。旧进度图保留失败和当时空白，不作为最终覆盖表。
- [最终canonical组成](results/canonical_composition_final_01/REPORT.md)：从完整P粒子数组计算质量、交叉协方差、分布尾部与二维概率图，误差基于独立重复。
- [t=0至128的canonical扫描](results/product_magic_canonical_gpu_n131072_01/REPORT.md)：k128的B约2.16916，仍高于精确最优成本2。[完整Hessian与收敛系数](PRODUCT_MAGIC_ASYMPTOTICS.md)导出B_k=2+22/k+O(k^-2)，几何预言与独立采样相符；尚无有限k余项包围。
- [径向消元](RADIAL_ELIMINATION.md)：消去对C/B无必要的标量Z噪声，保留完整角向重构验证；实际canonical半径以精确q的符号定义给出，未把数值半径称为认证。
- [保守高阶Nk界](CARTAN_BLOCK_BOUNDS.md)：Pi_(a+b)≤Pi_a⊗Pi_b使q_(a+b)≤q_a q_b。精确q4归一化给高阶可行witness，但不会认证较强的canonical Monte Carlo L点估计。
- [独立列生成](results/columns_n32_s5_01/REPORT.md)：全部实际分支在完整 Fock 空间验证；局部 pricing 不作为全局最优性证书。
- [精确Gaussian修补成本](results/column_cost_certificates_01/REPORT.md)：在明确IEEE舍入模型下，将浮点分支替换为协方差定义的相邻精确Gaussian并加入全部Fock修补，给九个参数点的严格构造上界；证明见[舍入证书](GAUSSIAN_ROUNDING_CERTIFICATES.md)。
- [t=0 物理组成必要性](PRODUCT_MAGIC_NECESSITY.md)：证明全字典尖锐界 `|<2M_AM_B|g>|≤1−eta/2`。U≤2.02 的任何分解中，eta≥0.2 的系数质量占比≤9.91%；最优分解必支撑于 S⁶×S⁶。
- [固定 witness 几何](results/fixed_geometry_n8192_01/REPORT.md)：t=0、9、11、20 到k=128；区分有效标度、软曲率与已认证维数。
- [能窗实验](results/energy_windows_01/REPORT.md)：336 个构造的全窗口约束与高精度重构通过；原理见 [ENERGY_WINDOW_THEORY.md](ENERGY_WINDOW_THEORY.md)。
- [最终高阶能窗证书](results/energy_window_witness_certificates_final_01/REPORT.md)：使用相同精确q4归一化与已认证独立构造成本。t11、Delta=.01的k32方向给窗内质量占比≤.943654，低阶同类界仍无信息。
- [跨参数解析必要性](PRODUCT_WITNESS_TRANSFER.md)：同一端点witness适用于所有t；t11、Delta=.01的窗内质量上界为.230621，强于当前高阶q4证书。九点严格成本区间、平均eta限制和84条窗口界见[独立验证结果](results/product_witness_transfer_01/REPORT.md)。
- [k2的完整六点预测](results/frozen_predictions_k2_01/REPORT.md)及[k2/k8两点比较](results/frozen_predictions_k2_k8_t11_t20_01/REPORT.md)：包含真实Husimi分布oracle、局部极大值oracle、HFB、物理先验、随机和列生成；每个核与实际项数均独立核验。使用cost_audited_aggregates.csv查看包含全部上游依赖的成本。
- [最终四阶六点预测](results/frozen_predictions_k2_k8_k16_k32_01/REPORT.md)：10650条数据，包含全部基线、预算、扰动、相干核与总成本。32项、epsilon=.2时，t11保真度由k2的.4518提升至k32的.9241；物理先验.9984、列生成.9952仍更强。
- `results/ladder*_01.json`保留逐阶推进记录；最终扫描与预测流水线状态均为complete，见`EXECUTION_STATE.json`。每个高阶仅在前阶五种子门槛与独立数组复核通过后推进。
- 独立种子并行CUDA模式由`--devices 0 1 2 3`开启。小规模k4训练及验证的134个数组与串行版本逐位相同；该回归不是小样本科学门槛通过声明。t20的k64串行部分保留在01，四卡运行02已完成并验证，两个生产级训练NPZ也与串行结果逐位一致。

旧的失败轮次和失败原因完整保留。最终完成审计区分实验执行、经验数值门槛和严格数学证书；非零t最优接触分类及高阶canonical积分的严格包围仍未解决。

## 命令

在本目录执行；每次指定尚不存在的输出目录。每次开始时冻结源码副本及 SHA256。

```bash
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=8 ../../../.venv/bin/python -W error::RuntimeWarning run_tests.py
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=8 ../../../.venv/bin/python scripts/run_benchmarks.py --output results/<new-run> --count 131072 --max-k 128
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=8 ../../../.venv/bin/python scripts/run_fk_gate.py --output results/<new-fk-run> --k 4 --t 0 11 20 --count 131072 --time-reversal --full-distribution
OPENBLAS_NUM_THREADS=1 ../../../.venv/bin/python scripts/make_report.py results/<run>
OPENBLAS_NUM_THREADS=1 ../../../.venv/bin/python scripts/verify_results.py results/<run>
```

可用 `--frozen results/<training-run>` 对已有 witness 做全新独立验证；必须保留原 witness 对称性设置。`--time-reversal` 的正确性由精确反幺正对称性推导和独立 Gaussian/算符测试支持，不是只取实部的任意降噪。查看 `summary.json` 的数值门槛，不能把 `status=complete`（一次运行完成）理解为实验方案或认证完成。
