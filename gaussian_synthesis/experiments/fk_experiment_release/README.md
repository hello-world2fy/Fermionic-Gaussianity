# FK 双团簇实验：代码与完整保存数据

发布日期：2026-10-24。本分支基于仓库 `main` 的 `a5183acb8f0a7d84a89149e1ad35e89b964a7b5f`，新增三个 FK 实验包及本发布索引。原有主库 API 与其他实验不在此次修改范围内。

## 入口与实际运行范围

| 实验包 | 内容 | 本次发布的运行 |
|---|---|---|
| [fk_two_cluster](../fk_two_cluster/README.md) | 完整八模模型、对称性、full/product Gaussian 后端、低阶 pilot、成本基线 | `development_smoke_01`、`pilot_20261023_01` |
| [fk_two_cluster_followup](../fk_two_cluster_followup/README.md) | 轨道平均、确定性 k=2 矩、有限轨道与消融后端 | **仅** `development_smoke_01` |
| [fk_canonical_structure](../fk_canonical_structure/README.md) | canonical 重构、独立分布/相干检验、无弧族输入的候选发现 | `development_smoke_01`、`canonical_20261024_01` |

中间包原 README 中的 `followup_20261023_01` 是未执行的正式运行路径，**该结果目录不存在**；该 README 保持历史原文，以维持已记录源码哈希。本索引澄清实际执行范围，不能把其开发结果称为三种子正式实验。

- [原始实验说明](original_experiment_spec.txt)：原样保存用户提供的说明，后续实施范围以各包协议和配置为准。
- [首轮完整报告](../fk_two_cluster/results/pilot_20261023_01/REPORT.md)。
- [最新结果解读](../fk_canonical_structure/results/canonical_20261024_01/INTERPRETATION.md)。
- [最新正式报告与图表](../fk_canonical_structure/results/canonical_20261024_01/REPORT.md)。
- [下一轮协议与验收门槛](../fk_canonical_structure/PROTOCOL.md)。

## 数据完整性与证据口径

三个实验包的303个有效文件原样保留，包括所有代码、测试、配置、依赖版本、五次运行的全部 JSON/NPZ、日志、图表和 `partial_*` 检查点。保留未通过验收与开发阶段的失败验证记录；没有挑选有利参数点。仅排除虚拟环境、Python字节码、字体/系统缓存等可重建文件。

“完整数据”指**全部实际保存产物**。连续采样采用流式充分统计量，未保存每个 Haar 样本的完整态向量；不要将归档描述为逐样本 ket 数据集。已保存的训练缓存、完整累加器、witness、发现原子、旋转轨迹、SOCP系数与残差均包含在内。

两个正式运行的已记录源码哈希与发布源码相符。三个历史 smoke 是开发期间快照，与后续修改过的源码存在差异；原 manifest 和原验证记录不作重写。`source_unchanged_during_run` 只描述对应运行期间，不保证历史 smoke 能用当前源码通过全部哈希检查。

最新正式实验：五个 k=2 点的128维重构与边缘分布稳定性通过；三个 k=4 点未通过独立重构；相干解释0/8；最终 Monte Carlo 重构统计误差包络0/24通过。t=20 的 witness 引导修补上界约1.4151，低于 eta≤0.25 字典下界约1.7268，但比人工弧族约1.3067仍高8.3%。这些成本是浮点数值证据，不是全 Gaussian 字典的最优性或严格区间证明。

## 安装与检查归档

使用 Python 3.12，在仓库根目录运行。三个包是自包含的实验依赖链，无须安装或修改主库：

```bash
python3.12 -m venv gaussian_synthesis/experiments/fk_two_cluster/.venv
gaussian_synthesis/experiments/fk_two_cluster/.venv/bin/python -m pip install -r gaussian_synthesis/experiments/fk_canonical_structure/results/canonical_20261024_01/requirements-lock.txt
gaussian_synthesis/experiments/fk_two_cluster/.venv/bin/python gaussian_synthesis/experiments/fk_experiment_release/verify_release.py --output /tmp/fk-release-audit-new
```

输出目录必须不存在。发布检查验证 `release_manifest.json` 列出的文件大小、SHA-256、Git blob SHA及 NPZ 成员可读性/CRC，并复核最新正式 canonical 保存数组；科学复核输出重定向至新目录，不覆盖已发布记录，也不重新采样或优化。`--integrity-only` 跳过科学复核。

旧 followup 验证器通过历史 manifest 的本机绝对 `source_run` 路径读取首轮结果，换机器不能直接运行。发布检查不自动复跑该历史 smoke，也不把它升级为正式结果。首轮结果的所有73个输入产物仍完整保存，以满足其来源审计。

开发测试可分别在各包目录运行：

```bash
# fk_two_cluster/
PYTHONPATH=. .venv/bin/python -W error -m unittest discover -s tests -v
# fk_two_cluster_followup/
../fk_two_cluster/.venv/bin/python -W error::RuntimeWarning run_tests.py
# fk_canonical_structure/
../fk_two_cluster/.venv/bin/python -W error::RuntimeWarning run_tests.py
```

再次运行最新正式实验时，从 `fk_canonical_structure/` 执行：

```bash
../fk_two_cluster/.venv/bin/python -W error::RuntimeWarning scripts/run_experiment.py --config configs/canonical.json --output results/new_canonical_run
../fk_two_cluster/.venv/bin/python -W error::RuntimeWarning scripts/verify_results.py results/new_canonical_run
```

使用新的输出目录。默认三种子、样本上限65536、单进程单线程、512 MiB、每阶段180秒、总预算30分钟；不会自动扩大扫描或增阶。跨机器的浮点结果与运行时间可能不同，证据等级和验收门槛应保持一致。

## 发布清单

`release_manifest.json` 逐项记录发布载荷的路径、字节数和校验和，包含本目录的说明、原始文本与验证工具；清单自身不递归包含自身。Git tree 同时覆盖清单。`publication_validation.json` 记录隔离副本中的验证状态；它与验证日志位于清单之外，避免自引用。
