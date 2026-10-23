# FK canonical 分解：构型发现与相干检验

本实验按已确认优先级执行：可信 canonical 重构 → 不输入人工构型的结构发现 → 相干物理解释与成本必要性。精确态仅作为给定输入与独立参考；人工饱和弧族只在物理先验对照组使用。

首轮 `../fk_two_cluster/` 和已有 `../fk_two_cluster_followup/` 保持只读。本目录复用它们的模型、Haar 后端和代数 k=2 矩，不改主库API。详细预注册指标见 [PROTOCOL.md](PROTOCOL.md)，参数见 [configs/canonical.json](configs/canonical.json)。

## 运行

在本目录执行，复用首轮 Python3.12 虚拟环境：

```bash
../fk_two_cluster/.venv/bin/python -W error::RuntimeWarning run_tests.py
../fk_two_cluster/.venv/bin/python -W error::RuntimeWarning scripts/run_experiment.py --output results/canonical_20261024_01
../fk_two_cluster/.venv/bin/python -W error::RuntimeWarning scripts/verify_results.py results/canonical_20261024_01
```

输出目录必须不存在；旧结果、失败状态、partial检查点均保留。正式运行前冻结源码、两个依赖包和候选witness哈希。三组随机流的训练、开发、构型发现、最终验证相互独立。全部数值流程为单进程单线程，阶段180秒、总计1800秒、512MiB。最终报告与PNG/PDF科学图由保存数据生成。

## 代码组织

| 模块 | 内容 |
|---|---|
| `fk3/canonical.py` | k=2代数审计、完整张量重构、k=4训练缓存 |
| `fk3/statistics.py` | 配对raw/RB积分、分布、ESS、独立半样本相干核 |
| `fk3/discovery.py` | Gaussian流形局部搜索、等候选数对照、轨道SOCP与必要性界 |
| `fk3/runner.py` | 顺序运行、预算、候选冻结、跨种子验收 |
| `fk3/report.py` | 只读数据汇总与图表 |
| `scripts/verify_results.py` | 保存数组和代数一致性复核，不重做Monte Carlo或SOCP |

## 解释边界

确定性 k=2 重构准确，不意味着半整数质量积分或分组向量准确。分布和相干解释使用各自的独立门槛；三个种子全部满足才进入解释。稀疏分箱需要足够局部ESS。开发稳定性与严格统计覆盖分别报告。

`B_k` 是原canonical连续分解的系数质量；`U_guided` 是witness引导候选再做SOCP后的Gaussian轨道表示上界。二者不可互换，也不是已求得完整字典的最优extent。归一化投影原子会错误降低成本，本实验保留其原始范数。轨道代表数不等于实际有限Gaussian分支数。

三组候选数相同，计算时间不同；报告实际采样、局部优化、系数拟合耗时。Gaussian重叠局部最大值不是全局上界，不作为全字典对偶可行性证书。所有成本分离默认是浮点数值证据，不是区间算术证明。
