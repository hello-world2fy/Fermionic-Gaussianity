# 双 FK 团簇的小规模 Gaussian 合成实验

本实验独立放在 `gaussian_synthesis/experiments/fk_two_cluster/`，不改变主库的四模 API。目标是核验两个 FK 团簇的精确物理与单 Gaussian 变分基线，然后比较完整八模 Gaussian 字典和局域乘积 Gaussian 字典的表示成本、有限 k 连续合成及跨团簇关联要求。

物理约定为 V=1，A 占低四个 occupation bits，B 占高四个 bits；完整 Fock 空间256维，全局偶宇称空间128维，经过算符与对称性检查的 witness 子空间为三维。Majorana 定义为 `gamma[2j]=c_j+c_j†`、`gamma[2j+1]=-i(c_j-c_j†)`；`Gamma[a,b]=i<gamma_a gamma_b>`，真空 `Gamma[0,1]=-1`。

## 环境与运行

以下命令均在本实验根目录运行。使用 Python 3.12 的独立虚拟环境；不要把系统 Python 3.9 当成主库支持的环境。

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install numpy scipy matplotlib cvxpy clarabel
PYTHONPATH=. .venv/bin/python -W error -m unittest discover -s tests -v
.venv/bin/python scripts/run_experiment.py --output results/pilot_20261023_01
```

每次运行指定一个新的输出目录。runner 默认拒绝覆盖已有结果；请保留失败验收、预算耗尽和采样校准结果。环境版本、实际配置和源码来源由结果目录记录。再次安装复现实验时，以对应运行保存的依赖版本为准。

仅重建已保存数据的报告：

```bash
.venv/bin/python scripts/make_report.py results/pilot_20261023_01
```

报告脚本使用 Matplotlib Agg，并把字体缓存写到该运行目录的 `.mplconfig/`；会更新 `REPORT.md` 和 `figures/`，不重跑物理或合成计算。

## 默认首轮范围

实际配置以 `configs/pilot.json` 及运行保存的 `config.json` 为准。首轮比较两个字典、3个独立种子、5个耦合点 `t/V = 0, 5, 9, 11, 20` 和4个阶数 `k = 1, 2, 3, 4`；训练样本阶梯为4096、16384、65536。内部生成批量256。配置设置每阶段180秒、总计1800秒、内存512 MiB，重构容差0.01；预算和验收门槛允许结果明确停在未通过状态。

执行顺序是精确模型与对称性核验、有限轨道字典成本上界、Gaussian Haar/frame 校准、三维 witness 连续合成、完整128维独立验证、报告。未通过前一阶段的条件时，不应把后续分布图当成可信物理结论。小规模首轮不自动增加到更高 k 或更大字典。

## 字典、成本与证据

- **full**：真实 Haar SO(16) 旋转，经 Givens Spin lift 作用于真空，覆盖全局偶宇称的八模纯 Gaussian rays。
- **product**：EE 与 OO 两个局域宇称分支独立等概率抽样；两个分支都必须包含。只取 EE 不能表示一般耦合态。局域 odd 原子通过固定 Majorana 的 Pin 变换构造。
- 两个字典的有限 k frame 常数不同：full 为 `d_(8,k)`，product 为 `2*d_(4,k)^2`；k=1 时均为128。
- 轨道字典 SOCP 提供可行成本上界，求解器的 optimal 只针对该有限字典。轨道投影原子不可重新归一化，残差通过 Gaussian Fock 原子修复。
- 有限 k 的系数质量 `B_k`、真实最小成本 `tau` 和 extent `xi=tau^2` 分开记录。`B_k^2` 不能直接标成精确 extent；未通过重构的质量估计不是目标态的可行上界。
- 受限 eta 字典与 full 的成本分离目前是浮点数值证据，不是区间算术证明。统计验证失败或预算不足保留原始状态。
- 分布解释需要独立完整重构与有效样本数等门槛。若没有通过的 k≥2 结果，报告不画、不解释高阶 Gaussian 分布。Gaussian 成分的经典平均也不能替代目标态的相干四点关联。

## 代码组织

| 文件或目录 | 职责 |
|---|---|
| `fk/conventions.py`、`model.py` | Clifford/Majorana 规范、完整 Hamiltonian 与三维基 |
| `fk/symmetry.py`、`exact.py`、`observables.py` | 不变空间核验、精确物理、Gaussian/Wick 观测量 |
| `fk/gaussian.py`、`dictionaries.py` | Haar Gaussian 原子、宇称、covariance、frame 常数 |
| `fk/moments.py`、`solver.py` | 三维重叠缓存、有限 k 目标与优化 |
| `fk/baselines.py`、`certificates.py` | 轨道 SOCP、残差修复、受限字典成本比较 |
| `fk/runtime.py` | 资源预算与可复现执行支持 |
| `fk/report.py` | 从结果 JSON 生成报告和静态科学图 |
| `scripts/` | 实验运行与报告命令入口 |
| `tests/` | 算符、Gaussian 后端、frame、优化与证据门槛检查 |
| `configs/`、`results/<run_id>/` | 配置及隔离的逐次结果 |

结果包括 `exact.json`、`baselines.json`、`calibration_*.json`、`pilot_*.json`、`summary.json`，以及原子、系数、witness、分布的 NPZ 数组。报告只读取保存的数据；物理峰、成本与残差均不手工填入图表。
