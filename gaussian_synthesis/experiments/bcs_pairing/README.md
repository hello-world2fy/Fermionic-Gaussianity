# BCS 基线与通用空间接口：第一批实现

本目录保存 BCS 专属代码、配置、测试和结果。通用的 `FermionSpace`、`OccupationSector`、`NumberSector`、状态 fidelity/距离位于主包 `src/gaussian_synthesis/`。现有四模 synthesis/extent 接口及采样器继续使用原实现。

本批完成：固定数目 seniority-zero 精确参考；可解 `1/t` 模型的渐近态及相位控制；常耦合淬火；投影 TD-BCS 与投影 product TDVP；多起点 onsite PBCS 拟合；二副本 minor 上界；数目守恒关联与能级分区熵。

一般跨能级 projected Gaussian、通用多模 Haar 与 Gaussian extent 合成属于后续批次，本次没有计算这些量。PBCS 拟合值是可实现 fidelity 下界，不是全局最优值；minor 上界仅针对 onsite PBCS。

## 环境与运行

要求 Python >=3.10。验证环境为 Python 3.12，依赖的准确版本保存在结果目录 `requirements-freeze.txt`。本机默认 `python3` 可能仍是 3.9，创建环境时应明确选择合适的解释器。

在仓库根目录执行：

```bash
python3.12 -m venv gaussian_synthesis/experiments/bcs_pairing/.venv
gaussian_synthesis/experiments/bcs_pairing/.venv/bin/python -m pip install -r gaussian_synthesis/experiments/bcs_pairing/requirements.txt
gaussian_synthesis/experiments/bcs_pairing/.venv/bin/python -m pip install -e gaussian_synthesis
```

测试与正式实验分开运行；脚本只定位实验包，通用库通过上述项目安装导入。

```bash
gaussian_synthesis/experiments/bcs_pairing/.venv/bin/python -W error::RuntimeWarning gaussian_synthesis/run_tests.py
gaussian_synthesis/experiments/bcs_pairing/.venv/bin/python -W error::RuntimeWarning gaussian_synthesis/experiments/bcs_pairing/scripts/run_tests.py
gaussian_synthesis/experiments/bcs_pairing/.venv/bin/python -W error::RuntimeWarning gaussian_synthesis/experiments/bcs_pairing/scripts/run_baselines.py --full --output gaussian_synthesis/experiments/bcs_pairing/results/my_l4_run
gaussian_synthesis/experiments/bcs_pairing/.venv/bin/python gaussian_synthesis/experiments/bcs_pairing/scripts/verify_results.py gaussian_synthesis/experiments/bcs_pairing/results/my_l4_run
gaussian_synthesis/experiments/bcs_pairing/.venv/bin/python gaussian_synthesis/experiments/bcs_pairing/scripts/plot_results.py gaussian_synthesis/experiments/bcs_pairing/results/my_l4_run
gaussian_synthesis/experiments/bcs_pairing/.venv/bin/python gaussian_synthesis/experiments/bcs_pairing/scripts/validate_all.py gaussian_synthesis/experiments/bcs_pairing/results/my_l4_run
```

输出目录及 figures 目录必须是新目录，不覆盖既有结果。verify 只读取已保存波函数和标量，重新计算验收量及 SHA-256，不重新拟合或积分。若需要在原结果环境之外核验，先核对源码快照及依赖版本。

`validate_all.py` 在新的 `validation/` 目录中归档主库/实验测试日志、结果核验、当前数值代码与快照的一致性，以及删行/篡改标量两项负测试；已有 validation 目录不会被覆盖。

## 范围和预算

首轮配置见 `configs/l4_pilot.json`，固定 `L=4,m=2,W=1`、均匀能级和 Dicke 初态。

- 淬火 `G=0.6`，即 `lambda=LG/W=2.4`；输出 `t∈[0,20]` 的 81 个时间点，在 6 个预定时刻拟合 PBCS。
- 渐近结构 11 个 `(eta,alpha)` 点；所有相位控制保持相应 eta 下的 Fock 概率不变。
- `1/t` 演化使用两个 t0、四个末时刻及额外的积分容差收紧检查。
- 每个拟合 16 个固定种子派生的起点；包含已知 MF/PBCS 候选及 Fock 边界候选。保存全部 restart 状态。
- 单进程、BLAS 单线程；数值运行至多 180 秒/512 MiB，拟合每点至多 60 秒。源码快照和环境收集在数值计时之前。异常保存配置、源码及 failure 信息；失败结果不标记通过。
- 单元测试预算：主库沿用 30 秒/256 MiB；实验 60 秒/512 MiB。测试不自动运行完整扫描。

## 结果文件

- `states.npz`：原始精确态、两种 MF spinor、投影态、拟合态、关联函数、熵、渐近态及 `1/t` 参考。
- `baselines.csv`、`asymptotic.csv`：可实现 fidelity、minor 上界、拟合状态及时间。
- `optimization_starts.json`：各次拟合的 seed、初值、终值、收敛标记及停止原因。
- `checks.json`、`timings.json`、`manifest.json`：冻结阈值、验收结果、资源数据及内容哈希。
- `source_snapshot/`：包含未提交实现的通用库、实验代码、测试与配置；仅凭基准 Git commit 无法复现新增代码。
- `figures/`：从已保存数据生成的 PNG/SVG 及绘图来源信息。

本轮记录见 [REPORT.md](REPORT.md)；完整设计与阶段边界见 [PLAN.md](PLAN.md)，数学及实现约定见 [METHOD.md](METHOD.md)。
