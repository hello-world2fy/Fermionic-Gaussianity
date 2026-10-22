# 四模 Gaussian 构型分布实验

这是独立于通用库的实验目录。实验只单向调用现有 `gaussian_synthesis` 包，未修改通用库的源码、公共 API、依赖或测试入口。

## 环境与复现

需要 Python 3.10+；本次在独立 `.venv` 中使用 Python 3.12.14、NumPy 2.5.4 和 Matplotlib 3.11.2。系统默认 `python3` 可能过旧，请先检查版本。

在本目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python run_tests.py
.venv/bin/python ../../run_tests.py
.venv/bin/python static_cat.py --full --output results/my_static
.venv/bin/python interaction_quench.py --full --output results/my_quench
.venv/bin/python plotting.py --input results/my_static
.venv/bin/python plotting.py --input results/my_quench
```

完整实验必须显式传入 `--full`；`--seed` 默认 `20261022`。已有输出目录不会被覆盖。`requirements.txt` 是可用依赖范围，`requirements-lock.txt` 固定本次使用的依赖版本。所有运行都会记录实际版本与源码哈希；旧数值结果不因后续绘图或报告修订而改变。

绘图仅读取已有结果，单独生成 PNG/PDF，不重新采样或训练。数值计算完成后会生成中文 `report.md`；重新绘图后可执行下面的命令刷新报告中的图片链接：

```bash
.venv/bin/python -c "from common import write_report; write_report('results/my_static'); write_report('results/my_quench')"
```

## 代码组织

| 模块 | 职责 |
|---|---|
| `reference.py` | cat/quench 态、解析 witness 与 extent、Schmidt Gaussian 变换、二次 Hamiltonian 对照 |
| `exact_solver.py` | 一般四模偶宇称态的精确 k=2 目标与完整 16 实维阻尼 Newton 求解 |
| `observables.py` | 流式分布统计、辅助矩阵、直方图、TV、重复标准误与交叉纯度 |
| `common.py` | 整体预算、独立随机流、扫描、SGD 锚点、保存和中文报告 |
| `static_cat.py` / `interaction_quench.py` | 两个独立的命令行入口 |
| `plotting.py` | 从保存数据生成科学图表 |
| `diagnostics.py` | 重放原有样本，诊断动态拉回直方图的事后回归标记 |
| `tests/` / `run_tests.py` | 实验专属测试；不会进入主库测试的自动发现范围 |

## 数学与数值约定

使用主库 bitmask 顺序 `[0,3,5,6,9,10,12,15]`，模式 j 对应 bit j，产生算符按模式编号升序排列。内积对第一个向量取共轭。逻辑 pair 基 `[00,01,10,11]` 对应 masks `[0,12,3,15]`，不能直接使用升序的四个非零分量。

Haar 测度归一化为 1，`d_(4,2)=35`。优化目标及其梯度为

\[
J_2(u)=q_2(u)/4-\operatorname{Re}\langle u|\psi\rangle,
\quad\nabla J_2=C_2(u)-\psi,
\]
\[
q_2(u)=\|u\|^4-|u^TJu|^2/8,
\quad C_2(u)=\|u\|^2u-(u^TJu)J\bar u/8.
\]

确定性求解从目标态本身初始化，对完整 8 维复空间求解，残差目标为 `1e-10`、最多 100 次迭代。解析参考不作为求解器或 SGD 的初值。

静态态是 `sqrt(1-r)|vac>+sqrt(r)|full>`；解析 witness 用 `t=b/a` 的严格单调标量方程二分求根。动态态是 `(vac+12+34+exp(-i phi)full)/2`。Schmidt 逻辑酉矩阵以合法 Gaussian 酉扩展到整个八维空间：pair 偶扇区使用 W，奇扇区使用 `sqrt(det W) I`。该扩展保留完整相位和 Gaussian quadric。

分布为 `p ∝ |<g|u>|³`、`h ∝ |<g|psi>|³`。矩阵 `R=E_p |g><g|` 是辅助 ensemble，不是目标物理纯态的密度矩阵。连续系数质量 `B=35 E_mu |<g|u>|³`，`B²` 与解析 Gaussian extent 分开报告。

magic 点的精确 purity 基准来自 `X=2|<g|M>|² ~ Beta(1,3)`。三次振幅重加权后 `X ~ Beta(5/2,3)`，因此 R 的谱为 `5/22` 和七个 `17/154`，purity 为 `116/847`。

## 运行配置与验收

- 静态扫描 51 点 `r=0..0.5`；SGD 锚点为 `0,0.1,0.25,0.5`。
- 动态扫描 81 点 `phi=0..2pi`；SGD 锚点为 `0,pi/2,pi,3pi/2,2pi`。静态预测在每个准确 `r(phi)` 上计算，不插值。
- 二次对照使用 `H_Q=sum(j*n_j), j=1..4`，从同一 Gaussian 初态演化。
- 每组分布观测使用 4 条独立随机流，各 65,536 个样本；每条流内部 p/h 和参数点共用样本。直接观测、静态预测、自由演化、SGD 训练、诊断与重构验证相互分流。
- 每个完整数值实验共用 180 秒 / 512 MiB 预算；单进程、数值库线程数为 1，单批最多 8192 个构型。绘图独立于此数值预算。
- 九个 SGD 锚点各分配失败概率 `0.05/9`，每点最多 1,200 万正式验证样本。验收要求 witness 误差不超过 `0.01`、独立重构上界不超过 `0.05`。
- 采样曲线与直方图使用 `max(6*combined_SE, 0.002)` 作预先指定的数值回归阈值；这不构成整条曲线的严格联合置信保证。没有更换随机种子以筛选通过结果。

整个运行的 `acceptance_passed` 要求所有预定检查通过。即使优化与正式重构均成功，少数经验直方图比较超阈值仍会保留 `failed_acceptance`，退出码为 1。它不表示数值数据缺失；`completed_stages` 与各个检查会明确指出完成情况。

## 数据口径

每个结果目录保存：

- `config.json`：配置、环境、源码哈希、随机流及统计约定。
- `summary.json` / `summary.csv`：验收状态、资源使用、逐点指标。
- `vectors.npz`：参数、目标、完整空间解、解析参考和残差；动态另存完整 Gaussian 变换。
- `observations.npz`：直接分布统计；动态另有 `reference_observations.npz`、`control_observations.npz` 和 `control_vectors.npz`。
- `anchors.csv` / `anchors/`：SGD witness、训练记录、冻结后独立重构与证书。
- `source_snapshot/`：数值运行时顶层实验 Python 源码快照，用于溯源；绘图或报告后续修订不会回写旧快照。
- `figures/` / `report.md`：PNG、PDF 图及中文分析。

`hist_n_*` 和 `hist_xy_*` 是每个 bin 的概率质量，不是密度。N 分为 40 个区间 `[0,4]`；二维坐标是构型的真空、满占据概率，各分为 40 个区间 `[0,1]`。`*_pulled_*` 使用 `U_G†g` 的坐标。

每个 block 都独立归一化。常规输出为四个 block 估计的等权均值；`*_se` 为 block 间标准误，`blocks_*` 保留各 block 指标。`R_*` 是平均矩阵，`purity_*` 是块纯度的平均，`spectrum_*` 是排序后块谱的平均，`spectrum_of_mean_R_*` 才是平均矩阵自身的谱。不同有限样本对象不可混用。

`purity_cross_*` 是独立 block 矩阵交叉乘积迹的平均，标准误由 delete-one jackknife 给出。它移除了同一经验矩阵的自配对项，但不能消除自归一化比值的全部有限样本偏差。ESS 按一个 block 的 65,536 个样本报告。

TV 使用两遍相同随机流：第一遍得到归一化积分，第二遍重放相同样本累积差值，无需保存全部构型。直方图和辅助矩阵均流式计算，不构造样本两两重叠矩阵。

## 本次结果入口

先阅读 [结果与分析](RESULTS.md)，再按需查看 [静态完整报告](results/static/report.md)、[动力学完整报告](results/quench/report.md)。首次运行结果和所有失败标记均予保留。

对动力学少数直方图回归标记的附加诊断可单独运行：

```bash
.venv/bin/python diagnostics.py --input results/quench
```

这只重放已记录的随机流，不更换种子，不新增正式样本，不修改原始验收。输出的近似误差评估只用于解释，不用于事后修改原阈值或追认通过。
