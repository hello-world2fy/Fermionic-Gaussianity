# Normal-state pairing formation

本实验实现用户说明“六”的三阶段研究流程。最新结论和未完成项见 [REPORT_MULTIPAIR_NOISE.md](REPORT_MULTIPAIR_NOISE.md)，
双转移 Gaussian 标签、相干噪声及相位循环读出见 [MULTIPAIR_SYMBOLS_AND_READOUT.md](MULTIPAIR_SYMBOLS_AND_READOUT.md)，
第十六轮见 [REPORT_PAIR_SYMBOLS.md](REPORT_PAIR_SYMBOLS.md)，
完整 Gaussian 配对场、正常交换项和 Slater 相干对比量见 [PAIR_SYMBOL_PHASES.md](PAIR_SYMBOL_PHASES.md)，
第十五轮见 [REPORT_OCCUPATION_RESPONSE.md](REPORT_OCCUPATION_RESPONSE.md)，
完整 Gaussian 单模式占据矩、有限秩限制与角向密度响应见 [OCCUPATION_SYMBOL_RESPONSE.md](OCCUPATION_SYMBOL_RESPONSE.md)，
第十四轮见 [REPORT_TEMPORAL_CUBIC.md](REPORT_TEMPORAL_CUBIC.md)，
时间相干三次流形、实辛传播及受限流形距离界见 [TEMPORAL_CUBIC_GEOMETRY.md](TEMPORAL_CUBIC_GEOMETRY.md)，
第十三轮见 [REPORT_COHERENT_RESPONSE.md](REPORT_COHERENT_RESPONSE.md)，
完整相干响应与暗—亮返回模见 [COHERENT_RESPONSE_AND_RETURN.md](COHERENT_RESPONSE_AND_RETURN.md)，
第十二轮见 [REPORT_DARK_PAIRS.md](REPORT_DARK_PAIRS.md)，
多次转移暗子空间、短时展开和完整相干相位矩见 [DARK_PAIR_COHERENCE.md](DARK_PAIR_COHERENCE.md)，
第十一轮见 [REPORT_ROTATION_GEOMETRY.md](REPORT_ROTATION_GEOMETRY.md)，
非线性场、同维数对照及保存有限矩阵的系数成本定义见 [ROTATION_GEOMETRY.md](ROTATION_GEOMETRY.md)，
第十轮见 [REPORT_COHERENT_BINS.md](REPORT_COHERENT_BINS.md)，
完整 Gaussian 条件相干积分与筛选定义见 [COHERENT_GAUSSIAN_BINS.md](COHERENT_GAUSSIAN_BINS.md)，
第九轮见 [REPORT_CLUSTER_FEEDBACK.md](REPORT_CLUSTER_FEEDBACK.md)，
任意加权单体涨落界及原首脉冲的精确有理弱证书见 [WEIGHTED_GAUSSIAN_BOUND.md](WEIGHTED_GAUSSIAN_BOUND.md)，
连通两次转移传播定义见 [CLUSTER_FEEDBACK.md](CLUSTER_FEEDBACK.md)，
第八轮见 [REPORT_NONLINEAR_CHANNELS.md](REPORT_NONLINEAR_CHANNELS.md)，
非线性 pair-CC 的 Pauli 修正与对照定义见 [NONLINEAR_PAIR_CC.md](NONLINEAR_PAIR_CC.md)，
第七轮见 [REPORT_COHERENT_RETURN.md](REPORT_COHERENT_RETURN.md)，
竞争通道的 Pauli 源与误判控制见 [PAULI_CHANNELS.md](PAULI_CHANNELS.md)，
第六轮初态对照见 [REPORT_ORBIT_CONTROLS.md](REPORT_ORBIT_CONTROLS.md)，
任意轨道的计数界与相干证书见 [PROJECTED_GAUSSIAN_COUNT_BOUND.md](PROJECTED_GAUSSIAN_COUNT_BOUND.md)，
第五轮报告见 [REPORT_INTERBAND.md](REPORT_INTERBAND.md)，
第五轮公式与观测量见 [INTERBAND_THEORY.md](INTERBAND_THEORY.md)，
第四轮见 [REPORT_JASTROW.md](REPORT_JASTROW.md)，
第三轮见 [REPORT_CONDITIONAL.md](REPORT_CONDITIONAL.md)，
第二轮见 [REPORT_COLUMNS.md](REPORT_COLUMNS.md)，
第一轮结果保留在 [REPORT.md](REPORT.md)，
公式与字典范围见 [THEORY.md](THEORY.md)。所有模型从正常费米海出发。

## 环境

从 `/home/tiger/Fermionic-Gaussianity-main` 运行以下命令。
CPU 部分需要 `requirements.txt` 中的 NumPy、SciPy、Matplotlib；成本修复另用 CVXPY/Clarabel。
GPU 部分使用环境已有的 PyTorch 2.9.1/CUDA；脚本兼容本机系统 Torch 安装路径。
迁移到其他机器时，需另行安装适配其 CUDA 的 Torch。
CPU 拟合使用多个独立进程；每进程数值库线程数设为 1，避免线程过度订阅。

```bash
.venv/bin/python -m unittest discover -s gaussian_synthesis/experiments/pairing_formation/tests -v
```

## 重现

以下路径变量仅为缩短命令。所有输出目录必须事先不存在；不要覆盖先前证据。

```bash
experiment=gaussian_synthesis/experiments/pairing_formation

.venv/bin/python "$experiment/scripts/run_physics.py" --output "$experiment/results/physics_screen_02"
.venv/bin/python "$experiment/scripts/run_physics.py" --levels 16 --couplings 2 4 --output "$experiment/results/physics_L16_02"

.venv/bin/python "$experiment/scripts/fit_key_times.py" --output "$experiment/results/key_fits_L12_G2_02"
.venv/bin/python "$experiment/scripts/fit_cc.py" --output "$experiment/results/cc_L12_G2_02"
.venv/bin/python "$experiment/scripts/fit_general_gaussian.py" --onsite-fits "$experiment/results/key_fits_L12_G2_02" --output "$experiment/results/general_gaussian_L12_G2_02"
.venv/bin/python "$experiment/scripts/run_stochastic.py" --output "$experiment/results/stochastic_L12_G2_02"

.venv/bin/python "$experiment/scripts/run_replica.py" --output "$experiment/results/replica2_L12_G2_02"
.venv/bin/python "$experiment/scripts/sample_full_gaussian.py" --source "$experiment/results/replica2_L12_G2_02/t1.npz" --per-device 4096 --output "$experiment/results/full_haar_L12_t1_pilot_02"

.venv/bin/python "$experiment/scripts/freeze_prediction.py" --replica "$experiment/results/replica2_L12_G2_02/t1.npz" --output "$experiment/results/frozen_L12_G2_02"
.venv/bin/python "$experiment/scripts/evaluate_prediction.py" --frozen "$experiment/results/frozen_L12_G2_02" --output "$experiment/results/prediction_L12_G2_02"
```

`run_stochastic.py` 和 Haar 采样默认使用全部 8 块 GPU；一般 projected Gaussian 拟合使用 6 块。
目前大部分小矩阵任务在 CPU 上已很快，使用全部 174 个可用 CPU 不会自动提高吞吐。

第十七轮检查实际八费米 Gaussian 标签，并以两次转移噪声和相位循环后的计数为物理读出：

```bash
.venv/bin/python "$experiment/scripts/check_multipair_symbols.py" --output "$experiment/results/multipair_symbol_checks_02" --seed 2110
.venv/bin/python "$experiment/scripts/screen_multipair_noise.py" --output "$experiment/results/multipair_noise_screen_02"
.venv/bin/python "$experiment/scripts/pilot_multipair_symbols.py" --stage train --output "$experiment/results/multipair_symbol_pilot_02"
.venv/bin/python "$experiment/scripts/pilot_multipair_symbols.py" --stage predict --output "$experiment/results/multipair_symbol_pilot_02"
.venv/bin/python "$experiment/scripts/pilot_multipair_symbols.py" --stage evaluate --output "$experiment/results/multipair_symbol_pilot_02"
.venv/bin/python "$experiment/scripts/readout_multipair_coherence.py" --output "$experiment/results/multipair_readout_02"
```

Haar 核查及标签训练均使用八块 H20；读出使用六个 CPU 进程。
训练复用 `pair_symbol_pilot_01` 和 `pair_contrast_01`；这没有新增合成或未知参数预测。
`diagnose_multipair_symbols.py --output 新目录`、`summarize_multipair_noise.py --output 新目录`
汇总固定 `_01` 结果及通过的 `multipair_symbol_tests_03.log`；其他输入须同步调整目录。
原始协议相对路径按本实验目录解释。123 项测试通过；28 条新轨迹加 16 条复用控制，十个优势条件均失败。
相位循环检验显示了低阶数据隐藏的相干读出，未证明 canonical 特有机制或新物理优先权。

第十六轮直接积分完整 Gaussian 配对场与正常交换标签，再比较相同初态的冻结传播和嵌套删除：

```bash
.venv/bin/python "$experiment/scripts/check_pair_symbols.py" --output "$experiment/results/pair_symbol_checks_03" --per-device 131072 --seed 1910
.venv/bin/python "$experiment/scripts/pilot_pair_symbols.py" --stage train --output "$experiment/results/pair_symbol_pilot_02"
.venv/bin/python "$experiment/scripts/pilot_pair_symbols.py" --stage predict --output "$experiment/results/pair_symbol_pilot_02"
.venv/bin/python "$experiment/scripts/pilot_pair_symbols.py" --stage evaluate --output "$experiment/results/pair_symbol_pilot_02"
.venv/bin/python "$experiment/scripts/pilot_pair_contrast.py" --stage freeze --source "$experiment/results/pair_symbol_pilot_02" --output "$experiment/results/pair_contrast_02"
.venv/bin/python "$experiment/scripts/pilot_pair_contrast.py" --stage evaluate --output "$experiment/results/pair_contrast_02"
```

两批 Haar 核查和八组确定性训练均使用八块 H20；传播小矩阵使用单线程数值库。
117 项测试通过，36 条主轨迹和 8 条事后对比量轨迹的 14 个优势条件全部未通过。
正常／异常项的 2∶1 比例已存在于 Gaussian 费米海，不能当作新机制。
原始运行工作目录为本实验目录，旧协议中的相对路径据此解释。
`diagnose_pair_symbols.py --output 新目录` 和 `summarize_pair_symbols.py --output 新目录`
读取固定 `_01` 证据（Haar 还包括 `_02`）；重跑其他输入时须同步调整这两个汇总脚本的目录。

第十五轮直接计算完整 Gaussian 构型的物理占据坐标，并进入六个角向响应扇区：

```bash
.venv/bin/python "$experiment/scripts/check_occupation_symbols.py" --output "$experiment/results/occupation_symbol_checks_02" --per-device 131072
.venv/bin/python "$experiment/scripts/pilot_occupation_response.py" --stage train --output "$experiment/results/occupation_response_02"
.venv/bin/python "$experiment/scripts/pilot_occupation_response.py" --stage predict --output "$experiment/results/occupation_response_02"
.venv/bin/python "$experiment/scripts/pilot_occupation_response.py" --stage evaluate --output "$experiment/results/occupation_response_02"
.venv/bin/python "$experiment/scripts/diagnose_occupation_response.py" --source "$experiment/results/occupation_response_02" --output "$experiment/results/occupation_response_diagnostics_02"
```

Haar 核查和确定性占据矩训练均使用八块 GPU；有限脉冲诊断使用十二个 CPU 进程。
训练复用同一 results 下的 `temporal_cubic_01` 四个早时刻 witness，
不读取未来响应态。`summarize_occupation_response.py --output 新目录`
汇总固定的 `_01` 证据；独立重跑须同步修改汇总输入。
第十五轮原始运行工作目录为本实验目录，旧协议中的相对输入路径也据此解释。
109 项测试通过；十二个优势判据均未通过，没有新增物理参数留出预测。

第十二轮的物理暗子空间、完整 Gaussian 合成及相干积分：

```bash
.venv/bin/python "$experiment/scripts/screen_dark_pairs.py" --output "$experiment/results/dark_pairs_02"
.venv/bin/python "$experiment/scripts/control_dark_pairs.py" --output "$experiment/results/dark_pair_controls_02"
.venv/bin/python "$experiment/scripts/check_dark_onsets.py" --output "$experiment/results/dark_onsets_02"
.venv/bin/python "$experiment/scripts/train_dark_gaussian.py" --output "$experiment/results/dark_gaussian_training_02"
.venv/bin/python "$experiment/scripts/integrate_dark_gaussians.py" --stage setup --source "$experiment/results/dark_gaussian_training_02" --output "$experiment/results/dark_gaussian_moments_03"
.venv/bin/python "$experiment/scripts/integrate_dark_gaussians.py" --stage integrate --output "$experiment/results/dark_gaussian_moments_03" --per-device 32768 --seed 2952
.venv/bin/python "$experiment/scripts/assess_dark_gaussians.py" --source "$experiment/results/dark_gaussian_moments_03" --output "$experiment/results/dark_gaussian_assessment_03"
```

相干积分默认八块 GPU；第二批归档使用每 GPU 524288 原子、种子 1952 至 1959，
通过 setup 的 `--reuse-calibration results/dark_gaussian_moments_01` 复用首批固定坐标。
`pilot_dark_phase_parts.py` 依次运行 `--stage train/predict/evaluate --output 新目录`，
训练时以 `--source` 指定合成目录，直接计算复标签的完整 Cartan 相干矩。
`refine_dark_phase_parts.py --output 新目录` 做四条数值复核，
`summarize_dark_pairs.py --output 新目录` 汇总第十二轮固定归档。
后两者及物理对照读取代码中明确的 `_01`／`_02` 数据；完整重跑须同步修改这些输入。
新增 GPU Cartan 后端限定 L≤14；原 CPU 后端 L≤12 的保护不变，均不是多项式成本实现。
L=22 只完成路径截断严格覆盖所需阶数的短时分块计算，没有声称完成其长时动力学。

第十一轮的局部模式和普通非线性场对照：

```bash
.venv/bin/python "$experiment/scripts/search_gaussian_modes.py" --output "$experiment/results/gaussian_modes_02" --workers 16
.venv/bin/python "$experiment/scripts/check_general_gaussian_modes.py" --source "$experiment/results/gaussian_modes_02" --output "$experiment/results/general_gaussian_modes_02"
.venv/bin/python "$experiment/scripts/screen_rotation_geometry.py" --stage predict --output "$experiment/results/rotation_geometry_02"
.venv/bin/python "$experiment/scripts/screen_rotation_geometry.py" --stage evaluate --output "$experiment/results/rotation_geometry_02"
```

一般 Gaussian 模式检查默认使用八块 GPU，每块四次局部初始化；不是完整测度抽样。
旋转脚本默认两个 CPU 进程，L=14、18，16 方向、种子 861、奇异值阈值 1e-9。
另外四个已完成集合分别使用阈值 1e-10，64 方向／种子 862，
`--matching shifted`，`--matching independent`；每个新目录均先 predict 后 evaluate。
`audit_rotation_mechanism.py`、`certify_rotation_dictionary_cost.py`、
`compare_rotation_rank_controls.py` 和 `summarize_rotation_geometry.py`
均以 `--output 新目录` 运行，并读取代码中固定的本轮 `_01` 数据。
同维数对照的第一次序列化失败已保留；完成并被汇总的版本是 `rotation_rank_controls_02`。
完整独立重跑须同步修改各脚本的输入和汇总路径；只改输出名不会切换整条证据链。
所有本轮条件都已查看，属于机制与表示容量检查。

第十轮的完整 Gaussian 分组积分与同初态先导传播命令：

```bash
.venv/bin/python "$experiment/scripts/screen_chirality_spectrum.py" --output "$experiment/results/chirality_spectrum_02"
.venv/bin/python "$experiment/scripts/integrate_gaussian_bins.py" --stage setup --output "$experiment/results/gaussian_bins_L10_02"
.venv/bin/python "$experiment/scripts/integrate_gaussian_bins.py" --stage integrate --per-device 131072 --output "$experiment/results/gaussian_bins_L10_02"
.venv/bin/python "$experiment/scripts/assess_gaussian_bins.py" --source "$experiment/results/gaussian_bins_L10_02" --output "$experiment/results/gaussian_bins_assessment_02"
```

积分阶段默认八块 GPU，准备阶段仅需一块 GPU。四个先验物理条件都已查看过。
`scripts/pilot_gaussian_bins.py` 按 `--stage train/predict/evaluate` 顺序运行；
`scripts/check_bin_prediction_noise.py` 复核独立半池及块重采样；
`scripts/summarize_coherent_bins.py` 汇总第十轮固定归档。
这些后续脚本读取明确写入代码的 `_01` 训练和精确参考路径；
完整重跑必须同步修改输入路径，不能仅改输出编号就视为新一轮独立证据。

第九轮 `scripts/run_cluster_feedback.py --stage train/predict/evaluate --output 新目录`
依次完成两次转移几何训练、同初态传播、评价；`predict` 默认使用八个 CPU 进程。
`scripts/screen_orbital_fluctuations.py --output 新目录` 在八块 GPU 上做 512 个局部轨道搜索。
`scripts/check_weighted_count_bound.py --output 新目录` 核查加强的加权界；
`scripts/certify_weighted_pulse.py --output 新目录` 用精确整数和分数认证固定有理脉冲条件。
这些脚本依赖已有 `_01` 训练/先导归档；完整重跑须同步修改输入路径。
`scripts/validate_cluster_feedback.py` 支持 `--metric-solver svd --error-control physical`，
及显式容差、metric 阈值、步长、`--compare-directory` 参数。
两级完成记录为 `cluster_feedback_physical_validation_01` 和 `_02`；
原坐标误差控制的两组未完成记录保留 `termination.json`。
`scripts/summarize_cluster_feedback.py --output 新目录` 汇总固定第九轮归档，
包括 `weighted_count_bound_02`、`rational_pulse_certificate_01` 及 73 项测试日志。

第八轮主脚本为 `scripts/run_nonlinear_channels.py`，依次运行 `--stage init`、
`--stage cc`、`--stage stochastic`、`--stage evaluate`。
`cc` 与 `stochastic` 可并行；必须等待两者完成后再运行 `evaluate`，后者会先冻结预测哈希。
随机阶段以 `--workers 8` 使用全部八块 GPU，每次运行保存 64 个独立误差块。
默认每个条件/种子采样 262144 条轨迹。

第八轮固定归档复核和诊断脚本包括 `check_nonlinear_channels.py`、
`refine_L18_channels.py`、`diagnose_channel_spans.py`、`pilot_angular_canonical.py`、`check_angular_aliases.py`、
`summarize_nonlinear_channels.py` 和 `finalize_nonlinear_audit.py`。
它们读取明确写在脚本中的本轮 `_01` 数据，不能仅修改输出目录就宣称重跑了另一整轮。
主预测之外的 Gaussian 空间、canonical 排序及混叠检查均标记为事后机制诊断。

`summarize.py` 是第一轮 `_01` 数据的固定归档脚本，默认创建 `assessment_01`，
其中已有 PNG/PDF 图、审计 JSON 和生成时的文件哈希快照。后续轮次应显式更新输入路径及新输出名。

第二轮脚本以已有 `_01` 数据为固定输入，输出必须使用新目录；以下命令依赖已有第一轮归档。
`freeze_response_columns.py` 同样读取归档的 `response_columns_01`，并不自动读取上条命令的 `_02`。
若要完整重跑新一轮，应先显式修改脚本中的输入路径，避免混用实验编号。

```bash
.venv/bin/python "$experiment/scripts/run_columns.py" --output "$experiment/results/adaptive_full_L12_G2_02"
.venv/bin/python "$experiment/scripts/assess_column_costs.py" --output "$experiment/results/column_costs_02"
.venv/bin/python "$experiment/scripts/freeze_columns.py" --output "$experiment/results/column_freeze_02"
.venv/bin/python "$experiment/scripts/evaluate_columns.py" --frozen "$experiment/results/column_freeze_02" --output "$experiment/results/column_predictions_02" --workers 24
.venv/bin/python "$experiment/scripts/train_response_columns.py" --output "$experiment/results/response_columns_02"
.venv/bin/python "$experiment/scripts/freeze_response_columns.py" --output "$experiment/results/response_freeze_02"
.venv/bin/python "$experiment/scripts/evaluate_columns.py" --frozen "$experiment/results/response_freeze_02" --output "$experiment/results/response_predictions_02" --workers 24
.venv/bin/python "$experiment/scripts/run_sector_diagnostics.py" --output "$experiment/results/sector_diagnostics_02" --workers 24
.venv/bin/python "$experiment/scripts/summarize_columns.py" --output "$experiment/results/assessment_columns_02"
```

`run_columns.py` 使用六块 GPU，响应训练使用四块；每个 GPU 任务的 CPU 数值库线程数为 1。
预测和规模/能谱对照使用 24 个独立 CPU 进程。资源按独立任务数量配置，不为了占满硬件重复计算。

## 模块

| 模块 | 内容 |
|---|---|
| `model.py` | 稀疏精确 Hamiltonian、计数统计、相干四点函数 |
| `linear.py` | 有限尺寸响应、Riccati 压缩矩阵、费米 CC lift |
| `structure.py` | 条件 Schmidt 界、单方向径向诊断、onsite 计数界 |
| `fitting.py` | PBCS、任意径向系数及转移数依赖轨道的可行拟合 |
| `coupled_cluster.py` | 优化 rank-1/2/full-rank pair-CC |
| `projected_gaussian.py` | 完整 fixed-N 范数的一般 Gaussian 拟合 |
| `replica2.py` | 完整 Gaussian 字典的确定性二阶合成 |
| `gaussian.py` | 完整 SO(2n)-Haar、带 pivot 的 Pfaffian |
| `stochastic.py` | Gaussian-noise SMF、离散 TWA 及计数符号 |
| `frozen.py` | 冻结集体方向、Gaussian 生成空间及 Hamiltonian 传播 |
| `columns.py` | 完整 Fock 范数、局部多起点 Gaussian 对称轨道搜索 |
| `column_prediction.py` | 实际列空间冻结、自由输运、相干核及删除控制 |
| `costs.py` | Fock 残差修复 SOCP、解析支持界的 magic witness |
| `sectors.py` | 转移数分辨电流、保持计数的相位控制及条件秩一替代 |
| `conditional.py` | 转移数条件轨道、密度导数空间、能量分辨观测量 |
| `conditional_dynamics.py` | 共同/条件轨道的 Hamiltonian-only TDVP，含 Berry 相位补偿 |
| `jastrow.py` | 两体密度 Jastrow、计数多项式轨道、组合族及同初态 TDVP |
| `density_cubes.py` | 独立于局部拟合的八构型概率约束及解析保真度界 |
| `multiorbit.py` | 每个转移数的多个相干轨道乘积、变量投影拟合及 Hamiltonian TDVP |
| `interband.py` | 固定计数纠缠、局域滤波与 Pauli 层析、PPT 见证、法向源与局域删除 |
| `orbit_response.py` | 早期态与 Hamiltonian 响应的共同轨道提取、训练内模型选择 |
| `projected_count_bound.py` | 任意轨道全 fixed-N 计数界、独立 Pfaffian 核查、QFI 与有限保真度界 |
| `coherent_return.py` | 低秩混态的计数相干、完整 QFI、回波和精确平坦集体链 |
| `pairing_channels.py` | 一般 Hermitian 配对核、有限尺寸线性响应与角向竞争通道观测量 |
| `cluster_dynamics.py` | 冻结连通转移簇、Euler 递推、自由复径向参数及 Hamiltonian DF 传播 |
| `orbital_fluctuations.py` | 任意轨道计数方差，保留破对输出和干涉；GPU 可微版本 |
| `weighted_count_bound.py` | 固定总数的一般单体加权界、全 fixed-N 有限保真度障碍 |
| `rational_evolution.py` | 整数 Taylor 作用与 Hermitian 实时间演化的严格有理余项 |

第三轮入口是 `fit_conditional_orbitals.py`、`fit_radial_cc.py`，
随后分别用 `freeze_conditional.py` / `evaluate_conditional.py` 做静态对照，
`freeze_conditional_dynamics.py` / `evaluate_conditional_dynamics.py` 做轨道运动预测。
`freeze_conditional_scale.py --levels 8`（或 16）产生规模复核初值，交给同一个动态评价脚本；
`freeze_shared_conditional.py` 产生完全相同初始态的结构删除控制。
所有冻结/拟合脚本都需要 `--output 新目录`；评价脚本需要 `--frozen 冻结目录 --output 新目录`。
`validate_conditional_dynamics.py --levels 12`（或 16）另检验步长、容差与 metric 截断，
`summarize_conditional.py --output 新目录` 归档固定 `_01` 数据并保存独立测试日志。
这些脚本的输入默认指向归档 `_01` 文件；迁移实验编号时需明确修改输入路径，避免误混。

第四轮入口是 `fit_jastrow.py`（默认四族，扩展族用 `--families collective_3 conditional_two_body`），
`freeze_jastrow.py` / `evaluate_jastrow.py`，以及 `freeze_jastrow_scale.py --levels 8`（或 16）。
`validate_jastrow.py --levels 12`（或 16）加严积分与 metric 截断；
`check_density_cubes.py` 检查概率约束，推导见 [DENSITY_CONSTRAINT.md](DENSITY_CONSTRAINT.md)。
`summarize_jastrow.py --output 新目录` 汇总固定第四轮 `_01` 数据、图和独立测试日志。
这些冻结和汇总脚本同样以已有归档为固定输入，完整重跑须同步修改输入目录，而不只是输出编号。

第五轮入口是 `fit_multiorbit.py`、`freeze_interband.py` / `evaluate_interband.py`。
`diagnose_interband.py` 计算离线 Schmidt、法向源及局域去相干控制；
`freeze_interband_scale.py --levels 8`（或 16）产生尺寸对照，使用同一评价脚本。
`freeze_reversal_confirmation.py` 冻结反转确认条件与两个拟合种子。
加严拟合用 `fit_multiorbit.py --times 1 --maxiter 5000`，随后将新的目录名传给
`freeze_reversal_confirmation.py --fits-directory 目录名 --refinement`；
此操作在原确认网格做数值复核，不产生新的独立条件。

`validate_multiorbit.py --levels 12`（或 16）检查积分与 SVD 截断；
`evaluate_interband_linear_controls.py --frozen 冻结目录 --output 新目录` 增加散射及 Riccati 对照。
`freeze_entanglement_witnesses.py` / `evaluate_entanglement_witnesses.py` 冻结预测选择的 PPT 见证并生成合成测量；
其中部分转置见证与 Gaussian 合成的向量 witness 不同。
`summarize_interband.py --output 新目录` 核查第五轮固定 `_01` 归档、运行完整测试并生成图。
所有冻结/拟合脚本需要 `--output 新目录`，未来态评价脚本需要 `--frozen 冻结目录 --output 新目录`。
多数训练、冻结和汇总入口仍指向固定归档输入；完整重跑须同步改输入，而非只换输出编号。

第六轮入口为 `train_orbit_response.py --output 新目录 --workers 24`，
`freeze_orbit_controls.py --mode response`（或 `swaps`）`--output 新目录`，
以及 `evaluate_orbit_controls.py --frozen 冻结目录 --grid known`（或 `unseen`）
`--output 新目录 --workers 24`。交换冻结只有 known 网格。
`validate_orbit_controls.py --output 新目录` 加严四种模型的积分；
`check_projected_count_bound.py --output 新目录 --workers 12` 核查解析界并评价保存的精确轨迹；
`evaluate_count_echo.py --output 新目录` 生成两副本与计数的合成测量；
`summarize_orbit_controls.py --output 新目录` 核查固定第六轮归档、运行测试并生成图。
这些脚本仍以已有 `_01` 数据为固定输入；完整独立重跑须同步修改输入路径及输出编号。
尤其响应冻结读取 `orbit_response_training_01`，积分验证和汇总也读取固定的第六轮目录，
不能只改输出名就假定整条流程使用了新的训练结果。

第七轮先用 `run_replica.py --G 4 --times .5 1 --output 新目录` 产生早期 witness，
然后运行 `run_coherent_return_pilot.py --output 新目录 --workers 8`，
`predict_coherent_return.py --output 新目录 --workers 8` 和
`validate_coherent_return.py --output 新目录`。
角向模型初筛用 `screen_pairing_channels.py --output 新目录 --workers 6`，
随后 `diagnose_channel_geometry.py`、`check_pauli_channels.py`、`check_channel_closure.py`
各自使用 `--output 新目录`。最后 `summarize_coherent_return.py --output 新目录` 生成审计和图。
这些脚本依赖固定 `_01` 归档：先导读取 `replica2_L12_G4_return_01`，
新预测和验证读取 `coherent_return_pilot_01`，角向几何读取 `pairing_channels_screen_01`。
完整独立重跑必须同步修改这些输入，不能只改输出编号。

研究状态持续记录在 [STATUS.json](STATUS.json)。晚时均匀模型的回波可由常规三通道空间预测；
竞争通道初筛发现了可解析追踪的 Pauli 诱导谐波，同时保留 Hamiltonian 精确闭合空间这一替代解释。
单实轨道反例排除了由负四配对算符关联直接推断手性的做法。
第八轮已加入多种强非线性基线，第九轮新增了同初态的连通二次转移对照；
canonical 独有的前瞻删除控制仍未通过，重要新物理和方法独特性仍未建立。
原首脉冲现在有严格但极弱的一般 PG 排除；局部拟合和低阶合成残差仍不是全局最优证书。
第十一轮进一步加入普通非线性场和同维数 Hamiltonian 块响应；后者在所测条件下更准确。
L=14 四谐波对照的有效对称维数是 116，原 2233 维环境不是本初态的必要有效成本。
本轮精确有理有限列成本证书只适用于保存矩阵，不是完整 Gaussian extent 下界。
第十二轮的同初态删除与短时展开定位了连通多次转移的暗成分，但没有建立新颖性或长寿命。
正实人口坐标的相干积分未通过方向稳定门槛；无 Monte Carlo 噪声的复相位矩传播也未越过强响应对照。
研究仍须在真正未见条件中完成 canonical 结构独有的物理预测，不能以这些诊断代替第三阶段。

## 第十三轮复现

在本实验目录运行；以下新目录必须不存在。

```bash
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/pilot_coherent_response.py --stage train --output results/coherent_response_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/pilot_coherent_response.py --stage predict --output results/coherent_response_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/pilot_coherent_response.py --stage evaluate --output results/coherent_response_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/check_dark_paths.py --workers 7 --output results/dark_paths_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/check_return_pair_mode.py --output results/return_pair_mode_02
```

完整响应训练使用八个 GPU 进程，逐方向计算完整 Cartan 导数；没有随机积分样本。
训练默认读取第十二轮固定 `_01` 训练、修复和公共空间文件。
预测阶段只读冻结空间和 Hamiltonian，评价阶段才读取已知参考轨迹。
路径及返回模检查依赖第十一／十二轮的固定对称空间和参考文件。
汇总入口 `summarize_coherent_response.py --output 新目录` 默认核查本轮 `_01` 归档，
并复制 `results/coherent_response_tests_01.log`；完整迁移重跑须同步修改这些输入。

第十三轮的完整响应选模仍未通过强对照。显式五阶返回模通过有限尺寸核查，
但一般尺寸证明未完成；r=6 的范数有三倍混叠修正。
L=62 仅计算短时系数，不代表该规模长时动力学已经完成。

## 第十四轮复现

在本实验目录运行，每个新输出目录须事先不存在。

```bash
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/pilot_temporal_cubic.py --stage train --output results/temporal_cubic_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/pilot_temporal_cubic.py --stage tensors --output results/temporal_cubic_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/pilot_temporal_cubic.py --stage predict --output results/temporal_cubic_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/pilot_temporal_cubic.py --stage evaluate --output results/temporal_cubic_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/pilot_temporal_cubic.py --stage refine --output results/temporal_cubic_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/diagnose_temporal_cubic.py --source results/temporal_cubic_02 --output results/temporal_cubic_diagnostics_02
/home/tiger/Fermionic-Gaussianity-main/.venv/bin/python scripts/diagnose_late_witness.py --source results/temporal_cubic_02 --output results/late_witness_02
```

训练只新传播至 t=1.35；合成和张量各使用八个 GPU 进程。
传播使用实 Hamiltonian 变分方程，处理映射对复坐标及其共轭的共同依赖。
局部拟合和晚期 witness 都是评价之后的诊断，不能回填训练或称为预测。
普通历史三次对照共同包含终点 witness，以保证两个起始态都完全相同。
评价默认读取既有 `nonlinear_channels_01` 参考，训练读取 `rotation_geometry_01` 的精确对称基。
汇总入口 `summarize_temporal_cubic.py --output 新目录` 固定读取本轮 `_01` 数据及
`results/temporal_cubic_tests_01.log`，完整迁移重跑须同步更改这些输入。

第十四轮的非线性时间历史方案仍未越过强预测对照。
新增强单调性不等式针对固定数目、无未配对电子振幅空间中的完整 Gaussian k=2 映射；
其数值代入不是精确物理态的区间认证，也不是一般 projected Gaussian 保真度界。
重要新物理、canonical 独有结构和真正未见条件中的第三阶段成功仍未建立。

## 最新强控制补充复现

在本实验目录运行（Python 使用仓库 `.venv/bin/python` 的绝对路径）：

```bash
python scripts/stochastic_multipair_readout.py --output results/stochastic_multipair_readout_02
python scripts/assess_stochastic_multipair.py --source results/stochastic_multipair_readout_02 --output results/stochastic_multipair_assessment_02
python scripts/uniform_multipair_controls.py --stage snapshots --output results/uniform_multipair_controls_02
python scripts/uniform_multipair_controls.py --stage stochastic --output results/uniform_multipair_controls_02
python scripts/uniform_multipair_controls.py --stage assess --output results/uniform_multipair_controls_02
```

角向随机评估复用既有 `multipair_readout_01` 精确参考；均匀模型脚本复用既有最佳保真度拟合。最新汇总位于 `results/assessment_multipair_noise_02`。汇总脚本固定读取原始 `_01` 实验输入及测试日志 `_03`；重跑其他输入时须同步调整。两组随机实验合计 2097152 条独立轨迹，同种子步长重复不重复计数。当前目标仍未完成，详见最新报告补充。
