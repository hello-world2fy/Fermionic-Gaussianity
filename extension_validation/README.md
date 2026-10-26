# extension_validation — 扩展验证复现与审计根目录

本目录是 fermionic Gaussianity（费米高斯凸合成框架）三项扩展验证的**自包含复现根**。
所有新增代码、脚本与产物只落在本目录内；对 `Gaussian_Convex_Synthesis_Algorithm.pdf`、
`paper_v2/`、`gaussian_synthesis/` 一律**只读**，不改动、不移动、不删除任何既有文件。

> **当前状态（终版）**：三方生产者（A/B/C）结果已冻结，独立 review 已确认全部最终数字。
> 本 README 记录目录结构、复现接口与固定参数；最终数值结论以权威交付物
> `report/推广验证报告.html` 为准。核心结果：A Fock 解析 sandwich、交互 BH 区间诚实放宽；
> B 固定预算无系统优势（个别点有改善）、dimer 到机器精度；C `R(ρ_M)∈[3.0, 3.000028]` 夹逼（非精确 R=3）。
>
> **覆盖范围（最终修正版）**：A 完成 Fock 基准、Bose–Hubbard 成本界、真正凝聚分数和单格点纠缠熵对照，未做临界标度外推。B 当前证据是 `slater/results/timed_bench.csv` 的 **456 条 `ritzfix_v2` 运行**：Hubbard 固定 n=8/12、H₂ 三键长固定 n=2/3/4，以及 0.3/0.8/1.5 秒实际截止实验；FCI 只用于评分。旧 `complete_ci.csv` 及旧计时结果因非正交 Ritz 选择目标错误全部撤回。C 当前为 **16 个 Gibbs 点、2856 个独立审计合法的 Gaussian 原子**；有限特征合成已完成，仍无严格连续 witness。
>
> **Git 轻量发布**：保留代码、最终 CSV/JSON、报告及关键 Gaussian 分解 NPZ。虚拟环境、ZIP、截图、图像、旧诊断、逐次 Slater checkpoint 和大批哈希清单仅保留在本地，不上传。完整轨迹验证日志描述的是本地保存的检查点；克隆后若要重做该验证，请先运行 `slater/timed_bench.py` 生成检查点。最终结果以报告、`timed_bench.csv` 与 `hubbard_curve_records.npz` 为准，以下旧阶段说明仅作历史背景。
>
> **非终版材料警示**：`review/*findings.md`、`review/review.json` 的早期版本与 `slater/results/diagnostics/`、
> 旧 Hubbard 多点曲线均为**过程/失败诊断，非终版结论**；一切以 `review/review.json`（终版）与 `report/推广验证报告.html` 为准。

---

## 一、三项实验 A / B / C

| 代号 | 目录 | 验证对象 | 数学性质 | 连续证书 vs 有限近似 |
|---|---|---|---|---|
| **A** | `hartree/` | 玻色 Hartree / 相干帧提升 `H=Sym^N C^m` | 确定性 Gauss-Jacobi/FFT 求积，**无采样**；解析 Fock 见证 `u_k=q_k(n)^{-1/(2k-1)}\|n>` | 解析 sandwich `t_k^2 ≤ ξ ≤ B_k^2` 为连续证书；m≥3 的 SOCP 上界为有限字典近似 |
| **B** | `slater/` | Slater 残差合成（half-filled Hubbard 链）+ dimer extent 解析基准 + H2/STO-3G | dimer `ξ_S=1+U/√(U²+16t²)` 由独立 2-fermion Takagi 分解到机器精度 | dimer 为解析基准；决定性测试是 `run_complete.py` 的 **Fock-complete** 可行性（全外部 Fock 基+40 非正交 Slater 原子，span 精确含残差，KKT~1e-6），仍为**固定行列式数 n=8/12、固定池**的有限字典，不是连续 Grassmannian 证书 |
| **C** | `signed_gaussian/` | 有符号高斯算子凸合成（密度矩阵 TV 范数 R） | 连续见证 `W=4ρ_M−P_+` 给出解析下界 `R≥3`；最终有限确定性 J 由 `run_exact_convex.py`（L-BFGS/闭式）求解 | 下界为连续证书；`R=3` 的精确连续上界未认证，终值为夹逼 `R(ρ_M)∈[3.0, 3.000028]`——有限字典 LP 是连续 R 的**有效数值上界**（原子构成全高斯流形子字典、全矩阵重构，浮点残差 ~2e-15），**但不是连续最优性证书**；white-noise 仅 3 点与解析线一致、平台尚需 hull 论证（未证） |

**最终口径备忘（已落实进报告）**
- Hubbard **局部投影字典**下 full-KKT ≈0.95 是**失配 demo**（局部 40 原子字典只张成残差的 ~5–30%），**不是对算法的否定**；旧诊断已移入 `results/diagnostics/`。决定性结论以 Fock-complete 的 `complete_ci.csv`（KKT 中位数 ~1.4e-6、max ~3e-4）为准。
- B 固定预算结论为**无系统优势**（个别点与 CIPSI 相当或略好），不是「never beats」也不是「退化为 CIPSI」；wall钟 replica 慢约 3–12×（实测累计 2.6–11.9×）。
- C 的有限池 witness 比值随池加密收缩，连续 `‖W‖` 界下不成立；metric-fix 后少数可观测量**特征匹配**（k=1 err 1.2e-11、k=2 err 4.7e-9，signed ensemble mass 2.46/1.65）是 `Tr O σ` 对齐，**不是**全 ρ 重构，也**无严格连续 witness**。
- **数据状态**：全部以 producer 最终版为准（B `complete_ci.csv` 576 行；C `magic_curve.csv`/`exact_convex_summary.json` 为 flatten(64) 修复后重跑，取代旧 Imag 残差版本）。

---

## 二、目录结构（浅层）

```
extension_validation/
├── README.md                  ← 本文件（复现根入口）
├── requirements.txt           ← 复现环境冻结（见第五节）
├── run_all.py                 ← 安全快照式复现驱动（默认 dry-run，绝不覆盖已存结果）
├── .venv/                     ← 共享 Python 环境（CPython 3.14.7）
│
├── hartree/                   ← A：玻色 Hartree 帧
│   ├── README.md              ← A 的详细说明与复现命令
│   ├── code/                  ← boson_frame / exp1_moments / exp2_convex / exp3_bosehubbard / exp4_rdm（真凝聚分数）/ make_figures / aggregate_summary / run_tests
│   └── results/               ← A 产物（CSV/JSON/figures/）+ exp4_rdm.csv/json（γ₁ RDM 真分数/熵）+ summary.json
│
├── slater/                    ← B：Slater 残差合成 / dimer / H2
│   ├── run_benchmark.py run_complete.py complete_ci.py plot_results.py timed_bench.py make_summary.py
│   ├── results/               ← complete_ci.csv（576 行）/ timed_bench.csv（456 行：fixed_n+fixed_time，Hubbard n8/12、H2 n2/3/4）/ dimer_extent.csv / h2_demo.csv / diagnostics/
│   ├── artifacts/             ← 重绘 PNG
│   ├── tests/test_slater.py
│   └── summary.json
│
├── signed_gaussian/           ← C：有符号高斯算子合成
│   ├── README.md              ← C 的详细说明与复现命令
│   ├── sggen/                 ← 独立库 gaussian / synthesis / hubbard / convex
│   ├── scripts/               ← run_magic / run_hubbard / run_convex / run_exact_convex / make_figs / assemble_summary / selftest_*
│   └── artifacts/             ← summary.json / magic_curve.csv / hubbard_curve.csv / magic_summary.json / exact_convex_summary.json / *.npz / figs/ / logs/
│
├── review/                    ← 独立只读审查（早期 findings 保留，非最终报告来源）
│   ├── a_findings.md b_findings.md c_findings.md
│   ├── *_independent_check.py / extension_frame_derivation.py / independent_formula_check.py
│   ├── source_audit.json
│   └── before_manifest.json   ← ★ 原始文件指纹（项目根相对路径 → SHA256），审计基准
│
├── report/                    ← 最终交付：推广验证报告.html（自包含 ECharts，已冻结）+ artifacts/ + _shots/
│
└── runs/                      ← run_all.py 快照复现输出根（运行时创建；临时，不进 committed 结果）
```

---

## 三、两种复现方式

### 方式 1（推荐默认）：安全快照复现 —— `run_all.py`

三方脚本**全部以 `__file__` 为基准写盘**（`*/results`、`*/artifacts`）。`run_all.py` 把
`hartree/`、`slater/`、`signed_gaussian/` **整树快照**到一个全新的 `runs/<UTC时间戳>/`
目录，再在快照内依次执行；所有输出落在快照里，committed 的 `results/`、`artifacts/`
**不会被改写**。

```bash
cd extension_validation

.venv/bin/python run_all.py --list     # 列出 A/B/C 阶段与脚本（不执行）
.venv/bin/python run_all.py            # dry-run：打印快照目标与将执行命令（默认，安全）
.venv/bin/python run_all.py --run      # 真跑：快照到 runs/run_<ts>/ 并依次执行
.venv/bin/python run_all.py --only B_slater --run   # 只复现某一方
```

- 默认 **dry-run**；不加 `--run` 绝不执行任何生产代码。
- `runs/run_<ts>/` 若同名已存在则**直接报错退出**（防覆盖）。
- 任一阶段失败即停止，committed 结果保持原样；快照内写 `run_all_manifest.json` 记录各阶段返回码。

### 方式 2：直跑（仅当有意刷新 committed 结果时使用）

各子目录 README 的 “Reproduce” 块，例如：

```bash
VENV=.venv
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 $VENV/bin/python hartree/code/exp1_moments.py
# …其余见 hartree/README.md、signed_gaussian/README.md、slater/run_benchmark.py 顶部 docstring
```

> 注意：方式 2 会直接写 committed 的 `results/`、`artifacts/`。复现实验请优先用方式 1（`run_all.py` 快照到 `runs/<ts>/`），避免改动已冻结的 committed 结果。

---

## 四、固定种子与复现参数（勿遗漏）

| 方 | 种子 | 关键固定参数 |
|---|---|---|
| A | `20261026` | m=2..4；矩实验 N≤6、k=1..12（共 864 个 (state,k) 对）；凸求积 m≤3、N≤4、k≤5；Bose-Hubbard 单满填充 M=2,3，U/J∈{0.5,2,6}；单进程单线程 |
| B | 起始行列式 3 种（Neel / anti-Neel / 随机 Sz=0；`run_benchmark.py` 随机种子 `default_rng(2026)`，决定性 `run_complete.py` 用 `default_rng(7)`） | Hubbard 开链 t=1、半填充 Sz=0；L∈{4,6}、U∈{1,8}；每步加 1 个原子、固定预算 n=8/12；外部池=全外部正交 Fock+40 非正交 Slater；方法 cipsi（PT2）/ greedy_noci（2×2 Ritz-drop）/ replica_coef / replica_energy（real convex-J k=2 的两种选取口径）；H2/STO-3G 在 R=0.74 / 1.5 / 2.0 Å |
| C | `20261026`（c_legal_v2） | 4 模偶宇称块；算符 annihilation/vacuum/CAR/n 已修正。Hubbard 合法池 LP：16 点 (U,β)∈{0,2,4,8}×{0.5,1,2,8}，共享 **2856 合法 kets/atoms**（wedges/contact/Haar），每点 weights/rho/reconstructed/frob 同存（hubbard_curve_records.npz，SHA256 be5d52e5…ee04042）；U=0 全 β 由 free wedge 解析正权 Σ=1 得 R=1（frob~1e-16），U8β8 R≈2.897（frob 1.3e-14）。feature：550 合法池（feature_attempt.npz）actual matching mass 2.444/1.609。magic LP 用 3488 合法原子（R 夹逼 [3,3.000028]）。旧非法 Gibbs 池（含非 quadric-1 atom）与旧 creation-H 数字仅 `results/diagnostics/` 历史；三者池勿混。合法性 fail-closed 校验：audit_ket（shape 8/isfinite/norm）+ audit_projector（原 shape/finite/Hermitian trace/purity-PSD/rank-1/outer/Gaussian），solve_signed_lp 在 flatten/LP 前逐原子校验；非法输入（非 Gaussian/2σ/mixed I8/NaN/非 Hermitian/错 shape）立即拒绝（selftest_validation.py 10 reject PASS，legal tiny R=1 frob 0）。单线程 |


统一环境变量：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。

---

## 五、环境

权威环境为本目录 `.venv`（CPython 3.14.7），直接依赖见 `requirements.txt`：
numpy 2.5.4、scipy 1.18.1、matplotlib 3.11.2、cvxpy 1.9.3（后端 scs/osqp）、pyscf 2.14.0；
报告截图工具 pymupdf 2.0。无 GPU、无区间算术——所有数值为浮点可复现，非区间认证。

---

## 六、审计与防覆盖约定

- **原始指纹**：`review/before_manifest.json`（项目根相对路径 → SHA256，含 `gaussian_synthesis/`、
  `paper_v2/` 等只读源）。
- 交付前做 **before vs after SHA256 比对**，排除 `.git/`、`.venv/`、`__pycache__/`、`runs/`、
  `report/artifacts/`；**只记录差异，不清理**。
- 最终交付仅一个总 HTML：`report/推广验证报告.html`（自包含 ECharts/SVG，数据从最终
  CSV/JSON 内联，无本地 fetch、无动画装饰）；其证据图与中间产物在 `report/artifacts/`。
- 终版状态：三方结果已冻结并经独立 review 确认；本 README 状态行与报告数值均以最终 summary/review 为准。
