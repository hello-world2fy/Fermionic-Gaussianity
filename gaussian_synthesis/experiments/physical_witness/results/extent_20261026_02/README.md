# Gaussian extent：当前最好结果（2026-10-26 批次）

这是三小时阶段结束后的轻量数据发布。“最好”指当前已计算候选中的选定结果，不表示已证明全局最优。

- [总报告](REPORT.md)：66 态目录，其中 27 态已有大阶结果，39 态待扩展。
- [CSV](dataset.csv) / [JSON](dataset.json)：全部状态、连续重构指标与补充 extent 界。
- `states/<态>/vectors.npz`：选定的 psi、u_k、w_k、direction、k、masks、support、n。
- `states/<态>/metrics.json`：同一冻结向量的指标、独立验证记录和基底约定。
- `states/<态>/target_metadata.json`：目标态与 Hamiltonian 参数。
- [有限分解界](finite_bounds.json)：10 态的分解上界、独立数值下界估计及四倍采样比较。
- `supporting_data/<态>/`：补充界所用的小型分解、方向、积分摘要和既有认证记录。
- [发布索引](publication_index.json)：原始来源路径到发布文件的映射及大文件排除说明。

连续分解阶段验收通过 4 态；原标准数值 A 为 2 态、B 为 1 态。10 项补充上界／数值下界比值中 9 项小于 1.20。补充界不替代连续重构验收。有限上界认证针对保存的 binary64 目标向量；积分下界仍是数值估计，尚无高阶严格最优性认证。

## 大文件与复现范围

每个发布文件小于 2 MiB。Hubbard 十二模 U=4 的原始有限分解文件约 12.46 MB，其中 (395, 2048) 的稠密 atoms 数组未上传；`decomposition_compact.npz` 仅保留方向、系数与 Fock 残差，不能单独复现完整分解。原文件保留在计算机上，路径及原有哈希见发布索引。

原始结果文件保持原样，其来源字符串及既有哈希沿用；本次发布未重跑哈希校验。`delivery_manifest.json` 描述原始交付目录中的 63 个文件（不包含清单自身），不覆盖此次新增的说明、目标元数据与 supporting_data。`REPRODUCE.md` 中的计算命令面向完整本地实验环境；本分支仅发布数据，未上传完整运行目录、采样数组、历史归档或完整算法源码快照。

## 读取向量

在仓库根目录运行（需要 NumPy）：

```python
from pathlib import Path
import numpy as np

root = Path("gaussian_synthesis/experiments/physical_witness/results/extent_20261026_02")
with np.load(root / "states/bcs_n8_G3/vectors.npz", allow_pickle=False) as data:
    psi, u, w = data["psi"], data["u_k"], data["w_k"]
    masks, k = data["masks"], int(data["k"])
    print(k, psi.shape, np.linalg.norm(psi))
```

`masks` 给出对应的 Fock 位串；详细基底约定见同目录 metrics.json。
