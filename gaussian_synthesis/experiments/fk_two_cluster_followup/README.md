# FK 双团簇：第二轮有界实验

本目录独立于首轮 `../fk_two_cluster/`。复用首轮 Python 3.12 虚拟环境与模型、Gaussian 后端，首轮源码和结果均作为只读输入；不改变核心库 API，也不提交或推送。

## 本轮问题

1. 冻结首轮最大训练量的 witness，使用全新的独立验证随机流，比较普通 Haar 重构和合法的 Spin(7) 轨道平均重构。对称投影原子不归一化，每个投影原子按完整 Gaussian 轨道平均解释。
2. 建立 full 字典的确定性 k=2 矩参考，区分训练误差和验证积分方差。精确整数矩不能直接给出半整数幂的系数质量积分。
3. 对已保存候选做 Fock、Fock+结构弧、Fock+Haar、完整候选的消融。
4. 将 t/V=20 的三类有效轨道转成有限个真实 Gaussian ket。测试有针对性的球面设计，以及总分支数为8、16、32、64、128的随机轨道方案；每个结果均以完整128维残差和 Fock 修补成本计价。

保留 t/V=0,11,20、k=2,4、三个种子以及样本上限65536。主要对照保持首轮 witness 不变；新 k=2 解析 witness 单独标记。每阶段180秒、首轮式总预算1800秒、512 MiB、单数值进程与单数值线程。正式实验不根据结果扩大扫描或增加 k。

## 运行

从本目录执行：

```bash
../fk_two_cluster/.venv/bin/python -W error::RuntimeWarning run_tests.py
../fk_two_cluster/.venv/bin/python -W error::RuntimeWarning scripts/run_followup.py --output results/followup_20261023_01
../fk_two_cluster/.venv/bin/python scripts/verify_results.py results/followup_20261023_01
```

输出目录必须不存在。实际运行配置、环境版本、源码与首轮输入文件哈希、原始向量、随机流、逐阶段资源和未通过验收的状态均保存。`partial_*.json/npz` 是检查点，不能当作完整或认证结果。脚本不会覆盖首轮数据。

正式结果在 `results/followup_20261023_01/REPORT.md`。从保存的实际 Gaussian 分支重建主结果：

```python
import numpy as np
a = np.load('results/followup_20261023_01/finite_orbits.npz')
g = a['exact_simplex_kets']             # shape (24, 128), normalized Gaussian kets
c = a['exact_simplex_coefficients']     # relative complex phases are essential
r = a['exact_simplex_fock_repair_coefficients']
psi = c @ g + r                        # complete even-sector reconstruction
cost = np.abs(c).sum() + np.abs(r).sum()
```

该 `cost` 是已保存数值目标的修补成本；JSON中的上界还加上真正目标的本征态误差估计。128维坐标的 Fock 掩码顺序保存在 `model.npz` 中。即使修补系数仅为舍入量级，成本计算仍保留它们。

## 证据口径

- 原始重构与轨道平均重构是两种表示；不能把简单丢弃补空间残差称为原始有限展开成功。
- 样本方差、经验误差和跨种子稳定性是数值诊断，不自动构成95%联合统计保证。
- 确定性 k=2 projector 来自代数表示，但本程序仍用浮点计算其矩阵和解；不声称有向舍入的区间证明。
- 有限 Gaussian 展开的修补上界包含所有 Fock 残差系数。主分支数与修补后的总分支数分别记录。
- 有限候选 SOCP 的最优状态只针对当前候选，不证明完整 Gaussian 原子范数的全局最优值。
- 适用于指定活跃轨道的球面设计需逐个验证完整投影，不能推广为对任意 Gaussian 都成立的设计。

## 结构

`fk2/rb.py`：成对重构估计；`exact_moments.py`：确定性 k=2 参考；`ablation.py`：轨道字典消融；`orbits.py`：实际有限 Gaussian 展开；`runner.py`：预算、冻结输入与检查点；`report.py`：从原始结果生成报告。详细推导见 `THEORY.md`。
