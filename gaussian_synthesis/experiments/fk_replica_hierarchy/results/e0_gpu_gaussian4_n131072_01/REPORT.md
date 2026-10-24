# E0 高阶解析基准

运行状态：`complete`。

使用通用 Gaussian 流形采样后端，未向积分器提供目标态的高阶解析积分。独立种子间 Student-t 区间是经验误差诊断，不是严格覆盖证明。

| 对象 | β | Z 相对误差均值 | 95% 重复半宽 | 1%经验门槛 | Gaussian纯度 |
|---|---:|---:|---:|---|---|
| gaussian4 | 0.5 | -0.0006603 | 0.001454 | True | True |
| gaussian4 | 1 | -0.00104 | 0.002408 | True | True |
| gaussian4 | 1.5 | -0.0008406 | 0.003321 | True | True |
| gaussian4 | 2 | -0.000996 | 0.002747 | True | True |
| gaussian4 | 3.5 | -0.001441 | 0.004654 | True | True |
| gaussian4 | 4 | -0.00145 | 0.004175 | True | True |
| gaussian4 | 7.5 | -0.0007103 | 0.005678 | True | True |
| gaussian4 | 8 | -0.0007776 | 0.005483 | True | True |
| gaussian4 | 15.5 | -0.0006938 | 0.007493 | True | True |
| gaussian4 | 16 | -0.000607 | 0.007429 | True | True |
| gaussian4 | 31.5 | -0.0003652 | 0.006942 | True | True |
| gaussian4 | 32 | -0.0003633 | 0.006947 | True | True |
| gaussian4 | 63.5 | -0.0001151 | 0.006507 | True | True |
| gaussian4 | 64 | -0.0001341 | 0.006538 | True | True |
| gaussian4 | 127.5 | -0.0002501 | 0.008195 | True | True |
| gaussian4 | 128 | -0.0002625 | 0.008172 | True | True |

![解析基准与采样误差](figures/analytic_benchmarks.png)

图中空心点为数值积分估计。纯 Gaussian 的真实最小成本及解析 L 均为 1；canonical B 在有限阶大于 1 是分解本身的成本，不能解释为非 Gaussian 资源。

本报告只覆盖 E0 的解析对象。FK 的 k=4 独立重构、E1–E5 和相应认证仍须逐项完成；此基准运行完成不等于整个方案完成。
