# E0 高阶解析基准

运行状态：`complete`。

使用通用 Gaussian 流形采样后端，未向积分器提供目标态的高阶解析积分。独立种子间 Student-t 区间是经验误差诊断，不是严格覆盖证明。

| 对象 | β | Z 相对误差均值 | 95% 重复半宽 | 1%经验门槛 | Gaussian纯度 |
|---|---:|---:|---:|---|---|
| gaussian8 | 0.5 | -0.0001512 | 0.001416 | True | True |
| gaussian8 | 1 | -0.0003066 | 0.001874 | True | True |
| gaussian8 | 1.5 | -0.0008439 | 0.002414 | True | True |
| gaussian8 | 2 | -0.001006 | 0.001934 | True | True |
| gaussian8 | 3.5 | 0.0005767 | 0.003236 | True | True |
| gaussian8 | 4 | -0.0002119 | 0.00349 | True | True |
| gaussian8 | 7.5 | 7.455e-05 | 0.006431 | True | True |
| gaussian8 | 8 | -0.0001829 | 0.006337 | True | True |
| gaussian8 | 15.5 | -0.001504 | 0.006296 | True | True |
| gaussian8 | 16 | -0.001444 | 0.006285 | True | True |
| gaussian8 | 31.5 | -0.00235 | 0.00913 | True | True |
| gaussian8 | 32 | -0.002288 | 0.009373 | True | True |
| gaussian8 | 63.5 | -0.001234 | 0.009056 | True | True |
| gaussian8 | 64 | -0.001241 | 0.00905 | True | True |
| gaussian8 | 127.5 | -0.0001992 | 0.01152 | False | True |
| gaussian8 | 128 | -0.0001891 | 0.01155 | False | True |

![解析基准与采样误差](figures/analytic_benchmarks.png)

图中空心点为数值积分估计。纯 Gaussian 的真实最小成本及解析 L 均为 1；canonical B 在有限阶大于 1 是分解本身的成本，不能解释为非 Gaussian 资源。

本报告只覆盖 E0 的解析对象。FK 的 k=4 独立重构、E1–E5 和相应认证仍须逐项完成；此基准运行完成不等于整个方案完成。
