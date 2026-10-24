# FK optimal Gaussian geometry

按 [原始方案](EXPERIMENT_SPEC.txt) 执行 A–F。此前 `fk_replica_hierarchy` 保持只读，复用已有 SMC、Cartan、Gaussian 优化及列生成后端。

- [本轮总报告](REPORT.md)：严格成本、组成必要性、物理误差、消融、相干核、新扰动和第二模型。
- [逐项验收与开放理想结果](COMPLETION_AUDIT.md)：原方案执行要求与证据对应。
- [机器可读状态](EXECUTION_STATE.json) 与 [验收证据哈希](results/final_requirement_audit.json)。

实验与有限预算的认证探索已完成。FK 精确 primal–dual matching、全局轨道商距离稳定常数及更强高阶认证仍开放，没有把这些理想结果标为已证明。

本次提交包含此前提交 3301a29 之后新增的实验代码、结果与核验。按本目录 `.gitignore`，原始 `.npz` / `.npy` 数组、源码快照和缓存保留在本地；清单中的这些路径不随 Git 提交。失败记录与求解器精度警告保留。
