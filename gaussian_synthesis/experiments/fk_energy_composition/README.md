# 能量分辨连续 Gaussian 组成

执行 [原始方案](EXPERIMENT_SPEC.txt)，范围包括 M1–M4 与 G1–G3。
本目录复用既有采样和算符实现；此前实验目录保持只读。

M1 数值积分已通过，当前进入 M2 加权合成与独立全局认证。
最终成功需要分布自动发现、全字典必要性、对强基线的独立增益及跨模型检验。
运行结束或产生解析参考证书不等于通过这些门槛。

- [预注册设置](configs/preregistration.json)：参数、重复、误差标准和强基线权限。
- [执行协议](PROTOCOL.md)：证据范围、里程碑和失败处理。
- [加权理论](WEIGHTED_SYNTHESIS.md)：加权重构、成本和全局证书的区别。
- [当前状态](EXECUTION_STATE.json)。
- [进展报告](REPORT.md) 与 [完整需求账本](REQUIREMENTS.json)。

大体积原始采样数组和源码快照保留本地；代码、报告、摘要、核验记录、小型冻结参数、
条件能量分箱向量和核矩阵纳入版本管理。推送范围与运行快照见 PUBLICATION.json。
M1 百万样本实验保存全部 ket。后续大网格支持显式 compact 存储：保存全部粒子的
相位、能量、特征、singlet overlap 和完整重构向量，并保存随机选取的 8,192 个
实际 Gaussian ket。其余 ket 可按冻结代码和随机种子重建；不会把这些未保存的
ket 或对称投影向量冒充为一个有限构造中的 Gaussian 项。
