# 复现

在仓库根目录设置 `PYTHONPATH=gaussian_synthesis/experiments/physical_witness`，使用 `.venv/bin/python -m physical_witness report` 重建总表。

每个 metrics.json 的 source 指向不可变尝试；job.json 包含算法参数，seeds.json 分离训练、校准和验证。 复算使用 runs/<campaign>/provenance/<source_snapshot>/source 作为 PYTHONPATH，并运行 worker --job <绝对 job.json>。 已存在的积分文件按参数核对后复用；新随机验证应建立新尝试，禁止覆盖已验收记录。

限时阶段结束后，旧任务中的截止时间仍有效。获得新计算时限后，在当前批次配置登记新截止时间并运行当前入口的 `resume`；程序从旧检查点新建尝试并保留来源，不修改旧 job.json。

`verify` 检查归档 SHA256、目标态、冻结向量、独立随机流和源码快照。数值 A/B 不是带失败概率的保证或确定性认证。
