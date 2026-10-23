# BCS L=4 发布前复核

日期：2026-10-23。发布分支：`codex/bcs-pairing-l4-baselines`，基于 `main` 的 `a5183acb8f0a7d84a89149e1ad35e89b964a7b5f`。

本次在独立工作区重新运行主库和 BCS 测试，并只读核验两次已保存运行。使用原实验虚拟环境的依赖，通过 `PYTHONPATH` 明确加载发布工作区的通用库；实际导入路径记录在 `summary.json`。未重新拟合或修改原始结果。

- 主库 49 项、BCS 15 项测试通过。
- 两次结果各 35 项验收通过。
- 最终运行的 68 个交付文件 SHA-256 和 46 个数值源码 SHA-256 匹配。
- 本分支只添加 BCS 实验、通用空间接口及对应测试和说明；其他实验不属于本次提交。

原始 stdout/stderr 与命令见本目录日志和 `summary.json`。`release_manifest.json` 列出本次交付文件的 SHA-256（不包含它自身）。

本次完成精确参考、投影 TD-BCS、投影乘积态 TDVP、onsite PBCS 可实现下界及 onsite minor 上界。尚未计算一般数投影 Gaussian 拟合或 Gaussian extent，也没有证明 onsite 优化达到全局最优。原始运行、图表、源码快照及其限制原样保留。
