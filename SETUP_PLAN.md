# 开发与实验顺序

更新：2026-10-28。范围见 [课程计划](SPARK_MU_COURSE_PROJECT_PLAN.md)，具体操作按 [统一实操指南](docs/START_HERE.md)，验收证据集中在 [知识库](docs/knowledge-base/KNOWLEDGE.md)。

| 顺序 | 工作 | 当前状态 |
|---|---|---|
| 1 | 内置数据、CPU 与 Spark local MU，已有 CUDA 参考 | 已实现；本轮 VMware local 已核对，CUDA 为历史验证 |
| 2 | Docker 硬限额、同总额度 CPU 基线、两个 worker 参与 | 已实现并有真实运行和资源记录 |
| 3 | 1／2 worker、固定预算和 1／2／4 分区对照 | 正式 MU 六组完成，24 次运行含六次预热 |
| 4 | 第一试次训练、第二试次测试，RMS 与固定训练 MU 模板特征 | 已完成；通道、窗口、标签和模板对齐 |
| 5 | CPU MLlib RMS＋Ridge、MU＋GBT，补充 MU＋Ridge | 单次验收与 local／standalone 正式对照完成 |
| 6 | 每配置预热一次、正式三次，汇总耗时、误差与图表 | 完整批次 12 组、48 次运行、36 条正式记录已核对 |
| 7 | 课程报告与交付 | 报告、七张配图和精选结果已整理到 docs/report；待复核、补作者信息及提交 |

当前对照实操指南第 10 节复核已有报告初稿，无需重跑已成功的实验。源码、输入或资源条件发生变化时另建批次，保留原始结果。

CUDA 只用于 MU 参考，ML 全程 CPU；Transformer、区间预测、网络限速和 HDFS 不列为当前必做项。
