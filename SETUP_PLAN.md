# 开发与实验顺序

范围见 [课程计划](SPARK_MU_COURSE_PROJECT_PLAN.md)，进度见 [知识库](docs/knowledge-base/KNOWLEDGE.md)，接口与分工见 [开发说明](docs/DEVELOPMENT.md)。

1. 保留现有 CPU/CUDA/Spark local MU 流程。按 [运行说明](README.md) 先跑 CPU，确认内置算法、输入和参数。已有验证不等于换机器后自动可用。
2. 按 [环境说明](environment/README.md) 验证 Docker。补容器 CPU/内存限额、同环境 CPU 基线；内置算法和数据已随包提供。先过环境算例，再运行真实 MU。
3. 做 1/2 worker 对照：每节点固定额度、worker 总预算固定。记录 driver/master 额度。分区数另行调整，保存 worker 参与的任务记录。
4. 确认 LDX 0° 训练/测试试次。复用原读取和特征函数，准备 RMS、固定 MU 模板特征和对应力标签。
5. CPU MLlib 运行 RMS＋Ridge、MU＋GBT；补充 MU＋Ridge 时复用原函数。保存指标、预测曲线和分项耗时。
6. 同配置预热一次、测三次，整理资源表、图和报告，解释计算与通信开销。

第 2–6 项尚未完成。当前 Compose 默认执行环境算例，未加硬限额；预测脚本尚不存在。不要把计划接口当作已实现命令。

CUDA 只用于 MU 优化参考。ML 全程 CPU；Transformer、区间预测和网络限速暂不列为必做项。
