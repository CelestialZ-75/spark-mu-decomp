# 项目架构

更新：2026-10-28。已内置 MU 算法和两份 EMG/力数据，受限 Docker MU／预测及正式资源对照已完成。实现状态见 [知识库](KNOWLEDGE.md)。

## 现有数据流

```text
参数 + 内置 EMG/力试次 + 内置 CKC（可选外部覆盖）
  → run_experiment：选通道/片段，扩展矩阵
  → 原 CKC 主循环
      CPU/GPU：原 Torch 运算
      Spark：driver 控制循环 → 时间块 worker → 合并返回 driver
  → parameters.json + mu_results.npz
  → plot_results：MU 脉冲图

benchmark：顺序预热和重复上述流程 → timings.csv → 耗时图
```

只在相关矩阵、投影、加权求和处接入 SparkOps。driver 保留完整输入、求逆、候选选择、质量判断与全局更新。worker 使用 NumPy；原算法只需要在 driver 可用。Spark 输出是 driver CPU Torch 张量，CPU/GPU 路径直接使用原设备张量。

时间分块包含全部扩展通道。相关矩阵先归约再除总样本数；投影按块起点排序拼接；加权求和使用相应时间段的权重。详见 [算子接口](../BACKEND_CONTRACT.md)。这不是完整的分布式内存算法。

## 文件职责

| 路径 | 职责 |
|---|---|
| src/mu_course/run_experiment.py | CLI 参数、内置/可选外部算法、输入、MU 调用、保存和计时 |
| src/mu_course/ckc/ | 内置分解、扩展、预处理、放电率编码与质量检查 |
| src/mu_course/data/、dataset.py | 两份完整同步 EMG/力试次及轻量读取 |
| scripts/plot_example_data.py | 查看内置 EMG 与力曲线 |
| scripts/prepare_example_data.py | 维护者重新导出数据，队友运行无需源仓库 |
| src/mu_course/spark_ops.py | 时间块缓存与三个 Spark 算子 |
| src/mu_course/benchmark.py | 顺序预热、重复计时和 CSV |
| src/mu_course/plot_results.py | 脉冲与平均耗时图 |
| scripts/submit.py | Spark 提交时的薄入口 |
| scripts/environment_smoke.py | 环境算例，不运行 MU |
| scripts/bootstrap_wsl.sh | 项目 uv/Python 与锁定依赖安装 |
| scripts/check_mu_results.py、check_prediction_results.py | Ubuntu 读取已生成的第 6–8 步结果并保存检查记录 |
| scripts/run_formal_group.sh | 按批次目录执行正式配置，预热一次、正式三次 |
| scripts/check_formal_mu.py、summarize_formal.py | 正式 MU 配对与质量检查、预热排除、指标和耗时汇总 |
| environment/ | CPU 镜像、Compose、环境说明 |
| src/mu_course/prepare_prediction.py、predict_force.py | RMS／固定训练 MU 特征，CPU MLlib 训练与独立测试 |
| scripts/predict_submit.py | 容器 spark-submit 的预测薄入口 |
| src/mu_course/runtime_info.py | 运行环境与 cgroup 限额读取 |
| scripts/run_vm_stage.sh | 单次容器环境、MU 和预测验收 |
| tests/ | 分块数值、输入、CUDA、预测特征／模板和小型模型保存／载入检查 |
| pyproject.toml、uv.lock | Python 3.11、Torch、PySpark 等依赖 |
| .private/ | 本地路径、验证脚本、来源快照和历史计划 |

Python、uv、Java 17、PySpark 3.5.7、Torch 2.8.0 是现有技术栈。CPU/CUDA extras 互斥；CUDA 版环境也可选择 CPU 运算。

## 运行边界

当前使用 VMware Ubuntu，WSL 保留为本地开发与历史 CUDA 参考。Compose 已定义 master、worker-1、worker-2、client driver，共享 CPU 镜像和内部网络；数据只读、输出单独挂载，UI 仅发布到本机回环地址。

基础 Compose 的 driver 默认运行环境算例，单独使用没有容器硬限额。`compose.limits.yaml` 增加 CPU 服务和各角色限额，`run_vm_stage.sh` 叠加两份配置提交真实任务；构建、真实任务和正式对照已有结果证据。默认 worker 各声明一个任务槽，executor heap 512 MiB；阶段／正式提交将 driver heap 从基础默认 1 GiB 改为 768 MiB。完整容器配额见 [环境说明](../../environment/README.md)。算法与示例数据随包安装，无需外部挂载。

正式实验分别比较增加 worker 总资源、固定 worker 总预算及 1／2／4 分区。driver/master 额度单独列出；分区数不等同于 worker 数。单机共享硬件不模拟真实跨服务器网络。

## 预测数据流

三试次 cascade 对照的独立入口为 `src/mu_course/cascade_experiment.py`，科研版分解和细化函数在 `src/mu_course/ckc/cascade_decomposer.py`。CPU 逐任务顺序执行；Spark `local[3]` 调度试次级任务，并用单独进程合并。前一级合并与下一级 Track A 可重叠，但 Track B 必须等待前一级池。当前为同一机器上的进程模拟，不是四台机器或四个 Spark worker。详情见 [cascade 对照说明](../CASCADE_COMPARISON.md)。

`src/mu_course/prepare_cascade_prediction.py` 将 cascade 池按试次投影为脉冲，再生成每条 MU 的六项特征，输出沿用原 `predict_force.py` 的特征表接口；它不修改回归模型。模板池若使用测试肌电，输出只可作为探索性评估。

```text
LDX 0° 训练试次 → RMS / 训练 MU 模板及放电率 → 训练特征表
LDX 0° 测试试次 → RMS / 应用训练模板及放电率 → 测试特征表
  → CPU MLlib Ridge / GBT → 预测表、RMSE/MAE、计时和曲线
```

复用试次读取、模板应用与 MU 编码，以窗口 RMS 或 MU 放电率特征文件连接模型。固定通道、模板、窗口、标签单位和时间映射。prepare_prediction.py 生成 features.npz、CSV、元数据，MU 模式另存完整 templates.npz；predict_force.py 用 CPU MLlib 训练，保存独立测试预测、指标、模型与图。旧 mu_results.npz 不作为完整模板。

MLlib 负责经典模型训练，Spark 负责数据分区和执行；不自建训练调度器。ML 全部 CPU，GPU 仅用于 MU。字段约定与团队工作包见 [开发说明](../DEVELOPMENT.md)。
