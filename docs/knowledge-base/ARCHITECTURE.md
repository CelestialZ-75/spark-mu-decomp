# 项目架构

更新：2026-10-20。当前是少量 MU 实验代码；受限 Docker 实验与 CPU MLlib 预测是下一阶段。实现状态见 [知识库](KNOWLEDGE.md)。

## 现有数据流

```text
参数 + 已预处理 EMG + 外部原算法目录
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
| src/mu_course/run_experiment.py | CLI 参数、外部导入、输入、MU 调用、保存和计时 |
| src/mu_course/spark_ops.py | 时间块缓存与三个 Spark 算子 |
| src/mu_course/benchmark.py | 顺序预热、重复计时和 CSV |
| src/mu_course/plot_results.py | 脉冲与平均耗时图 |
| scripts/submit.py | Spark 提交时的薄入口 |
| scripts/environment_smoke.py | 环境算例，不运行 MU |
| scripts/bootstrap_wsl.sh | 项目 uv/Python 与锁定依赖安装 |
| environment/ | CPU 镜像、Compose、环境说明 |
| tests/ | 分块数值、输入选择与 CUDA 检查 |
| pyproject.toml、uv.lock | Python 3.11、Torch、PySpark 等依赖 |
| .private/ | 本地路径、验证脚本、来源快照和历史计划 |

Python、uv、Java 17、PySpark 3.5.7、Torch 2.8.0 是现有技术栈。CPU/CUDA extras 互斥；CUDA 版环境也可选择 CPU 运算。

## 运行边界

WSL 用于本地开发与已有 CUDA 参考。Compose 已定义 master、worker-1、worker-2、client driver，共享 CPU 镜像和内部网络；数据只读、输出单独挂载，UI 仅发布到本机回环地址。

当前 worker 各声明 1 core、1 GiB 可供 Spark 调度内存，executor heap 512 MiB、driver heap 1 GiB。这些不是容器硬限额。当前 driver 默认运行环境算例，未配置原算法挂载，没有普通 CPU 基线服务，实际构建与双 worker 运行仍待验证。

后续增加容器硬限额和同环境 CPU 基线，分别比较增加 worker 总资源与固定 worker 总预算。driver/master 额度单独列出；分区数不等同于 worker 数。单机共享硬件不模拟真实跨服务器网络。

## 规划中的预测数据流（未实现）

```text
LDX 0° 训练试次 → RMS / 训练 MU 模板及放电率 → 训练特征表
LDX 0° 测试试次 → RMS / 应用训练模板及放电率 → 测试特征表
  → CPU MLlib Ridge / GBT → 预测表、RMSE/MAE、计时和曲线
```

复用原读取和特征代码，通过简单特征文件连接模型。固定通道、模板、窗口、标签单位和时间映射。当前结果尚未导出完整模板应用参数，预测数据准备需要补齐。新模块名称未定，不把规划目录当作已有代码。

MLlib 负责经典模型训练，Spark 负责数据分区和执行；不自建训练调度器。ML 全部 CPU，GPU 仅用于 MU。字段约定与团队工作包见 [开发说明](../DEVELOPMENT.md)。
