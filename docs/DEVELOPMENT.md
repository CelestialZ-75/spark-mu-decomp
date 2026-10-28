# 开发说明

更新：2026-10-28。进度以 [知识库](knowledge-base/KNOWLEDGE.md) 为准，范围见 [课程计划](../SPARK_MU_COURSE_PROJECT_PLAN.md)。

## 代码风格

- 简约、易读、快速可运行优先。先接通真实例子，再补必要参数和检查。
- 复用原算法、数据读取和开源库。新增函数只承担当前实验需要的工作。
- Python 3.11、PEP 8、四空格、snake_case，行宽沿用 Ruff 的 100。
- 参数集中在入口 argparse 或一个简单配置文件，不增加配置继承。
- 用普通函数组织读取、特征、训练和保存；确实需要保存状态时才用小类。
- 设备显式选择，GPU 不可用时报错；ML 和 Spark worker 都使用 CPU。
- 函数说明写清形状、单位和时间基准。注释解释计算原因与时间补偿，不逐句翻译代码。
- 检查路径、空输入、形状和非有限值等实际错误，不吞掉异常继续生成结果。
- 不为未来假设增加协议、工厂、注册器、抽象基类或统一任务框架。
- 给出简短运行命令、默认参数和输出位置，先保证正确，再优化实测瓶颈。

接口规范用于约定数据与参数含义，不需要增加抽象层。不能为了简短省略标签单位、训练/测试隔离或计时同步。

## 已实现接口

| 入口 | 输入 | 输出 |
|---|---|---|
| python -m mu_course / mu-decomp | 默认使用内置算法与数据，--core-dir/--data 可选，其余见 --help | run_experiment(args) 保存结果并返回报告字典 |
| dataset.load_trial(path) | 内置 NPZ | 同步 EMG、力、采样率、通道和单位 |
| load_emg(path, fsamp, start, duration, adapter) | 文件、采样率、片段和阵列 | float32 [channels, samples] |
| SparkOps | CPU Torch 扩展矩阵、session、分区数 | CPU Torch 张量；见 [算子接口](BACKEND_CONTRACT.md) |
| python -m mu_course.benchmark | MU 参数、--backends、--repeat、--partition-counts | 预热后顺序运行，保存逐次结果和 timings.csv |
| python -m mu_course.plot_results | 实验或比较目录 | MU 脉冲图或平均总计算耗时图 |

MAT 必须含 emg_clean，按 good_channels 选一个 64 通道阵列。NPY/NPZ 已完成预处理，NPZ 键为 emg。adapter 从 0 编号，时间单位为秒，相对于输入记录起点。不重新滤波，不用 emg_env 代替分解输入。采样率由参数传入。

当前输出：

- parameters.json：参数、设备、MU 数、worker host 和计时，新增 runtime.cgroup 字段记录进程实际可见限额；完整服务分配另由阶段脚本保存。
- mu_results.npz：pulses 是拼接后的 MU 样本索引，offsets 是切分位置；另含 pnrs、sils、ipts、separation_vectors。
- timings.csv：backend、partitions、run、mu_count 和 fit/startup/distribution/total_seconds。

脉冲索引相对于扩展后矩阵，绘图补偿 extender 裁掉的起始样本。跨试次特征须复核时间补偿。total_seconds 从扩展前计到 fit 后，不含文件读写与清理，不是完整进程耗时。同输出目录重跑会覆盖。

MU 实验 NPZ 不包含完整跨试次模板。预测准备入口单独保存 templates.npz，含投影、质心、固定训练方向、通道、采样率及训练片段。

## 已实现的预测接口

用简单文件连接数据准备和 MLlib；当前已完成真实容器预测及正式对照，结果见知识库。

- 特征表每行一个窗口，分为 train.csv／test.csv，含 trial_id、split、window_start_s、window_end_s、label、固定顺序的 feature_0 ... feature_n。RMS/MU 分表，窗口时间对应；label 的单位由 metadata.json 指定为 g。
- 当前标签为同窗口中指屈曲力均值，单位 g；第一试次训练、第二试次测试。窗口 410 样本、步长 102 样本，两端排除 430 样本；不能与 %MVC 混用。
- 一份小 JSON 记录通道编号、采样率、窗口、特征列含义、标签单位和时间处理。
- 预处理参数与 MU 模板只用训练数据获得，测试使用固定通道顺序与模板。MU 没有放电也保留特征列；缺少必需通道时明确报错，或预先确定共同通道后重新训练。
- 复用 MU 编码前检查平滑方向。居中平滑含未来信息时，标记为离线预测。
- MLlib 用 VectorAssembler 组装特征，LinearRegression 的 L2 配置实现 Ridge，GBTRegressor 实现 GBT，全部 CPU。
- 预测表含 trial_id、window_start_s、window_end_s、label_g、prediction_g，另存 RMSE/MAE、训练均值基线和启动/分发/训练/预测耗时；metrics.json 的 runtime.cgroup 记录进程限额。

入口为 `python -m mu_course.prepare_prediction --features rms|mu --output ...` 和 `python -m mu_course.predict_force --features 特征目录 --model ridge|gbt --output ...`。完整命令与容器提交见 [统一实操指南](START_HERE.md#step-6)。特征与预测输出目录必须为空；模型保存不包含在计算计时内。

## 团队工作包

| 工作包 | 输入 | 交付与验收 | 依赖 |
|---|---|---|---|
| 环境与调度 | 当前 Dockerfile/Compose、内置 MU/数据入口 | CPU 基线和 1/2 worker 配置，真实限额、两节点参与记录、同输入结果 | Docker 可运行 |
| 数据与特征 | 原项目数据/读取函数、MU 代码 | 训练/测试 RMS 与 MU 表、时间/单位说明、训练模板 | 已固定试次并实现模板导出／应用；修改后复核对齐 |
| MLlib 与展示 | 特征表与上述约定 | Ridge/GBT 预测、指标、曲线、资源耗时表 | 已有真实特征与预测；后续修改仍检查独立测试输出 |

改公共字段前先对齐。不顺带重构原研究仓库，不覆盖已有不相关改动。本页不指定人员，也不自动创建其他任务。

## 检查与提交

在 Ubuntu 项目根目录运行，先按环境说明建立 .venv-wsl：

```bash
.venv-wsl/bin/python -m mu_course --help
.venv-wsl/bin/ruff check src tests scripts
.venv-wsl/bin/pytest -q
```

测试覆盖算子数值、输入、真实特征／模板对齐、静默列、固定方向和小型 MLlib 模型保存／载入；真实 MU 与正式预测另读原始文件检查。CUDA 检查实际设备并同步计时。Docker 环境算例与真实 MU 分开验收。仅改文档时检查链接与参数，不必重跑算法。

用项目 uv 和锁文件维护 Python 3.11 环境，CPU/CUDA extras 二选一。用户授权的两份 src/mu_course/data 精简试次随包提供；其他真实数据、环境、缓存、凭据与输出不入 Git。提交和推送需用户授权。
