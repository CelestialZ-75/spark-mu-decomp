# 资源受限的 Spark 肌电计算实验

在一台电脑上模拟资源受限的 Spark 集群，观察真实肌电任务的计算、调度和通信开销。任务包括 MU 分解效率对照，以及 CPU MLlib 指力预测。GPU 仅用于 MU 分解参考。

仓库已内置 CKC 代码和三份 LDX 0° 中指梯形试次的同步 EMG/力数据，不需要 FingerForce_Decode 或额外算法目录。运行环境为 Linux x86_64，本轮使用 VMware Ubuntu，WSL2 也可用于本地运行；需要 Python 3.11，Spark 另需 Java 17。

当前已完成受限容器 MU、RMS＋Ridge／MU＋GBT／MU＋Ridge 预测和正式资源对照，共 12 组、36 条正式记录。课程报告已有 Markdown 初稿，待复核与交付；验证范围与关键结果见 [实际进度](docs/knowledge-base/KNOWLEDGE.md)。

首次接手按 [从零到交付实操指南](docs/START_HERE.md) 操作，涵盖 SSH、代理、安装、容器实验、报告和结果回传。下面是 Linux 本地运行的最短示例。

## 从拉取到运行

在 Ubuntu 的仓库根目录执行：

```bash
# 缺 Java 时安装；这是系统依赖，需本机管理员权限
sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless

# 创建项目环境，首次需要联网下载依赖，不修改全局 Python
bash scripts/bootstrap_wsl.sh cpu
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
export PYSPARK_PYTHON="$PWD/.venv-wsl/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
unset SPARK_HOME

# 内置真实数据、内置算法，默认取第 3 秒开始的 2 秒片段
.venv-wsl/bin/python -m mu_course --backend cpu --output outputs/cpu
.venv-wsl/bin/python -m mu_course.plot_results outputs/cpu

# 查看数据中的 EMG 和同步力曲线
.venv-wsl/bin/python scripts/plot_example_data.py

# 使用 Spark local，分区数和输出目录可直接修改
.venv-wsl/bin/python -m mu_course --backend spark --partitions 2 --output outputs/spark
```

已有 Python 3.11 和 uv 的 Linux 环境也可执行 `uv sync --locked --extra cpu --extra plots`，随后使用 `uv run --no-sync python -m mu_course`。此方式默认环境为 .venv，勿与上面的 .venv-wsl 路径混用。

参数见 `python -m mu_course --help`，默认值集中在 run_experiment.py 的 experiment_parser()。例如修改片段：

```bash
.venv-wsl/bin/python -m mu_course --start 3 --duration 5 --n-mu-max 5 --n-iter 30 --output outputs/cpu-5s
```

GPU 用户先执行 `bash scripts/bootstrap_wsl.sh gpu`，再选 `--backend gpu`。该环境也能运行 CPU 和 Spark，GPU 不可用时报错。ML 仍只使用 CPU。

## 数据与算法

- [内置数据](src/mu_course/data/README.md)：LDX 0° 的 session0_1、session0_2，共同有效通道，完整同步力轨迹。数据在 Python 包中，安装包与 Docker 镜像也会携带。
- [代码来源](src/mu_course/ckc/SOURCE.md)：只复制必需 CKC 模块，已接好 Spark 算子与包内导入。
- 默认数据为 ldx_0deg_1.npz，第二试次可用 `--data src/mu_course/data/ldx_0deg_2.npz`。
- 自有数据仍可用 --data；只有需要测试其他 CKC 版本时才用可选 --core-dir。

NPY/NPZ 应已预处理，emg 形状为 [channels, samples]，NPZ 键为 emg。MAT 使用 emg_clean 和 good_channels，不再滤波，不以包络替代。adapter 从 0 编号，时间相对输入记录起点，单位秒。内置数据只保留 adapter 0 的共同有效通道，因此选 adapter 0。

`mu_course.dataset.load_trial()` 可直接读取内置 EMG、力、采样率和通道编号。力为 [2,N]，顺序是中指屈曲、伸展，单位 g；屈曲 %MVC 可用 force[0] / flexion_mvc_g * 100 计算。当前预测使用第一试次训练、第二试次测试，标签为同窗口中指屈曲力均值，单位 g；特征与训练入口见 [实操指南第 7–8 节](docs/START_HERE.md#step-7)。

## 结果与对照

输出 parameters.json、mu_results.npz；同目录重跑会覆盖。脉冲索引相对扩展矩阵，绘图补偿 extender 裁掉的起始样本。没有 MU 时提示检查输入和参数。

```bash
.venv-wsl/bin/python -m mu_course.benchmark \
    --backends cpu spark --repeat 3 --partition-counts 1 2 4 \
    --start 3 --duration 2 --output outputs/comparison
.venv-wsl/bin/python -m mu_course.plot_results outputs/comparison
```

有 CUDA 时可在 --backends 中加 gpu。每配置先预热，再顺序测三次，保存 timings.csv。总计算耗时含扩展、Spark 启动/分发与 fit，不含读写、保存和清理。CUDA 计时同步，Spark 等待 action。

本地 benchmark 用于 local 对照。受限 Docker 正式实验使用 `compose.yaml` 与 `compose.limits.yaml`，由 `scripts/run_formal_group.sh` 逐配置预热一次、正式三次；不要把上面的 local 结果混入同一组统计。配置说明见 [环境说明](environment/README.md)，完整流程见 [实操指南第 9 节](docs/START_HERE.md#formal-comparison)。

完整原始结果保存在 Git 忽略的 `outputs/`。本次选定的 [课程报告](docs/report/COURSE_REPORT.md)、七张配图和 [精选结果](docs/report/results/RESULTS_GUIDE.md) 已整理到 `docs/report/`，可随代码一起提交；包含全部正式 CSV、MU 配对记录和固定 run-1 原始示例。完整模型、重复原始文件与日志继续保留在本地。

## 文档入口

- [从零到交付：VMware Ubuntu 实操指南](docs/START_HERE.md)
- [课程报告](docs/report/COURSE_REPORT.md) 与 [精选结果说明](docs/report/results/RESULTS_GUIDE.md)
- [课程计划](SPARK_MU_COURSE_PROJECT_PLAN.md) 与 [开发顺序](SETUP_PLAN.md)
- [完整试次 CKC 对照](docs/FULL_TRIAL_EXPERIMENT.md)
- [三试次粗粒度 Cascade 对照与 VMware 待办](docs/CASCADE_COMPARISON.md)
- [架构](docs/knowledge-base/ARCHITECTURE.md) 与 [实际进度](docs/knowledge-base/KNOWLEDGE.md)
- [开发风格、接口与团队工作包](docs/DEVELOPMENT.md)
- [Spark 算子](docs/BACKEND_CONTRACT.md)

代码优先简约、易读、快速可运行，参数集中，接口写清形状、单位和时间基准。
仓库当前为 private。用户授权的三份精简科研数据随仓库提供，本次选定的报告、配图与精选结果放在 `docs/report/`；其他真实数据、环境、缓存、凭据与完整生成输出仍不跟踪。未来公开前单独审查源码、数据与历史。.private/ 的旧路径和检查脚本仅供历史追溯，队友运行不需要它们。
