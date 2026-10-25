# 资源受限的 Spark 肌电计算实验

在一台电脑上模拟资源受限的 Spark 集群，观察真实肌电任务的计算、调度和通信开销。任务包括 MU 分解效率对照，以及后续 CPU MLlib 指力预测。GPU 仅用于 MU 分解参考。

仓库已内置 CKC 代码和两份同步 EMG/力数据，不需要 FingerForce_Decode 或额外算法目录。运行环境为 Linux x86_64，Windows 队友使用 WSL2 Ubuntu；需要 Python 3.11，Spark 另需 Java 17。

## 从拉取到运行

在 WSL / Ubuntu 的仓库根目录执行：

```bash
# 缺 Java 时安装；这是系统依赖，需本机管理员权限
sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless

# 创建项目环境，首次需要联网下载依赖，不修改全局 Python
bash scripts/bootstrap_wsl.sh cpu

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

`mu_course.dataset.load_trial()` 可直接读取内置 EMG、力、采样率和通道编号。力为 [2,N]，顺序是中指屈曲、伸展，单位 g；屈曲 %MVC 可用 force[0] / flexion_mvc_g * 100 计算。这里只提供数据，不代表预测模型已实现。

## 结果与对照

输出 parameters.json、mu_results.npz；同目录重跑会覆盖。脉冲索引相对扩展矩阵，绘图补偿 extender 裁掉的起始样本。没有 MU 时提示检查输入和参数。

```bash
.venv-wsl/bin/python -m mu_course.benchmark \
    --backends cpu spark --repeat 3 --partition-counts 1 2 4 \
    --start 3 --duration 2 --output outputs/comparison
.venv-wsl/bin/python -m mu_course.plot_results outputs/comparison
```

有 CUDA 时可在 --backends 中加 gpu。每配置先预热，再顺序测三次，保存 timings.csv。总计算耗时含扩展、Spark 启动/分发与 fit，不含读写、保存和清理。CUDA 计时同步，Spark 等待 action。

Docker CPU 运行与双 worker 环境见 [环境说明](environment/README.md)。硬资源限额、真实双 worker MU 验收和 MLlib 预测仍待完成，不能用本地运行代替受限集群结果。

## 文档入口

- [课程计划](SPARK_MU_COURSE_PROJECT_PLAN.md) 与 [开发顺序](SETUP_PLAN.md)
- [架构](docs/knowledge-base/ARCHITECTURE.md) 与 [实际进度](docs/knowledge-base/KNOWLEDGE.md)
- [开发风格、接口与团队工作包](docs/DEVELOPMENT.md)
- [Spark 算子](docs/BACKEND_CONTRACT.md)

代码优先简约、易读、快速可运行，参数集中，接口写清形状、单位和时间基准。
仓库当前为 private。用户授权的两份精简科研数据是随仓库提供的例外；其他真实数据、环境、缓存、凭据与生成输出仍不跟踪。未来公开前单独审查源码、数据与历史。.private/ 的旧路径和检查脚本仅供历史追溯，队友运行不需要它们。
