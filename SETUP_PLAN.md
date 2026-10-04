# 分阶段搭建方案

依据本地完整课程计划拆分近期可执行步骤。原计划留在本地，此公开文档描述工程搭建与验收。

## 1. 部署方案

```text
Windows + NVIDIA 驱动
└── WSL2 Ubuntu 24.04
    ├── Python 3.11 独立环境：CPU / CUDA / Spark local 开发
    └── Docker Desktop WSL2 集成（待安装）
        └── Compose 内部网络
            ├── driver：全局 CKC 状态、候选选择、归一化、收敛、结果保存
            ├── master：资源调度
            ├── worker-1：1 core，时间块 1 / 更多分区
            └── worker-2：1 core，时间块 2 / 更多分区
```

采用同一 CPU 镜像运行 driver/master/worker。driver 使用 client 模式，向 executor
发布可解析的 `driver` 地址。首版 Spark 使用 CPU worker；GPU 是独立比较后端，
不要求 Spark 调度 GPU。两个 worker 在一台机器上验证架构，不等于物理集群扩展。

先用 WSL 原生 CPU/GPU/local 排除运行环境问题，再验收 Compose。正式三后端主对照
优先统一到同一容器运行条件；如果 CPU/GPU 原生运行、Spark 容器运行，则分别标明，
并补充同镜像 CPU 基线，不能把容器差异解释成纯后端加速比。

## 2. 引入哪些依赖

| 依赖 | 用途 | 当前处理 |
|---|---|---|
| NumPy / SciPy | 数组、信号操作 | 公共依赖 |
| PyTorch | CPU / CUDA / worker 块计算 | 固定同一版本；cpu/gpu 二选一 |
| PySpark 3.5.7 | local 与 Standalone | pip 包自带 Spark 分发文件，无需重复下载 Spark |
| h5py | MATLAB v7.3 输入 | 公共依赖 |
| PyYAML | 配置 | 公共依赖 |
| Matplotlib | 质量与性能图 | plots extra |
| pytest / Ruff | 验证与代码检查 | dev group |
| 私有 MU 核心 | CKC 全局控制与块计算 | 下一阶段独立打包，尚未接入 |

暂不引入 HDFS、Kafka、Kubernetes、数据库、Web 前端、torchvision、torchaudio。
没有仅为安装已有功能而克隆整个开源库的必要。

现有 MU 示例的 `requirements_decompose.txt` 只声明三项，h5py 和绘图依赖需要补齐。
其辅助报告/预处理代码仍有原项目路径引用和相对导入，不能直接使用 editable 安装。

## 3. 顺序与通过条件

| 阶段 | 实际工作 | 小规模验收 |
|---|---|---|
| S0 公共依赖 | 解析锁文件；安装隔离环境；检查版本与 CPU 张量 | CPU 计算、读取依赖可导入 |
| S1 WSL/CUDA | Java 17；CPU/CUDA 同版 Torch；真实 CUDA 张量 | GPU 名称、实际 device、同步后结果 |
| S2 Spark 环境 | local[2]；Compose master + 2 worker + driver | 分区求和正确；两个 worker 容器确有任务证据 |
| S3 仓库关联 | 用户建公开仓库后关联 origin；审核待提交内容 | URL、分支、远程状态正确；首次推送后 tracking 生效 |
| S4 私有核心 | 来源快照、独立包、冻结策略、CPU/GPU 接口 | 原实现与抽取结果回归；不混入不同版本语义 |
| S5 分布式内核 | 时间分块、缓存、广播、reduce、全局索引 | 相关矩阵/投影/梯度/更新数值等价 |
| S6 完整小样本 | 同一平台 2 秒 smoke，再 5 秒正式实验 | local 与两 worker 跑通；正式样本得到有效 MU |
| S7 性能与交付 | 三后端、CPU 1/2 线程、Spark 1/2/4 分区 | 每组预热 + 3 次；配置、版本、质量、耗时齐全 |

S0–S2 的环境算例不调用 CKC，也不计入 MU 性能结论。完整 CLI 在 S4–S6 逐步实现。

## 4. 计算分工

扩展矩阵沿时间轴切分，每块保留全部扩展通道。不可变块分发一次并缓存。
worker 执行相关矩阵局部贡献、投影和加权求和，driver 完成全局合并与 CKC 状态更新。
全局归一化在聚合后进行，不能把每个块独立分解后拼接成“分布式 CKC”。

首版数据规模：64 通道 × 2048 Hz × 5 秒，扩展因子 10，float32 约 25 MiB。
driver 保留数据和全局一维序列是有意取舍；不宣称解决超单机内存问题。

课程知识映射：RDD 分区、persist 与 action、broadcast、reduce、task/executor、
Standalone 调度与资源限制、序列化与网络开销、可复现实验与性能分析。

## 5. 仓库与私有核心

用户提供的现有示例作为抽取来源。下一阶段先记录来源 commit、工作区差异和文件哈希，
在另一个私有项目中打包，不在公开课程目录中维护第二份算法。
开发可 editable 安装；正式实验使用固定 commit 或带哈希 wheel。
所有 executor 和 driver 必须安装同一版本，私有镜像不公开推送。

公开仓库建议创建为空仓库（不预生成 README 或 LICENSE），避免首次关联时出现两套历史。
已验证用户提供的远程为空，初始化本地 `main` 并配置 `origin`。`origin` 是远程名，分支的 upstream
是如 `origin/main` 的追踪关系，不必另外命名一个 `upstream` remote。
用户已授权首次提交和推送环境文件；原始课程计划留在本地，不进入公开历史。

公开前检查原计划中的科研路径、历史与策略说明是否适合发布，Git ignore 本身不是发布审核。
私有源码、真实数据、模型、wheel、日志和机器配置不纳入公开提交。

## 6. 正式计时前提

- Linux 文件系统工作副本运行实验；当前 `/mnt/d` 路径只用于环境准备，不能当正式性能结果。
- 三后端相同数据哈希、策略、dtype、候选数、最大迭代和早停规则。
- CPU/GPU/Spark 顺序运行；Spark 各任务 BLAS/Torch 线程数为 1。
- CUDA 边界同步；Spark 必须等待 action；分别报告启动、分发、迭代和总耗时。
- 记录实际迭代数与 MU 匹配质量。Spark 慢、无 MU 或差异过大都保留并解释。

## 7. 官方依据

- [Spark 3.5.7：Java 17 / Python 兼容范围](https://spark.apache.org/docs/3.5.7/)
- [Standalone 部署与 Python client 模式](https://spark.apache.org/docs/3.5.7/spark-standalone.html)
- [Docker Desktop WSL2 集成](https://docs.docker.com/desktop/features/wsl/)
- [NVIDIA WSL 指南：沿用 Windows 驱动](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [PyTorch 官方历史版本与 CUDA wheel](https://pytorch.org/get-started/previous-versions/)
- [uv 的 PyTorch 独立索引与互斥 extras](https://docs.astral.sh/uv/guides/integration/pytorch/)
