# Spark 运动单位分解课程项目实施计划

> 历史备份：当前实施以根目录简化后的计划为准。本文件的早期架构要求不再作为必做清单。

日期：2026-10-04
状态：设计计划，尚未实施。本文中的新仓库、模块、配置和命令均为拟定接口，不代表已经创建或验证。
适用课程：基于大数据架构的数据处理和分析系统。第二门课程不在本计划实施范围内。

## 1. 项目目标与边界

建议题目：**基于 Apache Spark 的高密度肌电运动单位分解系统及 CPU GPU 性能对比**。

基于现有 CKC 运动单位分解实现，提取独立私有核心库；在独立课程仓库实现单次分解内部的 Spark 并行计算。在普通 Windows 电脑上的 Ubuntu 环境中，以小规模发力平台期 EMG 跑通 CPU、GPU 和 Spark 三种后端，验证数学计算与分解结果，比较运行时间。

### 1.1 必须完成

- 独立私有核心库和独立课程仓库，通过固定版本依赖建立关系。
- 统一终端入口，显式选择 `cpu`、`gpu`、`spark`。
- CPU/GPU 共享算法控制流程，Spark 参与同一次分解的矩阵计算。
- 单机 Spark local 模式开发，并完成 Standalone 两个 worker 的实际运行验证。
- 使用同一份预处理数据、同一个平台期片段、同一算法策略进行三后端对照。
- 连通测试、数值一致性测试、小型完整分解实验、分区数量对比。
- 自动保存运行配置、版本、分解结果、耗时和实验汇总。
- 提供结果可视化、项目报告材料与课堂演示流程。

### 1.2 不作为验收前提

- Spark 必须快于 GPU 或 CPU。
- 多台物理机器、大内存服务器、大规模数据或新硬件采购。
- 全量受试者、全部手指、全部腕角度实验。
- HDFS、Kafka、流式采集、完整 Web UI、用户登录系统。
- 跨会话 cascade、目标域滤波器适配、力预测模型重构。
- 公开真实数据或私有分解核心。

项目贡献定位为：**在保留 CKC 全局更新规则的前提下，实现时间分块矩阵计算、统一执行后端和可复现实验对照**。单机多 worker 验证的是执行架构，不据此宣称物理集群横向扩展能力。

## 2. 已确认的原项目事实

本节记录迁移依据，避免后续执行者把旧示例与最新科研主线混为一谈。

### 2.1 代码来源与版本差异

- 示例目录：`example/gpu_MU_extract/`。
- 首次入库：`b3d467a`，2026-07-15 14:18 +0800。
- 当前磁盘文件创建日期是 2026-08-10，不能将其视为代码产生时间。
- 对比 `src/models/v3/ckc_decomposer.py`，示例的主要分离向量优化循环相同，但外围处理已经落后。

| 后续记录 | 修改 | 课程库处理建议 |
|---|---|---|
| `003e7ed`，2026-07-16 | 分段数量 floor 改为 ceil | 如保留分段功能则纳入，并测试尾段 |
| `4ec1f45`，2026-07-16 | RoA 阈值 0.3 改为 0.5 | 显式配置并冻结，属于结果策略变化 |
| `e3b386d`，2026-07-17 | 模板回投影释放 GPU 张量 | 如纳入回投影则保留 |
| `fa4bc7a`，2026-07-18 | 新增 `refine_filters()` | 首版不纳入 |
| `0391234`，2026-09-13 | 全局去重按 PNR 顺序遍历 | 如使用对应去重流程则纳入 |

最新 `v3_1` 还包含脉冲数规则、PNR 门控和诊断接口变化。2026-08-27 知识库中的“PNR 仅诊断”记录不能直接覆盖当前代码中的门控策略。新库必须形成独立、明确的 `algorithm_policy`，不能逐段复制不同版本后假定语义一致。

### 2.2 数据与范式

- 输入应为已经完成共同预处理的、整流前 EMG，例如 `emg_clean`。
- 不把 `emg_env` 包络替代为分解输入；缺少目标字段时明确报错。
- 原项目典型采样率为 2048 Hz，典型单阵列为 64 通道；实际以数据元信息为准。
- MATLAB v7.3 使用 HDF5，读取二维数组时需要确认转置语义。
- 数据目录不纳入 Git。
- 2026-08-29 决策记录中，已核实范式的平台相对同步起点为：10% MVC `[3,9)`、20% MVC `[13,19)`、30% MVC `[23,29)` 秒。
- 这些区间是特定范式的依据，不能未经确认套用到所有日期、导出版本和会话。

## 3. Git 仓库搭建和依赖管理

### 3.1 仓库关系

建议名称均为占位名，实际建库时可调整。

```text
mu-decomp-core                 私有，分解核心的唯一维护来源
    ├── FingerForce_Decode     原科研仓库，未来通过适配器依赖
    └── spark-mu-course        独立课程仓库，计划公开工程部分
```

原科研仓库改为依赖核心库是后续单独迁移步骤，不是创建课程仓库时自动执行的操作。历史科研入口可继续保留用于复现，但需明确标为冻结版本，避免两边持续独立修改同一核心。

### 3.2 私有核心库包含什么

- 基础 CKC 全局控制流程。
- 延迟扩展、相关矩阵处理、代价权重、向量更新、收敛判定。
- 脉冲提取、质量统计、明确版本化的去重规则。
- CPU/CUDA 张量计算后端。
- 供 Spark 调用的块级计算接口与统一结果类型。
- 合成测试数据、数值测试、算法来源与变更记录。

不引入科研训练器、GUI、受试者配置、全套预处理链、DANN、回归器、cascade 或 MUAP 实验目录。

### 3.3 公开课程仓库包含什么

- Spark 计算后端、分区与聚合逻辑。
- 统一 CLI、数据包校验、平台片段导出、运行记录。
- 基准测试、数值和脉冲一致性比较、可视化。
- Ubuntu/Spark 环境说明、合成示例、知识库与报告。

公开边界采用“核心算法私有、分布式调度公开”的方案：Spark 调用私有库提供的块计算接口。若将优化公式和实现完整放入公开后端，应接受这些算法细节随之公开，不能仅凭 GPU 文件私有就声称算法未公开。

### 3.4 抽取过程

1. 在原仓库只读记录当前 commit、相关工作区差异和候选文件哈希。
2. 对照示例、v3 和 v3_1，列出首版采用的功能与必要修复。
3. 明确首版来源快照。不要为了迁移而提交或清理无关的科研改动。
4. 在独立目录创建私有包，保留原作者和已有许可说明，记录来源路径与提交。
5. 优先采取“快照抽取＋来源清单”，避免在原仓库重写历史。
6. 如果必须保留逐行历史，在一次性克隆中做路径历史提取；不要在原工作目录执行历史改写。
7. 先验证抽取前后 CPU/GPU 行为，再改造成可插拔计算后端。
8. 建立首个版本标签，如 `v0.1.0`，并记录完整 commit SHA。

建议私有库保存 `PROVENANCE.md`：来源仓库标识、完整 commit、文件映射、已纳入修复、排除功能、语义改变。原工作区未提交但需要纳入的内容必须另记文件哈希及补丁来源，不能归因于已有 commit。

### 3.5 依赖方式

首选可安装 Python 包，课程仓库依赖固定完整 commit，环境锁文件保存解析结果。公开 README 可用 `<PRIVATE_CORE_GIT_URL>` 占位，授权安装说明私下提供；如保留真实私有仓库 URL，应确认允许公开该地址，URL 本身不授予访问权。

- 开发时：在授权电脑上对相邻私有库使用 editable 安装。
- 正式实验：安装冻结版本，不使用移动的分支头或未记录的 editable 工作区。
- 核心修复：提交到私有库，生成新版本，再由使用方显式升级。
- 同学和教师：授予必要的私有依赖访问权，或私下提供对应版本 wheel。
- wheel 是交付形式，不是源码保密技术；普通 Python wheel 通常包含源码。
- driver 和所有 executor 必须安装相同核心包版本，运行前主动探测。

submodule 可作为备选，但首版不同时使用包依赖与 submodule 管理同一代码。公开仓库缺少授权依赖时，允许查看文档和预生成示例；完整 CKC 实验须明确提示依赖未安装，不能假装已运行。

### 3.6 分支和公开发布

- 主分支保持可运行，功能分支建议 `codex/core-extraction`、`codex/spark-backend` 等。
- 提交按抽取、接口、后端、测试、文档分开，保留 Conventional Commits 风格。
- 初始课程仓库可先私有，完成数据、日志、凭据和依赖边界检查后再公开。
- `.gitignore` 从第一次提交前建立，覆盖 `data/`、`outputs/`、`cache/`、`.venv/`、`.env`、私有 wheel 和 Spark 事件日志。
- 公开小型示例仅放在明确审核的 `examples/synthetic/` 与 `docs/examples/`。
- 不把访问 token、SSH 私钥、网盘提取码写入依赖 URL、镜像层或公开文档。
- ignore 规则不能清除已经提交的数据；发布前需检查待提交文件及历史。

## 4. Ubuntu 开发与运行环境

### 4.1 推荐路径与备选路径

推荐：Windows + WSL2 Ubuntu，以同一个 Ubuntu 用户环境运行 CPU/GPU/Spark。前提是现有电脑有可用的 NVIDIA CUDA GPU，并确认课程允许该 Ubuntu 形式。

备选：保留附件的 VMware Ubuntu 做 CPU/Spark，Windows 或 WSL2 做 GPU。这样仍可完成架构演示，但三后端耗时表必须注明操作系统和虚拟化差异，不解释为纯后端加速比。

附件环境的 2 核、4–6 GB 内存和 50 GB 磁盘是起步配置，不代表两 worker 的保证配置。先测进程实际内存，不要求扩充硬件。没有 CUDA 设备时显式报告 GPU 不可用，不把 CPU 回退结果写成 GPU 结果；完整三后端验收需在现有可访问 GPU 上补跑。

### 4.2 拟定软件组合

| 项目 | 首版建议 | 说明 |
|---|---|---|
| Ubuntu | 24.04 系列 | WSL2 或课程虚拟机 |
| Python | 3.11.x | 新项目独立环境，不要求复制原项目全部依赖 |
| Java | OpenJDK 17 | 与选定 Spark 组合匹配 |
| Spark / PySpark | 3.5.7，所有节点一致 | 本计划采用已查阅官方兼容说明的版本；实施时验证安装来源并锁定 |
| PyTorch | 经实际 CUDA 验证后固定版本 | CPU/GPU 尽量使用同版本，不在计划中猜测具体 CUDA wheel |
| 数据计算 | NumPy、SciPy、h5py、PyYAML | 实施时解析并锁定完整版本 |
| 展示与测试 | Matplotlib、pytest；pandas 按需 | 不引入原项目无关依赖 |
| 环境管理 | uv + pyproject.toml + lock | 记录 uv 版本及 CPU/GPU 安装配置 |

Spark 3.5.7 官方列出的运行范围包含 Java 17 和 Python 3.8+，Python 3.11 是本项目拟定组合，不代表整个依赖栈已经实测。

### 4.3 环境建立顺序

1. 记录 Windows、Ubuntu、CPU 核心数、内存和 GPU 型号。
2. 安装并验证 Ubuntu、Python 和 Java。
3. 在 Ubuntu Linux 文件系统内建立项目目录和虚拟环境，避免把 `/mnt/c` 或 `/mnt/d` 的文件访问差异混入性能实验。
4. 安装课程包与授权核心包，验证 CPU。
5. 验证 `torch.cuda.is_available()`、GPU 名称及一个实际 CUDA 计算；WSL2 按 NVIDIA 官方流程配置，不在其中照搬完整 Linux 显卡驱动安装流程。
6. 启动 Spark `local[2]`，验证 Python worker 能导入核心库。
7. 建立 Standalone master 和两个 worker，driver 使用 client 模式提交 Python 应用。
8. 检查 executor 的 Python 路径、核心版本、线程限制和资源配置。

先用同一 Ubuntu 中的独立进程启动 master/worker。Docker Compose 是可选封装，环境链路稳定后再加；不要求同时维护两套集群部署。

### 4.4 小资源 Spark 设置

- local 模式从 `local[2]` 开始；真实架构演示使用一个 master、两个 worker。
- 每个 worker 宣告 1 个 CPU core，每个 executor 1 core，应用总 core 限制为 2。
- 初始可试 driver heap 1 GiB、每 executor heap 512 MiB；这是待实测起点，额外 Python/JVM/native 内存不包含在此数值中。
- 不使用绕过 Spark 内存检查的测试配置伪装资源足够。
- NumPy/BLAS/PyTorch 每任务默认 1 线程，避免两个任务分别占满所有 CPU。
- 基础数据两分区；分区数量和 worker 数是两个独立实验变量。
- worker 数不足时先缩小数据和并发；4 GB 环境必要时先完成 local，随后分时运行精简 Standalone 验证。
- 不依赖 Python worker 内可变全局状态跨任务保存算法状态。
- 对外不开放无认证集群；本机演示将 UI 端口限制在本机访问。

### 4.5 若使用 Docker

- driver、master、worker 放在同一个可互通网络中。
- driver 地址必须可被 executor 访问，不能广播一个只对宿主有效的 localhost。
- Python Standalone 使用 client 部署模式，不设计为 Python Standalone cluster 部署模式。
- 数据只读挂载到各容器相同路径，例如 `/workspace/data`，输出另挂载可写路径。
- 每个运行节点预装相同 Python、PyTorch 和核心包；不把整个虚拟环境通过 `--py-files` 传递。
- 含私有核心的镜像视为私有制品，不推送到公共镜像仓库。

## 5. 预期文件目录

### 5.1 私有核心库

```text
mu-decomp-core/
├── AGENTS.md
├── README.md
├── PROVENANCE.md
├── CHANGELOG.md
├── pyproject.toml
├── uv.lock
├── src/mu_decomp_core/
│   ├── __init__.py
│   ├── contracts.py          # 数据类型、计算后端协议、结果规范
│   ├── config.py             # 算法参数与质量策略
│   ├── engine.py             # 唯一 CKC 全局控制流程
│   ├── extension.py          # 延迟扩展与样本索引映射
│   ├── kernels.py            # 投影、局部梯度等块计算
│   ├── activity.py           # 活动度、初始化与禁止区
│   ├── spikes.py             # 原始脉冲检测与周期细化
│   ├── quality.py            # SIL、PNR、去重
│   └── backends/torch_ops.py # CPU/CUDA 实现
├── tests/
│   ├── test_reference.py
│   ├── test_extension.py
│   ├── test_kernels.py
│   └── test_quality.py
└── docs/knowledge-base/
    ├── KNOWLEDGE.md
    ├── ARCHITECTURE.md
    └── DECISIONS.md
```

### 5.2 课程仓库

```text
spark-mu-course/
├── AGENTS.md
├── README.md
├── IMPLEMENTATION_PLAN.md    # 迁入本计划后可改名
├── pyproject.toml
├── uv.lock
├── .gitignore
├── configs/
│   ├── algorithm_v1.yaml
│   ├── smoke.yaml
│   ├── plateau_small.yaml
│   ├── benchmark.yaml
│   └── spark/
│       ├── local.yaml
│       └── standalone.yaml
├── src/mu_course/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── routing.py
│   ├── runner.py
│   ├── backends/
│   │   ├── local.py
│   │   └── spark.py
│   ├── distributed/
│   │   ├── session.py
│   │   ├── partitioning.py
│   │   └── reductions.py
│   ├── data/
│   │   ├── manifest.py
│   │   ├── mat_reader.py
│   │   ├── plateau.py
│   │   └── synthetic.py
│   ├── evaluation/
│   │   ├── numerical.py
│   │   ├── spike_matching.py
│   │   └── benchmark.py
│   ├── artifacts/
│   │   ├── provenance.py
│   │   └── writer.py
│   └── visualization/
│       ├── plateau_check.py
│       ├── decomposition.py
│       └── performance.py
├── scripts/
│   ├── cluster_start.sh
│   ├── cluster_stop.sh
│   └── submit.py             # spark-submit 的薄入口
├── environment/
│   ├── README.md
│   ├── spark-defaults.conf.example
│   ├── Dockerfile            # 可选
│   └── compose.yaml          # 可选
├── tests/
│   ├── test_contracts.py
│   ├── test_plateau.py
│   ├── test_partitioning.py
│   ├── test_backend_equivalence.py
│   └── test_cli_errors.py
├── examples/synthetic/
├── data/                    # 私下下载；整个目录 Git 忽略
│   ├── manifests/
│   ├── processed/
│   └── segments/
├── outputs/                 # Git 忽略
│   ├── runs/<run_id>/
│   └── comparisons/<comparison_id>/
├── cache/                   # Git 忽略
└── docs/
    ├── knowledge-base/
    │   ├── KNOWLEDGE.md
    │   ├── ARCHITECTURE.md
    │   └── DECISIONS.md
    ├── DATA_CONTRACT.md
    ├── BACKEND_CONTRACT.md
    ├── BENCHMARK_PROTOCOL.md
    ├── ENVIRONMENT.md
    ├── MIGRATION_MAP.md
    └── report_outline.md
```

这些是职责边界，不要求第一天创建全部空文件。仅在相应阶段增加实际需要的模块，避免只有骨架没有行为。

## 6. 模块职责与统一后端路由

### 6.1 一套控制流程，多个计算后端

```text
CLI → 配置解析 → 数据与平台片段校验 → backend factory
                                              ↓
                                  私有核心 CKC engine
                                              ↓
                    CPU Torch / CUDA Torch / Spark operations
                                              ↓
                                  标准化结果与运行记录
```

`engine.py` 管理候选顺序、禁止区域、周期细化、向量更新、收敛和质量规则。Spark 后端只实现计算协议，不能复制一份略有不同的 CKC 主循环。

抽取实施顺序必须先包装现有类得到参考结果，再逐步替换矩阵操作。一次性重写控制循环会失去可验证的迁移基准。

### 6.2 关键模块责任

| 模块 | 负责 | 不负责 |
|---|---|---|
| `cli` | 命令、参数、退出码 | 分解数学公式 |
| `config` | 合并、类型检查、冻结有效配置 | 隐式选择硬件 |
| `routing` | 后端工厂、能力检测 | 在 GPU 失败时偷偷用 CPU |
| `runner` | 一次运行生命周期、计时、资源释放 | 候选 MU 的科学筛选 |
| `core.engine` | 所有后端共享的算法控制 | Spark 部署 |
| `core.kernels` | 可复用矩阵块计算 | 读取受试者路径 |
| `backends.spark` | 实现统一计算协议 | 改变停止阈值或质量门控 |
| `partitioning` | 时间块、全局索引、缓存释放 | 对每块独立训练分离向量 |
| `data.plateau` | 核验范式、选择并冻结片段 | 根据某后端最终 MU 数反向挑数据 |
| `evaluation` | 数值、脉冲和性能比较 | 把 CPU 输出视为生理真值 |
| `artifacts` | 版本、配置、结果、哈希 | 默认发布真实数据 |
| `visualization` | 绘制已有运行记录 | 绘图时重跑或修改分解 |

### 6.3 计算协议草案

协议重点覆盖：准备输入、计算相关矩阵、投影、加权信号求和、活动度、同步、资源关闭。可拟定为 `prepare`、`correlation`、`project`、`weighted_sum`、`activity`、`synchronize`、`close`。

- 输入 `emg` 统一为 `[channels, samples]`，显式采样率、通道编号、dtype。
- 扩展信号统一为 `[extended_channels, valid_samples]`。
- `project` 返回带明确全局时间索引的一维投影，不能依赖 Spark 收集顺序。
- `weighted_sum` 返回未归一化的全局梯度贡献；归一化只在汇总后执行。
- 控制器的张量和 NumPy 转换、dtype、设备迁移必须显式，不能让不同后端采用不同公式。
- 并行任务返回数值结果，不直接向同一输出文件追加。

统一结果至少包括：MU 脉冲索引、分离向量、SIL/PNR、候选状态、实际迭代次数、时间索引映射、阶段耗时、有效配置、环境与版本。使用 NPZ 加 JSON 等清晰格式，避免为交换基础数组引入任意 pickle 反序列化。

### 6.4 统一 CLI 草案

安装后入口拟为 `mu-decomp`，等价入口为 `python -m mu_course`。以下命令是待实现契约。

```bash
mu-decomp doctor --backend cpu
mu-decomp doctor --backend gpu
mu-decomp doctor --backend spark --runtime configs/spark/local.yaml

mu-decomp data inspect --manifest data/manifests/course_v1.json
mu-decomp data prepare --config configs/plateau_small.yaml

mu-decomp run --backend cpu --config configs/smoke.yaml
mu-decomp run --backend gpu --config configs/smoke.yaml
mu-decomp run --backend spark --config configs/smoke.yaml --runtime configs/spark/local.yaml

mu-decomp run --backend cpu --config configs/plateau_small.yaml
mu-decomp run --backend gpu --config configs/plateau_small.yaml
mu-decomp run --backend spark --config configs/plateau_small.yaml --runtime configs/spark/standalone.yaml

mu-decomp benchmark --config configs/benchmark.yaml
mu-decomp compare --runs outputs/runs/run_cpu outputs/runs/run_gpu outputs/runs/run_spark
mu-decomp visualize --comparison outputs/comparisons/example
```

Standalone 的薄入口将参数传给同一个 CLI，不另写算法脚本：

```bash
spark-submit --master spark://localhost:7077 --deploy-mode client \
  scripts/submit.py run --backend spark --config configs/plateau_small.yaml \
  --runtime configs/spark/standalone.yaml
```

上面的地址只适用于同一 Ubuntu 的进程式部署，容器环境改为集群内可解析地址。`benchmark` 首版按配置顺序串行运行三个后端，不让它们争抢资源。

配置优先级：内置默认值 < 算法配置 < 实验配置 < 显式 CLI 覆盖。运行前保存完整 resolved config；检测到不一致的 master、backend、dtype 或数据哈希时直接报错。

建议退出码：0 成功，2 配置或数据错误，3 后端不可用，4 执行失败，5 比较未通过。零 MU 可作为有效算法结果保存，但不能自动判为完整分解验收通过。

## 7. Spark 单次分解内部的并行设计

### 7.1 分块单位

只沿扩展信号的时间维分块，所有块保留完整扩展通道：

`X = [X_1, X_2, ..., X_P]`。

首版使用 2 个分区，块内用 PyTorch CPU 向量化，与 CPU/GPU 参考的计算语义尽量一致。一个采样点不形成一个 Spark task。不同 MU 候选仍按原算法顺序执行。

### 7.2 初始化

1. driver 读取冻结平台片段，执行同一延迟扩展，生成明确的有效样本映射。
2. driver 切分扩展矩阵，一次分发并缓存；使用显式 action 物化缓存。
3. 各分区计算 `R_p = X_p X_p^T`，汇总后除以总样本数 N。
4. driver 对全局 R 求伪逆，再用于各块投影。先采用统一 CPU 伪逆有利于隔离迭代后端差异。
5. driver 保留小型全局活动度序列及控制状态，初版允许 driver 保留原输入用于随机列访问。

首版不宣称可处理 driver 内存放不下的数据。今后若改为 worker 分块读盘，再处理扩展重叠区与分布式随机列读取。

### 7.3 一次普通迭代

当前全局分离向量为 c：

1. 计算 `a = R_plus^T c`，worker 计算 `s_p = a^T X_p`。
2. 按原算法产生权重与禁止区域掩码；分区使用全局索引，不能把每个块的端点当作整个信号的端点。
3. 分区计算 `g_p = X_p w_p`，全局相加得到 g。
4. 汇总后归一化 g，按原代价方向、步长和归一化规则更新 c。
5. 计算更新后的投影，检查与旧投影的收敛程度。

局部梯度不能先归一化再求和。涉及均值的统计量按样本数加权，不能对不等长分区的均值直接平均。伪逆阈值、dtype、零范数处理、边缘屏蔽规则必须三后端一致。

### 7.4 周期细化与全局操作

- 活动度 FFT、峰区搜索、全局 K-means 脉冲提取、不应期约束、周期性细化和最终去重首版保留在 driver。
- 为此按时间顺序收集一维投影或活动度，而不在每轮收集完整扩展矩阵。
- 64 通道、扩展因子 10、2048 Hz、5 秒的 float32 扩展矩阵约 25 MiB，一维投影约 40 KiB。这个取舍适合小型课程实验。
- 禁止直接对多个时间块分别执行完整 `CKCDecomposer.fit()` 后拼接，作为本方案“单次分解内部并行”的实现。
- 细化可能需要额外投影和同步，实际 stage/job 数作为实验结果记录，不承诺固定每轮一个 job。

### 7.5 后续性能优化顺序

先完成逐步对应原算法的实现，再合并同一分区内可独立完成的投影、权重和统计计算。仅在等价测试通过后启用融合路径。

- 缓存不可变 X 块，及时释放过期广播和 RDD。
- 用明确的返回值及 reduction 计算梯度，不用 accumulator 承载要求精确一次的算法状态。
- task 重试不得产生重复的结果文件或重复 MU 计数。
- 不依赖分区到 Python 进程的固定映射。
- 日志区分分区数、worker 数、实际参与 executor 数。
- 对照 1、2、4 分区寻找本机合适粒度，不把“更多分区”直接视为优化。

## 8. 数据分发和平台期片段选择

### 8.1 私下数据包

真实预处理数据通过受限网盘分发，解压到项目根目录 `data/`。公开文档仅描述结构，不包含分享链接和提取码。

数据包附清单：匿名受试者编号、会话、手指、条件、阵列、fs、通道列表、形状、dtype、预处理 profile、时间原点、文件 SHA-256、平台区间与单位。姓名缩写、私人绝对路径和无关原始元信息不进入公开材料。

对三种后端，分解只读取 EMG。同步力信号用于确认平台和绘图，不用来为某一后端额外优化候选筛选。

### 8.2 平台选择流程

1. 优先从一个已知有效的 Trapezoid 会话、一个手指、一个 64 通道阵列开始。
2. 优先核验 20% MVC 平台；它是拟定起点，不保证一定优于 10% 或 30%。
3. 确认 aligned 数据的第零样本是否仍对应同步起点，是否发生重裁剪。
4. 结合范式元信息和实际力曲线，确认平台中没有上升、下降、明显漂移或掉线。
5. 若当前记录确实匹配 `[13,19)` 秒的平台，建议正式片段采用 `[13.5,18.5)` 秒，保留两侧 0.5 秒裕量。
6. 连通片段可从同一平台中心选 `[15,17)` 秒。所有这些时间仅对已确认的同一时间原点有效。
7. 保存确认图：实际力曲线、平台边界、选中片段和 EMG 幅值摘要。
8. 冻结片段 ID、起止样本索引、通道列表和哈希，之后 CPU/GPU/Spark 全部复用。

原平台通常只有 6 秒，不能为了得到“10 秒测试片段”跨入斜坡或拼接不同平台。若要增加规模，可对同一平台测试 2、3、5 秒，或独立运行多个平台；不得把多段拼接冒充连续信号。

### 8.3 自动检查与人工确认

- 时间范围在数据有效区间内，无空数组、NaN 或 Inf。
- 目标阵列映射正确，坏通道处理一致，不临时为某后端删通道。
- 显示实际平台力均值、标准差和趋势；阈值通过数据检查后冻结，不在计划中虚构已验证门限。
- 平台无有效发力时阻止正式实验，不能仅因文件名含 Trapezoid 就通过。
- force 若为 grams，正确读取 MVC 后才能显示 `%MVC`；不把单位变换混入 EMG 分解输入。
- 预处理只做一次，裁剪前的滤波 profile 记录清楚，不在每个短片段上重复滤波。

### 8.4 配置草案

```yaml
schema_version: 1
experiment_id: plateau_small_v1
algorithm_policy: ckc_course_v1
data:
  manifest: data/manifests/course_v1.json
  record_id: pilot_plateau_01
  field: emg_clean
  adapter_id: 0
  sample_rate_hz: 2048
  # 仅在时间原点及实际平台核验后启用这些值。
  time_origin: sync_onset
  plateau_s: [13.0, 19.0]
  crop_s: [13.5, 18.5]
  require_approved_segment: true
algorithm:
  extension_factor: 10
  max_candidates: 3
  max_iterations: 25
  cost_function: 3
  refinement_interval: 20
  dtype: float32
  seed: 42
execution:
  backend: cpu
  cpu_threads: 1
  partitions: 2
output:
  root: outputs/runs
```

`max_candidates` 映射旧代码的 `n_mu_max`，不能简单解读为最终 MU 上限：周期提取、候选接受和去重会影响最终数量。质量门控、pinv 容差与收敛规则放入 `algorithm_v1.yaml` 明确定义，不依赖不同版本的隐藏默认值。

## 9. 测试与验收计划

### 9.1 P0 环境检查

检查 Python/Java/Spark 版本，核心包完整 commit，driver/executor 导入路径，CPU 线程设置，CUDA 实际计算和数据读取权限。缺少授权核心或 CUDA 明确失败，不自动替换后端。

### 9.2 P1 数学操作测试

先用小型确定性数组，不依赖真实数据，覆盖：

- 时间分块重组无遗漏、无重复，非整除分区和末块正确。
- 全局相关矩阵、投影、梯度、一次向量更新与单机一致。
- 块边界附近禁止区、边缘屏蔽和加权统计正确。
- rank-deficient 矩阵伪逆、全零输入、非有限输入行为明确。
- 延迟扩展后的脉冲索引能映射回原始裁剪片段及会话时间。

建议 float64 单步测试起始容差 `rtol=1e-8, atol=1e-10`；float32 起始容差 `rtol=1e-4, atol=1e-5`。这些是待验证的工程容差，需对病态输入和不同设备报告误差，不因失败任意放宽。

### 9.3 P2 连通测试

同一有效平台的 2 秒片段、64 通道、扩展因子 10、最多 1–2 个候选、3–5 次迭代。依次运行 CPU、GPU、Spark local，再运行 Standalone。

验收：

- 命令退出成功，结果文件、有效配置和 provenance 齐全。
- 所有数值有限，脉冲索引合法，输出 shape 符合协议。
- Spark 确有分区任务执行，不是 driver 独自完成后伪装为 Spark。
- GPU 输出设备记录确为 CUDA，并记录同步后的时间。
- 可接受零 MU，但标记 `smoke_only`，不能用于 MU 质量结论。

### 9.4 P3 原实现到新核心的回归验证

先固定一份参考快照和同一平台片段，在 CPU 上比较旧入口与新 engine。对比初始峰位置、代价权重、逐轮投影、迭代次数和脉冲输出。

必要修复如果改变了算法结果，要分别记录“原实现复现”和“新冻结策略”的结果，不能把有意变更误报为数值误差。

### 9.5 P4 小规模完整分解

正式起点：同一平台中心 5 秒、64 通道、扩展因子 10、最多 3 个候选、最多 25 次迭代。可在所有后端同时调整为 5 个候选或 45 次迭代，禁止只给某后端更少工作量。

- 至少保留一个参考配置可以得到非空、通过冻结质量规则的 MU 结果。
- 记录实际是否经过第 20 次迭代细化；早停可能导致没有触发。
- 为分支覆盖另设测试：固定迭代诊断模式或单元测试，明确与科学默认早停实验分开，不为覆盖分支暗改正式结果。
- 若片段没有有效 MU，先核验发力、通道、时间和原参考实现，再按预先记录的顺序尝试其他平台。片段选择冻结后重跑全部后端。
- 不以无限降低 SIL/PNR 阈值换取非空结果。

### 9.6 P5 三后端结果比较

数值比较与科学输出比较分开：单步矩阵误差可严格检查；非线性迭代后的 MU 顺序可能变化，需先匹配。

- 以脉冲时间一致性构造 MU 两两相似度，再做一对一匹配。
- 容差建议从 1 ms 起，明确按采样率转换；不允许通过大范围平移掩盖索引错误。
- 记录匹配对的 RoA、precision/recall/F1、未匹配数量、SIL/PNR 差异。
- 比较迭代次数、初始候选和是否跨越阈值，定位数值误差放大的来源。
- 建议将匹配脉冲 F1 0.95 作为首轮调查阈值，不是预先保证的性能；未达时保留结果并解释，不能删除失败案例。
- 真实数据参考一致性不等于生理真值准确率。合成数据的已知脉冲只验证受控场景。

### 9.7 P6 小型性能矩阵

| 实验 | 固定项 | 变化项 | 重复 |
|---|---|---|---|
| 三后端主对照 | 同一 5 秒片段和算法策略 | CPU 1 线程、GPU、Spark 2 worker/2 分区 | 每配置 3 次 |
| CPU 公平补充 | 同一任务 | CPU 1 与 2 线程 | 每配置 3 次 |
| Spark 分区粒度 | 固定 2 worker、总 2 core | 1、2、4 分区 | 每配置 3 次 |
| 片段长度 | 同一平台中心及算法策略 | 2、3、5 秒 | 资源允许时每配置 3 次 |

长度实验的生理数据内容会变化，因此只能用来分析运行成本与结果变化，不要求不同长度提取完全相同 MU。

计时规则：

1. 每后端预热一次，预热不计入重复结果。
2. CUDA 计时在相关边界调用同步，避免把异步提交时间当执行时间。
3. Spark 缓存必须物化，计时覆盖 action 完成。
4. 分别记录会话初始化、读数据、延迟扩展/分发、相关矩阵/伪逆、迭代、脉冲/去重、输出耗时。
5. 给出冷启动端到端时间和已初始化会话的任务时间；后者仍包含每次任务的数据准备与计算。缓存复用另列，不能给 Spark 免费复用却不给其他后端相同说明。
6. 顺序运行，记录运行次序，避免 CPU/GPU/Spark 互相抢资源。
7. 采用自然早停实验展示真实成本；另加固定迭代微基准隔离不同迭代次数的影响，二者分表呈现。
8. 汇报中位数及最小/最大值，3 次重复不做强统计显著性推断。

记录 wall time、内存采样峰值、GPU 峰值分配内存、Spark task/job 数、可获得的 executor 指标。采样峰值与实际绝对峰值有区别；没有测得的指标标为不可用，不填 0。

Standalone 验收需证明两个不同 worker/executor 实际承担计算，例如任务记录与事件日志。两个分区不自动意味着用了两个 worker。

## 10. 结果文件与可视化

每次运行生成唯一目录，默认不覆盖：

```text
outputs/runs/<run_id>/
├── resolved_config.yaml
├── provenance.json
├── result.npz
├── mu_metrics.csv
├── iteration_trace.csv
├── timing.json
├── resources.json
├── run.log
└── figures/
```

provenance 至少记录：课程 commit 和 dirty 状态、核心版本/commit、输入及片段哈希、算法策略、软件版本、OS/虚拟化、CPU/GPU、线程数、Spark master/worker/executor、分区数、时间戳、预热及缓存口径。

首版四类图：

1. 平台选择确认图：力曲线、平台范围、最终裁剪范围。
2. 分解结果图：EMG 摘要和 MU raster、SIL/PNR。
3. 一致性图：匹配 MU 的脉冲对照和误差统计。
4. 性能图：分阶段耗时、三后端总耗时、分区数量对比。

Spark UI 截图用于展示任务调度，不能替代结果图。首版不做交互 UI；时间允许再添加只读取已完成实验记录的轻量仪表板。真实波形、MU 模板和事件日志默认视为私有实验产物，公开前逐项审核。

## 11. 新仓库知识库迁移

两个新仓库均采用 `docs/knowledge-base/`，由各自 `AGENTS.md` 显式声明。这里只规划未来目录，不改动当前仓库知识库。

### 11.1 迁移原则

- 迁移的是经过核验、与新项目相关的知识，不整体复制旧日志。
- 旧结论写明来源与适用版本；计划项不能标记为已实现。
- 公开知识库不带受试者标识、私有链接、未公开科研结果或机器绝对路径。
- 原文件存在过时条目时，对照代码和较新决策再写入。
- `AGENTS.md` 存行为规则；三份知识文件存项目事实，二者分开。

### 11.2 应迁移的内容

| 原项目来源 | 迁移内容 | 目标 |
|---|---|---|
| `KNOWLEDGE.md` Environment / Conventions | Python 风格、配置驱动、依赖锁定、相对路径 | 两库 KNOWLEDGE，按新环境重写 |
| `KNOWLEDGE.md` Data Handling / Python Gotchas | v7.3/HDF5、二维转置、采样率、通道与力单位 | 课程 KNOWLEDGE + DATA_CONTRACT |
| `ARCHITECTURE.md` preprocessing / data loading | Stage 3 `emg_clean` 输入契约、同步后裁剪、无重复滤波 | 课程 DATA_CONTRACT |
| `DECISIONS.md` 2026-08-29 | 同步时间原点、平台区间修正及适用范围 | 课程平台选择决策 |
| `DECISIONS.md` 2026-07-14/15 与后续 Git 修复 | 单阵列分解、延迟扩展、分段/去重/显存问题 | 核心 KNOWLEDGE + PROVENANCE |
| GPU 示例用法文档 | 输入输出与基础 CKC 参数 | 核心 README，按新 API 重写 |
| 旧分解代码与 MATLAB 来源说明 | 算法来源、许可、索引语义 | 私有 PROVENANCE |
| 本计划新增设计 | 后端协议、Spark 控制边界、计时和发布策略 | 两库 ARCHITECTURE / DECISIONS |

### 11.3 不迁移或仅按需引用

- 不复制全量 Session Log、论文进展、跨角度结果表和受试者效果排名。
- 不复制 v0/v1/v2/v4 的架构、GUI/LSL/硬件控制说明。
- 不复制原项目整个 `pyproject.toml`、Torch 下载源或依赖版本，先做新环境验证。
- 不将旧“PNR 不作门控”写成所有版本通用规则；首版算法策略独立冻结。
- 不复制腕角正负方向的旧冲突解释；首版只保留数值与数据包定义，确需解剖方向时引用已核实元数据。
- 不将平台时间硬编码为所有数据集通用真理。
- 不把 MU firing-rate 回归、CFR 居中泄漏等下游知识塞入基础分解库；以后加入力解码时再迁移。

### 11.4 三份知识文件的具体内容

**KNOWLEDGE.md**：最短运行路径、私有依赖安装方式、数据放置规则、平台确认状态、已验证环境、常见报错、当前阶段与未完成事项。

**ARCHITECTURE.md**：两库依赖、目录与模块责任、CLI→engine→backend 数据流、矩阵维度、driver/executor 边界、结果格式。

**DECISIONS.md**：至少建立以下决策记录，实施前标为 Proposed，通过后更新为 Accepted：

1. 私有核心作为唯一维护来源，使用固定 commit 包依赖。
2. CPU/GPU/Spark 共享 CKC 控制器。
3. Spark 按时间块并行、driver 保留小型全局序列。
4. 首版允许 driver 保留输入，不宣称超单机内存能力。
5. 平台片段由数据与力曲线确认后冻结。
6. 全局质量策略统一，GPU 不静默降级。
7. 单机多 worker 与物理集群结果分开表述。
8. 冷启动、任务和迭代微基准分别计时。
9. 数据、私有依赖与公开示例的分发边界。

在 `MIGRATION_MAP.md` 记录每条知识的原相对路径、标题/日期、来源完整 commit、新位置、采用/改写/排除原因。不要仅记录容易变化的行号。

## 12. 分阶段实施与完成条件

| 阶段 | 工作 | 完成条件 |
|---|---|---|
| M0 冻结范围 | 确认平台数据、环境路线、算法策略、公开边界 | 决策和数据清单齐全 |
| M1 抽取核心 | 打包基础 CKC、纳入必要修复、保留来源 | 原参考与新核心回归可解释 |
| M2 统一入口 | CPU/GPU 路由、结果契约、环境检查 | 两后端同一 CLI 跑通，GPU 无静默回退 |
| M3 准备数据 | 私有包校验、平台核验、片段冻结 | 平台确认图与片段哈希齐全 |
| M4 Spark 数学计算 | 分块、缓存、相关矩阵、投影、梯度 | 单步与分区边界测试通过 |
| M5 完整分解 | 接入共同控制器、周期细化与去重 | local 与两 worker 正式小样本运行 |
| M6 实验 | 三后端、CPU 补充、分区对照 | 原始记录完整、比较规则一致 |
| M7 展示与交付 | 图、报告、环境说明、演示流程 | 授权同学在新目录复现小规模任务 |
| M8 可选科研接回 | 原仓库通过适配器引用核心 | 单独授权并完成科研回归后进行 |

如果由两人合作，可由一人负责核心/CPU/GPU，一人负责 Spark/环境/展示；先冻结计算协议和样例输入输出。此分工是后续人员建议，不授权自动创建任务或联系他人。

## 13. 风险与退阶策略

| 情况 | 处理 |
|---|---|
| Spark 小任务比 CPU/GPU 慢 | 保留真实结果，解释启动、同步和调度成本；作为架构研究结论 |
| 内存不足 | 缩短同一平台片段、减少候选/并发；三后端共同更新参数 |
| 2 worker 环境暂时不稳定 | local 继续开发，Standalone 验收保持未完成，不混称集群已跑通 |
| CUDA 不可用 | CPU/Spark 继续实现，GPU 单独标为待补，不伪造三后端完成 |
| 短片段没有有效 MU | 核验平台与原实现，尝试预先定义的其他有效平台并统一重跑 |
| 迭代数不同导致耗时差异 | 同时报实际迭代数，增加固定迭代微基准 |
| 浮点误差改变阈值附近候选 | 保存逐轮诊断和匹配结果，定位而不是只放宽阈值 |
| 私有依赖无法给教师访问 | 提前安排授权或私下版本包交付，公开 README 明确限制 |
| 原知识库与当前代码冲突 | 标明版本与作用路径，以冻结策略和验证结果作为本项目准则 |

## 14. 最终验收清单

- [ ] 本项目与原科研仓库的维护边界清楚，未复制持续漂移的第二份核心。
- [ ] 核心版本、来源修复和算法策略可追溯。
- [ ] `cpu`、`gpu`、`spark` 使用同一命令协议。
- [ ] GPU 实际使用 CUDA，Spark 实际有 executor 执行核心块计算。
- [ ] 两 worker 分担任务的证据可查。
- [ ] 平台期经过确认，无跨斜坡、错误时间原点或跨平台拼接。
- [ ] 三后端使用完全相同的数据片段与算法参数。
- [ ] 分区边界、全局归一化、周期细化和索引映射有有效测试。
- [ ] 至少一组小型真实平台实验得到可比较的非空 MU 输出。
- [ ] 性能记录区分初始化、任务、迭代，并正确处理 GPU 同步和 Spark action。
- [ ] 结果差异和失败情况保留，不预设 Spark 更快。
- [ ] 数据与私有制品不在公开仓库及公开镜像中。
- [ ] 新知识库只迁移必要、核实且适合其公开范围的内容。
- [ ] 授权同学能在新目录按说明复现实验。

## 15. 参考与后续执行注意事项

官方技术参考：

- [Spark 3.5.7 运行环境与入口](https://spark.apache.org/docs/3.5.7/)
- [Spark 3.5.7 Standalone 部署](https://spark.apache.org/docs/3.5.7/spark-standalone.html)
- [Spark 3.5.7 RDD 编程指南](https://spark.apache.org/docs/3.5.7/rdd-programming-guide.html)
- [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [pip Git 依赖支持](https://pip.pypa.io/en/stable/topics/vcs-support/)
- [Git submodule 说明](https://git-scm.com/docs/gitsubmodules)

本计划暂存于原项目根目录，迁入课程项目后可改名为 `IMPLEMENTATION_PLAN.md`。文中原项目相对路径是来源索引，不表示新仓库运行时必须依赖原项目目录。原环境配置附件仅作为运行环境参考，不将附件中的安装命令视为本次已执行操作。

后续开始实施时，先完成 M0 的环境能力检查和来源策略冻结，再按阶段执行。新建远程仓库、推送、公开发布、系统环境安装和原科研库接回都属于未来操作；本文件的生成不表示这些操作已经执行。
