# 项目知识与实际进度

## 2026-11-12 cascade 对照

**VMware 待办**：粗粒度 cascade 目前仅在 WSL 单机 `local[3]` 加独立合并进程完成真实数据探索，尚未在 VMware/Docker 的资源受限多 worker 环境运行或验收。当前入口使用 Spark local 模式，既有 Compose 也只有两个 worker，不能直接当作三 worker 的 cascade 集群结果；需先完成 standalone 调度、三 worker / 合并进程资源配置与日志核验，再执行基线和重复资源对照。仓库只有已授权的两个试次，本机三试次结果使用的第三试次在 Git 忽略目录，队友克隆后不能直接复现；不得擅自将第三试次加入仓库，应先按数据授权准备输入。完整原始实验输出也未提交。

已接入逐 MU 六特征（CFR、FR、dCFR、CVISI、TSLS、ACTIVE）与原有 MLlib GBT/Ridge。52 个模板生成 312 列；单次 GBT、Ridge 的 RMSE 为 91.84、93.23 g。池包含 session0_2 测试试次肌电且投影逐试次适配，这仅是探索性回归，不能称作严格独立测试；运行命令、缺失值处理及结果见 [说明](../CASCADE_COMPARISON.md)。

已新增三试次、三个力等级的 CPU 顺序和 Spark 粗粒度流水实验，参数与运行范围见 [cascade 对照说明](../CASCADE_COMPARISON.md)。正式 `40/45/5` 单次对照已完成：CPU 424.50 秒、Spark 210.02 秒，最终各 52 个模板；三级模板、PNR 和谱系数组一致，实际存在流水重叠。两组资源并行度不同，尚未进行重复试验或逐脉冲验收；Spark 停止时临时目录清理报错。第三试次只在本机忽略目录供本次实验使用，不能视为已授权加入仓库。

核对日期：2026-10-28。2026-10-25 已补齐独立运行所需的 CKC 与两份授权 EMG/力数据。旧环境/性能记录仍标记为历史验证，2026-10-26 以来的实现、验收和文档整理合并见会话记录。

## 项目定位与阅读入口

单机用 Docker 模拟资源受限的 Spark 集群，研究 MU 分解与 CPU 指力预测的计算/通信开销。数据少是实验规模设定，不据此放弃资源实验；也不声称已验证超大数据或真实多服务器性能。

- 范围与步骤：[课程计划](../../SPARK_MU_COURSE_PROJECT_PLAN.md)。
- 当前结构：[架构](ARCHITECTURE.md)。
- 风格、接口、团队工作包：[开发说明](../DEVELOPMENT.md)。
- 运行方法：[README](../../README.md)、[环境说明](../../environment/README.md)、[统一实操指南](../START_HERE.md)。
- 设计沿革：[决策](DECISIONS.md)。原型时期的后端协议已删除，不再照此实现。

代码优先简约、易读、快速可运行。参数集中，少量普通函数和必要小类，复用原算法及 Spark 调度。ML 全部 CPU，GPU 仅为已有 MU 优化参考。

## 当前代码进度

| 内容 | 状态 | 证据与边界 |
|---|---|---|
| 内置算法与同步示例数据 | 已补齐 | src/mu_course/ckc 和 data；两试次各约 14 MB，60 个共同有效通道 |
| 默认 CPU、数据读取和绘图 | 本轮 VMware 已验证 | 历史内置例子记录 3 个；本轮默认 CPU 得到 4 个，数据图与原始结果已检查 |
| 独立安装包内容 | 本次已验证 | 干净 wheel 含五个 CKC 模块和两份 NPZ；临时目录中从 wheel 解包导入并完成 CPU 分解 |
| WSL、依赖、CPU/Spark 环境算例 | 已验证（历史） | Python 3.11.15、Java 17，local[2] 两个 Python worker 进程，不是两个主机 |
| MU 入口、输入选择、保存、绘图 | 已实现、已验证（历史） | run_experiment.py、plot_results.py 与本地真实片段 |
| 三个 Spark 算子及原循环接入 | 本轮 VMware local 已验证 | CPU 与 Spark 1/2/4 分区均为 4 个 MU，配对脉冲最大偏差 1 样本 |
| 实际 CUDA MU | 已验证（历史） | 原 Torch 循环，实际设备与同步计时，无自写 CUDA kernel |
| 预热、重复计时、CSV | 本轮 local 与受限容器已验证 | 本地 16 次／12 条正式；受限正式批次 48 次／36 条，不混合统计 |
| Docker CPU 镜像与双 worker 配置 | 已有真实运行与文件证据 | 环境算例、真实 MU 和预测输入任务记录通过核对 |
| 容器硬限额、同总额度 CPU 基线 | 已完成受限正式对照 | 2 CPU／4736 MiB；服务划分不同，inspect 与 cgroup 可追溯 |
| 预测特征、跨试次模板、MLlib 模型 | 已实现并有真实结果 | 固定训练模板；RMS／MU 对齐；Ridge／GBT 模型、预测、指标与图齐全 |
| 受限资源对照、预测图与报告 | 对照完成，报告与精选结果已整理 | 正式 12 组、36 条记录；报告、七张配图及精选证据位于 docs/report，待复核与提交 |

## 历史验证结果

2026-10-05 的真实片段：0831 LDX 中立位中指梯形试次，emg_clean，2048 Hz，adapter 0，第 3 秒开始取 2 秒，63 个有效通道，扩展 10，5 候选、30 迭代、cost 1、SIL 0.8。

- CPU 与 Spark local[2] 各得到 4 个 MU，对应脉冲在 ±1 样本内一致。
- CPU/GPU 各预热一次、记录三次，均为 4 个 MU，配对脉冲在 ±1 样本内一致。
- CPU/GPU 平均总计算耗时约 0.887/0.595 秒，GPU 峰值 Torch 分配显存约 40.7 MiB。这是该本机小样本结果，不是当前资源实验结论。
- 当时 9 项测试通过，含真实 CUDA，无 CUDA 跳过。新数据改为两个试次共同有效的 60 通道，不能直接复用旧 63 通道的 MU 数量和耗时。

历史环境为 WSL2 Ubuntu 24.04.3，约 15 GiB RAM，28 逻辑 CPU，RTX 4070 Laptop 8 GB，Torch 2.8.0+cu128。当前环境可用性仍以实际运行检查为准。

命令与原始输出位置见 [.private/LOCAL_EXPERIMENT.md](../../.private/LOCAL_EXPERIMENT.md)。输出目录忽略，不保证其他机器具备这些文件。

## 接入注意事项

- 原算法来自 FingerForce_Decode/example/gpu_MU_extract；只复制必需五模块并修复包内导入，见 src/mu_course/ckc/SOURCE.md。默认不依赖研究仓库，外部版本覆盖为可选参数。
- MAT 用 Stage 3 emg_clean，不用 emg_env 代替；旧 0528 文件缺字段，历史验证改用 0831 数据。
- driver 仍保留完整输入与全局运算，不能声称已支持超单机内存。
- 总计算计时不含文件 I/O、保存和清理；同输出目录重跑覆盖结果。
- 当前预测使用 session0_1 训练、session0_2 测试，中指屈曲力 g；窗口 410、步长 102 样本，两端排除 430 样本。固定训练模板与 400 ms 居中 Hanning 平滑，限定离线预测。
- 当前仓库为 private。用户授权的两个精简真实试次纳入 src/mu_course/data，本次选定的报告、配图与精选结果纳入 docs/report；其他真实数据、环境、完整输出和凭据仍排除。未来公开须另审文件与历史。

## 下一步

实验与正式对照已完成并回传，容器已关闭。[报告初稿](../report/COURSE_REPORT.md)、七张配图和 [精选结果](../report/results/RESULTS_GUIDE.md) 已整理到 `docs/report/`，准备随私有仓库提交；完整 outputs 和最初报告副本保留。对照 [统一指南第 10 节](../START_HERE.md#course-report) 复核、补充作者信息，再由用户暂存、提交和推送。复制交付材料时保留整个 docs/report。成功实验无需重跑；当前未执行本次 git add、commit／push。Transformer 与区间预测仍为可选扩展。


## 会话记录

### 2026-11-12 — 单试次 n_mu_max=100 对照（用户停止）

按用户要求，在 WSL 对第一试次全部 63582 样本顺序执行 CPU 和 Spark local[2]，n_mu_max=100、n_iter=30，其余参数与前次完整试次一致。CPU 使用 2 个 Torch 线程；Spark 使用两个分区、单线程数值计算和 4 GiB Java 堆，未使用相同 Docker 资源限额。每组一次，无预热重复。

CPU 已完成：去重前 173 条、去重后 20 MU；total 235.911 秒、fit 235.534 秒，进程墙钟 288.01 秒；输出有限，SIL 0.946–0.998。用户要求停止后，已终止本次 Spark Python 和 JVM，确认工作进程退出，并删除周期检查。Spark 未完成，没有最终结果、算法总耗时或配对结论；终止时日志中的错误不能当成完整实验结果。结果及日志保存在忽略的 `outputs/n100-single-20261112/`。未修改算法，未提交或推送。

### 2026-11-12 — 完整试次 CKC 探索

读取队友 PPT 与 START_HERE.md。新增正式脚本的 `MU_START` / `MU_DURATION` / `MU_N_MU_MAX` 可选环境变量，默认仍为原来的 3 秒 / 2 秒 / 5 次候选；用新的完整试次运行目录，未覆盖正式实验。说明见 `docs/FULL_TRIAL_EXPERIMENT.md`。

本机 WSL 的 session0_1 全部 63582 样本、60 通道、2048 Hz，扩展 10、候选迭代上限 5、迭代 30、SIL 0.8、seed 42：CPU 单次得到 4 个 MU，total 约 11.94 秒，fit 约 11.42 秒。`/usr/bin/time` 的完整进程墙钟约 64.98 秒，Python 进程峰值 RSS 约 841 MiB。该单次 WSL 结果不等于原 VMware/Docker 受限正式对照。

Spark local[2] 首次默认 1 GiB JVM 堆在分发时 `OutOfMemoryError: Java heap space`，已终止卡住的本次进程并保留日志。4 GiB 堆重试成功，得到 4 个 MU，total 197.099 秒、fit 169.905 秒、启动 16.196 秒、分发 8.740 秒，进程墙钟 260.48 秒。与 CPU 以 ±1 样本配对，四个脉冲序列 F1 为 1.000、0.981、0.999、0.992，不是完全一致。相同 WSL 下 3–5 秒补充基线：CPU 4 MU、1.697 秒；Spark 3 MU、160.913 秒，三个配对 MU 的脉冲一致，CPU 另有一条未配对。两组线程配置不同，未设置相同容器限额，均为单次探索。当前没有本机 Docker 命令；不得把 local 测试称为受限双 worker 集群结果。原始文件在忽略的 `outputs/full-trial-20261112/`。

完整试次 CPU 候选次数从 5 增至 20，得到 9 MU、total 40.849 秒、fit 40.095 秒，SIL 0.956–0.989、PNR 21.36–29.06 dB；输出均为有限值。原 4 条中有 3 条在候选 20 结果中逐脉冲一致，另一条没有高重合匹配，不能把后者当作原结果的简单超集。候选 20 未运行 Spark，也未输出或验收预测所需固定模板。

### 2026-10-26～2026-10-28 — 受限实验、预测与正式对照完成

补齐 compose.limits.yaml、run_vm_stage.sh、prepare_prediction.py、predict_force.py、predict_submit.py 和 runtime_info.py。CPU 基线与默认集群总上限均为 2 CPU／4736 MiB，服务分配和线程布局分别记录。固定第一试次训练、第二试次测试；保存完整训练 MU 模板、阈值、方向与时间参数，apply_templates 新增可选固定极性参数，旧调用保持原行为。RMS／MU 共享窗口和标签；静默 MU 保留列，平滑限定离线使用。

正式实验与检查代码保存为 run_formal_group.sh、check_mu_results.py、check_prediction_results.py、check_formal_mu.py、summarize_formal.py，提供单次验收、预热、重复运行、配对检查、指标重算和汇总绘图。Ubuntu 用户截图确认 Ruff 通过、pytest 为 17 passed、1 skipped、1 warning；本机早期 Windows MLlib worker 崩溃不作为 Ubuntu 验收结果。最初文档和合成输出检查不代表真实实验，后续以传回原始文件核对补齐证据。

<a id="current-validation"></a>

本轮正式批次为 `outputs/vm-results/formal-20261027-204319`。用户在 Ubuntu 运行并传回 Windows，助手读取原始文件核对，没有重新运行正式实验：12 组都有 SUCCESS，共 48 次运行；排除每组一次预热后为 36 条正式记录，耗时均值与样本标准差和原始 JSON 一致。12 项输入快照、配置和正式脚本哈希全部匹配，两份数据 SHA256 与特征元数据一致。

24 次 MU 运行（含预热）均为 4 个 MU，与 CPU run-1 配对后脉冲最大偏差 1 样本，最大 PNR 差约 0.003266 dB、最大 SIL 差约 0.000200。预测 CSV 重算 RMSE／MAE 与记录一致，模型目录和曲线存在，cluster2 输入记录包含两个 worker。RMS 为 60 维，MU 预测为 1 维，训练／测试为 611／612 窗口；预测模板列数与短片段分解得到的 4 个 MU 不混写。

RMS＋Ridge、MU＋GBT、MU＋Ridge 的测试 RMSE／MAE 分别约为 97.606／73.683、253.316／207.204、289.594／245.535 g；训练均值基线 RMSE 约 383.853 g。三次正式总计算均值 ± 样本标准差：MU CPU 0.6268 ± 0.0367 秒，双 worker 两分区 76.6370 ± 1.4979 秒；RMS＋Ridge local 6.9781 ± 0.4297 秒，集群 25.6673 ± 0.1174 秒。完整统计保留在 formal_summary.csv；本轮集群未优于 CPU／local，应据实分析小输入、调度、通信及资源划分。

脉冲匹配只证明实现一致性；input_worker_hosts 只证明输入任务参与，未逐个重新载入全部真实模型，也未测得峰值内存余量。两个容器共享一台电脑，不代表真实多服务器或超单机内存能力；三次重复不是三份独立测试数据。

将 START、VMware 环境和 SSH 第 6–8 步三份指南合并为 START_HERE.md，并同步 README、计划、环境、架构、开发与代码来源说明。依据正式批次生成 Markdown 报告初稿、七张配图和三次原始耗时附表，数值与图片引用已核对。按用户选定范围，将报告、配图、两份正式 CSV、MU 正确性记录、固定 run-1 原始示例、资源／特征元数据及哈希证据整理到 docs/report，并附 RESULTS_GUIDE.md。完整 outputs 与最初报告副本保留，未执行本次暂存、提交或推送。

### 2026-10-25 — VMware 本地结果验收

用户复用 ubuntu-bigdata：Ubuntu 24.04.5，2 vCPU、6 GB、50 GB 虚拟磁盘、NAT。用户在 SSH 内完成安装，报告 Java 17、Python 3.11.15、Torch 2.8.0+cpu、PySpark 3.5.7；现有测试截图为 11 passed、1 skipped（CUDA），Ruff 通过。

助手读取 Windows 的 outputs/vm-results，核对两份单次结果和 16 份 benchmark 原始结果，以及 12 条正式 CSV。各次输入路径、[60,4096] 形状和算法参数相同。18 份结果全部为 4 个 MU，每 MU 脉冲数为 7、25、25、44；先按 ±1 样本脉冲重合评分作一对一匹配，再核对全部放电时刻，最大时差 1 样本（约 0.488 ms）。最低 SIL 为 0.935172，全部超过 0.8，最低 PNR 约 20.0785 dB；相对 CPU 最大 SIL 差约 0.000197、PNR 差约 0.00340 dB。所有保存数组均为有限值，索引和形状合法。四张 PNG 完整可读且非空白，未作人工图像内容判断。

正式三次平均 total_seconds：CPU 0.982 秒，Spark 1/2/4 分区为 32.266/33.934/57.971 秒，样本标准差分别为 0.046/0.451/0.419/5.525 秒。CSV 的每行与对应 parameters.json 一致，run-0 预热未进入正式统计。这些是当前小输入、单虚拟机 local 结果，不是受限 Docker 或多服务器性能结论。

本地验收记录保存为 outputs/vm-results/validation/CHECK_REPORT.md 和 audit.json（均在忽略的输出目录，不保证其他机器有这些文件）。此次只检查传回文件，没有重跑算法、启动 VM／Docker。虚拟机输入文件未单独导出哈希，仅核对所报输入路径、形状和参数。历史 3 个 MU 的差异原因仍未复现；本轮实现一致性通过不等于生理真实 MU 总数已知。可进入 Docker 环境算例，Docker／硬限额／预测仍待完成。未提交或推送。

### 2026-10-25 — 独立运行补齐

用户明确授权 session0_1、session0_2 的真实 EMG/力进入当前私有课程仓库。导出共同有效的 60 通道，保留各约 31 秒完整轨迹，2048 Hz、两路同步力，float32 压缩，记录源文件和导出 SHA256。原文件未修改。

复制五个必要 CKC 模块并适配包内导入；默认算法和数据随包提供。补 dataset.load_trial、数据绘图和维护者导出脚本，README 不再要求占位路径。12 项测试全部通过，含实际 CUDA 数值检查；默认 CPU 和 Spark local[2] 各得到 3 个 MU，对应脉冲在 ±1 样本内一致，数据图和脉冲图已检查。wheel 构建通过，并在仓库外临时目录从解包后的模块读取数据、完成 CPU 分解，依赖复用现有环境，不是全新机器安装验收。Ruff、排版、锁文件一致性和文档链接检查通过。结果在忽略的 outputs/standalone/。Docker 实际运行仍未验证。

离线隔离构建因缺少缓存 setuptools 失败，复用项目环境构建成功。首次发现旧 build/ 带入已删除框架文件，已移到忽略的 outputs/standalone/build-before-clean-package 后干净重建。未修改原研究仓库，未提交或推送。

### 2026-10-20 — 文档重组

核对现有源码和配置，重写课程计划、架构与开发说明，区分已实现、历史已验证和待办。记录受限集群前提、CPU ML 范围及简约风格。10 份当前文档的 34 个本地链接与代码围栏检查通过。本次未修改计算代码，未重跑历史实验。

### 2026-10-10 — 私有仓库内容

用户允许将课程代码、计划和本地说明提交推送。历史最后记录为 4e70fe4；外部研究仓库不属于本仓库提交范围。

### 2026-10-05 — 最小实现与 CUDA

删除早期协议/工厂/后端层，接通真实 CPU/Spark local，随后验证原 CUDA 实现。详细结果见上表与本地实验说明。更早的通用矩阵接口测试数量不作为当前测试进度。
