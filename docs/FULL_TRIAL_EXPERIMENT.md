# 完整试次 CKC 对照

本实验用第一试次的全部 63582 个采样点，采样率 2048 Hz，时间为 `[0, 31.0458984375)` 秒；输入为 60 个共同有效通道。扩展因子 10、候选上限 5、迭代 30、cost 1、SIL 0.8、seed 42。与原 3–5 秒实验保持算法参数一致，仅修改输入时间范围。

## WSL 单次探索

在项目根目录，先按 [运行说明](../README.md) 准备环境。两次运行顺序执行，保留各自原始结果：

```bash
.venv-wsl/bin/python -m mu_course --backend cpu --threads 2 \
  --start 0 --duration 31.0458984375 --output outputs/full-trial/cpu
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_SUBMIT_ARGS='--driver-memory 4g pyspark-shell' \
  .venv-wsl/bin/python -m mu_course --backend spark \
  --master 'local[2]' --partitions 2 --threads 1 \
  --start 0 --duration 31.0458984375 --output outputs/full-trial/spark
```

WSL 本地测试不等同于队友的 VMware/Docker 限额实验；CPU 线程数和 Spark worker 线程布局也不同。单次结果仅用于探索和排查，不应与原有三次均值混合。

`parameters.json` 的 total_seconds 从信号扩展前计到 fit 完成，包含 Spark 启动和分发，不包含加载文件、写结果和清理；fit_seconds 包含计算和 Spark 多次任务执行，不代表纯通信时间。比较时记录实际资源配置、进程墙钟时间和内存。保存的 `mu_results.npz` 包含各 MU 的脉冲、SIL 与 PNR；结果数相同不代表脉冲一定相同，需要按放电时刻配对。

## 原有 Docker 受限配置

按 [实操指南第 9 节](START_HERE.md#formal-comparison) 建立新的正式批次和输入证据。勿重用已有批次。脚本默认仍是原 3–5 秒，设置环境变量可指定本次完整试次：

```bash
MU_START=0 MU_DURATION=31.0458984375 bash scripts/run_formal_group.sh "$FORMAL_DIR" mu cpu
MU_START=0 MU_DURATION=31.0458984375 bash scripts/run_formal_group.sh "$FORMAL_DIR" mu two-p2
```

每组运行一次预热和三次正式重复。可用 `MU_N_MU_MAX=20` 改变候选次数，默认为 5；CPU 和 Spark 必须使用同一数值。参数记录在各次 `parameters.json`，脚本和资源配置需随批次存档。优先完成这两组，确认内存够用、结果匹配后再增加节点或分区对照。完整输入会增加 driver、Spark 序列化与缓存占用，原 2 秒限额未必足够；失败时保留日志和实际限额，不把失败或缩小输入记作正式完成。

CKC 当前最多寻找的候选数由 `--n-mu-max` 控制，完整片段不保证得到更多 MU。长输入也不保证 Spark 加速，需要以结果为准。

## 2026-11-12 本机试运行

在 WSL2、Python 3.11.15、Torch 2.8.0+cu128、PySpark 3.5.7 下顺序执行 CPU 与 Spark local[2]；两者均使用 CPU。系统有约 15 GiB 内存，未设置 Docker 硬限额。CPU Torch 使用 2 线程；Spark driver 和每个 worker 的数值线程为 1，使用 2 个分区。每组仅一次，并非正式重复统计。

| 完整试次 | MU 数 | 算法总时间 | fit 时间 | 进程墙钟 |
|---|---:|---:|---:|---:|
| CPU | 4 | 11.940 s | 11.419 s | 64.98 s |
| Spark local，4 GiB Java 堆 | 4 | 197.099 s | 169.905 s | 260.48 s |

同次 Spark 启动 16.196 秒、数据分发 8.740 秒；Spark 算法总时间约为 CPU 的 16.51 倍。两组 CPU 线程与 Spark 进程分工不同，且没有同额度容器隔离，不能作严格公平的集群加速比，也不能与 VMware/Docker 的 2 秒三次均值直接相减或相除。墙钟包含程序启动、加载与保存，口径不同于算法总时间。

CPU/Spark 各 4 个 MU；按 ±1 样本一对一配对，四组脉冲 F1 分别为 1.000、0.981、0.999、0.992。部分脉冲数量不同，所以不能描述为逐脉冲完全一致。CPU SIL 约 0.956–0.988、PNR 约 21.36–28.00 dB；这不证明真实 MU 数量或生理准确性。

Spark 默认 1 GiB Java 堆首次在数据分发时报 `OutOfMemoryError: Java heap space`，原始日志保留。4 GiB 重试跑通；`time -v` 只显示被计时 Python 进程的峰值 RSS，不能把它当作包括 JVM 和 worker 的 Spark 总内存。此次没有单独测量通信或逐阶段净计算开销。

原始日志、结果与配对明细位于本机忽略目录 `outputs/full-trial-20261112/`；正式报告中的 2 秒 Docker 结果保持原样。

同一 WSL 环境后续补跑 3–5 秒片段，CPU 4 个 MU、total 1.697 秒；Spark 3 个 MU、total 160.913 秒。三个配对 MU 的脉冲均在 ±1 样本内匹配，CPU 另有一个未配对 MU。短片段 Spark/CPU 为 94.84 倍，完整试次为 16.51 倍，长输入下两者差距在本次测量中相对缩小，但 Spark 仍较慢。两组各一次，且输入从单个平台改变为包含静息和多力平台的完整轨迹；不能把差异单独归因于长度，更不能视作稳定加速规律。两组 MU 数不完全一致，性能结论应连同正确性限制一起报告。

完整试次再将 `n_mu_max` 从 5 改为 20，CPU 得到 9 个 MU，total 40.849 秒、fit 40.095 秒；输出数组均为有限值，9 个 MU 的 SIL 为 0.956–0.989、PNR 为 21.36–29.06 dB。按 ±1 样本配对，原来 4 条中有 3 条与新结果逐脉冲一致，另一条与候选 20 的结果没有高重合匹配；新结果不是原结果的简单超集。数量和质量指标不代表已验证生理真值，也不代表固定预测模板已准备完成。候选 20 尚未运行 Spark，不能拿它与候选 5 的 Spark 耗时做性能比较。
