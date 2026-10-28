# 精选实验结果说明

本目录供课程团队随私有仓库查看报告依据。来源为 `formal-20261027-204319`，对应 2026 年 10 月 27 日在 VMware Ubuntu 完成的正式批次。报告正文位于 [COURSE_REPORT.md](../COURSE_REPORT.md)，报告图片位于同级 `../figures/`。

## 筛选规则

两份正式 CSV 包含全部 12 组、36 条正式记录，没有只保留较快或误差较低的运行。每组实际运行四次：run-0 预热、run-1/2/3 正式统计。`mu-correctness.json` 记录全部 24 次 MU 运行，包含预热，因此行数不同于正式 MU 统计的 18 条。

原始结果示例统一固定选 run-1：MU 保留 CPU 与 two-p2；预测保留 cluster2 的 RMS＋Ridge、MU＋GBT、MU＋Ridge。它们用于重算指标、查看脉冲和重新绘图，不代表其余运行缺失或被排除。

## 文件对应关系

| 本目录内容 | 原始批次位置 | 用途 |
|---|---|---|
| [formal_runs.csv](formal_runs.csv) | 同名文件 | 全部 36 条正式耗时与误差记录 |
| [formal_summary.csv](formal_summary.csv) | 同名文件 | 12 组均值与样本标准差，预热排除 |
| [mu-correctness.json](mu-correctness.json) | 同名文件 | 全部 MU 脉冲配对、PNR、SIL 与偏差 |
| `mu-cpu-run-1/` | `mu-cpu/run-1/` | CPU 参数与 MU 原始 NPZ |
| `mu-spark-run-1/` | `mu-two-p2/run-1/` | 双 worker 两分区的参数与 MU 原始 NPZ |
| `prediction-run-1/rms-ridge/` | `ml-cluster2-rms-ridge/run-1/` | 测试预测 CSV 和 metrics.json |
| `prediction-run-1/mu-gbt/` | `ml-cluster2-mu-gbt/run-1/` | 同上 |
| `prediction-run-1/mu-ridge/` | `ml-cluster2-mu-ridge/run-1/` | 同上 |
| `feature-metadata/rms.json`、`mu.json` | `features-rms/metadata.json`、`features-mu/metadata.json` | 通道、窗口、划分、特征维数、准备时间和输入哈希 |
| `resource-config/` | `evidence/`、`config/` 和各组 `limits.txt` | 基础配置、硬限额、单 worker 集中预算覆盖及实际限额 |
| `environment/` | `evidence/` 中同名文件 | VM CPU、内核、运行前内存、Docker 和 Compose 版本 |
| [image-id.txt](image-id.txt) | 同名文件 | 运行镜像 ID，仅作追溯，不是可直接下载的公开镜像地址 |
| [source-input-sha256.txt](source-input-sha256.txt) | `input-sha256.txt` | 原始批次冻结时的输入、配置和脚本哈希 |
| [SOURCE_FILES.json](SOURCE_FILES.json) | 整理时生成 | 每份复制文件的原始相对路径、来源 SHA256 和交付副本 SHA256 |

`mu-spark-run-1` 是便于阅读的目录名，原配置严格对应 `mu-two-p2`。各次 JSON 的容器路径、host、计时和指标保留原值，没有为适配交付目录改写。

## 如何读取与复核

CSV 和 JSON 可以直接查看。统计每组 total 的均值与样本标准差时，用 formal_runs.csv 中 run-1/2/3，样本标准差分母为 n−1。预测 CSV 中 `label_g` 和 `prediction_g` 用于重算 RMSE 与 MAE。

MU NPZ 的 `pulses` 是拼接脉冲索引，`offsets` 为切分位置；另有 `pnrs`、`sils`、`ipts` 和 `separation_vectors`。先匹配脉冲，再比较 MU；不要默认相同行号对应同一个运动单位。索引相对扩展矩阵，绘图加回 10 个裁剪样本并除以 2048，得到所选两秒片段内的时间。

现有绘图入口可在仓库根目录读取本次精选 MU 文件；使用已安装的项目环境，不会重新分解：

```bash
.venv-wsl/bin/python -m mu_course.plot_results docs/report/results/mu-cpu-run-1
.venv-wsl/bin/python -m mu_course.plot_results docs/report/results/mu-spark-run-1
```

绘图会在示例目录额外生成 `mu_pulses.png`；该新图不属于当前校验清单。专用批次检查脚本按原 `outputs` 结构读取，不能直接把本精选目录当成完整批次运行。全部原始运行与模型训练复现见 [统一指南](../../START_HERE.md#formal-comparison)。

## 两种哈希记录的区别

`source-input-sha256.txt` 原样保留原始冻结路径，其中包含本精选目录没有复制的特征 NPZ、模板及脚本。它用于追溯原批次，不能直接在本目录执行整份校验。

`../SHA256SUMS.txt` 则覆盖当前交付目录中的报告、图片、精选结果和本说明，路径相对 `docs/report/`。在 Ubuntu 仓库根目录执行以下命令，可校验这些交付文件；校验清单不包含自身：

```bash
cd docs/report
sha256sum -c SHA256SUMS.txt
```

文件发生有意修改后，应重新生成交付校验清单。Markdown 报告在复制后更新了结果链接与交付说明，因此其来源哈希和交付副本哈希不同；数值与图表来源保持不变。

## 保留在本地的完整结果

完整批次仍位于 `outputs/vm-results/formal-20261027-204319/`，原报告生成副本仍在 `outputs/course-report/`。没有删除或移动这些文件。

本次不复制全部模型目录、重复 NPZ、日志、特征矩阵与模板，也不复制其他真实试次、环境和缓存。完整模型载入、模板重应用或检查全部运行时，需要原始批次或按统一指南重新运行。仓库原有两份授权输入数据仍位于 `src/mu_course/data/`。

本目录是用户选定用于提交的交付副本，创建目录和文件不等于已经执行 git add、commit 或 push。
