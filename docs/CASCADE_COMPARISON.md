# 三试次 cascade 对照

比较 CPU 顺序执行与 Spark 任务并行执行。两组调用相同的科研 CKC 和质量筛选：10%、20%、30% 平台分别取原始时间 `[3,9)`、`[13,19)`、`[23,29)` 秒；Track A 的候选次数 40、每候选迭代上限 45，Track B 细化 5 次，扩展因子 10，SIL > 0.8，PNR > 15 dB。质量规则包括 PNR 优先的模板去重、谱系去重、源试次脉冲一致性去重和静息段筛选。

每一级由三个试次的 Track A 提取新模板。20%、30% 的 Track B 必须等上一级池合并完成，再分别对三个试次细化。Spark 版允许上一级合并与下一级 Track A 同时运行；依赖上一级池的 Track B 不会提前运行。

当前运行采用 WSL `local[3]` 和单独的合并进程，模拟三个任务槽加一个汇总 worker；它们共享一台机器和文件系统，不是四台服务器或四个 Spark worker，也没有容器 CPU／内存硬限额。CPU 和 Spark 的每个数值进程均使用一个线程，但 Spark 可同时运行三个任务及合并，因此结果是不同资源分配下的单次探索，不应称为等资源公平加速比。两组顺序运行，避免互相竞争。

**VMware 尚待实现和验收**：目前 cascade 入口使用 `local[N]`，不能连接已有的 Spark standalone master；现有 Compose 仅配置两个 worker。需补充三 worker 的调度、独立合并进程及资源限制，并在 VMware 上核对实际进程位置、资源限额、参数和结果。不得将现有 WSL 单机数字称为 VMware 资源限额实验。此前的 VMware 结果属于旧版细粒度算子，不是本 cascade 的运行证据。

项目只随仓库提供两个已授权试次。本机三试次实验额外从科研项目加载第三试次，仅输出到被 Git 忽略的 `outputs/`，不将其加入仓库。三个输入仅保留共同有效通道，因而与科研项目完整电极通道配置不必然相同。三个试次仅用于分解和性能研究，不能将其中任一试次同时作为完全独立的预测测试集。

在 WSL 中、仓库根目录运行：

```bash
.venv-wsl/bin/python scripts/prepare_cascade_trials.py \
  --third-mat /path/to/session0_3/aligned_Middle_Trapezoid.mat \
  --output outputs/my-cascade/inputs
bash scripts/run_cascade_comparison.sh outputs/my-cascade
```

`status.txt` 表示当前阶段或失败；`cpu.log`、`spark.log` 保存原始日志，`parameters.json` 记录参数，`pool_0.npz` 至 `pool_2.npz` 保存各级模板及 PNR，`events.json` 保存已完成任务的起止时间和进程号，`summary.json` 保存总耗时和最终模板数。运行中不要修改算法文件。计时结果和模板数应结合质量、脉冲一致性和资源条件分析。

## 2026-11-12 本机单次结果

WSL 单机，三个试次共有 60 通道，CPU 和 Spark 参数一致。CPU 顺序总耗时 424.50 秒；Spark `local[3]` 与独立合并进程总耗时 210.02 秒，前者约为后者的 2.02 倍。结果是不同并行资源分配下的单次观察，尚未重复，也不是等资源容器或多服务器加速比。

三个等级的池分别包含 17、33、52 个模板；两组每一级模板、PNR 和谱系数组逐元素完全一致，最终 PNR 为 15.30–29.53 dB。该结果未单独验收逐脉冲一致性或生理真值。任务时间记录证实前两次合并均与下一级 Track A 存在时间重叠，Track B 均在前级合并结束后开始。Spark 运行结束时日志出现临时目录清理异常；结果文件和完成状态已写出，但不能将清理描述为完全正常。原始结果位于 `outputs/cascade-20261112/`。

## 逐 MU 扩展特征及回归探索

`prepare_cascade_prediction` 使用已生成的模板池，对每条 MU 提取科研项目 E5 的六个特征：CFR、FR、dCFR、CVISI、TSLS、ACTIVE，按 MU 顺序排列。窗口使用课程项目原有的 200 ms 窗和 50 ms 步长，CFR 为 400 ms 居中 Hanning，属于离线特征。CVISI 在过去一秒内不足四次放电时未定义，按同一列训练集的中位数填充；训练列完全缺失时填 0。输出中记录每列填充值和缺失数量。

```bash
.venv-wsl/bin/python -m mu_course.prepare_cascade_prediction \
  --pool outputs/cascade-20261112/spark/pool_2.npz \
  --pool-parameters outputs/cascade-20261112/spark/parameters.json \
  --output outputs/my-cascade-prediction/features
.venv-wsl/bin/python -m mu_course.predict_force \
  --features outputs/my-cascade-prediction/features --model gbt \
  --master 'local[2]' --output outputs/my-cascade-prediction/gbt
.venv-wsl/bin/python -m mu_course.predict_force \
  --features outputs/my-cascade-prediction/features --model ridge \
  --master 'local[2]' --output outputs/my-cascade-prediction/ridge
```

示例依赖本机先前产生的模板池；仓库本身不包含第三试次或此实验输出。投影复用科研版 `refine_filters(n_iter=0)`：每个试次分别计算协方差与放电检测阈值，因此是无监督试次适配，并非固定阈值的跨试次投影。

本机单次探索：session0_1 的 611 个窗口训练、session0_2 的 612 个窗口测试。52 MU 产生 312 列，两试次所有 MU 都有脉冲，各列在两试次均有变化。CVISI 原始缺失值训练 13501 个、测试 16578 个。原样使用 MLlib GBT 与 Ridge，RMSE/MAE 分别为 91.84/65.34 g、93.23/68.09 g；训练均值基线 RMSE 383.85 g。已用预测 CSV 重算指标。输出位于 `outputs/cascade-prediction-20261112/`。

**评估边界**：该 52 MU 池由 session0_1、session0_2、session0_3 的肌电共同分解而来，包含测试试次的无标签肌电；加上逐试次投影适配，上述分数不能作为严格独立测试的泛化结果，也不宜与旧版只用训练数据拟合模板的结果直接比较。若要做严格测试，需要仅用训练来源重新建立模板池、固定测试方案后重跑。

绘制本批次耗时点图、回归曲线和误差图：

```bash
.venv-wsl/bin/python scripts/plot_cascade_results.py
```

图位于 `outputs/cascade-visualization-20261112/`。脚本读取已有结果并核对预测曲线的标签与误差，不会重新运行实验；图中标注了资源差异与测试试次参与模板构建的限制。
