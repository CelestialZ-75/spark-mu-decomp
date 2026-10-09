# Spark MU 课程实验

用同一小段肌电数据比较 CPU、GPU 和 Spark 的 MU 分解结果与耗时。
沿用已有 CKC 主循环，CPU/GPU 使用原 PyTorch 计算，Spark 只替换三个矩阵算子。

## 运行一次实验

先按 [环境说明](environment/README.md) 安装依赖。私有算法目录通过参数提供，
需要含有 `ckc_decomposer.py`、`ckc_extender.py` 和 `ckc_preprocess.py`。
Spark 路径需要私有 `fit(eSig, ops=None)` 的少量算子接入改动，见
[算子说明](docs/BACKEND_CONTRACT.md)。本仓库目前设为 private，包含课程代码、计划和
本地实验说明；原算法仍通过外部目录引用，真实数据和实验输出不纳入 Git。

```bash
# Ubuntu / WSL，项目根目录；替换这两个本地路径
.venv-wsl/bin/python -m mu_course \
    --core-dir /path/to/private/mu-example \
    --data /path/to/aligned_trial.mat \
    --backend cpu --fsamp 2048 --start 3 --duration 2 --adapter 0 \
    --ext-factor 10 --n-mu-max 5 --n-iter 30 \
    --cost-fcn 1 --sil-threshold 0.8 --output outputs/cpu

# 查看 MU 脉冲图
.venv-wsl/bin/python -m mu_course.plot_results outputs/cpu
```

换成 `--backend gpu` 使用 CUDA；换成 `--backend spark --partitions 2` 使用 Spark local。
GPU 不可用时会报错。`python -m mu_course --help` 可以查看全部参数，默认值集中在
`src/mu_course/run_experiment.py` 的 `experiment_parser()` 中。

运行 GPU 前先执行 `bash scripts/bootstrap_wsl.sh gpu`。这个环境也能运行 CPU 和 Spark，
不需要在每次比较时切换安装选项。结果记录实际 device、显卡名称、Torch/CUDA 版本和
本次运行的 PyTorch 峰值分配显存。GPU 矩阵计算使用完整 float32 精度，沿用原算法中
部分在 CPU 上执行的峰值检测和质量判断。

MAT 输入直接读取已滤波的 `emg_clean`，按 `good_channels` 选择指定的 64 通道阵列，
不再次滤波，也不使用包络。NPY/NPZ 输入应已完成预处理，形状为 `[channels, samples]`，
NPZ 默认键为 `emg`。时间参数相对于输入记录起点；adapter 从 0 开始编号。

输出是 `parameters.json`、`mu_results.npz`。脉冲索引相对于扩展后的矩阵，绘图会补偿
原 extender 裁掉的起始样本。同一个输出目录重跑会覆盖结果，需要保留时换目录。

## 小规模对照

```bash
.venv-wsl/bin/python -m mu_course.benchmark \
    --core-dir /path/to/private/mu-example --data /path/to/aligned_trial.mat \
    --backends cpu gpu spark --repeat 3 --partition-counts 1 2 4 \
    --start 3 --duration 2 --output outputs/comparison
.venv-wsl/bin/python -m mu_course.plot_results outputs/comparison
```

每组先预热一次，再顺序运行三次。输出各次 MU 结果和 `timings.csv`，图中显示平均总计算耗时。
总计算耗时包含扩展、Spark 启动/分发和分解，不含读取输入、保存文件及清理资源。
CUDA 计时等待设备完成，Spark 算子等待 action。只运行已有环境时，可用 `--backends cpu spark`。

没有 MU 时会保留结果并提示检查输入和参数。小数据下 Spark 慢也是有效结果；
先确认分解结果，再解释计时差异。Docker 双 worker 的运行仍需单独验证。

当前已用真实两秒数据跑通 CPU 和 Spark local：各得到 4 个 MU，对应脉冲在
±1 个采样点内一致。CUDA 阶段也已完成：RTX 4070 Laptop GPU、Torch 2.8.0+cu128，
同样输入三次均得到 4 个 MU，与 CPU 脉冲在 ±1 个采样点内一致。9 项测试通过。
当前小样本 CPU/GPU 平均总计算耗时约 0.887/0.595 秒；这是本机这一组参数的阶段结果。
Docker 双 worker 和完整三后端性能对照仍待完成。

- [搭建顺序](SETUP_PLAN.md)
- [Spark 算子与原算法接入](docs/BACKEND_CONTRACT.md)
- [当前进度](docs/knowledge-base/KNOWLEDGE.md)
- [完整课程计划](SPARK_MU_COURSE_PROJECT_PLAN.md)
- [本地数据路径与运行命令](.private/LOCAL_EXPERIMENT.md)

`.private/` 中的检查脚本保留当前机器的路径，其他机器运行前需要调整。
长版计划备份仅供历史参考，当前实施以根目录简化计划为准。之后考虑部分开源时，
再单独审阅本地路径、科研来源说明和历史文档。
