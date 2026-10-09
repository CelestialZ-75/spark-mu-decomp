# 本地真实数据实验

数据：D:/meta-robotics/Code/FingerForce_Decode/data/processed_0831LDX/session0_1/aligned_Middle_Trapezoid.mat
算法：D:/meta-robotics/Code/FingerForce_Decode/example/gpu_MU_extract
依据：原项目知识库的 Stage 3 emg_clean 输入约定、同步时间平台期，以及 configs/v3_1_cross_angle_0831ldx.yaml。
旧 0528 文件没有 emg_clean，不拿 emg_env 代替。

WSL 运行：

```bash
.venv-wsl/bin/python -m mu_course \
  --core-dir /mnt/d/meta-robotics/Code/FingerForce_Decode/example/gpu_MU_extract \
  --data /mnt/d/meta-robotics/Code/FingerForce_Decode/data/processed_0831LDX/session0_1/aligned_Middle_Trapezoid.mat \
  --backend cpu --fsamp 2048 --start 3 --duration 2 --adapter 0 \
  --ext-factor 10 --n-mu-max 5 --n-iter 30 --cost-fcn 1 --sil-threshold 0.8 \
  --output outputs/minimal/cpu
```

Spark 使用 --backend spark --partitions 2，并换输出目录。完整平台可用 start=3,duration=6，后续正式对照可先试 5 秒。
CUDA 阶段已验证：当前环境为 Torch 2.8.0+cu128，CUDA 12.8，RTX 4070 Laptop GPU。同一环境可运行 cpu/gpu/spark。CPU/GPU 三次对照结果在 outputs/cuda-stage/comparison/；GPU 三次均得到 4 个 MU，与 CPU 的对应脉冲在 ±1 个采样点内一致。
本地私有 ckc_decomposer.py 的 fit 已加入可选 ops 参数，未提交到私有仓库。
