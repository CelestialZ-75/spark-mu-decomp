# 内置 EMG 与同步力数据

数据来源：processed_0831LDX/session0_1、session0_2 的 aligned_Middle_Trapezoid.mat，LDX、0°、中指梯形任务。用户于 2026-10-25 明确授权将这两份精简数据放入当前私有课程仓库，供团队拉取运行。其他数据不在此次范围内。

导出不裁剪时间、不降采样、不再次滤波。两个试次取 adapter 0 的共同有效通道，转换 float32 并压缩为 NPZ，避免复制各约 140 MB 的完整 MAT。精确形状、通道、时长、源文件与导出文件 SHA256 见 metadata.json。

| 文件 | 内容 |
|---|---|
| ldx_0deg_1.npz | session0_1 的 EMG 与同步力 |
| ldx_0deg_2.npz | session0_2 的 EMG 与同步力 |
| metadata.json | 来源、选择规则、形状与校验值 |

NPZ 字段：

- emg：float32 [C,N]，来自 emg_clean；C 是两个试次共同有效通道数。
- force：float32 [2,N]，已同步的中指屈曲和伸展力，单位 g；优先 force_corrected，否则 force_aligned，实际字段写入 metadata.json。
- fsamp：2048 Hz；EMG 与力共享采样点和起点。
- channel_ids：原始物理通道的零基编号，与 emg 行顺序一致。原 MAT 的 good_channels 为一基编号。
- trial_id：session0_1 或 session0_2；force_unit：g。
- flexion_mvc_g：2220.267，来自原配置的中指屈曲 MVC。屈曲 %MVC = force[0] / flexion_mvc_g * 100，不据此归一化伸展力。

时间 0 是原 aligned 试次起点。数据保留完整轨迹，MU 快速例子只取第 3–5 秒。计划中的一训一测配对仍应在预测实验前确定，不因文件名顺序自动成为最终实验结论。

```python
from mu_course.dataset import DATA_DIR, load_trial

trial = load_trial(DATA_DIR / "ldx_0deg_1.npz")
emg, force = trial["emg"], trial["force"]
print(emg.shape, force.shape, trial["fsamp"].item())
```

正常运行无需导出脚本或原研究仓库。只有维护者重新生成例子时才使用：

```bash
.venv-wsl/bin/python scripts/prepare_example_data.py --source-dir /path/to/processed_0831LDX
```

这些是真实科研数据，不是合成信号。它们随当前私有仓库共享，不构成公开发布许可。公开前应另行审查。
