# 当前矩阵算例说明

本文描述已经实现的代码，方便查看和运行。它不再作为后续算法接入必须遵守的接口。
完整 MU 分解尚未接入；后续优先保留原算法，以几个函数或小类封装 Spark 算子。

## 算例做了什么

输入 X 的形状是 `[features, samples]`，当前代码提供：

| 方法 | 计算 |
|---|---|
| `prepare(X)` | 保存输入并转换成所选浮点类型 |
| `correlation()` | `X @ X.T / N` |
| `project(a)` | `a @ X`，输出按时间顺序排列 |
| `weighted_sum(w)` | `X @ w` |
| `synchronize()` | GPU 等待计算完成；Spark 方法已等待 action |
| `close()` | 释放保存的数据，不停止外部 SparkSession |

CPU/GPU 使用 Torch，Spark 缓存连续时间块，用广播传递向量，用 treeReduce 合并块贡献。
相关矩阵只在全局求和后除以 N。投影按块起点排序后拼接。

当前接口都返回 NumPy 数组，因此 GPU 每次返回会发生设备到 CPU 的传输。
接入 MU 迭代时应重新处理这一点，尽量沿用原算法的设备张量，避免来回传输。
现有协议、工厂和验证层可以随代码简化，不要求继续扩展。

## 运行已有检查

```bash
.venv-wsl/bin/mu-decomp kernel-check --backend cpu
.venv-wsl/bin/mu-decomp kernel-check --backend spark --partitions 4
.venv-wsl/bin/pytest -q
```

检查比较三个算子与 NumPy 直接计算的结果，也覆盖不均匀分块等情况。
当前 float64 容差为 rtol=1e-8、atol=1e-10，float32 为 rtol=1e-4、atol=1e-5。
`--backend gpu` 需要安装 gpu extra 并有实际可用的 CUDA，不会自动改用 CPU。

CLI 还保留 `doctor`、JSON 输出和运行信息检查，这些是已有功能，后续不必复制到实验入口。
任务记录按块保存，记录数不等于 worker 数；local 模式也不等于双 worker 容器运行。

下一步先跑出原算法的 CPU 结果，再逐个替换算子。矩阵检查通过不能替代完整 MU 实验。
