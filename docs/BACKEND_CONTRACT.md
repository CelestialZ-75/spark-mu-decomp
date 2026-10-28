# Spark 算子与原算法接入

当前只有一个小型 `SparkOps`，实现相关矩阵、投影和加权求和。
原 CKC 主循环继续负责求逆、活动索引、候选选择、迭代和质量判断。

| 方法 | 计算 |
|---|---|
| `SparkOps(session, X, partitions)` | 按时间切块，缓存并执行一次 action |
| `correlation()` | 各块计算 `X_i @ X_i.T`，求和后除以总 N |
| `project(a)` | 广播 a，各块计算 `a @ X_i`，按起点拼接 |
| `weighted_sum(w)` | 各块取对应权重，计算 `X_i @ w_i`，求和 |
| `close()` | 释放块缓存 |

worker 使用 NumPy 矩阵计算，输出转成 driver 上的 CPU Torch 张量。
CPU/GPU 沿用原算法，不经过这个转换，迭代张量保留在所选设备。
Spark driver 持有完整输入，小数据版本不处理超单机内存问题。

## 私有算法中的少量改动

课程入口默认使用仓库内 `mu_course.ckc`，`--core-dir` 仅供外部版本对照。为了共用主循环，
私有 `CKCDecomposer.fit` 增加可选 `ops=None` 参数，只调整以下位置：

- 相关矩阵：ops 为空时执行原计算，否则调用 `ops.correlation()`。
- 三处投影：先算 `a = c_tx @ inv_corr_sig`，再执行原投影或 `ops.project(a)`。
- 梯度加权求和：原路径保持原计算；Spark 路径将边缘之外的权重置零，再调用 `ops.weighted_sum()`。

未传 ops 时保留原 CPU/GPU 行为。Spark 不需要在 worker 安装私有算法。
这些改动现已随内置 CKC 提供，队友无需自行修改。显式使用外部版本时，该版本也需支持相同调用点。

## 张量约定

令 X 的形状为 [C, N]，C 为扩展通道数、N 为扩展后时间样本数。输入为 CPU Torch
张量；当前主入口使用 float32，数值测试另覆盖 float64。

- correlation() 返回 [C, C]，归一化使用全局 N，不取分块均值的简单平均。
- project(a) 的 a 为 [C]，返回 [N]，保持原始时间顺序。
- weighted_sum(w) 的 w 为 [N]，返回 [C]；边缘屏蔽由调用方处理。
- 分区数为 1 到 N，每块包含全部 C；分区数不是 worker 数。
- 调用方在 finally 中 close()，并关闭自己创建的 session。

算子不改变候选选择、阈值或 MU 质量规则。CPU/GPU 不通过 NumPy 绕行。
预测模型使用 MLlib 现成实现，不扩展这里为通用机器学习后端。跨试次模板应用的固定极性参数属于预测辅助接口，不改变这里的三个 Spark 算子约定；修改范围见 [SOURCE](../src/mu_course/ckc/SOURCE.md)。

## 检查

```bash
.venv-wsl/bin/pytest -q
```

数值检查覆盖 float32/64、1/2/4 分区、不均匀分块、全局归一化、
投影顺序和边缘权重屏蔽。另检查输入片段选择。
完整 MU 结果还要查看脉冲和质量指标，浮点归约顺序可能影响迭代与阈值判断。

环境算例仍由 `scripts/environment_smoke.py` 提供，旧 doctor/kernel-check 入口已移除。
