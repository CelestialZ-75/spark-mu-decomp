# 统一矩阵后端 v0.1

本接口处理已经扩展的 `X[features, samples]`，不实现 CKC 候选选择、迭代更新、
质量门控、延迟扩展或脉冲索引映射。将来的私有 engine 调用本协议；当前 CLI
只验收通用矩阵计算，不能视为 MU 分解结果。

| 操作 | 输入 / 输出 | 语义 |
|---|---|---|
| `prepare(X)` | 非空二维实浮点数组 | 拷贝输入，按选定 float32/64 转换；拒绝 NaN/Inf |
| `shape` | `(features, samples)` | 未 prepare 或 close 后访问报错 |
| `correlation()` | `[features, features]` | `X @ X.T / N`，只按全局 N 归一化 |
| `project(a)` | `[features]` → `[samples]` | `a @ X`，按原始时间顺序返回 |
| `weighted_sum(w)` | `[samples]` → `[features]` | `X @ w`，不执行局部归一化 |
| `synchronize()` | 无 | CUDA 等待设备；Spark 方法已等待 action 完成 |
| `close()` | 无 | 释放本后端数据，可重复调用，不停止外部 SparkSession |

输出统一为 driver 上的 NumPy 数组。GPU 返回时包含回传成本；后续性能报告必须
说明这一边界，不将此实现的耗时直接解释为纯 GPU kernel 时间。大矩阵及全局 weights
仍由 driver 持有，这是首版小样本的内存边界。

## Spark 路径

1. 连续时间块包含所有 feature 行，分块保存起点；限制 `1 <= partitions <= N`。
2. RDD 使用 MEMORY_AND_DISK；实际 action 物化缓存并核对 Python/Torch/NumPy/课程包版本。
3. 相关矩阵与加权求和使用块内 Torch CPU 计算和 treeReduce；任务各用 1 个 Torch 线程。
4. 广播向量或权重；完成 action 后销毁广播；重新 prepare 或 close 时释放旧缓存。
5. 一维投影依据全局起点排序并校验覆盖，拒绝重叠、缺块和非有限计算结果。

诊断记录是一条/块，包含 partition、host 和 Python PID，不能把记录条数当 worker 数。
local 两进程检查不等于两台 worker 主机验收。Standalone 仍需镜像和真实任务验证。

## 运行与验收

```bash
bash scripts/bootstrap_wsl.sh cpu
.venv-wsl/bin/mu-decomp doctor --backend cpu
.venv-wsl/bin/mu-decomp kernel-check --backend cpu
.venv-wsl/bin/mu-decomp kernel-check --backend spark --partitions 4 \
    --output outputs/checks/spark-p4.json
.venv-wsl/bin/pytest -q
```

JSON 输出标记 `linear_algebra_only` 和 `ckc_engine: not_integrated`，输出文件已存在时拒绝覆盖。
`--backend gpu` 必须安装 gpu extra 且通过实际 CUDA 检查，不会回退到 CPU。
float64 单步容差为 rtol=1e-8、atol=1e-10；float32 为 1e-4、1e-5。

使用 spark-submit 时通过 `scripts/submit.py` 进入相同 CLI，并给 CLI 同样的
`--master` 值；Standalone 使用 client 模式，所有节点提前安装同一个课程包。

```bash
# 容器网络内，构建相同镜像后使用；尚未完成 Docker 运行验收
spark-submit --master spark://master:7077 --deploy-mode client \
    --conf spark.driver.host=driver --conf spark.cores.max=2 \
    --conf spark.executor.cores=1 scripts/submit.py kernel-check \
    --backend spark --master spark://master:7077 --partitions 2 --require-two-hosts
```

目前只验证 generic matrix primitives。接下来先完成私有核心包与原参考回归，再接入
全局 CKC 控制器、统一结果文件、真实平台数据和性能协议。
