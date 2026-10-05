# Spark MU 课程实验

基于已有 MU 分解算法，在一小段肌电数据上比较 CPU、GPU 和 Spark 的结果与耗时。
主要工作是将少量矩阵算子改成 Spark 分块计算，保留原算法主循环，并提供容易修改的实验参数。
小数据下 Spark 较慢也属于有效结果。

目前 WSL CPU、Spark local 和三个矩阵算子的检查已通过。完整 MU 算法尚未接入，
真实 CUDA 与 Docker 双 worker 运行尚未验证。现有通用后端是早期算例，后续会按实验需要简化。

- [搭建顺序](SETUP_PLAN.md)
- [WSL / Docker 运行说明](environment/README.md)
- [当前矩阵算例说明](docs/BACKEND_CONTRACT.md)
- [项目进度](docs/knowledge-base/KNOWLEDGE.md)

依赖使用 Python 3.11、Java 17、PySpark 3.5.7 和 PyTorch 2.8.0，
由 `pyproject.toml`、`uv.lock` 管理。CPU 与 CUDA 安装选项二选一。

```bash
# Ubuntu / WSL2，项目根目录
bash scripts/bootstrap_wsl.sh cpu
.venv-wsl/bin/python scripts/environment_smoke.py --backend cpu
# 需要 Java 17
.venv-wsl/bin/python scripts/environment_smoke.py --backend spark

# 当前矩阵算例，不是完整 MU 分解
.venv-wsl/bin/mu-decomp kernel-check --backend cpu
.venv-wsl/bin/mu-decomp kernel-check --backend spark --partitions 4
```

下一步先通过少量本地适配接入原算法，CPU 跑出结果后再接 GPU 和 Spark。
私有算法、真实数据和本地实验输出不放入公开仓库。完整计划保留在本地。
