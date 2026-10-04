# Spark MU course project

在小规模 HD-EMG 上实现单次 MU 分解内部的 Spark 并行，并对比 CPU、CUDA GPU
和 Spark 的结果与成本。小数据上 Spark 较慢也属于有效实验结果。

当前阶段：环境已完成 CPU/Spark local 验收；已实现统一矩阵后端和 CLI 检查入口。
私有核心包与完整 MU 分解尚未接入。

- [分阶段搭建与验收](SETUP_PLAN.md)
- [WSL / Docker 操作说明](environment/README.md)
- [统一后端接口与数值验收](docs/BACKEND_CONTRACT.md)
- [已核实状态](docs/knowledge-base/KNOWLEDGE.md)

原始完整课程计划保留在本地，不随公开仓库发布。

公共依赖由 `pyproject.toml` 和 `uv.lock` 管理。Python 3.11，Java 17，
PySpark 3.5.7；PyTorch 2.8.0 提供互斥的 CPU / CUDA 12.8 安装选项。
该 Torch 版本是环境起点，仍需私有核心回归验证后才能冻结为正式实验版本。

```bash
# Ubuntu / WSL2，项目根目录
bash scripts/bootstrap_wsl.sh cpu
.venv-wsl/bin/python scripts/environment_smoke.py --backend cpu
# Java 17 安装后
.venv-wsl/bin/python scripts/environment_smoke.py --backend spark
# 已安装课程包后的统一入口
.venv-wsl/bin/mu-decomp kernel-check --backend cpu
.venv-wsl/bin/mu-decomp kernel-check --backend spark --partitions 4
```

原科研示例目前是源码，尚非独立 Python 包。公开仓库不包含其源码或真实数据；
之后通过授权的私有 wheel 或固定 commit 包接入。环境检查只使用合成整数，
不代表分解数值一致性或科学有效性验收。
