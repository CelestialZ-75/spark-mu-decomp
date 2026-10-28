# Linux 与 Docker 环境说明

更新：2026-10-28。当前路线为 Windows 编辑 → VMware Ubuntu 运行 → Docker 模拟受限集群。VM 为 2 vCPU／6 GB；实际进度见 [知识库](../docs/knowledge-base/KNOWLEDGE.md)。SSH、代理、安装、权限、实验和回传命令统一维护在 [实操指南](../docs/START_HERE.md)，本页说明配置与运行边界。

## Python 与 Java

需要 Linux x86_64、Python 3.11、Java 17，锁定依赖包括 Torch 2.8.0、PySpark 3.5.7。`scripts/bootstrap_wsl.sh cpu` 是普通 Linux 安装流程，VMware 也可使用；`.venv-wsl` 只是沿用的目录名。项目 uv、缓存和 Python 环境均位于仓库内，不修改系统 Python。

首次安装和当前 shell 的 Java／PySpark 变量见 [实操指南第 2 节](../docs/START_HERE.md#step-2)，以后重连见 [第 12 节](../docs/START_HERE.md#vm-resume)。安装和下载不计入正式实验时间。

WSL2 仍可用于 Linux 本地开发和历史 CUDA MU 参考。CPU／GPU extra 互斥；CUDA 版环境也可选择 CPU／Spark 后端。VMware 当前路线使用 CPU，不能因为 Windows 有显卡就假定 Ubuntu VM 有 CUDA。

## Compose 文件的职责

- `compose.yaml` 定义 master、两个 worker 和 client driver；单独使用时没有容器硬限额，driver 默认只运行环境算例。
- `compose.limits.yaml` 叠加 CPU／内存硬限额，并提供独立 `cpu` 服务。
- `scripts/run_vm_stage.sh` 自动叠加两份文件，执行单次环境、真实 MU 和预测验收；正式重复使用 `scripts/run_formal_group.sh`，各次独立保存。

这两份配置及真实任务已在 Ubuntu 运行，原始结果已回传核对。复现时仍需验收自己的环境，环境算例通过不能代替真实 MU／预测检查。

| 服务 | CPU 上限 | 内存硬上限 | 执行配置 |
|---|---:|---:|---|
| master | 0.25 | 384 MiB | daemon heap 128 MiB |
| worker-1 | 0.5 | 1280 MiB | 一个任务槽，executor heap 512 MiB |
| worker-2 | 0.5 | 1280 MiB | 同上 |
| driver | 0.75 | 1792 MiB | 阶段／正式脚本提交时 heap 768 MiB |
| CPU 基线 | 2 | 4736 MiB | 集群停止后独立运行，Torch 两线程 |

集群与 CPU 基线总上限均为 2 CPU／4736 MiB，服务划分和线程布局不同。`memswap_limit` 与内存上限相同，禁止容器额外使用 swap。基础 Compose 默认 driver heap 为 1 GiB，实际阶段脚本会覆盖为表中 768 MiB；不要把 Spark 堆内存或任务槽当成容器硬限额。

正式 MU 还比较 `one-half`、`one-full`、`two-p1/p2/p4`；其中单 worker 集中预算配置在批次 `config/one-full.yaml` 中生成。完整资源表、控制变量及脚本命令见 [第 9 节](../docs/START_HERE.md#formal-comparison)。

## 网络、挂载与输出

Docker Engine 运行在 Ubuntu VM 内，复用已有安装；Windows 无需 Docker Desktop。WSL 用户选择 Docker Desktop 时，才需要相应 WSL integration。

master／worker／driver 通过内部网络通信。Master UI 绑定 Ubuntu `127.0.0.1:8080`，两个 worker UI 的主机端口为 8081／8082；Windows 查看页面使用 [SSH 转发](../docs/START_HERE.md#vm-ui)。两个容器共享单台电脑，不代表两台物理服务器，也没有 HDFS 分布式存储。

镜像内已安装 CKC 和两份授权数据，无需外部算法挂载。`data/` 只读挂载到 `/workspace/data`，供自定义输入；`outputs/` 挂载到 `/workspace/outputs`，保留各次结果。容器用户为 UID 10001，目录权限使用 [6.1 的 ACL 方法](../docs/START_HERE.md#step-6)。

Ubuntu shell、Docker daemon、构建容器的代理是不同配置层，具体排错见 [基础镜像下载代理](../docs/START_HERE.md#docker-pull-proxy)。构建成功后代码修改需重新构建镜像；仅编辑文档或宿主机结果检查脚本不需要重建。

完整结果、环境和缓存默认不入 Git；用户选定用于提交的报告、配图与精选结果见 [交付目录](../docs/report/results/RESULTS_GUIDE.md)。含科研源码／数据的镜像供当前私有课程项目运行，公开发布需另行核对许可。
