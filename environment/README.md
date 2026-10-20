# WSL 与 Docker 运行说明

WSL 用于开发和已有 CUDA 参考，Docker 用于单机模拟受限 CPU 集群。
下列 Docker 命令目前只运行环境算例，不代表完整 MU 分解或资源对照已完成。
命令在 Ubuntu 的项目根目录执行，首次安装耗时不计入实验。

## WSL 环境

复用现有 Ubuntu 24.04 / WSL2。依赖安装到项目环境，不修改全局 Python。
正式计时尽量使用 Linux 文件系统中的项目副本，减少挂载盘读写影响；迁移后重新创建环境。

```bash
# 仅在缺少 Java 时执行
sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless
java -version

bash scripts/bootstrap_wsl.sh cpu
.venv-wsl/bin/python scripts/environment_smoke.py --backend cpu
PYSPARK_PYTHON="$PWD/.venv-wsl/bin/python" \
    .venv-wsl/bin/python scripts/environment_smoke.py --backend spark

# 需要 GPU 时切换安装选项
bash scripts/bootstrap_wsl.sh gpu
.venv-wsl/bin/python scripts/environment_smoke.py --backend gpu
```

CPU/GPU extra 二选一。WSL 使用 Windows NVIDIA 驱动，不另装 Linux 显卡驱动。
`nvidia-smi` 能看到显卡后，还要运行 GPU 算例确认实际 CUDA 计算。
bootstrap 使用项目 uv 0.11.19 和现有锁文件，后续按运行需要调整依赖即可。

gpu extra 安装 CUDA 版 Torch；同一个环境仍可以选择 `--backend cpu` 或 `spark`。
完整 MU 实验使用 `--backend gpu`，实际设备和显存记录在 `parameters.json`。
计时前后等待 CUDA 完成，不将初次下载和依赖安装时间计入实验。

## 当前 Docker 双 worker 配置

安装并启动 Docker Desktop，在设置中启用 WSL2 engine 与 Ubuntu integration。
确认 Ubuntu 中 `docker version` 能连接服务，`docker compose version` 可以运行。

```bash
mkdir -p data outputs
docker compose -f environment/compose.yaml config --quiet
docker compose -f environment/compose.yaml build
docker compose -f environment/compose.yaml up -d master worker-1 worker-2
# master UI 显示两个 worker 后执行
docker compose -f environment/compose.yaml run --rm --use-aliases driver
docker compose -f environment/compose.yaml down
```

`--use-aliases` 让 executor 能通过 `driver` 地址回连临时容器。
Master UI 在 <http://localhost:8080>，worker UI 在 8081/8082。
当前 driver 默认执行环境求和算例。完整 MU 入口已经实现，但容器还需要只在 driver
挂载私有算法目录，并传入 `--core-dir` 和容器内的数据路径，再通过 `scripts/submit.py` 运行。
公共镜像不包含私有算法，这一步尚未实际验证。

查看输出是否出现 `worker-1`、`worker-2` 两个 host。若只有一个参与，先检查 UI 注册状态，
再重跑；环境脚本在这种情况下会报错。课堂演示保留 UI 截图或任务输出即可，无需建设日志平台。

当前两个 worker 各使用 1 core，应用总 core 为 2，executor heap 为 512 MiB，driver heap 为 1 GiB。
这些是 Spark 调度和堆内存设置，不是容器硬限额；Python 等进程还会占用额外内存。
当前 Compose 没有 cpus/mem_limit，因此还没有实现受限资源实验。
两个 worker 都位于同一台电脑。

数据只读挂载到 `/workspace/data`，输出挂载到 `/workspace/outputs`。
容器用户 UID 为 10001，遇到输出权限问题时调整该目录属主或权限。
Docker 镜像目前只有公共 CPU 依赖，实际构建和双 worker 运行尚未验证。
GPU 先在 WSL 中比较，GPU 容器不列为必做项。

## 算法接入

先让 driver 通过本地适配使用已有私有算法，worker 运行公开矩阵函数。
无需先建立私有制品库或镜像发布流程。若容器确实需要私有代码，再通过本地挂载或安装接入。
当前 Docker 构建上下文不包含私有源码、wheel、数据或实验输出。
原算法源码、真实数据和访问凭据继续留在本地；课程仓库当前为 private。

## 下一阶段资源配置（待实现）

先给容器设置实际 CPU/内存额度，再匹配 Spark cores 与 executor memory。
新增同 CPU 镜像的普通 CPU 基线，原算法只挂载到需要运行它的 driver/基线容器。
当前文件尚无这些配置，不提供未验证的运行命令。

分别测试每 worker 固定额度增加节点、固定 worker 总预算重新分配。
driver/master 额度同时记录；未固定整套预算时，不写成固定系统总资源。
先压缩 CPU，内存以能够稳定运行为起点，具体额度经过试运行再定。

每组记录配置、worker 参与、结果和分项耗时，配置间顺序运行。
节点数量和数据分区分别调整。单机共享主机不代表真实服务器网络，首版不加网络限速。
GPU 仅保留 WSL MU 参考，预测全部 CPU。见 [课程计划](../SPARK_MU_COURSE_PROJECT_PLAN.md)。
