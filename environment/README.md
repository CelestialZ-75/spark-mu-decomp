# WSL 与 Docker 环境

所有命令均在 Ubuntu 项目根目录执行。环境安装与实验分开；初次拉取较慢不计入性能。

## WSL 原生环境

现有 Ubuntu 24.04 / WSL2 可复用，不重装、不修改全局 Python。正式实验应在 Linux
文件系统建立 Git 工作副本，再重新生成环境；不能直接拷贝 Windows 或旧路径的 venv。

```bash
# 系统 Java 尚缺时，终端里输入自己的 sudo 密码；不要把密码发到聊天。
sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless
java -version

# 校验 uv 下载 SHA-256，依赖只写入 .tools、.cache、.venv-wsl
bash scripts/bootstrap_wsl.sh cpu
.venv-wsl/bin/python scripts/environment_smoke.py --backend cpu
PYSPARK_PYTHON="$PWD/.venv-wsl/bin/python" \
    .venv-wsl/bin/python scripts/environment_smoke.py --backend spark

# 替换为同版本 CUDA wheel；不会同时装 cpu/gpu 两个 extra
bash scripts/bootstrap_wsl.sh gpu
.venv-wsl/bin/python scripts/environment_smoke.py --backend gpu
```

WSL 中沿用 Windows NVIDIA 驱动，不安装 Linux 显卡驱动。`nvidia-smi` 的 CUDA Version
是驱动支持能力，不代表已安装同版本 Toolkit，也不决定必须选择同号 Torch wheel。
Torch CUDA 12.8 wheel 自带所需用户态运行库；当前 GPU 是否可计算以实际 smoke 为准。

uv 固定为 0.11.19；依赖更新时使用项目 `.tools/uv-0.11.19/uv` 更新锁文件并重跑验证。
bootstrap 使用 `--locked`，不会静默升级依赖。

## Docker Desktop 路线

先安装并启动 Docker Desktop for Windows，在设置中启用 WSL2 engine 和 Ubuntu
integration。不要同时在 Ubuntu 内另装一套 Docker Engine。
随后确认 Ubuntu 中 `docker version` 同时显示 client/server，`docker compose version` 正常。

```bash
mkdir -p data outputs
docker compose -f environment/compose.yaml config --quiet
docker compose -f environment/compose.yaml build
docker compose -f environment/compose.yaml up -d master worker-1 worker-2
# master UI 应显示两个 alive worker，再执行：
docker compose -f environment/compose.yaml run --rm --use-aliases driver
# 同容器 CPU 环境基线
docker compose -f environment/compose.yaml run --rm --no-deps driver \
    python scripts/environment_smoke.py --backend cpu
docker compose -f environment/compose.yaml down
```

`--use-aliases` 使临时 driver 容器以 `driver` 解析，executor 才能回连。
Master UI 为 <http://localhost:8080>，worker UI 为 8081/8082，仅绑定本机。
RPC 7077 不发布到宿主；各节点在内部网络通信。

检查输出 `partitions` 中出现 `worker-1`、`worker-2` 两个 host。脚本严格检查两容器
参与，若刚启动时调度只落到一个 worker 会报错；检查 UI 注册状态后重跑，不降低验收标准。
将来正式任务另保存 event log 和 executor 信息。

每 worker 宣告 1 core、1 GiB executor 可分配内存；应用每 executor heap 512 MiB，
driver heap 1 GiB，总 core 2。它们不是容器总内存限额，Python/native/JVM 还有额外开销。
当前 Dockerfile 安装公共 CPU 依赖，未含私有核心和 GPU 支持。
base image 目前固定版本系列而非 digest；验收后记录实际 image ID/digest。

数据只读挂到所有节点 `/workspace/data`，输出挂到 `/workspace/outputs`。
运行用户 UID 10001；正式写文件前需为输出目录设置相应属主或 ACL，不能用全局 777 解决。
当前环境 smoke 仅写标准输出。GPU 容器作为后续一致环境验收项，参考
[Compose GPU 配置](https://docs.docker.com/compose/how-tos/gpu-support/)。

## 私有核心接入条件

已有示例目录尚不能作为安装包引用。先独立打包、回归、冻结版本，之后再安装：
开发环境使用授权本地包，镜像构建使用私有制品和校验哈希。
当前 Docker 构建上下文采用白名单，不会自动包含 data、outputs、私有 wheel 或源码。
不要把 token 放入 build args、镜像层、公开依赖 URL 或 Compose 配置。
