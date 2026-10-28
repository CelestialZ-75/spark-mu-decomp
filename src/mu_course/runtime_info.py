"""Record effective Linux cgroup limits without assuming Docker is present."""

import platform
import socket
from pathlib import Path


def runtime_info():
    limits = {}
    for name in [
        "cpu.max",
        "memory.max",
        "memory.swap.max",
        "cpu/cpu.cfs_quota_us",
        "cpu/cpu.cfs_period_us",
        "memory/memory.limit_in_bytes",
    ]:
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            limits[name] = path.read_text().strip()
    return {"host": socket.gethostname(), "python": platform.python_version(), "cgroup": limits}
