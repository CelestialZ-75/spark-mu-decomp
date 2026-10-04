#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

backend="${1:-cpu}"
if [[ "$backend" != cpu && "$backend" != gpu ]]; then
    echo 'Usage: bash scripts/bootstrap_wsl.sh [cpu|gpu]' >&2
    exit 2
fi
if [[ "$(uname -m)" != x86_64 ]]; then
    echo 'This environment is locked for Linux x86_64.' >&2
    exit 2
fi

# Keep the bootstrap and its caches local; do not change shell profiles.
uv_version=0.11.19
uv_dir="$PWD/.tools/uv-$uv_version"
mkdir -p "$uv_dir"
if [[ ! -x "$uv_dir/uv" ]]; then
    archive=uv-x86_64-unknown-linux-gnu.tar.gz
    release="https://github.com/astral-sh/uv/releases/download/$uv_version"
    curl -fsSL --connect-timeout 10 --max-time 120 --retry 2 --retry-all-errors \
        "$release/$archive" -o "$uv_dir/$archive"
    curl -fsSL --connect-timeout 10 --max-time 30 --retry 2 --retry-all-errors \
        "$release/$archive.sha256" -o "$uv_dir/$archive.sha256"
    (cd "$uv_dir" && sha256sum -c "$archive.sha256")
    tar -xzf "$uv_dir/$archive" -C "$uv_dir" --strip-components=1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-wsl"
"$uv_dir/uv" sync --locked --extra "$backend" --extra plots
echo "Environment ready: $UV_PROJECT_ENVIRONMENT"
if ! command -v java >/dev/null; then
    echo 'Spark also needs Java 17: sudo apt-get install openjdk-17-jre-headless'
fi
