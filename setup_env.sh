#!/usr/bin/env bash
# =============================================================================
#  OpenHLM · 一键环境配置（仅 VLA 训练 + VLA 部署，openpi 侧）
#
#  适用环境
#    普通 Ubuntu 22.04 / 24.04 工作站（原生安装，不用 docker、不用预约显卡），
#    NVIDIA GPU：训练建议 >= 24 GB 显存，仅推理 >= 8 GB。
#    在「刚 clone 完仓库的新电脑」上，于仓库根目录执行：
#        bash setup_env.sh
#
#  不含：GR00T 的 C++ WBC / TensorRT 部署栈、遥操作（.venv_teleop）、HuMI 采集环境
#
#  脚本做了什么（每一步都可重复执行）
#    1. 预检：系统 / 网络 / 磁盘 / GPU / 内存
#    2. 安装 uv（若缺失，用官方脚本装到 ~/.local/bin）
#    3. 初始化 third_party 子模块（lerobot / dlimp / Muon，pyproject 里是 path 依赖）
#    4. uv sync 建立 src/openpi4OpenHLM/.venv（Python 3.11，torch 2.9 + jax 0.5.3）
#    5. 应用 transformers 补丁（src/openpi/models_pytorch/transformers_replace）
#    6. 补齐部署客户端依赖（pyzmq / rich 等，openpi-eval/main.py 需要）
#    7. 自检：torch / jax / cv2 / zmq / transformers / openpi / openpi_client，
#       以及部署客户端能否 import（openpi-eval/jaka_tabletop_env、sonic_g1_env、main）
#    8. 可选（--with-weights）：准备 pi05_base PyTorch 权重
#    9. 生成 env.sh（训练 / 部署要用的环境变量，source 即可）
#
#  常用用法
#    bash setup_env.sh                  # 装环境（不含预训练权重，最快）
#    bash setup_env.sh --with-weights   # 顺带下载 / 转换 pi05_base 权重（+20 GB 磁盘）
#    bash setup_env.sh --check          # 只自检，不安装
#    bash setup_env.sh --dry-run        # 只打印将要执行的命令
#    bash setup_env.sh --recreate       # 删掉 .venv 从零重建
#    bash setup_env.sh --help           # 全部参数
#
#  与官方 README 的对应关系
#    根 README「Environment Setup」+ openpi README「Installation / PyTorch Setup」
#    + doc/08-local-machine.md 里踩过的坑（pyarrow 冲突、transformers 补丁、权重转换）
# =============================================================================

# 被 `sh setup_env.sh` 调用时（dash 不支持 pipefail），自动转用 bash 重新执行
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -Eeuo pipefail

# ----------------------------------------------------------------------------
# 0. 路径与默认值
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
OPENPI_DIR="$REPO_ROOT/src/openpi4OpenHLM"
GROOT_DIR="$REPO_ROOT/src/GR00T-WholeBodyControl4OpenHLM"

PY_VERSION="3.11"
VENV_DIR="$OPENPI_DIR/.venv"

DRY_RUN=0
CHECK_ONLY=0
WITH_WEIGHTS=0
RECREATE=0
SKIP_SUBMODULES=0
SKIP_VERIFY=0
INSTALL_SYS_DEPS=0
WRITE_ENV_FILE=1
FORCE_CONVERT=0

PROXY="${http_proxy:-${HTTP_PROXY:-}}"
INDEX_URL=""
CHECKPOINT_DIR=""          # 本地 pi05_base（JAX）目录，留空则用默认缓存路径
DATA_DIR=""                # 数据集目录（HF_LEROBOT_HOME），默认 <repo>/data
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

# uv 行为：copy 链接模式可以避免 transformers 补丁污染 uv 全局缓存（官方 README 的 WARNING）
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

# ----------------------------------------------------------------------------
# 1. 输出与命令包装
# ----------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BLUE=$'\033[36m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_MAGENTA=$'\033[1;35m'; C_DIM=$'\033[2m'
else
  C_RESET=""; C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_MAGENTA=""; C_DIM=""
fi

log()     { printf '%s[openhlm]%s %s\n' "$C_BLUE" "$C_RESET" "$*"; }
ok()      { printf '%s[  ok  ]%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn()    { printf '%s[warn  ]%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
err()     { printf '%s[error ]%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }
section() { printf '\n%s==> %s%s\n' "$C_MAGENTA" "$*" "$C_RESET"; }
die()     { err "$*"; exit 1; }

# 141 = SIGPIPE（如 `--help | head`），不算错误，静默退出
trap 'rc=$?; if [[ $rc -ne 141 && $BASHPID -eq $$ ]]; then err "第 ${LINENO} 行执行失败（退出码 ${rc}），已中止。"; fi; exit $rc' ERR

# run <cmd...>：真正执行；--dry-run 时只打印
run() {
  printf '  %s$ %s%s\n' "$C_DIM" "$*" "$C_RESET"
  (( DRY_RUN )) && return 0
  "$@"
}

# run_in <dir> <cmd...>：在指定目录执行
run_in() {
  local dir="$1"; shift
  printf '  %s$ (cd %s && %s)%s\n' "$C_DIM" "${dir#"$REPO_ROOT"/}" "$*" "$C_RESET"
  (( DRY_RUN )) && return 0
  ( cd "$dir" && "$@" )
}

# run_sh <shell 字符串>：需要管道 / 重定向时用
run_sh() {
  local cmd="$1"
  printf '  %s$ %s%s\n' "$C_DIM" "$cmd" "$C_RESET"
  (( DRY_RUN )) && return 0
  bash -c "$cmd"
}

has() { command -v "$1" >/dev/null 2>&1; }

# ----------------------------------------------------------------------------
# 2. 参数解析
# ----------------------------------------------------------------------------
usage() {
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${BASH_SOURCE[0]}"
  cat <<'EOF'

可选参数
  --with-weights            准备 pi05_base PyTorch 权重（约 +20 GB 磁盘 / 需网络）
  --checkpoint-dir <dir>    pi05_base（JAX 格式）本地目录，配合 --with-weights
  --data-dir <dir>          数据集根目录（HF_LEROBOT_HOME），默认 <仓库>/data
  --force-convert           已有 PyTorch 权重也重新转换
  --recreate                删除 .venv 后从零重建
  --skip-submodules         跳过 third_party 子模块初始化
  --skip-verify             跳过最后的依赖自检
  --install-system-deps     用 apt 安装 opencv / 编译所需系统库（需要 sudo）
  --proxy <url>             下载走代理，如 http://127.0.0.1:7897
  --index-url <url>         PyPI 镜像，如 https://pypi.tuna.tsinghua.edu.cn/simple
  --python <ver>            Python 版本（默认 3.11）
  --venv <path>             虚拟环境路径（默认 src/openpi4OpenHLM/.venv）
  --no-env-file             不生成 env.sh
  --check                   只做预检 + 自检
  --dry-run                 只打印命令，不执行
  -h, --help                显示本帮助
EOF
}

while (( $# )); do
  case "$1" in
    --with-weights)        WITH_WEIGHTS=1 ;;
    --checkpoint-dir)      [[ $# -ge 2 ]] || die "--checkpoint-dir 需要参数"; CHECKPOINT_DIR="$2"; shift ;;
    --data-dir)            [[ $# -ge 2 ]] || die "--data-dir 需要参数"; DATA_DIR="$2"; shift ;;
    --force-convert)       FORCE_CONVERT=1 ;;
    --recreate)            RECREATE=1 ;;
    --skip-submodules)     SKIP_SUBMODULES=1 ;;
    --skip-verify)         SKIP_VERIFY=1 ;;
    --install-system-deps) INSTALL_SYS_DEPS=1 ;;
    --proxy)               [[ $# -ge 2 ]] || die "--proxy 需要参数"; PROXY="$2"; shift ;;
    --index-url)           [[ $# -ge 2 ]] || die "--index-url 需要参数"; INDEX_URL="$2"; shift ;;
    --python)              [[ $# -ge 2 ]] || die "--python 需要参数"; PY_VERSION="$2"; shift ;;
    --venv)                [[ $# -ge 2 ]] || die "--venv 需要参数"; VENV_DIR="$2"; shift ;;
    --no-env-file)         WRITE_ENV_FILE=0 ;;
    --check)               CHECK_ONLY=1 ;;
    --dry-run)             DRY_RUN=1 ;;
    -h|--help)             usage; exit 0 ;;
    *) die "未知参数：$1（--help 查看用法）" ;;
  esac
  shift
done

# 相对 --venv / --data-dir 按仓库根解析
[[ "$VENV_DIR" = /* ]] || VENV_DIR="$OPENPI_DIR/$VENV_DIR"
[[ -n "$DATA_DIR" ]] || DATA_DIR="$REPO_ROOT/data"
[[ "$DATA_DIR" = /* ]] || DATA_DIR="$REPO_ROOT/$DATA_DIR"

export GIT_LFS_SKIP_SMUDGE=1
if [[ -n "$PROXY" ]]; then
  export http_proxy="$PROXY" https_proxy="$PROXY" all_proxy="$PROXY"
  export HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY" ALL_PROXY="$PROXY"
fi
if [[ -n "$INDEX_URL" ]]; then
  export UV_DEFAULT_INDEX="$INDEX_URL" UV_INDEX_URL="$INDEX_URL"
fi

# ----------------------------------------------------------------------------
# 3. 预检
# ----------------------------------------------------------------------------
preflight() {
  section "[1/6] 预检"

  [[ -d "$OPENPI_DIR" ]] || die "找不到 $OPENPI_DIR，请在仓库根目录运行本脚本。"

  case "$(uname -s)" in
    Linux) ok "系统：$(uname -srm)（官方仅在 Ubuntu 22.04+ 验证）" ;;
    *)     die "仅支持 Linux（官方 README：Ubuntu 22.04，未支持其它系统）。" ;;
  esac

  local missing=()
  for c in git bash curl tar; do has "$c" || missing+=("$c"); done
  if (( ${#missing[@]} )); then
    die "缺少基础命令：${missing[*]}（Ubuntu: sudo apt install -y git curl ca-certificates）"
  fi
  ok "基础命令齐备：git $(git --version | awk '{print $3}') / curl $(curl --version | awk 'NR==1 {print $2}')"

  (( BASH_VERSINFO[0] >= 4 )) || die "需要 bash >= 4，当前 $BASH_VERSION"

  # 磁盘：venv 约 8-10 GB；带权重再 +20 GB
  local free_gb
  free_gb=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')
  if (( free_gb < 12 )); then
    warn "仓库所在分区仅剩 ${free_gb} GB，.venv 需要约 8-10 GB（带权重另需 20 GB）。"
  else
    ok "磁盘可用：${free_gb} GB"
  fi

  # 内存：官方转换脚本峰值 >28 GB，低内存版本约 13 GB
  local mem_gb
  mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  if (( mem_gb < 16 )); then
    warn "内存 ${mem_gb} GB：够训练（显存才是瓶颈），但转换 pi05_base 权重可能 OOM。"
  else
    ok "内存：${mem_gb} GB"
  fi

  # GPU
  if has nvidia-smi; then
    local gpu
    gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
            | awk 'NR <= 3 {printf "%s%s", sep, $0; sep = " ; "}' || true)
    if [[ -n "$gpu" ]]; then ok "GPU：$gpu"; else warn "nvidia-smi 无法读取 GPU 信息，检查驱动。"; fi
  else
    warn "未找到 nvidia-smi：训练 / 策略服务需要 NVIDIA 驱动，装好后再跑一次 --check。"
  fi

  # 网络（软检查，失败只提示）
  if has curl && ! curl -sSfI --max-time 10 https://astral.sh >/dev/null 2>&1; then
    warn "访问 astral.sh 失败：安装 uv / PyPI 可能不通，必要时加 --proxy 或 --index-url。"
  fi
}

system_deps() {
  (( INSTALL_SYS_DEPS )) || return 0
  section "[1.5/6] 系统依赖（apt）"
  has apt-get || die "--install-system-deps 需要 apt-get"
  run_sh "sudo apt-get update"
  run_sh "sudo apt-get install -y build-essential cmake ninja-build pkg-config \
            libgl1 libglib2.0-0 libgomp1 libsndfile1 ffmpeg"
}

# ----------------------------------------------------------------------------
# 4. uv
# ----------------------------------------------------------------------------
ensure_uv() {
  section "[2/6] 准备 uv"

  if has uv; then
    ok "已安装 uv $(uv --version | awk '{print $2}')"
    return 0
  fi

  log "未找到 uv，使用官方安装脚本（装到 ~/.local/bin）"
  run_sh "curl -LsSf https://astral.sh/uv/install.sh | sh"
  if (( ! DRY_RUN )); then
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    has uv || die "uv 安装后仍不在 PATH，请手动 export PATH=\"\$HOME/.local/bin:\$PATH\" 后重跑。"
    ok "uv $(uv --version | awk '{print $2}') @ $(command -v uv)"
  fi
}

# ----------------------------------------------------------------------------
# 5. 子模块
# ----------------------------------------------------------------------------
ensure_submodules() {
  section "[3/6] 初始化子模块（lerobot / dlimp / Muon）"

  if (( SKIP_SUBMODULES )); then
    warn "--skip-submodules：跳过子模块检查"
    return 0
  fi

  [[ -f "$REPO_ROOT/.gitmodules" ]] || die "缺少 .gitmodules：请先 git pull 到最新 dev 分支。"

  local pending=0
  while IFS= read -r line; do
    [[ "$line" == -* ]] && pending=1
  done < <(git -C "$REPO_ROOT" submodule status 2>/dev/null || true)

  if (( pending == 0 )); then
    ok "子模块已就绪"
    return 0
  fi

  log "发现未初始化子模块，开始拉取（lerobot 是 LFS 仓库，已设置 GIT_LFS_SKIP_SMUDGE=1）"
  run_in "$REPO_ROOT" git submodule update --init --recursive

  if (( ! DRY_RUN )); then
    local empty=()
    for m in third_party/lerobot third_party/dlimp third_party/Muon; do
      [[ -n "$(ls -A "$REPO_ROOT/$m" 2>/dev/null || true)" ]] || empty+=("$m")
    done
    (( ${#empty[@]} == 0 )) || die "子模块目录仍为空：${empty[*]}（检查网络 / 代理后重跑）。"
    ok "子模块就绪（pyproject 用 path 依赖引用这三个目录）"
  fi
}

# ----------------------------------------------------------------------------
# 6. openpi venv
# ----------------------------------------------------------------------------
venv_python() { printf '%s/bin/python' "$VENV_DIR"; }

venv_healthy() {
  [[ -x "$(venv_python)" ]] || return 1
  "$(venv_python)" -c 'import sys; assert sys.version_info[:2] >= (3, 10)' >/dev/null 2>&1
}

sync_openpi_env() {
  section "[4/6] 同步 openpi 依赖（uv sync）"

  local pyproject="$OPENPI_DIR/pyproject.toml"
  [[ -f "$pyproject" ]] || die "缺少 $pyproject"

  local stamp="$VENV_DIR/.openhlm-env-stamp"
  local want
  want="$( { sha256sum "$pyproject"; uv --version 2>/dev/null || true; } | sha256sum | cut -d' ' -f1 )"

  if (( RECREATE )); then
    [[ "$VENV_DIR" == "$OPENPI_DIR"/* ]] || die "--venv 指向仓库外，拒绝自动删除：$VENV_DIR"
    log "--recreate：删除旧的 $VENV_DIR"
    run rm -rf "$VENV_DIR"
  elif [[ -f "$stamp" ]] && [[ "$(cat "$stamp" 2>/dev/null)" == "$want" ]] && venv_healthy; then
    ok "依赖已同步（pyproject / uv 版本未变），跳过 uv sync"
    return 0
  fi

  # 官方 README：GIT_LFS_SKIP_SMUDGE=1 uv sync；GIT_LFS_SKIP_SMUDGE 已在前面 export
  run_in "$OPENPI_DIR" uv sync --python "$PY_VERSION"
  run_in "$OPENPI_DIR" uv pip install --python "$(venv_python)" -e .

  if (( ! DRY_RUN )); then
    venv_healthy || die "venv 创建失败：$VENV_DIR"
    printf '%s' "$want" > "$stamp"
    ok "venv 就绪：$VENV_DIR（Python $("$(venv_python)" -V 2>&1 | awk '{print $2}')）"
  fi
}

# ----------------------------------------------------------------------------
# 7. transformers 补丁
# ----------------------------------------------------------------------------
transformers_patch() {
  section "[5/6] 应用 transformers 补丁"

  local src="$OPENPI_DIR/src/openpi/models_pytorch/transformers_replace"
  [[ -d "$src" ]] || die "缺少补丁目录：$src"

  if (( DRY_RUN )); then
    printf '  %s$ cp -r %s/* <venv>/lib/python%s/site-packages/transformers/%s\n' \
      "$C_DIM" "${src#"$REPO_ROOT"/}" "$PY_VERSION" "$C_RESET"
    return 0
  fi

  local site="" d
  for d in "$VENV_DIR"/lib/python*/site-packages; do
    [[ -d "$d/transformers" ]] && site="$d"
  done
  [[ -n "$site" ]] || die "在 $VENV_DIR 里找不到 transformers，先完成 uv sync。"

  local stale=0 rel f
  while IFS= read -r -d '' f; do
    rel="${f#"$src"/}"
    [[ -f "$site/transformers/$rel" ]] || { stale=1; continue; }
    cmp -s "$f" "$site/transformers/$rel" || stale=1
  done < <(find "$src" -type f -print0)

  if (( stale == 0 )); then
    ok "transformers 补丁已是最新（AdaRMS / 精度 / KV cache）"
    return 0
  fi

  log "复制 transformers_replace 到 site-packages/transformers/"
  run cp -r "$src"/. "$site/transformers/"
  ok "transformers 补丁已应用"
}

# ----------------------------------------------------------------------------
# 8. 部署客户端依赖
# ----------------------------------------------------------------------------
deploy_extras() {
  section "[6/6] 补齐部署客户端依赖"

  # openpi-eval/main.py: cv2 numpy tqdm tyro openpi_client
  # jaka_tabletop_env.py: zmq scipy；sonic_g1_env.py: msgpack + gear_sonic(PYTHONPATH)
  local pkgs=("pyzmq" "rich")
  local missing=()

  if (( DRY_RUN )); then
    printf '  %s$ uv pip install --python %s %s%s\n' "$C_DIM" "$(venv_python)" "${pkgs[*]}" "$C_RESET"
    return 0
  fi

  local spec mod
  for spec in "${pkgs[@]}"; do
    mod="${spec//-/_}"
    "$(venv_python)" -c "import $mod" >/dev/null 2>&1 || missing+=("$spec")
  done

  if (( ${#missing[@]} )); then
    run_in "$OPENPI_DIR" uv pip install --python "$(venv_python)" "${missing[@]}"
    ok "已安装：${missing[*]}"
  else
    ok "部署依赖齐备（pyzmq / rich 已在 venv 中）"
  fi
}

# ----------------------------------------------------------------------------
# 9. 自检
# ----------------------------------------------------------------------------
verify_env() {
  section "[自检] 依赖与 GPU"
  (( SKIP_VERIFY )) && { warn "--skip-verify：跳过自检"; return 0; }
  (( DRY_RUN )) && { printf '  %s$ (venv python) import torch/jax/cv2/zmq/openpi_client ...%s\n' "$C_DIM" "$C_RESET"; return 0; }

  [[ -x "$(venv_python)" ]] || die "找不到 venv：$VENV_DIR
       环境还没装好，直接执行：bash setup_env.sh"

  local rc=0
  "$(venv_python)" - <<'PY' || rc=$?
import importlib
import sys

CORE = [
    ("torch", "torch"),
    ("jax", "jax"),
    ("flax", "flax"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("zmq", "pyzmq"),
    ("scipy", "scipy"),
    ("msgpack", "msgpack"),
    ("websockets", "websockets"),
    ("transformers", "transformers"),
    ("tyro", "tyro"),
    ("tqdm", "tqdm"),
    ("rich", "rich"),
    ("openpi", "openpi (editable)"),
    ("openpi_client", "openpi-client"),
]

print(f"  python        {sys.version.split()[0]}")
missing = []
for mod, label in CORE:
    try:
        importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001
        missing.append((label, exc))

for mod in ("torch", "jax", "numpy", "transformers"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<13} {getattr(m, '__version__', '?')}")
    except Exception:  # noqa: BLE001
        pass

try:
    import torch

    if torch.cuda.is_available():
        print(f"  cuda          {torch.version.cuda}  |  {torch.cuda.get_device_name(0)}")
    else:
        print("  cuda          x torch 看不到 GPU（装好驱动后可重跑 --check）")
except Exception:  # noqa: BLE001
    pass

try:
    import jax

    print(f"  jax devices   {[str(d) for d in jax.devices()]}")
except Exception as exc:  # noqa: BLE001
    print(f"  jax devices   x {exc}")

# transformers 补丁是否真的生效：pi0_pytorch 依赖替换后的 Gemma/PaliGemma 实现
try:
    importlib.import_module("openpi.models_pytorch.pi0_pytorch")
    print("  pi0_pytorch   OK 导入正常（transformers 补丁生效）")
except Exception as exc:  # noqa: BLE001
    print(f"  pi0_pytorch   ! 导入失败：{exc}")
    print("                -> 多为 transformers 补丁未生效，重跑 [5/6] 步（或直接重跑本脚本）")

if missing:
    print("\n  缺失依赖：")
    for label, exc in missing:
        print(f"    - {label}: {exc}")
    sys.exit(1)
print("\n  依赖自检通过（训练 + 部署）")
PY

  if (( rc != 0 )); then
    die "自检未通过（见上方列表）。可执行：uv pip install --python $(venv_python) <包名>"
  fi
}

# 部署客户端（openpi-eval）能否 import：需要 venv 依赖 + GR00T 仓库在 PYTHONPATH
verify_deploy_client() {
  section "[自检] 部署客户端（openpi-eval）"
  (( SKIP_VERIFY )) && { warn "--skip-verify：跳过"; return 0; }
  (( DRY_RUN )) && { printf '  %s$ (openpi-eval) import jaka_tabletop_env sonic_g1_env main%s\n' "$C_DIM" "$C_RESET"; return 0; }
  [[ -d "$GROOT_DIR/openpi-eval" ]] || { warn "找不到 $GROOT_DIR/openpi-eval，跳过部署客户端自检"; return 0; }

  if ( cd "$GROOT_DIR/openpi-eval" && PYTHONPATH="$GROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$(venv_python)" -c "import jaka_tabletop_env, sonic_g1_env, main" ); then
    ok "部署客户端可导入（jaka_tabletop_env / sonic_g1_env / main）"
  else
    die "部署客户端导入失败。常见原因：
       - 缺 pyzmq / scipy：uv pip install --python $(venv_python) pyzmq scipy
       - 未设置 PYTHONPATH：export PYTHONPATH=$GROOT_DIR"
  fi
}

# ----------------------------------------------------------------------------
# 10. 可选：pi05_base 权重
# ----------------------------------------------------------------------------
ensure_assets() {
  local jax_dir="$1" pt_dir="$2"
  (( DRY_RUN )) && return 0
  # doc/08 坑 4：上游转换脚本复制 assets 的路径有误，这里补一份 norm stats
  if [[ -d "$jax_dir/assets" ]] && [[ ! -d "$pt_dir/assets" ]]; then
    log "补拷 norm stats：$jax_dir/assets"
    cp -r "$jax_dir/assets" "$pt_dir/assets"
    ok "norm stats 已补齐"
  fi
}

prepare_weights() {
  (( WITH_WEIGHTS )) || return 0
  section "[可选] 准备 pi05_base 预训练权重"

  local ckpt_root="$OPENPI_DATA_HOME/openpi-assets/checkpoints"
  local jax_dir="${CHECKPOINT_DIR:-$ckpt_root/pi05_base}"
  local pt_dir="$ckpt_root/pi05_base_pytorch"

  if [[ -f "$pt_dir/model.safetensors" ]] && (( ! FORCE_CONVERT )); then
    ok "PyTorch 权重已存在：$pt_dir"
    ensure_assets "$jax_dir" "$pt_dir"
    return 0
  fi

  local mem_gb
  mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  if (( mem_gb < 28 )); then
    warn "转换 pi05_base 峰值内存约 28 GB（官方脚本），本机 ${mem_gb} GB 可能 OOM。"
    warn "低内存替代方案见 doc/08-local-machine.md 的 tools/convert_pi05_lowmem.py（峰值 ~13 GB）。"
  fi

  if [[ ! -d "$jax_dir" ]]; then
    if (( DRY_RUN )); then
      warn "dry-run：JAX 权重目录尚不存在（$jax_dir），实际执行时会提示下载。"
      return 0
    fi
    die "找不到 JAX 权重目录：$jax_dir
       先下载（二选一）：
         gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base <目标目录>
         huggingface-cli download <hf-repo> --local-dir <目标目录>
       然后用 --checkpoint-dir <该目录> 重跑本脚本。"
  fi

  local wfree_gb
  wfree_gb=$(df -Pk "$(dirname "$ckpt_root")" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}')
  if [[ -n "${wfree_gb:-}" ]] && (( wfree_gb < 20 )); then
    warn "权重目录所在分区仅剩 ${wfree_gb} GB，pi05_base + 转换产物约需 20 GB。"
  fi

  log "转换 JAX -> PyTorch：$jax_dir -> $pt_dir"
  run_in "$OPENPI_DIR" uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir "$jax_dir" \
    --config_name pi05_aloha \
    --output_path "$pt_dir"

  ensure_assets "$jax_dir" "$pt_dir"
  (( DRY_RUN )) || ok "PyTorch 权重就绪：$pt_dir"
}

# ----------------------------------------------------------------------------
# 11. env.sh
# ----------------------------------------------------------------------------
write_env_file() {
  (( WRITE_ENV_FILE )) || return 0
  section "生成 env.sh"

  local out="$REPO_ROOT/env.sh"
  (( DRY_RUN )) && { printf '  %s$ 写入 %s%s\n' "$C_DIM" "$out" "$C_RESET"; return 0; }

  cat > "$out" <<EOF
# 由 setup_env.sh 生成（$(date -Iseconds)）——不要提交到 git
# 用法：source env.sh
#   训练：  cd src/openpi4OpenHLM && .venv/bin/python scripts/train_pytorch.py ...
#   服务：  cd src/openpi4OpenHLM && .venv/bin/python scripts/serve_policy.py ...
#   客户端：cd src/GR00T-WholeBodyControl4OpenHLM/openpi-eval && python main.py ...
export OPENHLM_ROOT="$REPO_ROOT"
export OPENPI_DIR="$OPENPI_DIR"
export OPENPI_VENV="$VENV_DIR"

# openpi 训练 / 推理
export HF_LEROBOT_HOME="\${HF_LEROBOT_HOME:-$DATA_DIR}"
export OPENPI_DATA_HOME="\${OPENPI_DATA_HOME:-$OPENPI_DATA_HOME}"
export XLA_PYTHON_CLIENT_PREALLOCATE="\${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export WANDB_MODE="\${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

# 部署客户端要能 import gear_sonic / sonic_g1_env（无需另建 .venv_teleop）
export PYTHONPATH="$GROOT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"

# 让 .venv/bin/python 与 uv 直接可用
export PATH="$VENV_DIR/bin:\$HOME/.local/bin:\$PATH"
EOF
  ok "已生成 $out"
}

# ----------------------------------------------------------------------------
# 12. 收尾提示
# ----------------------------------------------------------------------------
summary() {
  section "完成 / 下一步"
  cat <<EOF
  仓库        $REPO_ROOT
  venv        $VENV_DIR
  env 文件    $REPO_ROOT/env.sh    （source env.sh 后即可跑训练 / 部署）

  pyproject 里 jaka_tabletop_pick 的 pytorch_weight_path 是按训练机写的绝对路径，
  换机器必须用命令行覆盖（safetensors 不展开 ~）：

  # 训练（单卡）
  source env.sh && cd src/openpi4OpenHLM
  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/train_pytorch.py jaka_tabletop_pick \\
    --exp-name jaka_pick_real_v1 --overwrite \\
    --batch-size 8 --num-train-steps 30000 --save-interval 5000 \\
    --use-8bit-adam --lr-schedule.decay-steps 30000 --lr-schedule.warmup-steps 1000 \\
    --pytorch-weight-path "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base_pytorch"

  # 策略服务（部署服务端）
  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/serve_policy.py --env JAKA --num-steps 10 \\
    policy:checkpoint --policy.config=jaka_tabletop_pick \\
    --policy.dir=checkpoints/jaka_tabletop_pick/jaka_pick_real_v1/30000

  # 部署客户端（无显示器时务必加 --no-opencv-visualize）
  cd src/GR00T-WholeBodyControl4OpenHLM/openpi-eval
  python main.py --env jaka_tabletop_pick --no-opencv-visualize --mock --use-fake-policy

  训练前按 doc/02-data.md 准备数据（HF_LEROBOT_HOME=$DATA_DIR）；改过数据要重算 norm stats。
  XLA_PYTHON_CLIENT_PREALLOCATE / WANDB_MODE / PYTHONPATH 已在 env.sh 里；
  CUDA_VISIBLE_DEVICES 按实际空闲的卡指定（多卡机器务必显式指定）。
EOF
}

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
main() {
  printf '%s\n' "=============================================================================="
  printf ' OpenHLM 环境配置 · VLA 训练 + VLA 部署\n'
  printf ' 仓库：%s\n' "$REPO_ROOT"
  (( DRY_RUN )) && printf ' 模式：dry-run（只打印，不执行）\n'
  (( CHECK_ONLY )) && printf ' 模式：check（只预检 + 自检）\n'
  printf '%s\n' "=============================================================================="

  preflight
  system_deps

  if (( CHECK_ONLY )); then
    verify_env
    verify_deploy_client
    summary
    return 0
  fi

  ensure_uv
  ensure_submodules
  sync_openpi_env
  transformers_patch
  deploy_extras
  verify_env
  verify_deploy_client
  prepare_weights
  write_env_file
  summary
}

main "$@"
