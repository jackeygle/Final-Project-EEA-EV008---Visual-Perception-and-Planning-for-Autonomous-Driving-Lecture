#!/usr/bin/env bash
# Cityscapes — 通过 Kaggle 社区镜像下载（非官网直链）。
#
# 使用前在网页打开对应数据集并点击同意规则（Rules）。
#
# 认证（任选其一）：
#   export KAGGLE_API_TOKEN='KGAT_...'     # Kaggle Account → API
#   或 ~/.kaggle/kaggle.json 含 api_token，或含 username + key（经典）
# 建议：pip install -U kaggle kagglesdk
#
# 用法：
#   export CITYSCAPES_DEST=/scratch/work/zhangx29/data/cityscapes
#   bash scripts/download_cityscapes_kaggle.sh
#
# 多包镜像（空格分隔）：
#   export KAGGLE_CITYSCAPES_SLUGS='owner1/slug1 owner2/slug2'
#
set -euo pipefail

DEST="${CITYSCAPES_DEST:-${HOME}/data/cityscapes}"
mkdir -p "${DEST}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: 未找到 kaggle 命令。执行: pip install -U kaggle kagglesdk"
  exit 1
fi

KJSON="${KAGGLE_CONFIG_DIR:-${HOME}/.kaggle}/kaggle.json"
if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "${KJSON}" ]]; then
  KAGGLE_API_TOKEN="$(
    python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('api_token') or '')" "${KJSON}"
  )" || true
  export KAGGLE_API_TOKEN
fi

_has_legacy_json() {
  [[ -f "${KJSON}" ]] || return 1
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get('username') and d.get('key') else 1)" "${KJSON}" 2>/dev/null
}

if [[ -z "${KAGGLE_API_TOKEN:-}" ]] && ! _has_legacy_json; then
  echo "Error: 请先配置 Kaggle 认证其一："
  echo "  export KAGGLE_API_TOKEN='KGAT_...'"
  echo "  或在 ~/.kaggle/kaggle.json 写入 api_token，或 username + key"
  exit 1
fi

SLUGS="${KAGGLE_CITYSCAPES_SLUGS:-hungngu1999/cityscapes-dataset}"

echo "下载目录: ${DEST}"
for slug in ${SLUGS}; do
  echo "==> kaggle datasets download -d ${slug}"
  kaggle datasets download -d "${slug}" -p "${DEST}"
done

shopt -s nullglob
for z in "${DEST}"/*.zip; do
  echo "==> unzip $(basename "${z}")"
  unzip -oq "${z}" -d "${DEST}"
done
shopt -u nullglob

# 推断 CITYSCAPES_ROOT（需存在 leftImg8bit/ 与 gtFine/）
suggest_root() {
  local d="$1"
  if [[ -d "${d}/leftImg8bit" && -d "${d}/gtFine" ]]; then
    echo "${d}"
    return 0
  fi
  return 1
}

CS_ROOT=""
if suggest_root "${DEST}"; then
  CS_ROOT="${DEST}"
elif suggest_root "${DEST}/cityscapes"; then
  CS_ROOT="${DEST}/cityscapes"
elif suggest_root "${DEST}/cityscapes_dataset"; then
  CS_ROOT="${DEST}/cityscapes_dataset"
else
  found="$(find "${DEST}" -maxdepth 5 -type d -name leftImg8bit 2>/dev/null | head -1 || true)"
  if [[ -n "${found}" ]]; then
    CS_ROOT="$(dirname "${found}")"
    if [[ ! -d "${CS_ROOT}/gtFine" ]]; then
      echo "Warning: 找到 leftImg8bit 但未在同层发现 gtFine，请手动检查目录结构。"
    fi
  fi
fi

echo ""
if [[ -n "${CS_ROOT}" ]]; then
  echo "下一步训练："
  echo "  export CITYSCAPES_ROOT=${CS_ROOT}"
  echo "  python3 train_cityscapes.py --cityscapes_root \"\${CITYSCAPES_ROOT}\" --save_dir ."
else
  echo "未自动定位 leftImg8bit。请 ls \"${DEST}\" 后手动设置 CITYSCAPES_ROOT。"
  ls -la "${DEST}" 2>/dev/null | head -25 || true
fi
