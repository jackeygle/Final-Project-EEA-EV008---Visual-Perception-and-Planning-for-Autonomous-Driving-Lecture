#!/usr/bin/env bash
# Download Cityscapes packages needed for train_cityscapes.py (fine labels + RGB).
# Requires a registered account: https://www.cityscapes-dataset.com/register/
#
# Usage (recommended: export in a private shell, do not commit passwords):
#   export CITYSCAPES_USERNAME='your_login'
#   export CITYSCAPES_PASSWORD='your_password'
#   export CITYSCAPES_DEST=/scratch/work/zhangx29/data/cityscapes   # optional
#   bash scripts/download_cityscapes.sh
#
# Downloads (~11 GB + ~0.24 GB zip) then unzips into CITYSCAPES_DEST:
#   leftImg8bit/   gtFine/
#
# Alternative: community mirrors via Kaggle CLI — scripts/download_cityscapes_kaggle.sh
# (requires Kaggle rules accepted + KAGGLE_API_TOKEN or username/key in kaggle.json).
#
set -euo pipefail

DEST="${CITYSCAPES_DEST:-${HOME}/data/cityscapes}"
USER_NAME="${CITYSCAPES_USERNAME:-}"
PASS="${CITYSCAPES_PASSWORD:-}"

if [[ -z "${USER_NAME}" || -z "${PASS}" ]]; then
  echo "Error: set CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD to your cityscapes-dataset.com account."
  echo "Register at https://www.cityscapes-dataset.com/register/ if needed."
  exit 1
fi

command -v wget >/dev/null || { echo "Error: wget not found."; exit 1; }
command -v unzip >/dev/null || { echo "Error: unzip not found."; exit 1; }

mkdir -p "${DEST}"
cd "${DEST}"

COOKIES="${DEST}/.cityscapes_cookies.txt"
rm -f "${COOKIES}"

echo "Logging in (cookie saved to ${COOKIES})..."
wget -q --keep-session-cookies --save-cookies="${COOKIES}" \
  --post-data "username=${USER_NAME}&password=${PASS}&submit=Login" \
  -O /dev/null \
  "https://www.cityscapes-dataset.com/login/"

# If login failed, file-handling will still redirect; user sees small HTML download
download_pkg() {
  local id="$1"
  local name="$2"
  echo "Downloading packageID=${id} (${name})..."
  wget -c --load-cookies="${COOKIES}" --content-disposition \
    "https://www.cityscapes-dataset.com/file-handling/?packageID=${id}"
}

# 3 = leftImg8bit_trainvaltest.zip (~11 GB), 1 = gtFine_trainvaltest.zip (~241 MB)
download_pkg 3 "leftImg8bit_trainvaltest.zip"
download_pkg 1 "gtFine_trainvaltest.zip"

for z in leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip; do
  if [[ -f "${z}" ]] && ! unzip -tqq "${z}" &>/dev/null; then
    echo "Error: ${z} is not a valid ZIP (wrong password or login failed). Check CITYSCAPES_USERNAME / PASSWORD."
    exit 1
  fi
done

echo "Unzipping (needs tens of GB free disk)..."
for z in leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip; do
  if [[ -f "${z}" ]]; then
    unzip -nq "${z}"
  else
    echo "Warning: expected ${z} not found in ${DEST}"
  fi
done

echo "Done. Point training at:"
echo "  export CITYSCAPES_ROOT=${DEST}"
echo "  python3 train_cityscapes.py --cityscapes_root \"\${CITYSCAPES_ROOT}\" ..."
