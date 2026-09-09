#!/usr/bin/env bash
# HSCL 的两臂：as-released（发布版，模型选择读测试集）与 clean（只看验证集）。
#
# 两臂的唯一差别是 scripts/patches/hscl_clean_selection.patch 那一行，理由见
# docs/investigations.md#hscl-test-leak。跑完自动 git checkout 还原，因此中断后
# 重跑是安全的；已有日志的 seed 会跳过。
#
# 用法：bash scripts/run_hscl_arms.sh [输出目录]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HSCL="$ROOT/.mmsa-reference/papers/HSCL"
OUT="${1:-$ROOT/outputs/hscl_arms}"
SEEDS=(42 43 44 45 46)

[ -d "$HSCL" ] || { echo "没有 $HSCL，先跑 scripts/setup_paper_repros.sh" >&2; exit 1; }

# 无论怎么退出都还原成发布版——留一份打了补丁的树，下一次"as-released"就是假的。
cleanup() { git -C "$HSCL" checkout -- src/solver.py 2>/dev/null || true; }
trap cleanup EXIT

run_arm() {
  local arm="$1"
  mkdir -p "$OUT/$arm"
  for seed in "${SEEDS[@]}"; do
    local log="$OUT/$arm/seed$seed.log"
    if [ -s "$log" ]; then echo "  跳过 $arm/seed$seed（已有日志）"; continue; fi
    echo "  $arm seed $seed  $(date +%H:%M:%S)"
    ( cd "$HSCL/src" && PYTHONPATH="$ROOT/.mmsa-reference/env/shim:$ROOT/.mmsa-reference/compat:." \
      "$ROOT/.venv/bin/python" -u -c "
import sys, hscl_compat
sys.argv = ['main.py', '--dataset', 'mosi', '--data_path', '../data/MOSI',
            '--bert_path', 'bert-base-uncased', '--seed', '$seed']
hscl_compat.run('main.py')" ) > "$log" 2>&1
  done
}

echo "== as-released（发布版）"
cleanup
run_arm as_released

echo "== clean（只按验证集选）"
git -C "$HSCL" apply "$ROOT/scripts/patches/hscl_clean_selection.patch"
git -C "$HSCL" diff --stat src/solver.py
run_arm clean
cleanup

echo
"$ROOT/.venv/bin/python" "$ROOT/scripts/report_hscl_arms.py" "$OUT"
