#!/usr/bin/env bash
# 把两篇有公开代码的对比学习论文（ConFEDE、HSCL）配置成能在本机跑的状态。
#
# 与 setup_mmsa_reference.sh 同样的定位：**产出物一律不入库**（`.mmsa-reference/`
# 在 .gitignore 里），换机器必须重跑这个脚本。名单与分档见
# docs/benchmark_contrastive.md。
#
# 关键设计与 setup_mmsa_reference.sh 一致：**torch 仍取自主 venv**，只把两篇论文
# 真正 import 的第三方包装进参照 shim。参照运行只有在两边共用同一个张量库时才有
# 意义——否则指标一差就先要怀疑 torch 版本。
#
# requirements.txt 里的钉死版本（ConFEDE 要 torch 1.8.1 / transformers 4.5.1 /
# pytorch-pretrained-bert；HSCL 要 torch 1.8.0+cu111 / transformers 2.10.0 /
# Python 3.7）**都不装**。逐个 grep 过它们的 import：ConFEDE 实际只用 numpy /
# sklearn / torch / tqdm / transformers / pytorch_metric_learning，HSCL 只多用
# pandas。因此不需要第二个解释器，两处 API 变更在 .mmsa-reference/compat/ 里补，
# 论文源码逐字节保持发布状态（见 investigations.md#paper-repro-compat）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPERS="$ROOT/.mmsa-reference/papers"
SHIM="$ROOT/.mmsa-reference/env/shim"
PY="$ROOT/.venv/bin/python"

[ -x "$PY" ] || { echo "没有 $PY，先跑 bash scripts/setup.sh" >&2; exit 1; }
[ -d "$SHIM" ] || { echo "没有 $SHIM，先跑 bash scripts/setup_mmsa_reference.sh" >&2; exit 1; }

mkdir -p "$PAPERS"

clone() {  # clone <目录名> <仓库地址>
  if [ -d "$PAPERS/$1/.git" ]; then
    echo "== $1 已存在，跳过 clone"
  else
    echo "== clone $1"
    git clone --depth 1 "$2" "$PAPERS/$1"
  fi
}

clone ConFEDE https://github.com/XpastaX/ConFEDE.git
clone HSCL     https://github.com/Turdidae810/HSCL.git
# UniMSE 只为留档：它跑不到一个 batch，原因不是兼容性，见
# investigations.md#unimse-partial-release。clone 它是为了让那份诊断可复核，
# 不是为了跑出数字——所以下面**不**给它下 t5-base（851MB）。
clone UniMSE  https://github.com/LeMei/UniMSE.git

echo "== 装 ConFEDE 需要的 pytorch-metric-learning 进 shim"
"$ROOT/.mmsa-reference/env/bin/python" -m pip install -q "pytorch-metric-learning==0.9.99"

echo "== ConFEDE：数据软链 + 补上它忘建的目录"
# 发布版的 config.py 对 model_path / result_path 调了 check_dir，独独漏了
# encoder_path，三个编码器都往那里存 —— 第一次 MAE 改善时才炸，已经烧掉几十轮。
# 见 investigations.md#confede-missing-ckpt-dir
mkdir -p "$PAPERS/ConFEDE/MOSI/data/MOSI/Processed" "$PAPERS/ConFEDE/MOSI/ckpt/fea_encoder"
ln -sfn "$ROOT/datasets/CMU-MOSI/Processed/unaligned_50.pkl" \
        "$PAPERS/ConFEDE/MOSI/data/MOSI/Processed/unaligned_50.pkl"

echo "== HSCL：把我们的特征改写成 SDK 布局 + 标签 CSV 软链"
# HSCL 读的是 CMU-MultimodalSDK/MMIM 那一系的 pickle，键名、id 格式、标签形状、
# **补零方向**都与 MMSA 的不同。方向是改我们的数据去适配它的代码，不改它的代码，
# 这样两边读的是同一份 sha256 校验过的特征。padding 那个坑见
# investigations.md#mmsa-padding-not-zero
mkdir -p "$PAPERS/HSCL/data/MOSI"
if [ -s "$PAPERS/HSCL/data/MOSI/mosi_data_noalign.pkl" ]; then
  echo "   mosi_data_noalign.pkl 已存在，跳过"
else
  "$PY" "$ROOT/scripts/adapt_features_sdk.py" --dataset mosi \
      --out "$PAPERS/HSCL/data/MOSI/mosi_data_noalign.pkl"
fi
ln -sfn "$ROOT/datasets/CMU-MOSI/label.csv" "$PAPERS/HSCL/data/MOSI/MOSI-label.csv"
# create_dataset.py 会把转换结果缓存成 train/dev/test.pkl。缓存按旧形状建的话
# 后面会以难懂的方式炸（labels 少一维 → collate_fn 里 IndexError），所以
# 每次重建源 pickle 都要连带删掉缓存。
rm -f "$PAPERS/HSCL/data/MOSI"/{train,dev,test}.pkl

echo "== UniMSE：只做数据软链（它跑不通，理由见 investigations.md#unimse-partial-release）"
mkdir -p "$PAPERS/UniMSE/datasets/MOSI"
ln -sfn "$PAPERS/HSCL/data/MOSI/mosi_data_noalign.pkl" \
        "$PAPERS/UniMSE/datasets/MOSI/mosi_data_noalign.pkl"
ln -sfn "$ROOT/datasets/CMU-MOSI/label.csv" "$PAPERS/UniMSE/datasets/MOSI/MOSI-label.csv"

cat <<'EOF'

配置完成。怎么跑：

  # ConFEDE：先三个编码器（约 17 分钟），再融合 5 seed（约 2 小时）
  cd .mmsa-reference/papers/ConFEDE/MOSI
  PYTHONPATH=../../../env/shim:../../../compat:. ../../../../.venv/bin/python -u -c \
    "import confede_compat; confede_compat.run('main.py', 'cuda:0')"
  PYTHONPATH=../../../env/shim:../../../compat:. ../../../../.venv/bin/python -u -c \
    "import confede_compat; confede_compat.run('main-fusion.py', 'cuda:0')"

  # HSCL：两臂各 5 seed（约 25 分钟），跑完自动出对照表
  bash scripts/run_hscl_arms.sh

  # UniMSE：跑不通，别花时间。它需要 ../t5-base/（851MB）和仓库里没有的预处理
  # pickle，而后者把论文的统一标签空间烘在里面。诊断已固化在
  # docs/investigations.md#unimse-partial-release，不需要重跑来确认。

注意 ConFEDE 的检查点约 2.9G（文本编码器单个 438MB），HSCL 的 SDK pickle 442MB。
本机磁盘余量一直在 5G 上下，跑之前先 df -h /。
EOF
