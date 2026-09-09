"""把我们校验过的 MMSA 特征改写成 CMU-MultimodalSDK 的 `*_data_noalign.pkl` 布局。

对标 HSCL（MMM 2024）用的。它的 `create_dataset.py` 读的是 MMIM/SDK 那一系的
pickle，键名与我们的 `unaligned_50.pkl` 不同。**不改它的源码，改我们的数据的
形状**——这样两边跑在同一份 sha256 校验过的特征上，指标差异不能归因于数据。

三处必须翻译，第三处是坑：

1. `id`：我们是 `<U16` 的 `'03bSnISJMiM$_$11'`，SDK 是 bytes、形状 (N,1)，
   HSCL 取 `[:,0]` 再 `decode('utf-8')`，然后用正则 `(.*)_(.*)` 拆成
   video/clip 去 CSV 里查。`$_$` 分隔符会让那个正则拆错，所以写成 `video_clip`。
2. `labels`：我们是 `regression_labels` 形状 (N,)，SDK 是 **(N, 1, 1)**。多出来的
   那一维不是冗余，是被读的：HSCL 的 `collate_fn` 取 `sample[1]`（于是拿到 (1,1)）、
   `torch.cat(dim=0)` 成 (B,1)，再判 `if labels.size(1) == 7` 来识别 MOSEI——SDK 的
   MOSEI 标签是 (N,1,7)，七种情绪，取第 0 列作情感强度。写成 (N,1) 的话
   `cat` 会压成一维，`size(1)` 直接 IndexError。
3. **补零方向要反过来，而且先得真的把零补上。** HSCL 用 `x[L - length:, :]` 取
   序列，也就是假定补零在**前端**（SDK/MMIM 的约定）。我们这份 MMSA 特征是
   **后端**放数据。直接喂过去，它取到的是纯零。这是约定 5 列的那类协议差异
   （padding），只看键名对不对发现不了。

   更细的一层：MMSA 的 `unaligned_50.pkl` 里，**记录长度之外的那段并不是零**。
   实测 vision 有 302/2199 条、audio 有 71/2199 条在 `*_lengths` 之后仍有非零帧
   （见 docs/investigations.md#mmsa-padding-not-zero）。`*_lengths` 才是权威——
   我们自己的 loader 一直按它遮罩，那段残值从没进过模型。而 HSCL 的
   `get_length` 是"数非全零帧"，喂原样数据会**多数**出来，切片就会把真实帧的
   开头丢掉。所以这里先按记录长度清零、再左移，让 SDK 那条不变式（真实序列
   之外全零）真的成立，HSCL 的启发式才能算回记录长度。

   残留 2/2199 条（train、test 各一）序列内部本来就有一帧全零，清零后
   `get_length` 会少数 1 帧。这是 HSCL 启发式自身的界，喂它原版 SDK pickle 也
   一样，不在这里补。

CSV 用 `datasets/CMU-MOSI/label.csv` 原样软链，它的列名（video_id / clip_id /
text）已经是 HSCL 要的。
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLIT_MAP = {"train": "train", "valid": "valid", "test": "test"}


def left_pad(features: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """把每条序列挪到序列维的尾部，前端留零。

    从 `features[row, :length]` 取——按记录长度截断，因此顺带丢掉了记录长度之外
    的残值。向量化不比循环快多少（N 是一千量级），循环胜在读起来就是这个语义。
    """
    out = np.zeros_like(features)
    total = features.shape[1]
    for row, length in enumerate(lengths):
        length = int(length)
        if length <= 0:
            continue
        out[row, total - length:] = features[row, :length]
    return out


def convert(source: Path, destination: Path) -> None:
    with source.open("rb") as handle:
        data = pickle.load(handle)

    converted = {}
    for sdk_split, our_split in SPLIT_MAP.items():
        split = data[our_split]
        vision = np.asarray(split["vision"], dtype=np.float32)
        audio = np.asarray(split["audio"], dtype=np.float32)
        vision_lengths = np.asarray(split["vision_lengths"])
        audio_lengths = np.asarray(split["audio_lengths"])

        # 记录长度必须落在序列维之内，且真实帧不能被判成全零——两者破了，
        # 下面的搬移就切错了。**不**要求"非零帧数 == 记录长度"：上面文档串里那
        # 302 条残值就是反例，清零本身是这里要做的事。
        for name, features, lengths in (("vision", vision, vision_lengths),
                                        ("audio", audio, audio_lengths)):
            if lengths.min() < 1 or lengths.max() > features.shape[1]:
                raise SystemExit(
                    f"{our_split}/{name}: 记录长度落在 [1, {features.shape[1]}] "
                    f"之外（实测 {lengths.min()}..{lengths.max()}）。"
                )
            nonzero = (np.abs(features).sum(-1) != 0).sum(1)
            interior = int((nonzero < lengths).sum())
            beyond = int((nonzero > lengths).sum())
            print(f"  {our_split:<6}{name:<7}记录长度之外仍有非零帧 {beyond} 条"
                  f"（清零）；序列内部含全零帧 {interior} 条"
                  f"（HSCL 的 get_length 会少数）")

        identifiers = np.array(
            [[identifier.replace("$_$", "_").encode("utf-8")]
             for identifier in split["id"]],
            dtype=object,
        )
        # (N, 1, 1)，不是 (N, 1)——理由在模块文档串第 2 条。
        labels = np.asarray(split["regression_labels"],
                            dtype=np.float32).reshape(-1, 1, 1)

        converted[sdk_split] = {
            "vision": left_pad(vision, vision_lengths),
            "audio": left_pad(audio, audio_lengths),
            "labels": labels,
            "id": identifiers,
            "text": np.asarray(split["text"], dtype=np.float32),
        }
        print(f"  {sdk_split:<6}{len(labels):>6} 条  vision {vision.shape[1:]}"
              f"  audio {audio.shape[1:]}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(converted, handle, protocol=4)
    size_mb = destination.stat().st_size / 1e6
    print(f"写出 {destination}  ({size_mb:.0f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--out", type=Path, required=True,
                        help="目标 *_data_noalign.pkl 的路径")
    args = parser.parse_args()

    folder = "CMU-MOSI" if args.dataset == "mosi" else "CMU-MOSEI"
    source = ROOT / "datasets" / folder / "Processed" / "unaligned_50.pkl"
    if not source.exists():
        raise SystemExit(f"没有 {source}，先跑 scripts/fetch_dataset.sh {args.dataset}")
    print(f"{source} → SDK 布局")
    convert(source, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
