"""Build the model comparison table — the one that goes into the paper.

`summary_table.py` answers "did we reproduce MMSA?" and prints shortfalls in
standard errors. This answers a different question: **what does the field look
like across nine years, measured the same way?** So it carries the year, the
idea in one line, the data setting, where the reference came from, and every
metric — not just the shortfall.

    python scripts/model_table.py                 # write docs/model_table.md
    python scripts/model_table.py --stdout        # print instead

Regenerated, never typed. Everything numeric comes from each group's
`summary.json` and every verdict from `check_acceptance.py`, so the table cannot
drift from the runs behind it.

Two columns exist because the table is misleading without them:

* **Data** — aligned and unaligned are different problems. MulT and ConFEDE run
  unaligned; most of the rest run aligned. A single column of numbers spanning
  both invites a comparison that is not sound.
* **Reference** — whether a model was checked against its authors' own code, a
  third-party reimplementation, or authors' code that needed a compatibility
  patch to run at all. Without it a reader cannot separate "reproduced
  faithfully" from "happened to land nearby".

Acc-7 is reported and is deliberately NOT a primary metric (docs/decisions.md,
2026-07-28): it slices a continuous prediction into seven bins, so one sample
crossing a boundary flips a whole cell. EF-LSTM's +-8.11 is what that looks like.
MAE and Corr are the only continuous metrics here and carry the verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from msa.config import OUTPUT_ROOT, PROJECT_ROOT

OUT_PATH = PROJECT_ROOT / "docs" / "model_table.md"

#: A row whose number does not yet mean what the model is capable of. Rendered as
#: a footnote and marked in the verdict column, because a table that shows 1.1549
#: next to 0.7290 without saying why invites exactly the wrong conclusion.
#:
#: Currently empty. ConFEDE lived here until its stage-one pretraining was
#: implemented, which moved it from 1.1549 to 0.7358 -- kept as the mechanism,
#: because the next half-finished model will need it too.
CAVEATS: dict[str, str] = {}

#: group -> (year, one-line idea, reference tier). Order is the story's, which is
#: chronological except that the two controls sit at the end where they belong.
#:
#: `year` is the publication year of the method, not of our run. Baselines we
#: built ourselves have none.
MODELS: list[tuple[str, str, str, str]] = [
    # LF-LSTM is stored per device -- the CUDA/CPU pair is the cross-device
    # comparison, and the CUDA one is what every other row was run on.
    ("lf_lstm_mosi_cuda", "—", "自建基线：各模态各一 LSTM，末状态拼接后回归", "control"),
    ("ef_lstm_mosi", "—", "早融合：逐步拼接三模态后过一个 LSTM", "mmsa"),
    ("lf_dnn_mosi", "—", "晚融合：各模态子网独立编码后接全连接", "mmsa"),
    ("tfn_mosi", "2017", "三路外积显式构造全部单/双/三模态交互项", "mmsa"),
    ("lmf_mosi", "2018", "低秩分解，不显式构造融合张量（19× 压缩）", "mmsa"),
    ("mfn_mosi", "2018", "三个 LSTMCell 同步推进，注意力写入共享记忆", "mmsa"),
    ("graph_mfn_mosi", "2018", "融合建成动态图，边权由数据定", "mmsa"),
    ("mctn_mosi", "2019", "循环翻译学联合表征，测试时只需源模态", "mmsa"),
    ("mfm_mosi", "2019", "表征因子化为判别因子 + 模态专属生成因子", "mmsa"),
    ("mult_mosi", "2019", "跨模态注意力，直接处理未对齐序列", "mmsa"),
    ("bert_mag_mosi", "2020", "音视频合成一次位移，加在 BERT 嵌入层之后", "mmsa"),
    ("misa_mosi", "2020", "模态不变/特有表征解耦（首个微调 BERT）", "mmsa"),
    ("self_mm_mosi", "2021", "自监督生成单模态伪标签，多任务联合训练", "mmsa"),
    ("mmim_mosi", "2021", "互信息下界 + 对比预测编码", "mmsa"),
    ("cenet_mosi", "2023", "音视频作为偏移注入 BERT 层间", "mmsa"),
    ("tetfn_mosi", "2023", "音视频交互经文本中介，不直接相见", "mmsa"),
    ("almt_mosi", "2023", "音视频只位移一个语言主导的超模态", "mmsa"),
    ("clgsi_mosi", "2024", "正负对不按标签相等划分，按情感强度距离加权", "author"),
    ("confede_mosi", "2023", "每模态投影为相似/相异两支，与检索来的伙伴样本对比", "author"),
    ("dmd_mosi", "2023", "专属/共享解耦后在可学习的图上互相蒸馏，边权决定谁教谁", "author-patched"),
    ("dlf_mosi", "2025", "音视频只作为被查询方补充语言（语言聚焦）", "author"),
    ("dpdf_lq_mosi", "2025", "双路并行：可学习查询取全局摘要 + 局部交叉注意力，门控加权", "author"),
    ("text_bert_mosi", "—", "对照组：只用文本，微调编码器", "mmsa"),
]

REFERENCE_LABEL = {
    "author": "作者代码",
    "author-patched": "作者代码+兼容重建",
    "mmsa": "MMSA 代码",
    "control": "本仓库基线",
}
#: Reported in this order. mae and corr are the primary pair.
METRICS = [("mae", "MAE ↓"), ("corr", "Corr ↑"), ("acc2_non0", "Acc-2"),
           ("f1_non0", "F1"), ("acc5", "Acc-5"), ("acc7", "Acc-7")]
PERCENT = {"acc2_non0", "f1_non0", "acc5", "acc7"}


def verdicts() -> dict[str, str]:
    """Group -> short verdict, read from check_acceptance rather than recomputed.

    Parsing another script's output is a coupling worth naming: the alternative
    is a second copy of the acceptance rules, and two copies of a rule drift.
    If this ever comes back empty, that script's output format changed.
    """
    done = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "check_acceptance.py"), "--all"],
        capture_output=True, text=True, check=False)
    found, group = {}, None
    for line in done.stdout.splitlines():
        header = re.match(r"=== (\S+) \(", line)
        if header:
            group = header.group(1)
        elif "VERDICT:" in line and group:
            verdict = line.split("VERDICT:", 1)[1].strip()
            if verdict.startswith("reproduced, but"):
                verdict = "复现成功（有标记）"
            elif verdict.startswith("reproduced"):
                verdict = "复现成功"
            elif "ADJUDICATED" in verdict:
                verdict = "差距已裁定"
            else:
                verdict = "未复现"
            found[group] = verdict
            group = None
    return found


def cell(stats: dict, metric: str) -> str:
    scale = 100 if metric in PERCENT else 1
    digits = 2 if metric in PERCENT else 4
    return f"{stats['mean'] * scale:.{digits}f} ± {stats['std'] * scale:.{digits}f}"


def build() -> str:
    judged = verdicts()
    header = ["年份", "模型", "思路", "数据", "seed", *[label for _, label in METRICS],
              "参照", "判定"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    missing = []
    for group, year, idea, tier in MODELS:
        path = OUTPUT_ROOT / group / "summary.json"
        if not path.exists():
            missing.append(group)
            continue
        payload = json.loads(path.read_text())
        stats = payload["summary"]
        name = group.rsplit("_", 1)[0]
        row = [year, f"**{name}**", idea,
               "aligned" if payload.get("aligned") else "unaligned",
               str(len(payload.get("seeds", []))),
               *[cell(stats[key], key) if key in stats else "—" for key, _ in METRICS],
               REFERENCE_LABEL[tier],
               "⚠️ 见脚注" if group in CAVEATS else judged.get(group, "—")]
        lines.append("| " + " | ".join(row) + " |")

    shown = [g for g, *_ in MODELS if g in CAVEATS and (OUTPUT_ROOT / g / "summary.json").exists()]
    if shown:
        lines.append("")
        for group in shown:
            lines.append(f"⚠️ **{group.rsplit('_', 1)[0]}**：{CAVEATS[group]}")
    return "\n".join(lines), missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = ap.parse_args()

    table, missing = build()
    document = f"""# 模型比较表（CMU-MOSI）

**由 `python scripts/model_table.py` 生成，不要手抄。** 每次有新结果就重跑它。

全部为 10 个种子（42-51）的 mean ± 样本标准差（ddof=1）。**主指标是 MAE 与 Corr**
——它们是仅有的连续指标；Acc-2/5/7 与 F1 照常汇报但不参与判定
（2026-07-28 变更，理由见 `decisions.md`）。**Acc-7 把连续预测切成七段，一个样本跨过边界
就翻转整格**，EF-LSTM 的 ±8.11 就是这么来的。

「参照」列指判定所对照的对象：`作者代码` 为我们在本机以同一协议跑作者原始实现；
`作者代码+兼容重建` 表示该 release 在当前 PyTorch 上无法启动，需要一处会改变数值的
兼容性补丁才能运行（见 `investigations.md#dmd-reference-patches`），**证据强度低一档**；
`MMSA 代码` 为第三方二次实现。**没有这一列，读者无法区分「复现得准」与「碰巧接近」。**

「数据」列必须看：**aligned 与 unaligned 是不同的问题设定**，跨设定直接比大小不成立。

{table}
"""
    if missing:
        document += ("\n**尚无结果的组**（跑完后重跑本脚本）："
                     + "、".join(f"`{g}`" for g in missing) + "\n")
    if args.stdout:
        print(document)
    else:
        OUT_PATH.write_text(document)
        print(f"wrote {OUT_PATH.relative_to(PROJECT_ROOT)}"
              + (f"  ({len(missing)} group(s) still missing)" if missing else ""))


if __name__ == "__main__":
    main()
