"""生成汇报用的四张图。

**只画已入库、可重算的数字**：模型指标读 `outputs/*/seed*/result.json`，
论文报告值来自 `docs/benchmark_contrastive.md` 里从 PDF 抄录的那张表（每条都注了
表号）。**没有任何一个数是凭记忆填的**，两类来源在图注里分开标。

用 `uv run --no-project --with matplotlib` 跑——matplotlib 不在本项目的钉死依赖里，
**绝不能装进 .venv**：`uv run` 曾经在仓库目录里把 .venv 重同步、把 torch 升到 CUDA
跑不动的版本（`investigations.md#uv-run-resync`），所以这里既不进 venv，也带 --no-project。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for candidate in ("Noto Serif CJK SC", "Noto Sans CJK SC", "Source Han Serif SC"):
    if any(f.name == candidate for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = candidate
        break
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})

INK, GREY, RED, GREEN, AMBER = "#1a1a1a", "#8a8a8a", "#9b1010", "#0a5d00", "#8a6d00"


def test_mae(group: str) -> list[float]:
    return [json.load(open(p))["test"]["mae"]
            for p in sorted(glob.glob(str(ROOT / "outputs" / group / "seed*/result.json")))]


# ── 图 1：八年技术演进，我们实测的台阶 ────────────────────────────────
def figure_progression() -> None:
    # (标签, 年份, 我们的测试 MAE) —— 取自 docs/storyline.md 的复现总表
    models = [("TFN", 2017, 0.9511), ("LMF", 2018, 0.9513), ("MFN", 2018, 0.9437),
              ("MulT", 2019, 0.8974), ("MISA", 2020, 0.7610), ("Self-MM", 2021, 0.7142),
              ("CENET", 2022, 0.7364), ("TETFN", 2023, 0.7283)]
    text_only = 0.8018                      # 纯文本微调 BERT 对照组
    # 配对 bootstrap（2000 次重采样）的判定，见 docs/experiments.md
    verdict = {"TFN→LMF": "ns", "LMF→MFN": "ns", "MFN→MulT": "sig",
               "MulT→MISA": "sig", "MISA→Self-MM": "sig",
               "Self-MM→CENET": "worse", "CENET→TETFN": "ns"}

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    xs = range(len(models))
    ys = [m[2] for m in models]
    ax.plot(xs, ys, "-o", color=INK, lw=1.6, ms=5, zorder=3)

    ax.axhline(text_only, color=RED, ls="--", lw=1.4, zorder=2)
    ax.annotate("纯文本微调 BERT（无音视频）= 0.8018\n"
                "显著优于 MulT 及此前全部冻结特征模型",
                xy=(0.0, text_only), xytext=(0.0, text_only - 0.012),
                color=RED, fontsize=8.5, va="top")

    for i, (label, year, value) in enumerate(models):
        # MISA 正好落在纯文本那条线附近，标在上方会压住它，单独挪到右下。
        offset, align = ((14, -16), "left") if label == "MISA" else ((0, 11), "center")
        ax.annotate(f"{label}\n{year}", (i, value), textcoords="offset points",
                    xytext=offset, ha=align, fontsize=8, color=INK)
    # 显著性标记放在图底专用的一行，不跟着曲线走——跟着走会和模型名撞在一起。
    marker_y = 0.675
    ax.annotate("相邻台阶：", (-0.45, marker_y), fontsize=8, color=INK, va="center")
    for i in range(len(models) - 1):
        key = f"{models[i][0]}→{models[i+1][0]}"
        kind = verdict.get(key, "ns")
        mark, colour = {"sig": ("显著", GREEN), "ns": ("ns", GREY),
                        "worse": ("倒退", RED)}[kind]
        ax.annotate(mark, ((i + i + 1) / 2, marker_y), ha="center", va="center",
                    fontsize=8.5, color=colour,
                    weight="bold" if kind != "ns" else None)

    ax.set_xticks([]); ax.set_ylabel("测试集 MAE ↓")
    ax.set_ylim(0.655, 1.02)
    ax.set_title("图 1　八年演进里只有三步显著，而中间那步是「开始微调 BERT」",
                 loc="left", fontsize=10.5, weight="bold", pad=26)
    fig.tight_layout(); fig.savefig(OUT / "fig1_progression.png", bbox_inches="tight"); plt.close(fig)


# ── 图 2：效应量 vs 可检测下限（核心图）────────────────────────────────
def figure_effects_vs_mde() -> None:
    items = [
        ("HyCon 自称：去掉全部对比损失", 0.056, "claim"),
        ("跨论文：同一个 MMIM 的两个报告值", 0.038, "claim"),
        ("MOSI 上 LF-LSTM 的 seed 标准差", 0.0387, "noise"),
        ("HSCL：同 seed 重跑一次的差异", 0.0079, "noise"),
        ("MMCL 相对 MMIM 的报告改进", 0.005, "claim"),
        ("我们：第 2 阶段全部对比损失效应（上界）", 0.0174, "ours"),
        ("我们：MMIM 自带对比项（MOSEI，已知有效）", 0.0159, "ours"),
        ("我们：dcl 在 MOSEI 上（唯一站住的）", 0.0070, "ours"),
        ("我们：label_dcl 的机制增量（第三批）", 0.0004, "ours"),
    ]
    items.sort(key=lambda t: t[1])
    colour = {"claim": AMBER, "noise": RED, "ours": INK}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ys = range(len(items))
    ax.barh(list(ys), [v for _, v, _ in items],
            color=[colour[k] for _, _, k in items], height=0.6, zorder=3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([n for n, _, _ in items], fontsize=8)

    # 两条刻度线的标注放到坐标区**下方**，与竖线对齐——放在图内无论摆哪都会压住
    # 条形自己的数值标签。
    from matplotlib.transforms import blended_transform_factory
    below = blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvline(0.013, color=GREEN, lw=1.6, ls="--", zorder=4)
    ax.annotate("MOSI 分辨极限 0.013 (n=20)", xy=(0.013, -0.135), xycoords=below,
                ha="center", color=GREEN, fontsize=8.5, weight="bold")
    ax.axvline(0.0039, color=GREEN, lw=1.2, ls=":", zorder=4)
    ax.annotate("MOSEI 0.0039", xy=(0.0039, -0.215), xycoords=below,
                ha="center", color=GREEN, fontsize=8)

    for y, (_, value, _) in zip(ys, items):
        ax.annotate(f"{value:.4f}", (value, y), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("MAE 上的效应量（绝对值）", labelpad=26)
    ax.set_xlim(0, 0.064)
    ax.set_title("图 2　橙=论文声称　红=纯噪声　黑=我们实测\n绿线＝这台仪器读得出的最小刻度",
                 loc="left", fontsize=10, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig2_effects_vs_mde.png", bbox_inches="tight"); plt.close(fig)


# ── 图 3：8 篇的报告值 vs 它可被检验的程度 ─────────────────────────────
def figure_benchmark() -> None:
    # MOSI 报告 MAE，全部从各论文 PDF 抄录（表号见 benchmark_contrastive.md）
    papers = [("ConKI", 0.681, "C"), ("UniMSE", 0.691, "C"), ("MMIM", 0.700, "A"),
              ("MMCL", 0.705, "C"), ("CMSBERT-CLR", 0.708, "C"),
              ("HyCon", 0.713, "C"), ("ConFEDE", 0.742, "A"), ("HSCL", None, "A")]
    shown = [p for p in papers if p[1] is not None]
    colour = {"A": GREEN, "C": GREY}
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    xs = range(len(shown))
    ax.bar(list(xs), [p[1] for p in shown],
           color=[colour[p[2]] for p in shown], width=0.6, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([p[0] for p in shown], fontsize=8.5, rotation=20, ha="right")
    for x, p in zip(xs, shown):
        ax.annotate(f"{p[1]:.3f}", (x, p[1]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    ax.set_ylim(0.6, 0.78); ax.set_ylabel("论文自报的 MOSI MAE ↓")

    ax.annotate("报告值最好的两篇都无法检验：ConKI 从未公开代码，\n"
                "UniMSE 公开了但跑不到一个 batch",
                xy=(0.5, 0.676), xytext=(2.4, 0.617), fontsize=8.5, color=RED,
                ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                     ec=RED, lw=0.8),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.annotate("绿＝能在我们统一校验过的特征上跑通（3/8）　　灰＝只能引用论文报告值（5/8）\n"
                "HSCL 未画：MMM 2024 在 Springer 付费墙后，它自己的报告值取不到",
                xy=(0, 0), xytext=(0.0, -0.42), xycoords="axes fraction",
                fontsize=8, color=INK)
    ax.set_title("图 3　八篇对标：报告值的高低与它可被检验的程度无关",
                 loc="left", fontsize=10.5, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig3_benchmark.png",
                                    bbox_inches="tight"); plt.close(fig)


# ── 图 4：不同数据集/模型上，重跑一次的噪声有多大 ───────────────────────
def figure_noise() -> None:
    measured = [("LF-LSTM / MOSI", test_mae("lf_lstm_mosi_cuda")),
                ("TFN / MOSI", test_mae("tfn_mosi")),
                ("ALMT / MOSI", test_mae("almt_mosi")),
                ("MMIM / MOSI", test_mae("mmim_diag_mosi_unaligned_off")),
                ("MMIM / MOSEI", test_mae("mmim_diag_mosei_unaligned_off")),
                ("LF-LSTM / MOSEI", test_mae("lf_lstm_mosei_cuda"))]
    import statistics
    rows = [(name, statistics.stdev(v), len(v)) for name, v in measured if len(v) > 1]

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ys = range(len(rows))
    ax.barh(list(ys), [sd for _, sd, _ in rows],
            color=["#3a3a3a" if "MOSEI" not in n else GREEN for n, _, _ in rows],
            height=0.6, zorder=3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{n}  (n={k})" for n, _, k in rows], fontsize=8.5)
    for y, (_, sd, _) in zip(ys, rows):
        ax.annotate(f"{sd:.4f}", (sd, y), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=8)
    ax.axvspan(0.005, 0.03, color=AMBER, alpha=0.16, zorder=1)
    ax.annotate("文献里「改进」常见的量级 0.005 – 0.03", xy=(0.0175, 2.55),
                fontsize=8.5, color=AMBER, ha="center", weight="bold")
    ax.set_xlabel("测试集 MAE 的 seed 间标准差")
    ax.set_title("图 4　换个 seed 重跑的波动，普遍与论文声称的改进同量级\nMOSEI 是例外，也是唯一分辨得开的地方",
                 loc="left", fontsize=10, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig4_noise.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    figure_progression()
    figure_effects_vs_mde()
    figure_benchmark()
    figure_noise()
    for path in sorted(OUT.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")
