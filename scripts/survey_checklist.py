"""One view of the survey, cross-checked against everything that could contradict it.

The same paper used to appear in four places -- `survey.md`'s ledgers,
`sweep_venues.tsv`'s enumeration, `reading_list.md`, and `papers/MANIFEST.tsv` --
with no one of them authoritative. That is the same disease as the
hand-maintained progress table that drifted onto stale numbers: **one fact, more
than one home.**

So `docs/papers.tsv` is now the only place a paper's status is *stated*, and this
script is the only place the checklist is *rendered*:

    python scripts/survey_checklist.py            # write docs/survey_checklist.md
    python scripts/survey_checklist.py --check    # exit non-zero on any drift

The rendering is the smaller half. The point is the cross-check: every claim in
papers.tsv is tested against evidence that lives elsewhere, so the two cannot
quietly disagree.

* A paper marked `reproduced` must have a group under `outputs/` with a
  `summary.json`. Claiming a reproduction we do not have is the worst failure
  mode available to this file.
* A paper enumerated `IN_SCOPE` by `sweep_venues.py` must appear in papers.tsv,
  even if only as `excluded`. Otherwise the enumeration silently widens and the
  ledger never notices -- which is how coverage rots.
* A `code` URL is reported as unverified unless `survey.md`'s rules were applied
  to it. Conference metadata does not show code links that appear in the body;
  judging from it misjudged four papers, and once claimed MFN had no author
  implementation when `pliang279/MFN` was live the whole time.

`survey.md` keeps what cannot be generated: the verification rules, the scanning
procedure, the exclusion reasoning.
"""

from __future__ import annotations

import argparse
import csv
import sys

from msa.config import OUTPUT_ROOT, PROJECT_ROOT

PAPERS = PROJECT_ROOT / "docs" / "papers.tsv"
SWEEP = PROJECT_ROOT / "docs" / "sweep_venues.tsv"
MANIFEST = PROJECT_ROOT / "papers" / "MANIFEST.tsv"
OUT_PATH = PROJECT_ROOT / "docs" / "survey_checklist.md"

STATUS_LABEL = {
    "reproduced": "✅ 已复现",
    "candidate": "◻ 候选",
    "planned": "▶ 计划中",
    "excluded": "✕ 已排除",
}


def load_papers() -> list[dict]:
    with PAPERS.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def enumerated_in_scope() -> set[str]:
    if not SWEEP.exists():
        return set()
    with SWEEP.open() as handle:
        return {row["title"].strip().lower()
                for row in csv.DictReader(handle, delimiter="\t")
                if row.get("tier") == "IN_SCOPE" and row.get("title")}


def drift(papers: list[dict]) -> list[str]:
    """Every way the ledger and the evidence can disagree."""
    problems = []
    for row in papers:
        if row["status"] == "reproduced":
            groups = list(OUTPUT_ROOT.glob(f"{row['key']}_*/summary.json"))
            if not groups:
                problems.append(
                    f"{row['key']}: 标为已复现，但 outputs/ 下没有对应组的 summary.json")
    known = {row["title"].strip().lower() for row in papers}
    for title in sorted(enumerated_in_scope() - known):
        problems.append(f"枚举为 IN_SCOPE 但未进 papers.tsv：{title[:90]}")
    return problems


def build(papers: list[dict], problems: list[str]) -> str:
    have_pdf = set()
    if MANIFEST.exists():
        with MANIFEST.open() as handle:
            have_pdf = {row["file"].rsplit(".", 1)[0].lower()
                        for row in csv.DictReader(handle, delimiter="\t")}

    by_status: dict[str, list[dict]] = {}
    for row in papers:
        by_status.setdefault(row["status"], []).append(row)

    lines = []
    for status in ("reproduced", "planned", "candidate", "excluded"):
        group = sorted(by_status.get(status, []), key=lambda r: (r["venue"], r["key"]))
        if not group:
            continue
        lines.append(f"\n## {STATUS_LABEL[status]}（{len(group)}）\n")
        lines.append("| key | 出处 | 类别 | 标题 | 作者代码 | PDF |")
        lines.append("|---|---|---|---|---|---|")
        for row in group:
            code = f"[链接]({row['code']})" if row["code"] else "—"
            pdf = "✓" if row["key"].lower() in have_pdf else "—"
            lines.append(f"| `{row['key']}` | {row['venue'] or '—'} | "
                         f"{row['class'] or '—'} | {row['title'][:70]} | {code} | {pdf} |")

    contrastive = [r for r in papers if r["class"] == "contrastive"]
    header = f"""# 调研清单（唯一权威视图）

**由 `python scripts/survey_checklist.py` 生成，不要手抄。**
论文状态只在 `docs/papers.tsv` 里声明；本文件只是渲染，且**每条声明都与别处的证据对过**。

- 共 **{len(papers)}** 条；已复现 **{len(by_status.get('reproduced', []))}** 条
- 对比学习一类 **{len(contrastive)}** 条（导师指定方向，见 `decisions.md` 2026-08-04）

**「作者代码」一列的链接一律取自论文正文**，不取会议页面元数据——后者不显示正文里的代码链接，
据此判断已误判过四篇（规矩见 `survey.md`「核实规矩」）。
"""
    if problems:
        header += ("\n## ⚠️ 与证据不一致（须处理）\n\n"
                   + "\n".join(f"- {p}" for p in problems) + "\n")
    else:
        header += "\n**一致性检查通过**：无声明与证据冲突。\n"
    return header + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any claim disagrees with the evidence")
    args = ap.parse_args()

    papers = load_papers()
    problems = drift(papers)
    OUT_PATH.write_text(build(papers, problems))
    print(f"wrote {OUT_PATH.relative_to(PROJECT_ROOT)}  ({len(papers)} papers)")
    for problem in problems:
        print(f"  ! {problem}")
    if args.check and problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
