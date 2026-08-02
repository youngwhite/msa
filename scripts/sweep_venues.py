"""Enumerate every paper title in a set of venues, then filter for this task.

Keyword search cannot promise completeness: reword the query and a paper
disappears with no signal that it did. This takes every title in every volume
and filters locally, so the only way to miss something is a filter you can read.

    python scripts/sweep_venues.py                  # all venues, refresh cache
    python scripts/sweep_venues.py --offline        # reuse the cached HTML
    python scripts/sweep_venues.py --venues acl     # one family

Three things this had to learn the hard way, all of which are silent failures:

* **Nested tags truncate titles.** ACL wraps acronyms in
  `<span class=acl-fixed-case>`, so matching a title as `[^<]+` stops at the
  first inner tag and drops every paper whose title contains an acronym. The
  first version returned fifteen plausible hits and had lost DEAR and MoLAN.
* **Word boundaries matter.** `mosi` without `\b` matches "MoSiC", an ICCV paper
  on self-supervised motion trajectories.
* **A title cannot establish the task.** D2R (EMNLP 2024) and Beyond Static
  Alignment (ACL 2026) both say "Multimodal Sentiment" and both evaluate on
  MVSA/HFM — image-text social media sentiment, not tri-modal video regression.
  Enumeration fixes *missing*; only reading the datasets fixes *wrongly
  included*, so IN_SCOPE here means "worth checking", not "confirmed".

Every run prints the enumeration total. A total that drops is an extractor that
broke, not a quiet year — check it before trusting the hit list.

ACM MM is absent on purpose: dl.acm.org answers 403 to scripted requests, so it
needs manual or institutional access. That is recorded in docs/survey.md as an
un-enumerated venue rather than silently omitted.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "docs" / "sweep_2024_2026.tsv"
DEFAULT_CACHE = Path("/tmp/msa_sweep_cache")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

#: ACL Anthology volumes. Rolling three years plus Findings, where a lot of
#: this task's work lands.
ACL_VOLUMES = [
    f"{year}.{track}"
    for year in (2024, 2025, 2026)
    for track in ("acl-long", "emnlp-main", "naacl-long", "findings-acl", "findings-emnlp")
]
#: CVF conferences. ICCV is biennial, WACV annual, CVPR annual.
CVF_CONFERENCES = ["CVPR2024", "CVPR2025", "CVPR2026", "ICCV2023", "ICCV2025",
                   "WACV2024", "WACV2025", "WACV2026"]

BROAD = re.compile(
    r"multimodal sentiment|multi-modal sentiment|multimodal emotion|multi-modal emotion"
    r"|\bmosi\b|\bmosei\b|multimodal affect|sentiment analysis|emotion recognition", re.I)
#: This project's task: tri-modal sentiment regression on the MOSI/MOSEI family.
NARROW = re.compile(r"multimodal sentiment|multi-modal sentiment|\bmosi\b|\bmosei\b", re.I)
#: Shares the vocabulary, different benchmark: aspect-based sentiment is
#: text-only, and emotion recognition in conversation uses IEMOCAP/MELD.
EXCLUDE = re.compile(r"aspect-based|aspect-term|in conversation|conversational", re.I)

#: ACL: <strong><a class=align-middle href=/ID/>TITLE</a></strong>, unquoted attrs.
ACL_ENTRY = re.compile(r"<strong><a class=align-middle href=/([^>]+)>(.*?)</a></strong>", re.S)
#: CVF: <dt class="ptitle"><br><a href="...">TITLE</a>
CVF_ENTRY = re.compile(r'<dt class="ptitle">\s*<br>\s*<a href="([^"]+)">(.*?)</a>', re.S)
#: AAAI (OJS): <h3 class="title"><a ...>TITLE</a>
AAAI_ENTRY = re.compile(r'class="title"[^>]*>\s*<a[^>]*>(.*?)</a>', re.S)
AAAI_ISSUE = re.compile(r'issue/view/(\d+)"[^>]*>\s*([^<]{4,90})')


def fetch(url: str, target: Path, offline: bool) -> str:
    """Cache a page. CVF needs a browser UA and serves gzip; OJS is large."""
    if target.exists() and (offline or target.stat().st_size > 4096):
        return target.read_text(errors="ignore")
    if offline:
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sL", "--compressed", "--max-time", "120",
                    "-A", UA, url, "-o", str(target)], check=False)
    return target.read_text(errors="ignore") if target.exists() else ""


def strip_tags(raw: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def tier_of(title: str) -> str | None:
    if not BROAD.search(title):
        return None
    if NARROW.search(title):
        return "adjacent" if EXCLUDE.search(title) else "IN_SCOPE"
    return "other_task"


def sweep_acl(cache: Path, offline: bool) -> tuple[int, list[tuple[str, str, str]]]:
    total, rows = 0, []
    for volume in ACL_VOLUMES:
        page = fetch(f"https://aclanthology.org/volumes/{volume}/",
                     cache / "acl" / f"{volume}.html", offline)
        for paper_id, raw in ACL_ENTRY.findall(page):
            total += 1
            title = strip_tags(raw)
            tier = tier_of(title)
            if tier:
                rows.append((tier, volume, title, paper_id.rstrip("/")))
    return total, rows


def sweep_cvf(cache: Path, offline: bool) -> tuple[int, list[tuple[str, str, str]]]:
    total, rows = 0, []
    for conference in CVF_CONFERENCES:
        # Recent editions split by day and only ?day=all lists everything.
        page = fetch(f"https://openaccess.thecvf.com/{conference}?day=all",
                     cache / "cvf" / f"{conference}.html", offline)
        if page.count("ptitle") < 50 and not offline:
            page = fetch(f"https://openaccess.thecvf.com/{conference}",
                         cache / "cvf" / f"{conference}-plain.html", offline)
        for href, raw in CVF_ENTRY.findall(page):
            total += 1
            title = strip_tags(raw)
            tier = tier_of(title)
            if tier:
                rows.append((tier, conference, title, href))
    return total, rows


def sweep_aaai(cache: Path, offline: bool) -> tuple[int, list[tuple[str, str, str]]]:
    """AAAI proceedings are one OJS issue per technical track, tens per year."""
    issues: dict[str, str] = {}
    for page_number in (1, 2, 3, 4):
        page = fetch(f"https://ojs.aaai.org/index.php/AAAI/issue/archive/{page_number}",
                     cache / "aaai" / f"archive-{page_number}.html", offline)
        for issue_id, label in AAAI_ISSUE.findall(page):
            label = html.unescape(label).strip()
            if re.search(r"AAAI-2[456]", label):
                issues[issue_id] = label
    total, rows = 0, []
    for issue_id, label in issues.items():
        page = fetch(f"https://ojs.aaai.org/index.php/AAAI/issue/view/{issue_id}",
                     cache / "aaai" / f"issue-{issue_id}.html", offline)
        for raw in AAAI_ENTRY.findall(page):
            total += 1
            title = strip_tags(raw)
            tier = tier_of(title)
            if tier:
                rows.append((tier, label.split()[0], title, f"issue/{issue_id}"))
    return total, rows


SWEEPS = {"acl": sweep_acl, "cvf": sweep_cvf, "aaai": sweep_aaai}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--venues", nargs="*", default=list(SWEEPS),
                        choices=list(SWEEPS))
    parser.add_argument("--offline", action="store_true",
                        help="reuse cached HTML, fetch nothing")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    all_rows, grand_total = [], 0
    for name in args.venues:
        total, rows = SWEEPS[name](args.cache, args.offline)
        grand_total += total
        all_rows.extend((name, *row) for row in rows)
        in_scope = sum(1 for r in rows if r[0] == "IN_SCOPE")
        print(f"{name:5s} enumerated {total:6d}  hits {len(rows):3d}  in scope {in_scope:3d}")
        if total < 500:
            print(f"  !! {name} enumerated only {total}; the extractor is probably "
                  f"broken, not the venue empty")

    all_rows.sort()
    args.out.write_text(
        "venue_family\ttier\tvenue\ttitle\tpaper_id\n"
        + "\n".join("\t".join(row) for row in all_rows) + "\n")
    total_in_scope = sum(1 for r in all_rows if r[1] == "IN_SCOPE")
    print(f"\nenumerated {grand_total} papers, wrote {len(all_rows)} hits "
          f"({total_in_scope} in scope) to {args.out.relative_to(PROJECT_ROOT)}")
    print("ACM MM not swept: dl.acm.org returns 403 to scripted requests "
          "(see docs/survey.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
