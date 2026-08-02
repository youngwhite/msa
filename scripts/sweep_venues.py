"""Enumerate every paper title in a venue-year and filter for this task.

Keyword search cannot promise completeness: reword the query and a paper
disappears with no signal that it did. This takes every title and filters
locally, so the only way to miss something is a filter you can read.

    python scripts/sweep_venues.py                    # DBLP + CVF, 2017-2026
    python scripts/sweep_venues.py --from-year 2024   # recent only
    python scripts/sweep_venues.py --offline          # reuse the cache

**DBLP is the primary source** and one parser covers ACL, EMNLP, NAACL, CVPR,
ICCV, ECCV, WACV, AAAI, IJCAI, ICML, ICLR, NeurIPS and ACM MM. Reaching it
through each publisher's own site instead cost a bespoke extractor per venue --
ninety-seven OJS issue pages for AAAI, gzip and `?day=all` for CVF -- and ACM's
DL answers 403 to scripts at all, which is what made ACM MM look unreachable
until DBLP turned out to have it for free.

**DBLP lags, so it cannot be the only source.** CVPR 2026 is published on CVF
and absent from DBLP, and its eight in-scope papers would vanish from a
DBLP-only sweep. CVF is therefore kept as a direct fallback for editions DBLP
has not indexed. An access barrier is a reason to find another route, never a
reason to drop coverage.

Three silent failure modes, each recorded with the case that produced it:

* **Nested tags truncate titles.** ACL wraps acronyms in
  `<span class=acl-fixed-case>`; matching a title as `[^<]+` drops every paper
  whose title contains an acronym. The first version returned fifteen plausible
  hits having lost DEAR and MoLAN.
* **Word boundaries matter.** `mosi` without `\b` matches MoSiC, an ICCV paper
  on motion trajectories.
* **A title cannot establish the task.** D2R and Beyond Static Alignment both
  say "Multimodal Sentiment" and both evaluate on MVSA/HFM -- image-text social
  media sentiment. IN_SCOPE means "worth checking", never "confirmed"; only the
  datasets in the body settle it. Enumeration fixes *missing* and does nothing
  about *wrongly included*.

Every run prints the enumeration total per venue. A total that drops means the
extractor broke, not that a year was quiet.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "docs" / "sweep_venues.tsv"
DEFAULT_CACHE = Path("/tmp/msa_sweep_cache")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

#: Venues to sweep. The per-year volume keys are DISCOVERED from each venue's
#: DBLP index rather than guessed: guessing cost 59 missing venue-years on the
#: first run, including emnlp2019 through emnlp2022 and acl2020, because the
#: main proceedings are keyed emnlp2019-1 in some years and emnlp2023 in others.
#: A guessed key that 404s looks exactly like a year with no papers.
DBLP_VENUES = ["acl", "emnlp", "naacl", "cvpr", "iccv", "eccv", "wacv",
               "aaai", "ijcai", "mm", "icml", "iclr", "nips"]
#: Volume suffixes that are not research tracks: demos, industry, tutorials,
#: student research workshops. Kept out so the enumeration total means papers.
NON_RESEARCH = re.compile(r"(?:\d)(?:d|i|t|s|w)$")
#: CVF editions DBLP has not indexed yet. Checked at every sweep, not assumed.
CVF_FALLBACK = ["CVPR2026", "WACV2026", "ICCV2025"]

BROAD = re.compile(
    r"multimodal sentiment|multi-modal sentiment|multimodal emotion|multi-modal emotion"
    r"|\bmosi\b|\bmosei\b|multimodal affect|sentiment analysis|emotion recognition", re.I)
NARROW = re.compile(r"multimodal sentiment|multi-modal sentiment|\bmosi\b|\bmosei\b", re.I)
EXCLUDE = re.compile(r"aspect-based|aspect-term|in conversation|conversational", re.I)

DBLP_TITLE = re.compile(r'<span class="title" itemprop="name">(.*?)</span>', re.S)
CVF_ENTRY = re.compile(r'<dt class="ptitle">\s*<br>\s*<a href="([^"]+)">(.*?)</a>', re.S)


def fetch(url: str, target: Path, offline: bool) -> str:
    if target.exists() and (offline or target.stat().st_size > 4096):
        return target.read_text(errors="ignore")
    if offline:
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sL", "--compressed", "--max-time", "150",
                    "-A", UA, url, "-o", str(target)], check=False)
    return target.read_text(errors="ignore") if target.exists() else ""


def strip_tags(raw: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip().rstrip(".")


def tier_of(title: str) -> str | None:
    if not BROAD.search(title):
        return None
    if NARROW.search(title):
        return "adjacent" if EXCLUDE.search(title) else "IN_SCOPE"
    return "other_task"


def discover_keys(venue: str, years: range, cache: Path, offline: bool) -> list[str]:
    """Read a venue's real volume keys off its DBLP index instead of guessing."""
    page = fetch(f"https://dblp.org/db/conf/{venue}/index.html",
                 cache / "dblp" / f"index-{venue}.html", offline)
    found = set(re.findall(rf"conf/{venue}/({venue}\d{{4}}[a-z0-9-]*)\.html", page))
    keys = []
    for key in sorted(found):
        year = re.search(r"\d{4}", key)
        if year and int(year.group()) in years and not NON_RESEARCH.search(key):
            keys.append(key)
    return keys


def sweep_dblp(years: range, cache: Path, offline: bool) -> tuple[int, list, list[str]]:
    total, rows, empty = 0, [], []
    for venue in DBLP_VENUES:
        for key in discover_keys(venue, years, cache, offline):
            page = fetch(f"https://dblp.org/db/conf/{venue}/{key}.html",
                         cache / "dblp" / f"{key}.html", offline)
            titles = DBLP_TITLE.findall(page)
            if not titles:
                empty.append(key)
                continue
            total += len(titles)
            for raw in titles:
                title = strip_tags(raw)
                tier = tier_of(title)
                if tier:
                    rows.append(("dblp", tier, key, title))
    return total, rows, empty


def sweep_cvf(cache: Path, offline: bool) -> tuple[int, list]:
    """Direct from CVF, for editions DBLP has not indexed."""
    total, rows = 0, []
    for conference in CVF_FALLBACK:
        page = fetch(f"https://openaccess.thecvf.com/{conference}?day=all",
                     cache / "cvf" / f"{conference}.html", offline)
        for _href, raw in CVF_ENTRY.findall(page):
            total += 1
            title = strip_tags(raw)
            tier = tier_of(title)
            if tier:
                rows.append(("cvf", tier, conference, title))
    return total, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-year", type=int, default=2017,
                        help="MSA as a field starts around TFN (2017)")
    parser.add_argument("--to-year", type=int, default=2026)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    years = range(args.from_year, args.to_year + 1)
    dblp_total, dblp_rows, missing = sweep_dblp(years, args.cache, args.offline)
    cvf_total, cvf_rows = sweep_cvf(args.cache, args.offline)

    rows = sorted(set(dblp_rows + cvf_rows))
    # A paper indexed by both sources appears once per source under different
    # venue keys; de-duplicate on the title so the count means papers.
    seen, unique = set(), []
    for source, tier, venue, title in rows:
        norm = re.sub(r"\W+", "", title.lower())
        if norm in seen:
            continue
        seen.add(norm)
        unique.append((source, tier, venue, title))

    in_scope = [r for r in unique if r[1] == "IN_SCOPE"]
    print(f"dblp  enumerated {dblp_total:6d}  ({len(missing)} venue-years absent)")
    print(f"cvf   enumerated {cvf_total:6d}  (editions DBLP has not indexed)")
    if dblp_total < 5000:
        print("  !! DBLP total looks low; the extractor is probably broken")
    args.out.write_text("source\ttier\tvenue\ttitle\n"
                        + "\n".join("\t".join(r) for r in unique) + "\n")
    print(f"\n{len(unique)} unique hits, {len(in_scope)} in scope -> "
          f"{args.out.relative_to(PROJECT_ROOT)}")
    if missing:
        print("volumes that returned no titles (check before trusting): "
              + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
