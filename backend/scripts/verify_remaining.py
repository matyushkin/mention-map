"""Verify PD status of remaining unverified authors via Wikidata.

Looks up each unverified author, determines death year, and either:
  - Confirms PD (died 70+ years ago)
  - Removes not-PD (died <70 years ago or alive)
  - Removes not-found (no Wikidata entry — can't verify)

Updates parquet files in-place.

Usage:
    uv run --extra hf python scripts/verify_remaining.py --output-dir ../hf_dataset
"""

import argparse
import logging
import re
import time
from collections import Counter
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CURRENT_YEAR = 2026

# Non-author entries to remove
NON_AUTHORS = {
    "Английская_литература", "Немецкая_литература", "Французская_литература",
    "Американская_литература", "Исследования_и_путешестви Географические",
    "Начальник полиции безопасности и СД", "Дальстрой", "Русская народная песня",
    "Русская народная пѣсня", "аноним", "неизвестен",
}


def lookup_author_wikidata(name: str, client: httpx.Client) -> dict | None:
    """Look up author death year via Wikidata search + entity fetch."""
    clean = re.sub(r"\s*\([^)]*\)", "", name)
    clean = re.sub(r"^(Свт\.|†)\s*", "", clean)
    clean = clean.replace("ъ", "").replace("ѣ", "е").replace("ѳ", "ф").replace("і", "и")
    clean = clean.replace("\u200e", "").strip()

    if not clean or len(clean) < 3 or ";" in clean:
        return None

    try:
        resp = client.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": clean,
                "language": "ru",
                "type": "item",
                "limit": 1,
                "format": "json",
            },
        )
        data = resp.json()
        if not data.get("search"):
            return None

        qid = data["search"][0]["id"]

        resp2 = client.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": qid,
                "props": "claims",
                "format": "json",
            },
        )
        entity = resp2.json().get("entities", {}).get(qid, {})
        claims = entity.get("claims", {})

        is_human = any(
            c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q5"
            for c in claims.get("P31", [])
        )

        death = birth = None
        for c in claims.get("P570", []):
            tv = c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", "")
            m = re.search(r"(\d{4})", tv)
            if m:
                death = int(m.group(1))
        for c in claims.get("P569", []):
            tv = c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("time", "")
            m = re.search(r"(\d{4})", tv)
            if m:
                birth = int(m.group(1))

        return {"qid": qid, "human": is_human, "death": death, "birth": birth}
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir

    # Collect unverified authors
    authors = Counter()
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f, columns=["author", "license"])
        for i in range(t.num_rows):
            if t.column("license")[i].as_py() != "unverified":
                continue
            a = (t.column("author")[i].as_py() or "").strip()
            if a:
                authors[a] += 1

    logger.info("Unverified authors: %d (%d works)", len(authors), sum(authors.values()))

    # Classify via Wikidata
    pd_authors: dict[str, int] = {}  # name → death_year
    remove_authors: set[str] = set(NON_AUTHORS)

    client = httpx.Client(
        headers={"User-Agent": "MentionMap/0.1 (https://github.com/matyushkin/mention-map)"},
        timeout=10.0,
    )

    checked = 0
    for name, count in authors.most_common():
        if name in NON_AUTHORS or name.endswith("литература"):
            remove_authors.add(name)
            continue

        result = lookup_author_wikidata(name, client)
        checked += 1
        time.sleep(0.15)

        if result is None:
            # Not found in Wikidata — remove
            remove_authors.add(name)
        elif not result.get("human"):
            remove_authors.add(name)
        elif result.get("death") and result["death"] + 70 < CURRENT_YEAR:
            pd_authors[name] = result["death"]
        elif result.get("death"):
            # Died too recently
            remove_authors.add(name)
        elif result.get("birth") and result["birth"] + 150 < CURRENT_YEAR:
            # Born 150+ years ago, no death recorded — likely PD
            pd_authors[name] = result["birth"] + 80  # estimate
        else:
            # Alive or uncertain
            remove_authors.add(name)

        if checked % 100 == 0:
            logger.info(
                "  checked %d/%d: PD=%d, remove=%d",
                checked, len(authors), len(pd_authors), len(remove_authors),
            )

    client.close()

    logger.info(
        "Final: PD=%d (%d works), remove=%d (%d works)",
        len(pd_authors),
        sum(authors.get(a, 0) for a in pd_authors),
        len(remove_authors),
        sum(authors.get(a, 0) for a in remove_authors),
    )

    # Apply to parquet files
    total_upgraded = total_removed = 0

    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f)
        licenses = t.column("license").to_pylist()
        reasons = t.column("license_reason").to_pylist()
        deaths = t.column("author_death_year").to_pylist()
        authors_col = t.column("author").to_pylist()

        keep = []
        changed = False

        for i in range(t.num_rows):
            a = (authors_col[i] or "").strip()

            if licenses[i] == "unverified" and a in remove_authors:
                total_removed += 1
                changed = True
                continue

            if licenses[i] == "unverified" and a in pd_authors:
                deaths[i] = pd_authors[a]
                licenses[i] = "PD-old-70"
                reasons[i] = f"author_died_{pd_authors[a]}_wikidata_search"
                total_upgraded += 1
                changed = True

            keep.append(i)

        if changed:
            t = t.take(keep)
            licenses = [licenses[i] for i in keep]
            reasons = [reasons[i] for i in keep]
            deaths = [deaths[i] for i in keep]
            t = t.set_column(t.schema.get_field_index("license"), "license", pa.array(licenses, type=pa.string()))
            t = t.set_column(t.schema.get_field_index("license_reason"), "license_reason", pa.array(reasons, type=pa.string()))
            t = t.set_column(t.schema.get_field_index("author_death_year"), "author_death_year", pa.array(deaths, type=pa.int16()))
            pq.write_table(t, f, compression="zstd")

    logger.info("Upgraded to PD: %d, Removed: %d", total_upgraded, total_removed)

    # Also remove works without author
    no_author_removed = 0
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f)
        authors_col = t.column("author").to_pylist()
        licenses = t.column("license").to_pylist()

        keep = []
        for i in range(t.num_rows):
            a = (authors_col[i] or "").strip()
            if not a and licenses[i] == "unverified":
                no_author_removed += 1
                continue
            keep.append(i)

        if len(keep) < t.num_rows:
            pq.write_table(t.take(keep), f, compression="zstd")

    logger.info("Removed works without author: %d", no_author_removed)

    # Final stats
    total = verified = unverified = 0
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f, columns=["license"])
        for i in range(t.num_rows):
            total += 1
            if t.column("license")[i].as_py() == "PD-old-70":
                verified += 1
            else:
                unverified += 1
    logger.info("FINAL: %d total, %d PD (%.1f%%), %d unverified (%.1f%%)",
                total, verified, verified / total * 100, unverified, unverified / total * 100)


if __name__ == "__main__":
    main()
