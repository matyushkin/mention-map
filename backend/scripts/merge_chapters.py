"""Merge chapter records into complete works.

Many works in the dataset are split into chapters (e.g.,
"Преступление и наказание (Достоевский)/Часть I/Глава I").
This script merges them into single records per work.

Usage:
    uv run --extra hf python scripts/merge_chapters.py --output-dir ../hf_dataset
"""

import argparse
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-chapters", type=int, default=2,
                        help="Minimum chapters to trigger merge")
    args = parser.parse_args()

    out = args.output_dir
    schema = pq.read_schema(str(sorted(out.glob("works-*.parquet"))[0]))

    # Collect all records
    parent_chapters = defaultdict(list)
    standalone = []

    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f)
        for i in range(t.num_rows):
            rec = {col: t.column(col)[i].as_py() for col in t.column_names}
            wid = rec["id"] or ""
            if "/" in wid:
                parent = wid.split("/")[0]
                parent_chapters[parent].append(rec)
            else:
                standalone.append(rec)

    to_merge = {k: v for k, v in parent_chapters.items() if len(v) >= args.min_chapters}
    for k, v in parent_chapters.items():
        if len(v) < args.min_chapters:
            standalone.extend(v)

    logger.info("Standalone: %d, to merge: %d works (%d chapters)",
                len(standalone), len(to_merge), sum(len(v) for v in to_merge.values()))

    # Merge
    merged = []
    for parent, chapters in to_merge.items():
        # Sort by chapter path
        chapters.sort(key=lambda r: natural_sort_key(r["id"]))

        first = chapters[0]
        text_parts = []
        for ch in chapters:
            # Chapter heading from ID suffix
            suffix = ch["id"].split("/", 1)[1] if "/" in ch["id"] else ""
            # Skip version/edition variants
            if re.match(r"^(Версия \d+|ДО|СО|ПСС|Редакци|Изд\.).*$", suffix):
                suffix = ""
            suffix = suffix.strip("/").strip()

            text = ch["text"] or ""
            if suffix and text:
                text_parts.append(f"{suffix}\n\n{text}")
            elif text:
                text_parts.append(text)

        merged_text = "\n\n".join(text_parts)

        rec = dict(first)
        rec["id"] = parent
        rec["text"] = merged_text
        rec["text_length"] = len(merged_text)
        rec["word_count"] = len(merged_text.split())

        # Take best metadata from any chapter
        for ch in chapters:
            if not rec.get("year_written") and ch.get("year_written"):
                rec["year_written"] = ch["year_written"]
            if not rec.get("source") and ch.get("source"):
                rec["source"] = ch["source"]
            if not rec.get("date_text") and ch.get("date_text"):
                rec["date_text"] = ch["date_text"]
            if not rec.get("year_published") and ch.get("year_published"):
                rec["year_published"] = ch["year_published"]

        merged.append(rec)

    logger.info("Merged: %d works", len(merged))

    # Combine
    all_records = standalone + merged
    all_records.sort(key=lambda r: r.get("id", ""))
    logger.info("Total after merge: %d", len(all_records))

    # Remove old files
    for f in sorted(out.glob("works-*.parquet")):
        os.remove(f)

    # Write new files
    batch_size = 5000
    for batch_start in range(0, len(all_records), batch_size):
        batch = all_records[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size
        table = pa.table(
            {f.name: [r.get(f.name) for r in batch] for f in schema},
            schema=schema,
        )
        pq.write_table(table, out / f"works-{batch_num:04d}.parquet", compression="zstd")

    total = sum(pq.read_metadata(str(f)).num_rows for f in sorted(out.glob("works-*.parquet")))
    logger.info("Final: %d records", total)

    # Stats
    big_works = sorted(merged, key=lambda r: -r["text_length"])[:10]
    logger.info("Biggest merged works:")
    for r in big_works:
        logger.info("  %10d chars  %s", r["text_length"], r["id"][:60])


if __name__ == "__main__":
    main()
