"""Export dataset sample as JSON for the HTML viewer.

Usage:
    uv run --extra hf python scripts/export_viewer_data.py --output-dir ../hf_dataset [--limit 500]
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500,
                        help="Max records to export (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.output_dir
    records = []

    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f)
        cols = t.column_names
        for i in range(t.num_rows):
            rec = {}
            for col in cols:
                val = t.column(col)[i].as_py()
                if col == "text" and val and len(val) > 10000:
                    val = val[:10000] + f"\n\n[...обрезано, всего {len(t.column(col)[i].as_py())} символов...]"
                if col == "text_tagged" and val and len(val) > 12000:
                    val = val[:12000] + f"\n\n[...обрезано...]"
                rec[col] = val
            records.append(rec)

    print(f"Total records: {len(records)}")

    if args.limit and args.limit < len(records):
        random.seed(args.seed)
        # Ensure we include some drama with tags
        dramas = [r for r in records if r.get("text_tagged")]
        others = [r for r in records if not r.get("text_tagged")]
        n_drama = min(len(dramas), args.limit // 5)
        n_other = args.limit - n_drama
        selected = random.sample(dramas, n_drama) + random.sample(others, min(n_other, len(others)))
        random.shuffle(selected)
        records = selected
        print(f"Sampled: {len(records)} (incl. {n_drama} drama with tags)")

    out_path = out / "viewer_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Written: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
