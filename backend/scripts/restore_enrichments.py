"""Restore enriched fields from overlay files after a rebuild.

If build_hf_from_dump.py overwrites works parquet with raw data,
this script merges back the enrichments (license, genre, source, etc.)
from overlays that are preserved separately.

Usage:
    uv run --extra hf python scripts/restore_enrichments.py --output-dir ../hf_dataset
"""

import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir
    overlays = out / "overlays"

    enrichments_path = overlays / "enrichments.parquet"
    if not enrichments_path.exists():
        logger.error("No enrichments overlay found at %s", enrichments_path)
        logger.error("Run build_annotations.py or manually create overlays first")
        return

    # Load enrichments
    enr = pq.read_table(enrichments_path)
    enr_map = {}
    for i in range(enr.num_rows):
        wid = enr.column("id")[i].as_py()
        enr_map[wid] = {
            "license": enr.column("license")[i].as_py() or "",
            "license_reason": enr.column("license_reason")[i].as_py() or "",
            "genre": enr.column("genre")[i].as_py() or "",
            "source": enr.column("source")[i].as_py() or "",
            "year_written": enr.column("year_written")[i].as_py(),
        }
    logger.info("Loaded enrichments for %d works", len(enr_map))

    # Apply to works
    restored = 0
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f)
        ids = t.column("id").to_pylist()

        fields_to_restore = ["license", "license_reason", "genre", "source", "year_written"]
        new_cols = {field: list(t.column(field).to_pylist()) if field in t.column_names else [""] * t.num_rows
                    for field in fields_to_restore}

        changed = False
        for i, wid in enumerate(ids):
            if wid in enr_map:
                e = enr_map[wid]
                for field in fields_to_restore:
                    old_val = new_cols[field][i]
                    new_val = e[field]
                    # Prefer enriched value if it's more filled
                    if new_val and (not old_val or old_val == "unverified" or old_val == "other"):
                        new_cols[field][i] = new_val
                        changed = True
                        restored += 1

        if changed:
            for field in fields_to_restore:
                if field in t.column_names:
                    dtype = t.schema.field(field).type
                    idx = t.schema.get_field_index(field)
                    t = t.set_column(idx, field, pa.array(new_cols[field], type=dtype))
            pq.write_table(t, f, compression="zstd")

    logger.info("Restored %d field values", restored)

    # Restore authors
    authors_overlay = overlays / "authors_enriched.parquet"
    if authors_overlay.exists():
        import shutil
        shutil.copy(authors_overlay, out / "authors.parquet")
        logger.info("Restored authors from overlay")

    # Restore annotations
    ann_overlay = overlays / "annotations.parquet"
    if ann_overlay.exists():
        import shutil
        shutil.copy(ann_overlay, out / "annotations.parquet")
        logger.info("Restored annotations from overlay")


if __name__ == "__main__":
    main()
