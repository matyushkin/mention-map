"""Import Pushdom Dataverse corpora into the dataset.

Usage:
    uv run --extra hf python scripts/import_pushdom.py --dataset-dir ../hf_dataset
"""

import argparse
import csv
import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hf_from_dump import WORK_SCHEMA
from import_libru import build_works_index, normalize_for_match


def read_text(path: Path) -> str | None:
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1251", "koi8-r"):
        try:
            text = raw.decode(enc)
            if re.search(r"[а-яА-ЯёЁ]{3,}", text):
                return text.strip()
        except (UnicodeDecodeError, ValueError):
            pass
    return None


def import_xix_prose(corpus_dir: Path, dataset_dir: Path) -> list[dict]:
    """Import Pushdom XIX century prose corpus."""
    metadata_file = corpus_dir / "metadata.tab"
    texts_dir = corpus_dir / "texts"

    if not metadata_file.exists():
        logger.error("metadata.tab not found in %s", corpus_dir)
        return []

    works_index = build_works_index(dataset_dir)
    records = []

    with open(metadata_file, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        for row in reader:
            filename = row["filename"].strip('"')
            raw_author = row["author"].strip('"')
            # "Фамилия, Имя Отчество" → "Имя Отчество Фамилия"
            parts = raw_author.split(", ", 1)
            author = f"{parts[1]} {parts[0]}" if len(parts) == 2 else raw_author

            title = row["title"].strip('"')
            birth = int(row["author_birth_year"].strip('"')) if row["author_birth_year"].strip('"').isdigit() else None
            death = int(row["author_death_year"].strip('"')) if row["author_death_year"].strip('"').isdigit() else None
            year = int(row["year"].strip('"')) if row["year"].strip('"').isdigit() else None

            txt_path = texts_dir / filename
            if not txt_path.exists():
                continue
            text = read_text(txt_path)
            if not text or len(text) < 100:
                continue

            norm_a = normalize_for_match(author)
            norm_t = normalize_for_match(title)
            if f"{norm_a}||{norm_t}" in works_index:
                continue

            records.append({
                "id": f"pushdom:{filename}",
                "title": title,
                "author": author,
                "author_id": "",
                "author_birth_year": birth,
                "author_death_year": death,
                "year_written": year,
                "year_published": None,
                "genre": "prose",
                "text": text,
                "text_length": len(text),
                "word_count": len(text.split()),
                "source": f"pushdom:doi:10.31860/openlit-2020.10-C004/{filename}",
                "categories": [],
                "interwiki": [],
                "quality": "",
                "license": "PD-old-70",
                "license_reason": f"author_died_{death}" if death else "pushdom_xix_century",
                "is_translation": False,
                "translator": "",
                "wikisource_page": "",
            })

    return records


def save_records(records: list[dict], dataset_dir: Path, label: str):
    if not records:
        return
    table = pa.table(
        {f.name: [r.get(f.name) for r in records] for f in WORK_SCHEMA},
        schema=WORK_SCHEMA,
    )
    existing = sorted(dataset_dir.glob("works-*.parquet"))
    last_num = int(existing[-1].stem.split("-")[1]) if existing else -1
    new_path = dataset_dir / f"works-{last_num + 1:04d}.parquet"
    pq.write_table(table, new_path, compression="zstd")
    total = sum(pq.read_metadata(str(f)).num_rows for f in sorted(dataset_dir.glob("works-*.parquet")))
    logger.info("%s: %d records → %s, total dataset: %d", label, len(records), new_path.name, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("/private/tmp/pushdom"))
    args = parser.parse_args()

    records = import_xix_prose(args.corpus_dir, args.dataset_dir)
    logger.info("New texts: %d", len(records))

    from collections import Counter
    for a, c in Counter(r["author"] for r in records).most_common(15):
        logger.info("  %3d  %s", c, a)

    save_records(records, args.dataset_dir, "pushdom_xix")


if __name__ == "__main__":
    main()
