"""Build a HuggingFace dataset of Russian literary texts from ru.wikisource.org.

Crawls categories (poetry by year, prose, drama) to discover works,
extracts structured metadata + clean text, and saves as Parquet.
Supports resuming from a checkpoint.

Usage:
    uv run --extra hf python scripts/build_hf_dataset.py
    uv run --extra hf python scripts/build_hf_dataset.py --resume
    uv run --extra hf python scripts/build_hf_dataset.py --push matyushkin/ru-wikisource-literature
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.wikisource import WikisourceClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Output paths ─────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "hf_dataset"
CHECKPOINT_FILE = "checkpoint.json"

# ── Categories to crawl ──────────────────────────────────────

POETRY_YEAR_RANGE = range(1700, 1930)

GENRE_CATEGORIES = {
    "poetry": [f"Поэзия_{y}_года" for y in POETRY_YEAR_RANGE],
    "prose": [
        "Романы",
        "Повести",
        "Рассказы",
        "Новеллы",
        "Очерки",
    ],
    "drama": [
        "Пьесы",
        "Трагедии",
        "Комедии",
        "Драмы",
    ],
    "fable": ["Басни"],
}

# Pages to skip (meta pages, disambiguation, etc.)
SKIP_PREFIXES = (
    "Категория:", "Автор:", "Обсуждение:", "Викитека:",
    "Участник:", "Шаблон:", "Справка:", "Медиавики:",
)

# ── Schema ───────────────────────────────────────────────────

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("title", pa.string()),
    ("author", pa.string()),
    ("author_birth_year", pa.int16()),
    ("author_death_year", pa.int16()),
    ("year_written", pa.int16()),
    ("year_published", pa.int16()),
    ("genre", pa.string()),
    ("text", pa.string()),
    ("text_length", pa.int32()),
    ("word_count", pa.int32()),
    ("source", pa.string()),
    ("categories", pa.list_(pa.string())),
    ("interwiki", pa.list_(pa.string())),
    ("quality", pa.string()),
    ("license", pa.string()),
    ("wikisource_page", pa.string()),
])


# ── Metadata parsing ────────────────────────────────────────

def parse_author_years(author_str: str) -> tuple[int | None, int | None]:
    """Extract birth/death years from author string like 'Автор (1799—1837)'."""
    match = re.search(r"\((\d{3,4})\s*[—–-]\s*(\d{3,4})\)", author_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"\((\d{3,4})\s*[—–-]\s*\)", author_str)
    if match:
        return int(match.group(1)), None
    return None, None


def parse_year(value: str) -> int | None:
    """Extract a year from strings like '1826', '1826 год', '[[1826]]'."""
    if not value:
        return None
    match = re.search(r"\d{3,4}", value)
    return int(match.group()) if match else None


def clean_wikilink(text: str) -> str:
    """Remove wiki markup: [[link|display]] → display, [[link]] → link."""
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    return text.strip()


def detect_license(categories: list[str]) -> str:
    """Detect license from category names."""
    for cat in categories:
        if "PD-old" in cat or "PD-Russia" in cat:
            return cat
    return "PD-old"


LICENSE_NOISE = re.compile(
    r"(?:"
    r"Поскольку Российская Федерация|"
    r"Исключительное право на это произведение|"
    r"Если произведение является переводом|"
    r"Это произведение было опубликовано|"
    r"Это произведение не охраняется|"
    r"Данное произведение|"
    r"Public domain|"
    r"Общественное достояние|"
    r"не является полным правопреемником|"
    r"Срок действия исключительного авторского права|"
    r"Это произведение перешло в общественное достояние"
    r").*?(?=\n\n|\Z)",
    re.DOTALL,
)


def strip_license_text(text: str) -> str:
    """Remove license boilerplate that leaks through HTML cleanup."""
    return LICENSE_NOISE.sub("", text).strip()


def detect_quality(categories: list[str]) -> str:
    """Detect text quality from categories (75%, 100%, etc.)."""
    for cat in categories:
        if re.match(r"^\d+%$", cat):
            return cat
    return ""


# ── Interwiki extraction ─────────────────────────────────────

def get_interwiki(client: WikisourceClient, page_title: str) -> list[str]:
    """Get interwiki language links for a page."""
    data = client._api_get(
        action="parse",
        page=page_title,
        prop="iwlinks",
    )
    links = data.get("parse", {}).get("iwlinks", [])
    return list({link.get("prefix", "") for link in links if link.get("prefix", "")})


# ── Main crawler ─────────────────────────────────────────────

class DatasetBuilder:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = output_dir / CHECKPOINT_FILE
        self.seen: set[str] = set()
        self.records: list[dict] = []
        self.batch_num = 0
        self._load_checkpoint()

    def _load_checkpoint(self):
        if self.checkpoint_path.exists():
            data = json.loads(self.checkpoint_path.read_text())
            self.seen = set(data.get("seen", []))
            self.batch_num = data.get("batch_num", 0)
            logger.info(
                "Resumed from checkpoint: %d pages seen, batch %d",
                len(self.seen), self.batch_num,
            )

    def _save_checkpoint(self):
        self.checkpoint_path.write_text(json.dumps({
            "seen": sorted(self.seen),
            "batch_num": self.batch_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, indent=2))

    def _flush_batch(self):
        """Write accumulated records as a Parquet file."""
        if not self.records:
            return

        table = pa.table({
            field.name: [r.get(field.name) for r in self.records]
            for field in SCHEMA
        }, schema=SCHEMA)

        out_path = self.output_dir / f"data-{self.batch_num:04d}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        logger.info(
            "Wrote batch %d: %d records → %s (%.1f MB)",
            self.batch_num, len(self.records), out_path.name,
            out_path.stat().st_size / 1e6,
        )
        self.records = []
        self.batch_num += 1
        self._save_checkpoint()

    def process_page(self, client: WikisourceClient, page_title: str, genre: str):
        """Extract a single work and add to the batch."""
        if page_title in self.seen:
            return
        if any(page_title.startswith(p) for p in SKIP_PREFIXES):
            return

        self.seen.add(page_title)

        try:
            # Get metadata
            meta = client.get_metadata(page_title)

            # Get text
            text = strip_license_text(client.get_page_text(page_title))
            if not text or len(text.strip()) < 50:
                logger.debug("Skipping %s — too short", page_title)
                return

            author_raw = clean_wikilink(meta.author)
            birth, death = parse_author_years(meta.author)

            # Get interwiki links
            iwlinks = get_interwiki(client, page_title)

            record = {
                "id": page_title,
                "title": clean_wikilink(meta.title) if meta.title != page_title else page_title,
                "author": re.sub(r"\s*\(\d.*\)", "", author_raw),
                "author_birth_year": birth,
                "author_death_year": death,
                "year_written": parse_year(meta.created),
                "year_published": parse_year(meta.published),
                "genre": genre,
                "text": text,
                "text_length": len(text),
                "word_count": len(text.split()),
                "source": clean_wikilink(meta.source),
                "categories": [
                    c for c in meta.categories
                    if not c.startswith("Статьи") and not c.startswith("Ссылка")
                    and not c.startswith("Викиданные") and not c.startswith("Страницы")
                ],
                "interwiki": iwlinks,
                "quality": detect_quality(meta.categories),
                "license": detect_license(meta.categories),
                "wikisource_page": page_title,
            }
            self.records.append(record)

            if len(self.records) >= 500:
                self._flush_batch()

        except Exception:
            logger.exception("Failed to process %s", page_title)

    def crawl_category(
        self, client: WikisourceClient, category: str, genre: str,
    ):
        """Crawl all pages in a category."""
        logger.info("Crawling category: %s (genre=%s)", category, genre)
        try:
            members = client.get_category_members(category, limit=500)
        except Exception:
            logger.warning("Category not found: %s", category)
            return

        pages = [
            m["title"] for m in members
            if not m["title"].startswith("Категория:")
        ]
        logger.info("  Found %d pages in %s", len(pages), category)

        for i, page in enumerate(pages):
            if page in self.seen:
                continue
            logger.info(
                "  [%d/%d] %s", i + 1, len(pages), page,
            )
            self.process_page(client, page, genre)

    def finalize(self):
        """Flush remaining records and remove checkpoint."""
        self._flush_batch()
        logger.info(
            "Dataset complete: %d total pages, %d batches",
            len(self.seen), self.batch_num,
        )


def push_to_hub(output_dir: Path, repo_id: str):
    """Combine all Parquet files and push to HuggingFace Hub."""
    from datasets import Dataset

    parquet_files = sorted(output_dir.glob("data-*.parquet"))
    if not parquet_files:
        logger.error("No parquet files found in %s", output_dir)
        return

    tables = [pq.read_table(f) for f in parquet_files]
    combined = pa.concat_tables(tables)
    ds = Dataset(combined)

    logger.info("Pushing %d rows to %s", len(ds), repo_id)
    ds.push_to_hub(repo_id, private=False)
    logger.info("Done! Dataset available at https://huggingface.co/datasets/%s", repo_id)


def main():
    parser = argparse.ArgumentParser(
        description="Build HF dataset from ru.wikisource.org",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--push", type=str, metavar="REPO_ID",
        help="Push to HuggingFace Hub (e.g., matyushkin/ru-wikisource-literature)",
    )
    parser.add_argument(
        "--genres", nargs="*",
        choices=list(GENRE_CATEGORIES.keys()),
        default=list(GENRE_CATEGORIES.keys()),
        help="Genres to crawl (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Records per Parquet file (default: 500)",
    )
    args = parser.parse_args()

    if args.push:
        push_to_hub(args.output_dir, args.push)
        return

    if not args.resume:
        checkpoint = args.output_dir / CHECKPOINT_FILE
        if checkpoint.exists():
            logger.warning(
                "Checkpoint exists at %s. Use --resume to continue or delete it to start fresh.",
                checkpoint,
            )
            return

    builder = DatasetBuilder(args.output_dir)

    # delay=2.0 is conservative; Wikimedia has no hard limit for reads
    # but recommends sequential requests. maxlag=5 backs off when servers are busy.
    with WikisourceClient(delay=2.0, maxlag=5) as client:
        for genre in args.genres:
            categories = GENRE_CATEGORIES[genre]
            for category in categories:
                builder.crawl_category(client, category, genre)
                builder._save_checkpoint()

        builder.finalize()

    logger.info("Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
