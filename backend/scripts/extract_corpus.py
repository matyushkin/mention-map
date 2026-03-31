"""Extract a corpus of Russian literary texts from ru.wikisource.org.

Saves each work as a JSON file with metadata and chapter structure.
Respects Wikimedia rate limits (1 request/second).

Usage:
    uv run python scripts/extract_corpus.py [--output-dir ../data]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.wikisource import WikisourceClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Curated list: works with rich character networks, good for mention mapping
WORKS = [
    {
        "page": "Война и мир (Толстой)/Том 1",
        "slug": "war-and-peace-v1",
        "title": "Война и мир. Том 1",
        "author": "Л. Н. Толстой",
        "year": 1869,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Война и мир (Толстой)/Том 2",
        "slug": "war-and-peace-v2",
        "title": "Война и мир. Том 2",
        "author": "Л. Н. Толстой",
        "year": 1869,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Война и мир (Толстой)/Том 3",
        "slug": "war-and-peace-v3",
        "title": "Война и мир. Том 3",
        "author": "Л. Н. Толстой",
        "year": 1869,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Война и мир (Толстой)/Том 4",
        "slug": "war-and-peace-v4",
        "title": "Война и мир. Том 4",
        "author": "Л. Н. Толстой",
        "year": 1869,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Анна Каренина (Толстой)",
        "slug": "anna-karenina",
        "title": "Анна Каренина",
        "author": "Л. Н. Толстой",
        "year": 1877,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Братья Карамазовы (Достоевский)",
        "slug": "brothers-karamazov",
        "title": "Братья Карамазовы",
        "author": "Ф. М. Достоевский",
        "year": 1880,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Преступление и наказание (Достоевский)",
        "slug": "crime-and-punishment",
        "title": "Преступление и наказание",
        "author": "Ф. М. Достоевский",
        "year": 1866,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Мёртвые души (Гоголь)",
        "slug": "dead-souls",
        "title": "Мёртвые души",
        "author": "Н. В. Гоголь",
        "year": 1842,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Евгений Онегин (Пушкин)",
        "slug": "eugene-onegin",
        "title": "Евгений Онегин",
        "author": "А. С. Пушкин",
        "year": 1833,
        "genre": "verse_novel",
        "language": "ru",
    },
    {
        "page": "Герой нашего времени (Лермонтов)",
        "slug": "hero-of-our-time",
        "title": "Герой нашего времени",
        "author": "М. Ю. Лермонтов",
        "year": 1840,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Отцы и дети (Тургенев)",
        "slug": "fathers-and-sons",
        "title": "Отцы и дети",
        "author": "И. С. Тургенев",
        "year": 1862,
        "genre": "novel",
        "language": "ru",
    },
    {
        "page": "Ревизор (Гоголь)",
        "slug": "inspector-general",
        "title": "Ревизор",
        "author": "Н. В. Гоголь",
        "year": 1836,
        "genre": "play",
        "language": "ru",
    },
    {
        "page": "Вишнёвый сад (Чехов)",
        "slug": "cherry-orchard",
        "title": "Вишнёвый сад",
        "author": "А. П. Чехов",
        "year": 1904,
        "genre": "play",
        "language": "ru",
    },
    {
        "page": "Горе от ума (Грибоедов)",
        "slug": "woe-from-wit",
        "title": "Горе от ума",
        "author": "А. С. Грибоедов",
        "year": 1825,
        "genre": "play",
        "language": "ru",
    },
]


def clean_author(author: str) -> str:
    """Remove wiki markup from author string."""
    author = re.sub(r"\[\[([^|\]]+)\|?[^\]]*\]\]", r"\1", author)
    author = re.sub(r"\([\d—]+\)", "", author)
    return author.strip()


def extract_work(client: WikisourceClient, work: dict, output_dir: Path) -> bool:
    slug = work["slug"]
    page = work["page"]
    output_file = output_dir / f"{slug}.json"

    if output_file.exists():
        logger.info("Skipping %s — already extracted", slug)
        return True

    logger.info("Extracting: %s (%s)", work["title"], page)

    try:
        # Get Wikisource metadata
        ws_meta = client.get_metadata(page)

        # Get chapter structure
        subpages = client.get_subpages(page)
        if not subpages:
            subpages = [
                link for link in client.get_links(page)
                if link.startswith(page + "/")
            ]

        chapters = []
        if subpages:
            for i, subpage in enumerate(subpages):
                logger.info(
                    "  [%d/%d] %s", i + 1, len(subpages),
                    subpage.rsplit("/", 1)[-1],
                )
                text = client.get_page_text(subpage)
                if text.strip():
                    chapters.append({
                        "title": subpage.rsplit("/", 1)[-1],
                        "number": i + 1,
                        "page_title": subpage,
                        "text": text,
                        "char_count": len(text),
                    })
        else:
            # Single-page work
            logger.info("  Fetching as single page")
            text = client.get_page_text(page)
            chapters.append({
                "title": work["title"],
                "number": 1,
                "page_title": page,
                "text": text,
                "char_count": len(text),
            })

        full_text = "\n\n".join(ch["text"] for ch in chapters)

        result = {
            "slug": slug,
            "title": work["title"],
            "author": work["author"],
            "year": work["year"],
            "genre": work["genre"],
            "language": work["language"],
            "wikisource_page": page,
            "wikisource_metadata": {
                "title": ws_meta.title,
                "author": clean_author(ws_meta.author),
                "created": ws_meta.created,
                "published": ws_meta.published,
                "source": ws_meta.source,
                "categories": ws_meta.categories,
            },
            "chapters": chapters,
            "stats": {
                "chapter_count": len(chapters),
                "total_chars": len(full_text),
                "total_words": len(full_text.split()),
            },
        }

        output_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "  Saved %s: %d chapters, %d chars",
            slug, len(chapters), len(full_text),
        )
        return True

    except Exception:
        logger.exception("Failed to extract %s", slug)
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract corpus from ru.wikisource.org")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "data",
    )
    parser.add_argument(
        "--works", nargs="*", help="Slugs to extract (default: all)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    works = WORKS
    if args.works:
        works = [w for w in WORKS if w["slug"] in args.works]

    logger.info("Extracting %d works to %s", len(works), args.output_dir)

    with WikisourceClient(delay=1.0) as client:
        results = {"ok": 0, "fail": 0}
        for work in works:
            if extract_work(client, work, args.output_dir):
                results["ok"] += 1
            else:
                results["fail"] += 1

    logger.info("Done: %d ok, %d failed", results["ok"], results["fail"])

    # Write corpus index
    index = []
    for f in sorted(args.output_dir.glob("*.json")):
        if f.name == "index.json":
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        index.append({
            "slug": data["slug"],
            "title": data["title"],
            "author": data["author"],
            "year": data["year"],
            "genre": data["genre"],
            "chapters": data["stats"]["chapter_count"],
            "chars": data["stats"]["total_chars"],
            "words": data["stats"]["total_words"],
            "file": f.name,
        })

    index_file = args.output_dir / "index.json"
    index_file.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Index written: %d works", len(index))


if __name__ == "__main__":
    main()
