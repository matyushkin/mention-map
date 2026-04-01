"""Enrich genre field using Wikipedia article lookup.

For each work with genre='other', searches Russian Wikipedia for the work
title + author, extracts genre from the article infobox or categories.

Usage:
    uv run --extra hf python scripts/enrich_genres.py --output-dir ../hf_dataset
"""

import argparse
import logging
import re
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GENRE_KEYWORDS = {
    "poetry": ["поэма", "стихотворение", "стих", "элегия", "баллада", "ода",
                "сонет", "песнь", "гимн", "эпиграмма", "послание", "poem"],
    "prose": ["роман", "повесть", "рассказ", "новелла", "очерк", "сказка",
              "притча", "эссе", "записки", "воспоминания", "мемуары", "проза",
              "novel", "story", "tale"],
    "drama": ["пьеса", "комедия", "трагедия", "драма", "водевиль", "сцены",
              "либретто", "сценка", "опера", "play"],
    "fable": ["басня", "fable"],
}


def classify_genre(text: str) -> str | None:
    """Classify genre from Wikipedia text/categories."""
    text_lower = text.lower()
    for genre, keywords in GENRE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return genre
    return None


def lookup_wikipedia_genre(title: str, author: str, client: httpx.Client) -> str | None:
    """Search Wikipedia for a work and extract genre."""
    # Clean title for search
    clean_title = re.sub(r"\s*\([^)]*\)\s*", " ", title).strip()
    clean_author = re.sub(r"\s*\([^)]*\)\s*", " ", author).strip()

    # Try search with title + author
    query = f"{clean_title} {clean_author}"

    try:
        resp = client.get(
            "https://ru.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            },
        )
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return None

        page_title = results[0]["title"]

        # Get page categories + extract
        resp2 = client.get(
            "https://ru.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": page_title,
                "prop": "categories|extracts",
                "clshow": "!hidden",
                "cllimit": 20,
                "exintro": True,
                "explaintext": True,
                "exsentences": 3,
                "format": "json",
            },
        )
        data2 = resp2.json()
        pages = data2.get("query", {}).get("pages", {})

        for page in pages.values():
            # Check categories
            cats = " ".join(
                c.get("title", "")
                for c in page.get("categories", [])
            )
            genre = classify_genre(cats)
            if genre:
                return genre

            # Check extract
            extract = page.get("extract", "")
            genre = classify_genre(extract)
            if genre:
                return genre

    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max works to check (0=all)")
    args = parser.parse_args()

    out = args.output_dir

    # Collect 'other' genre works
    other_works = []
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f, columns=["id", "title", "author", "genre"])
        for i in range(t.num_rows):
            if t.column("genre")[i].as_py() == "other":
                other_works.append({
                    "file": f,
                    "id": t.column("id")[i].as_py(),
                    "title": t.column("title")[i].as_py() or "",
                    "author": t.column("author")[i].as_py() or "",
                })

    logger.info("Works with genre='other': %d", len(other_works))

    if args.limit:
        other_works = other_works[:args.limit]
        logger.info("Limited to %d", args.limit)

    # Deduplicate by (author, title) — many chapters share same work
    seen = {}
    unique_queries = []
    for w in other_works:
        # Use parent title (before /)
        base_title = w["title"].split("/")[0].strip()
        key = (w["author"], base_title)
        if key not in seen:
            seen[key] = None
            unique_queries.append((w["author"], base_title, key))

    logger.info("Unique (author, title) to look up: %d", len(unique_queries))

    # Lookup genres from Wikipedia
    client = httpx.Client(
        headers={"User-Agent": "MentionMap/0.1 (https://github.com/matyushkin/mention-map)"},
        timeout=10.0,
    )

    checked = 0
    found = 0
    for author, title, key in unique_queries:
        genre = lookup_wikipedia_genre(title, author, client)
        if genre:
            seen[key] = genre
            found += 1
        checked += 1
        time.sleep(0.2)  # Respect Wikipedia rate limits

        if checked % 200 == 0:
            logger.info("  checked %d/%d, found genres: %d",
                        checked, len(unique_queries), found)

    client.close()
    logger.info("Wikipedia lookup done: %d checked, %d genres found", checked, found)

    # Apply to dataset
    # Build full map: for each work, find genre by (author, base_title)
    genre_map = {}
    for w in other_works:
        base_title = w["title"].split("/")[0].strip()
        key = (w["author"], base_title)
        if seen.get(key):
            genre_map[w["id"]] = seen[key]

    logger.info("Works to reclassify: %d", len(genre_map))

    reclassified = 0
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f)
        genres = t.column("genre").to_pylist()
        ids = t.column("id").to_pylist()
        changed = False
        for i in range(t.num_rows):
            if ids[i] in genre_map:
                genres[i] = genre_map[ids[i]]
                reclassified += 1
                changed = True
        if changed:
            t = t.set_column(
                t.schema.get_field_index("genre"), "genre",
                pa.array(genres, type=pa.string()),
            )
            pq.write_table(t, f, compression="zstd")

    logger.info("Reclassified: %d", reclassified)

    # Final stats
    from collections import Counter
    final = Counter()
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f, columns=["genre"])
        for i in range(t.num_rows):
            final[t.column("genre")[i].as_py()] += 1
    total = sum(final.values())
    for g, c in final.most_common():
        logger.info("  %s: %d (%.1f%%)", g, c, c / total * 100)


if __name__ == "__main__":
    main()
