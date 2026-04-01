"""Enrich genre field using Wikipedia batch API.

For works with genre='other', searches Russian Wikipedia in batches
of 50 titles, extracts genre from categories and article intro.

Usage:
    uv run --extra hf python scripts/enrich_genres.py --output-dir ../hf_dataset
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

GENRE_KEYWORDS = {
    "poetry": ["поэма", "поэзия", "стихотворение", "стих", "элегия", "баллада", "ода",
                "сонет", "песнь", "гимн", "эпиграмма", "послание", "poem", "поэт"],
    "prose": ["роман", "повесть", "рассказ", "новелла", "очерк", "сказка",
              "притча", "эссе", "записки", "воспоминания", "мемуары", "проза",
              "прозаик", "novel", "story", "tale"],
    "drama": ["пьеса", "комедия", "трагедия", "драма", "водевиль", "сцены",
              "либретто", "сценка", "опера", "драматург", "play"],
    "fable": ["басня", "баснописец", "fable"],
}


def classify_genre(text: str) -> str | None:
    text_lower = text.lower()
    for genre, keywords in GENRE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return genre
    return None


def batch_lookup_genres(titles: list[str], client: httpx.Client) -> dict[str, str]:
    """Look up genres for up to 50 titles in one API call."""
    results = {}
    titles_str = "|".join(titles[:50])

    try:
        resp = client.get(
            "https://ru.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": titles_str,
                "prop": "categories|extracts",
                "clshow": "!hidden",
                "cllimit": "max",
                "exintro": True,
                "explaintext": True,
                "exsentences": 3,
                "format": "json",
            },
        )
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})

        for page in pages.values():
            page_title = page.get("title", "")
            if page.get("missing") is not None:
                continue

            cats = " ".join(c.get("title", "") for c in page.get("categories", []))
            genre = classify_genre(cats)
            if not genre:
                extract = page.get("extract", "")
                genre = classify_genre(extract)

            if genre:
                results[page_title] = genre

    except Exception as e:
        logger.debug("Batch error: %s", e)

    return results


def search_titles(queries: list[tuple[str, str]], client: httpx.Client) -> dict[str, str]:
    """Search Wikipedia for (title, author) pairs, return title → Wikipedia page title mapping."""
    mapping = {}

    # Batch search: use opensearch API which supports one query at a time but is faster
    for title, author in queries:
        clean = re.sub(r"\s*\([^)]*\)\s*", " ", title).strip()
        if len(clean) < 3:
            continue

        try:
            resp = client.get(
                "https://ru.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": clean,
                    "limit": 1,
                    "format": "json",
                },
            )
            data = resp.json()
            if len(data) >= 2 and data[1]:
                wiki_title = data[1][0]
                mapping[f"{author}||{title}"] = wiki_title
        except Exception:
            pass

        time.sleep(0.05)  # Light rate limit for opensearch

    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Max unique queries (0=all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N unique queries")
    args = parser.parse_args()

    out = args.output_dir

    # Collect 'other' genre works, deduplicate by (author, base_title)
    other_works = []
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f, columns=["id", "title", "author", "genre"])
        for i in range(t.num_rows):
            if t.column("genre")[i].as_py() == "other":
                other_works.append({
                    "id": t.column("id")[i].as_py(),
                    "title": t.column("title")[i].as_py() or "",
                    "author": t.column("author")[i].as_py() or "",
                })

    logger.info("Works with genre='other': %d", len(other_works))

    # Deduplicate by (author, base_title)
    seen = {}
    unique_queries = []
    for w in other_works:
        base_title = w["title"].split("/")[0].strip()
        key = (w["author"], base_title)
        if key not in seen:
            seen[key] = None
            unique_queries.append((base_title, w["author"]))

    logger.info("Unique (author, title) to look up: %d", len(unique_queries))

    if args.offset:
        unique_queries = unique_queries[args.offset:]
        logger.info("After offset %d: %d remaining", args.offset, len(unique_queries))
    if args.limit:
        unique_queries = unique_queries[:args.limit]
        logger.info("Limited to %d", args.limit)

    client = httpx.Client(
        headers={"User-Agent": "MentionMap/0.1 (https://github.com/matyushkin/mention-map)"},
        timeout=15.0,
    )

    # Step 1: Search Wikipedia for page titles (opensearch, fast)
    logger.info("Step 1: Searching Wikipedia for page titles...")
    title_mapping = search_titles(unique_queries, client)
    logger.info("Found %d Wikipedia pages", len(title_mapping))

    # Step 2: Batch lookup genres (50 titles per request)
    wiki_titles = list(set(title_mapping.values()))
    logger.info("Step 2: Batch genre lookup for %d Wikipedia pages...", len(wiki_titles))

    genre_results = {}
    for batch_start in range(0, len(wiki_titles), 50):
        batch = wiki_titles[batch_start:batch_start + 50]
        batch_genres = batch_lookup_genres(batch, client)
        genre_results.update(batch_genres)
        time.sleep(0.5)

        if batch_start % 500 == 0 and batch_start > 0:
            logger.info("  %d/%d batches, found %d genres",
                        batch_start // 50, len(wiki_titles) // 50, len(genre_results))

    client.close()
    logger.info("Found genres for %d Wikipedia pages", len(genre_results))

    # Step 3: Map back to works
    # key (author, base_title) → genre
    work_genres = {}
    for w in other_works:
        base_title = w["title"].split("/")[0].strip()
        key = f"{w['author']}||{base_title}"
        if key in title_mapping:
            wiki_title = title_mapping[key]
            if wiki_title in genre_results:
                work_genres[w["id"]] = genre_results[wiki_title]

    logger.info("Works to reclassify: %d", len(work_genres))

    # Step 4: Apply
    reclassified = 0
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f)
        genres = t.column("genre").to_pylist()
        ids = t.column("id").to_pylist()
        changed = False
        for i in range(t.num_rows):
            if ids[i] in work_genres:
                genres[i] = work_genres[ids[i]]
                reclassified += 1
                changed = True
        if changed:
            t = t.set_column(t.schema.get_field_index("genre"), "genre",
                             pa.array(genres, type=pa.string()))
            pq.write_table(t, f, compression="zstd")

    logger.info("Reclassified: %d", reclassified)

    # Stats
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
