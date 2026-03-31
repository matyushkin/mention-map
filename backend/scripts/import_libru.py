"""Import texts from a local lib.ru (Moshkov) archive into the dataset.

Parses .txt.html files, extracts author+title from <title> tags,
cleans HTML, unwraps hard-wrapped prose lines, and matches to
the existing Wikisource-based catalogue.

Usage:
    uv run --extra hf python scripts/import_libru.py \
        --libru-dir /Users/leo/Downloads/lib.ru.28.03.09/book \
        --dataset-dir ../hf_dataset
"""

import argparse
import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Directories that contain Russian literature (public domain)
LITERARY_DIRS = {
    "LITRA",       # Russian classical literature
    "RUSSLIT",     # More Russian literature
    "POEZIQ",      # Poetry
    "POEEAST",     # More poetry
    "GOGOL", "BUNIN",  # Some authors are top-level
}

# Skip these — not literary texts or not public domain
SKIP_DIRS = {
    "COMPULIB", "CISCO", "CHESS", "BRIDGE", "ASTROLOGY",
    "ALPINISM", "TURIZM", "AWARDS", "BIBLIOGR", "ACKNOWLEDGEMENT",
    "COPYRIGHT", "SONGS", "KSP", "AQUARIUM",
}

# Hard line-wrap detection: if many lines cluster near the same length,
# the text is hard-wrapped prose and needs unwrapping.
# lib.ru uses various widths: 64, 72, 75, 80 chars.
WRAP_DETECT_THRESHOLD = 0.3  # 30% of lines near any wrap width → prose wrapping


# ── HTML parsing ─────────────────────────────────────────────

def parse_libru_file(path: Path) -> dict | None:
    """Parse a lib.ru .txt.html file into {author, title, text}."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None

    # Detect encoding
    text = None
    for enc in ("utf-8", "koi8-r", "cp1251"):
        try:
            text = raw.decode(enc)
            # Verify it looks like Russian text
            if re.search(r"[а-яА-ЯёЁ]{3,}", text):
                break
            text = None
        except (UnicodeDecodeError, ValueError):
            continue
    if not text:
        return None

    # Extract title
    title_match = re.search(r"<title>([^<]+)</title>", text, re.IGNORECASE)
    if not title_match:
        return None

    raw_title = title_match.group(1).strip()

    # Parse "Автор. Название" pattern
    # Handle variations: "И.А.Бунин. Господин из Сан-Франциско"
    #                    "Федор Михайлович Достоевский. Бесы"
    author, title = "", raw_title
    dot_match = re.match(r"^(.+?)\.\s+(.+)$", raw_title)
    if dot_match:
        candidate_author = dot_match.group(1).strip()
        candidate_title = dot_match.group(2).strip()
        # Verify author looks like a name (has Cyrillic, not too long)
        if re.search(r"[А-ЯЁ]", candidate_author) and len(candidate_author) < 80:
            author = candidate_author
            title = candidate_title

    # Extract body text (inside <pre>...</pre> or after <hr>)
    # Remove HTML tags but preserve structure
    body = text

    # Remove everything before the first <hr> (header/metadata)
    hr_parts = re.split(r"<hr\s*/?>", body, flags=re.IGNORECASE)
    if len(hr_parts) > 1:
        body = "<hr>".join(hr_parts[1:])  # keep everything after first <hr>

    # Preserve paragraph structure: <ul>, <h*>, <p> → double newlines
    body = re.sub(r"<ul[^>]*>", "\n\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</ul>", "\n\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n\n\1\n\n", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<p[^>]*>", "\n\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</p>", "\n\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    # Remove remaining tags
    body = re.sub(r"<a[^>]*>", "", body)
    body = re.sub(r"</a>", "", body)
    body = re.sub(r"</?pre[^>]*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"</?font[^>]*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"</?[a-zA-Z][^>]*>", "", body)

    # Decode HTML entities
    body = body.replace("&nbsp;", " ")
    body = body.replace("&amp;", "&")
    body = body.replace("&lt;", "<")
    body = body.replace("&gt;", ">")
    body = body.replace("&quot;", '"')
    body = body.replace("&laquo;", "«")
    body = body.replace("&raquo;", "»")
    body = body.replace("&#160;", " ")

    # Remove footnote markers
    body = re.sub(r"\(\d+\)", "", body)

    body = body.strip()
    if len(body) < 100:
        return None

    # Unwrap hard-wrapped lines for prose
    body = unwrap_prose(body)

    # Final cleanup
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = body.strip()

    return {
        "author": author,
        "title": title,
        "text": body,
        "file": str(path),
    }


def unwrap_prose(text: str) -> str:
    """Rejoin hard-wrapped lines in prose texts.

    lib.ru texts are often wrapped at ~75-80 chars.
    Poetry lines are shorter and shouldn't be joined.

    Strategy: detect if text is hard-wrapped, then join lines within
    paragraphs while preserving paragraph breaks (empty lines).
    """
    lines = text.split("\n")
    if len(lines) < 10:
        return text

    # Detect hard wrapping: check if many lines cluster near any common width
    content_lines = [l for l in lines if l.strip()]
    if not content_lines:
        return text

    lengths = [len(l) for l in content_lines]
    # Check common wrap widths: 60-80
    near_wrap = 0
    for w in range(60, 85):
        count = sum(1 for ln in lengths if w - 3 <= ln <= w + 3)
        near_wrap = max(near_wrap, count)
    wrap_ratio = near_wrap / len(content_lines)

    if wrap_ratio < WRAP_DETECT_THRESHOLD:
        # Not hard-wrapped (poetry, or short text) — keep as is
        return text

    # Rejoin hard-wrapped lines within paragraphs.
    # Paragraph boundaries in lib.ru prose:
    #   1. Empty lines
    #   2. Lines starting with 5+ spaces (indented first line of paragraph)
    #   3. Short lines followed by indented lines (end of paragraph)
    paragraphs = []
    current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Empty line = paragraph boundary
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append("")
        elif line.startswith("     ") and current:
            # Indented line with existing paragraph = new paragraph
            paragraphs.append(" ".join(current))
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        paragraphs.append(" ".join(current))

    # Collapse multiple spaces left by hard-wrap alignment
    result = "\n".join(paragraphs)
    result = re.sub(r"  +", " ", result)
    return result


# ── Matching ─────────────────────────────────────────────────

def normalize_for_match(s: str) -> str:
    """Normalize a string for fuzzy matching."""
    s = s.lower()
    s = s.replace("ё", "е")
    # Remove initials like "И.А." or "Ф.М."
    s = re.sub(r"\b[а-яa-z]\.\s*", "", s)
    # Remove punctuation
    s = re.sub(r"[.,;:!?\"'«»()—\-]", " ", s)
    s = " ".join(s.split())
    return s.strip()


def build_works_index(dataset_dir: Path) -> dict[str, dict]:
    """Build (normalized_author, normalized_title) → work_id index."""
    index = {}
    for f in sorted(dataset_dir.glob("works-*.parquet")):
        t = pq.read_table(f, columns=["id", "author", "title", "text_length"])
        for i in range(t.num_rows):
            author = normalize_for_match(t.column("author")[i].as_py() or "")
            title = normalize_for_match(t.column("title")[i].as_py() or "")
            if author and title:
                key = f"{author}||{title}"
                index[key] = {
                    "id": t.column("id")[i].as_py(),
                    "text_length": t.column("text_length")[i].as_py(),
                }
    return index


# ── Main ─────────────────────────────────────────────────────

def import_libru(libru_dir: Path, dataset_dir: Path):
    logger.info("Scanning lib.ru archive at %s", libru_dir)

    # Collect all .txt.html files from literary directories
    files = []
    for subdir in sorted(libru_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        # Include literary dirs and their subdirs
        if name in LITERARY_DIRS:
            for f in subdir.rglob("*.txt.html"):
                files.append(f)
        elif name in SKIP_DIRS:
            continue
        else:
            # Top-level author dirs — include if they have .txt.html files
            for f in subdir.glob("*.txt.html"):
                files.append(f)

    logger.info("Found %d .txt.html files", len(files))

    # Parse all files
    parsed = []
    for f in files:
        result = parse_libru_file(f)
        if result and result["author"]:
            parsed.append(result)

    logger.info("Parsed %d files with author+title", len(parsed))

    # Build index of existing works
    logger.info("Building works index...")
    works_index = build_works_index(dataset_dir)
    logger.info("Works index: %d entries", len(works_index))

    # Match and find gaps
    matched = 0
    improved = 0
    new_texts = []

    for p in parsed:
        norm_author = normalize_for_match(p["author"])
        norm_title = normalize_for_match(p["title"])
        key = f"{norm_author}||{norm_title}"

        if key in works_index:
            matched += 1
            existing = works_index[key]
            # lib.ru text is longer → could improve
            if len(p["text"]) > existing["text_length"] * 1.2:
                improved += 1
        else:
            new_texts.append(p)

    logger.info("Matched to existing: %d", matched)
    logger.info("Could improve (lib.ru text >20%% longer): %d", improved)
    logger.info("New texts not in Wikisource: %d", len(new_texts))

    # Show sample of new texts
    if new_texts:
        logger.info("Sample new texts:")
        for p in new_texts[:20]:
            logger.info("  %s — %s (%d chars)", p["author"], p["title"], len(p["text"]))

    # Save new texts as a separate parquet for review
    if new_texts:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_hf_from_dump import WORK_SCHEMA
        records = []
        for p in new_texts:
            records.append({
                "id": f"libru:{Path(p['file']).stem}",
                "title": p["title"],
                "author": p["author"],
                "author_id": "",
                "author_birth_year": None,
                "author_death_year": None,
                "year_written": None,
                "year_published": None,
                "genre": "other",
                "text": p["text"],
                "text_length": len(p["text"]),
                "word_count": len(p["text"].split()),
                "source": p["file"],
                "categories": [],
                "interwiki": [],
                "quality": "",
                "license": "unverified",
                "license_reason": "libru_no_metadata",
                "wikisource_page": "",
            })

        table = pa.table(
            {f.name: [r.get(f.name) for r in records] for f in WORK_SCHEMA},
            schema=WORK_SCHEMA,
        )
        out_path = dataset_dir / "libru-new.parquet"
        pq.write_table(table, out_path, compression="zstd")
        logger.info("Saved %d new texts → %s (%.1f MB)",
                     len(records), out_path.name, out_path.stat().st_size / 1e6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libru-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    import_libru(args.libru_dir, args.dataset_dir)


if __name__ == "__main__":
    main()
