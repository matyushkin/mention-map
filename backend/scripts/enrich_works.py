"""Post-processing: enrich works with author data and PD validation.

Matches works to authors by name, fills in death years,
and recomputes license status.

Usage:
    uv run --extra hf python scripts/enrich_works.py --output-dir ../hf_dataset
"""

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CURRENT_YEAR = 2026


def normalize_name(name: str) -> str:
    """Normalize author name for fuzzy matching."""
    # Remove wiki markup leftovers
    name = re.sub(r"\[\[[^\]]*\]\]", "", name)
    name = re.sub(r"\|[^|]*=", "", name)
    name = re.sub(r"<[^>]+>", "", name)
    name = re.sub(r"\{\{[^}]*\}\}", "", name)
    # Remove parenthetical notes (but keep text before)
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    # Remove prefix like "~предполож."
    name = re.sub(r"^[~≈]?\s*предполож\.?\s*", "", name)
    # Normalize unicode
    name = unicodedata.normalize("NFC", name)
    # Lowercase, collapse whitespace
    name = " ".join(name.lower().split())
    # Remove «ё» → «е» ambiguity
    name = name.replace("ё", "е")
    # Normalize doreform orthography → modern
    name = name.replace("ъ", "").replace("ѣ", "е").replace("ѳ", "ф").replace("і", "и")
    # Normalize common transliteration variants
    name = name.replace("й", "й")  # combining breve normalization
    return name.strip()


# Common name variants that should match
NAME_ALIASES = {
    "вильям шекспир": "уильям шекспир",
    "вильям шекспиръ": "уильям шекспир",
    "шекспир": "уильям шекспир",
    "артур конан дойль": "артур конан дойл",
    "жюль верн": "жюль верн",
    "м е салтыков щедрин": "михаил евграфович салтыков-щедрин",
    "м. е. салтыков щедрин": "михаил евграфович салтыков-щедрин",
    "салтыков-щедрин": "михаил евграфович салтыков-щедрин",
    "салтыков щедрин": "михаил евграфович салтыков-щедрин",
    "максим горький": "максим горький",
    "горький": "максим горький",
    "ильф и петров": "ильф и петров",
    "льюис кэрролл": "льюис кэрролл",
    "льюис керролл": "льюис кэрролл",
    "марк твен": "марк твен",
    "о генри": "о. генри",
    "г х андерсен": "ханс кристиан андерсен",
    "ганс христиан андерсен": "ханс кристиан андерсен",
}


def build_author_index(authors_path: Path) -> dict:
    """Build normalized_name → {author_id, death_year, birth_year} index."""
    t = pq.read_table(authors_path)
    index = {}

    for i in range(t.num_rows):
        aid = t.column("author_id")[i].as_py()
        name = t.column("name")[i].as_py()
        death = t.column("death_year")[i].as_py()
        birth = t.column("birth_year")[i].as_py()

        if not name:
            continue

        entry = {"author_id": aid, "death_year": death, "birth_year": birth}

        # Index by full normalized name
        norm = normalize_name(name)
        if norm:
            index[norm] = entry

        # Also index by "Автор:Name" page title (without prefix)
        page_name = aid.removeprefix("Автор:")
        norm_page = normalize_name(page_name)
        if norm_page and norm_page != norm:
            index[norm_page] = entry

        # Index by family_name + given_names
        family = t.column("family_name")[i].as_py() or ""
        given = t.column("given_names")[i].as_py() or ""
        if family:
            norm_family = normalize_name(family)

            # "Фамилия Имя Отчество" (reversed order)
            if given:
                reversed_name = normalize_name(f"{family} {given}")
                if reversed_name not in index:
                    index[reversed_name] = entry

            # Index by family name + first given name only
            # "Толстой Лев" for matching "Лев Николаевич Толстой"
            if given:
                first_given = given.split()[0] if given else ""
                short = normalize_name(f"{family} {first_given}")
                if short not in index:
                    index[short] = entry
                short_rev = normalize_name(f"{first_given} {family}")
                if short_rev not in index:
                    index[short_rev] = entry

        # Also add dehyphenated version of name
        dehyph = norm.replace("-", " ")
        dehyph = " ".join(dehyph.split())
        if dehyph != norm and dehyph not in index:
            index[dehyph] = entry

    # Add reverse aliases
    for alias, canonical in NAME_ALIASES.items():
        norm_alias = normalize_name(alias)
        norm_canonical = normalize_name(canonical)
        if norm_canonical in index and norm_alias not in index:
            index[norm_alias] = index[norm_canonical]

    return index


def match_author(author_str: str, index: dict) -> dict | None:
    """Try to match an author string to the index."""
    if not author_str:
        return None

    norm = normalize_name(author_str)
    if not norm or len(norm) < 3:
        return None

    # Exact match
    if norm in index:
        return index[norm]

    # Try alias
    if norm in NAME_ALIASES:
        alias = normalize_name(NAME_ALIASES[norm])
        if alias in index:
            return index[alias]

    # Try removing "Автор:" prefix
    if norm.startswith("автор:"):
        cleaned = norm.removeprefix("автор:").strip()
        if cleaned in index:
            return index[cleaned]

    # Try removing hyphens/dashes (Салтыков-Щедрин → Салтыков Щедрин)
    dehyphenated = norm.replace("-", " ").replace("—", " ")
    dehyphenated = " ".join(dehyphenated.split())
    if dehyphenated != norm and dehyphenated in index:
        return index[dehyphenated]

    # Try family name only match (last word) — only if unambiguous
    words = norm.split()
    if len(words) >= 2:
        # Try "Фамилия Имя" → "Имя Фамилия" and vice versa
        reversed_name = " ".join(words[1:] + words[:1])
        if reversed_name in index:
            return index[reversed_name]

    return None


def compute_license(death_year: int | None, license_template: str) -> tuple[str, str]:
    """Compute license and reason."""
    if death_year and death_year + 70 < CURRENT_YEAR:
        return "PD-old-70", f"author_died_{death_year}"
    if license_template and license_template != "unverified":
        return license_template, "wikisource_template"
    if death_year:
        return "unverified", f"author_died_{death_year}_too_recent"
    return "unverified", "no_death_year"


def enrich_works(output_dir: Path):
    authors_path = output_dir / "authors.parquet"
    if not authors_path.exists():
        logger.error("authors.parquet not found")
        return

    logger.info("Building author index...")
    index = build_author_index(authors_path)
    logger.info("Author index: %d entries", len(index))

    works_files = sorted(output_dir.glob("works-*.parquet"))
    if not works_files:
        logger.error("No works files found")
        return

    stats = {"total": 0, "matched": 0, "upgraded": 0, "already_verified": 0}

    for wf in works_files:
        t = pq.read_table(wf)
        authors = t.column("author").to_pylist()
        author_ids = t.column("author_id").to_pylist()
        birth_years = t.column("author_birth_year").to_pylist()
        death_years = t.column("author_death_year").to_pylist()
        licenses = t.column("license").to_pylist()
        reasons = t.column("license_reason").to_pylist()

        changed = False
        for i in range(t.num_rows):
            stats["total"] += 1

            if licenses[i] == "PD-old-70":
                stats["already_verified"] += 1
                continue

            # Try to match author
            match = match_author(authors[i], index)
            if not match:
                continue

            stats["matched"] += 1

            # Fill in missing data
            if not author_ids[i]:
                author_ids[i] = match["author_id"]
                changed = True
            if not birth_years[i] and match["birth_year"]:
                birth_years[i] = match["birth_year"]
                changed = True
            if not death_years[i] and match["death_year"]:
                death_years[i] = match["death_year"]
                changed = True

            # Recompute license
            old_license = licenses[i]
            new_license, new_reason = compute_license(
                death_years[i],
                old_license if old_license != "unverified" else "",
            )
            if new_license != old_license:
                licenses[i] = new_license
                reasons[i] = new_reason
                stats["upgraded"] += 1
                changed = True

        if changed:
            # Rebuild table with updated columns
            t = t.set_column(
                t.schema.get_field_index("author_id"),
                "author_id", pa.array(author_ids, type=pa.string()),
            )
            t = t.set_column(
                t.schema.get_field_index("author_birth_year"),
                "author_birth_year", pa.array(birth_years, type=pa.int16()),
            )
            t = t.set_column(
                t.schema.get_field_index("author_death_year"),
                "author_death_year", pa.array(death_years, type=pa.int16()),
            )
            t = t.set_column(
                t.schema.get_field_index("license"),
                "license", pa.array(licenses, type=pa.string()),
            )
            t = t.set_column(
                t.schema.get_field_index("license_reason"),
                "license_reason", pa.array(reasons, type=pa.string()),
            )
            pq.write_table(t, wf, compression="zstd")
            logger.info("Updated %s", wf.name)

    logger.info(
        "Done: %d total, %d matched to authors, %d license upgraded, %d already verified",
        stats["total"], stats["matched"], stats["upgraded"], stats["already_verified"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    enrich_works(args.output_dir)


if __name__ == "__main__":
    main()
