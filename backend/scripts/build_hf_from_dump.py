"""Build a HuggingFace dataset from a Wikimedia XML dump of ru.wikisource.org.

Single-pass streaming parser that extracts:
  - authors.parquet  — author catalogue with metadata
  - works data files — literary texts with structured metadata

Usage:
    # Download dump and build dataset
    uv run --extra hf python scripts/build_hf_from_dump.py

    # Use a previously downloaded dump
    uv run --extra hf python scripts/build_hf_from_dump.py --dump-file /path/to/dump.xml.bz2

    # Enrich with Wikidata after building
    uv run --extra hf python scripts/build_hf_from_dump.py --enrich-wikidata

    # Push to HuggingFace Hub
    uv run --extra hf python scripts/build_hf_from_dump.py --push matyushkin/ru-wikisource-literature
"""

import argparse
import bz2
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "hf_dataset"
DUMP_URL = "https://dumps.wikimedia.org/ruwikisource/latest/ruwikisource-latest-pages-articles.xml.bz2"
DUMP_FILENAME = "ruwikisource-latest-pages-articles.xml.bz2"
MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"

MAX_OUTPUT_BYTES = 100 * 1024 * 1024 * 1024  # 100 GB

# ── Schemas ──────────────────────────────────────────────────

AUTHOR_SCHEMA = pa.schema([
    ("author_id", pa.string()),        # page title (Автор:Имя Фамилия)
    ("name", pa.string()),             # full name
    ("family_name", pa.string()),
    ("given_names", pa.string()),
    ("birth_year", pa.int16()),
    ("death_year", pa.int16()),
    ("description", pa.string()),      # e.g. "русский поэт и прозаик"
    ("wikidata_id", pa.string()),      # Q-number, filled by Wikidata enrichment
    ("works_count", pa.int32()),       # filled after works extraction
    ("categories", pa.list_(pa.string())),
])

WORK_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("title", pa.string()),
    ("author", pa.string()),
    ("author_id", pa.string()),        # link to authors table
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
    ("license_reason", pa.string()),
    ("wikisource_page", pa.string()),
])

# ── Skip rules ───────────────────────────────────────────────

SKIP_NS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
           "12", "13", "14", "15", "100", "101", "103",
           "828", "829"}
# Note: ns=102 is "Автор:" namespace — we keep it for author extraction

NONLITERARY_PREFIXES = (
    "Закон ", "Федеральный закон ", "Указ ", "Приказ ", "Постановление ",
    "Распоряжение ", "Решение ", "Информационное письмо ",
    "ЭСБЕ/", "ТСД/", "БСЭ1/", "РСКД/", "МЭСБЕ/", "ПБЭ/", "ВЭ/", "НЭС/",
    "Конституция ", "Кодекс ", "Устав ",
    "Донесение ", "Спецсообщение ", "Директива ",
    "Постановление ГКО", "Приказ НКО",
)

CURRENT_YEAR = 2026


def _is_nonliterary(title: str) -> bool:
    return any(title.startswith(p) for p in NONLITERARY_PREFIXES)


# ── Wikitext parsing ─────────────────────────────────────────

def parse_template_params(wikitext: str, template_name: str) -> dict[str, str]:
    """Extract parameters from a wikitext template like {{Отексте|...}}."""
    pattern = rf"\{{\{{\s*{re.escape(template_name)}\s*\n?(.*?)\}}\}}"
    match = re.search(pattern, wikitext, re.DOTALL)
    if not match:
        alt_name = template_name[0].lower() + template_name[1:]
        pattern = rf"\{{\{{\s*{re.escape(alt_name)}\s*\n?(.*?)\}}\}}"
        match = re.search(pattern, wikitext, re.DOTALL)
    if not match:
        return {}

    params = {}
    for param_match in re.finditer(
        r"\|\s*([А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z_]*)\s*=\s*(.*?)(?=\n\s*\||$)",
        match.group(1), re.DOTALL,
    ):
        params[param_match.group(1).strip()] = param_match.group(2).strip()
    return params


def clean_wikitext(text: str) -> str:
    """Convert wikitext to plain text."""
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\{\{[^{}]*\{\{[^{}]*\}\}[^{}]*\}\}", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:Категория|Category):[^\]]+\]\]", "", text)
    text = re.sub(r"\[\[[a-z]{2,3}:[^\]]+\]\]", "", text)
    for old, new in [
        ("&nbsp;", " "), ("&#160;", " "), ("&mdash;", "—"),
        ("&laquo;", "«"), ("&raquo;", "»"), ("&amp;", "&"),
        ("&lt;", "<"), ("&gt;", ">"),
    ]:
        text = text.replace(old, new)
    # Remove leftover "Категория:..." lines (without brackets)
    text = re.sub(r"^Категория:[^\n]+$", "", text, flags=re.MULTILINE)
    # Remove __NOEDITSECTION__ and similar magic words
    text = re.sub(r"__[A-Z]+__", "", text)
    # Remove leftover section headers (== Title ==)
    text = re.sub(r"^=+\s*[^=\n]+\s*=+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _unwrap_literary_templates(text: str) -> str:
    """Convert literary templates to plain text BEFORE removing all templates.

    Handles: {{f1|title|text|}}, {{poem|title|text}}, {{poem-on|title}},
    {{лесенка|line1|line2|...}}, {{стих|text}}, etc.
    """
    # FIRST: unwrap inner templates that appear inside poem/f1 blocks
    # {{лесенка|line1|line2|...}} → join lines with spaces
    def _unwrap_lesenka(m: re.Match) -> str:
        parts = m.group(1).split("|")
        return " ".join(p.strip() for p in parts if p.strip())
    text = re.sub(
        r"\{\{[Лл]есенка\|(.*?)\}\}", _unwrap_lesenka, text,
    )

    # THEN: unwrap outer poem/f1 containers
    # {{f1|Title|body text|}} → body text
    text = re.sub(
        r"\{\{f1\|[^|]*\|(.*?)\|\}\}", r"\1", text, flags=re.DOTALL,
    )
    # {{poem|Title|body text}} → body text
    text = re.sub(
        r"\{\{poem\|[^|]*\|(.*?)\}\}", r"\1", text, flags=re.DOTALL,
    )
    # {{poem-on|Title}} → remove
    text = re.sub(r"\{\{poem-on\|([^}]*)\}\}", "", text)
    # {{poem-off}} → remove
    text = re.sub(r"\{\{poem-off\}\}", "", text)
    # {{стих|text}} → text
    text = re.sub(r"\{\{стих\|(.*?)\}\}", r"\1", text, flags=re.DOTALL)
    # {{center|text}} → text
    text = re.sub(r"\{\{center\|(.*?)\}\}", r"\1", text, flags=re.DOTALL)
    # {{right|text}} → text
    text = re.sub(r"\{\{right\|(.*?)\}\}", r"\1", text, flags=re.DOTALL)
    # {{smaller|text}} → text
    text = re.sub(r"\{\{smaller\|(.*?)\}\}", r"\1", text, flags=re.DOTALL)
    # {{larger|text}} → text
    text = re.sub(r"\{\{larger\|(.*?)\}\}", r"\1", text, flags=re.DOTALL)
    # {{разрядка|text}} → text
    text = re.sub(r"\{\{разрядка\|(.*?)\}\}", r"\1", text, flags=re.DOTALL)
    # <poem>text</poem> → text
    text = re.sub(r"<poem>(.*?)</poem>", r"\1", text, flags=re.DOTALL)
    return text


def extract_clean_body(wikitext: str) -> str:
    # Remove header/metadata templates
    body = re.sub(
        r"\{\{(?:Отексте|Обавторе|Об авторе|header|Header)[^{}]*(?:\{\{[^{}]*\}\}[^{}]*)*\}\}",
        "", wikitext, flags=re.DOTALL | re.IGNORECASE,
    )
    body = re.sub(
        r"\{\{(?:PD|pd|ОД|Public domain|Лицензия|License|simple|аудиостатья)[^{}]*\}\}",
        "", body, flags=re.IGNORECASE,
    )
    # Remove notes/references sections
    body = re.sub(
        r"^==+\s*(Примечания|Комментарии|Источники|Ссылки|См\. также)\s*==+.*",
        "", body, flags=re.MULTILINE | re.DOTALL,
    )
    # Unwrap literary templates BEFORE generic template removal
    body = _unwrap_literary_templates(body)
    return clean_wikitext(body)


def parse_author_years(author_str: str) -> tuple[int | None, int | None]:
    match = re.search(r"\((\d{3,4})\s*[—–-]\s*(\d{3,4})\)", author_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"\((\d{3,4})\s*[—–-]\s*\)", author_str)
    if match:
        return int(match.group(1)), None
    return None, None


def clean_author_name(author_raw: str) -> str:
    author = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", author_raw)
    author = re.sub(r"\[\[([^\]]+)\]\]", r"\1", author)
    author = re.sub(r"\s*\([\d—–\-\s]+\)\s*", "", author)
    author = re.sub(r"<[^>]+>", "", author)
    return author.strip()


def parse_year(value: str) -> int | None:
    if not value:
        return None
    match = re.search(r"\d{3,4}", value)
    return int(match.group()) if match else None


def detect_genre(categories: list[str]) -> str:
    cats_lower = " ".join(categories).lower()
    if any(w in cats_lower for w in ["поэзия", "стихотворени", "стих"]):
        return "poetry"
    if any(w in cats_lower for w in ["роман", "повест", "рассказ", "новелл", "очерк", "проз"]):
        return "prose"
    if any(w in cats_lower for w in ["пьес", "комеди", "трагеди", "драм"]):
        return "drama"
    if "басн" in cats_lower:
        return "fable"
    return "other"


def detect_license(categories: list[str]) -> str:
    for cat in categories:
        if "PD-old" in cat or "PD-Russia" in cat or "PD-Rus" in cat:
            return cat
    return ""


def compute_license_reason(
    death_year: int | None,
    license_template: str,
) -> tuple[str, str]:
    """Return (license, license_reason) based on author death and templates."""
    if death_year and death_year + 70 < CURRENT_YEAR:
        return "PD-old-70", f"author_died_{death_year}"
    if license_template:
        return license_template, "wikisource_template"
    if death_year:
        return "unverified", f"author_died_{death_year}_too_recent"
    return "unverified", "no_death_year"


def detect_quality(categories: list[str]) -> str:
    for cat in categories:
        if re.match(r"^\d+%$", cat):
            return cat
    return ""


def extract_categories(wikitext: str) -> list[str]:
    return re.findall(r"\[\[Категория:([^\]|]+)", wikitext)


def extract_interwiki(wikitext: str) -> list[str]:
    return list(set(re.findall(r"\[\[([a-z]{2,3}):[^\]]+\]\]", wikitext)))


def filter_categories(categories: list[str]) -> list[str]:
    return [
        c for c in categories
        if not c.startswith("Статьи") and not c.startswith("Ссылка")
        and not c.startswith("Викиданные") and not c.startswith("Страницы")
    ]


# ── Author page parsing ─────────────────────────────────────

def parse_author_page(title: str, wikitext: str) -> dict | None:
    """Parse an Автор: page into an author record."""
    params = parse_template_params(wikitext, "Обавторе")
    if not params:
        params = parse_template_params(wikitext, "Об авторе")
    if not params:
        return None

    name_parts = {
        "family_name": params.get("ФАМИЛИЯ", ""),
        "given_names": params.get("ИМЕНА", ""),
    }
    full_name = f"{name_parts['given_names']} {name_parts['family_name']}".strip()
    if not full_name:
        full_name = title.removeprefix("Автор:")

    # Extract years from the author string or page content
    description = clean_wikitext(params.get("ОПИСАНИЕ", ""))
    birth, death = None, None

    # Try from template variants field
    variants = params.get("ВАРИАНТЫИМЁН", "")
    combined = f"{full_name} {variants} {wikitext[:500]}"
    birth, death = parse_author_years(combined)

    # Try from categories
    categories = extract_categories(wikitext)
    if not birth:
        for cat in categories:
            m = re.search(r"Родившиеся в (\d{3,4}) году", cat)
            if m:
                birth = int(m.group(1))
                break
    if not death:
        for cat in categories:
            m = re.search(r"Умершие в (\d{3,4}) году", cat)
            if m:
                death = int(m.group(1))
                break

    return {
        "author_id": title,
        "name": full_name,
        "family_name": name_parts["family_name"],
        "given_names": name_parts["given_names"],
        "birth_year": birth,
        "death_year": death,
        "description": description,
        "wikidata_id": "",  # filled by Wikidata enrichment
        "works_count": 0,   # filled after works pass
        "categories": filter_categories(categories),
    }


# ── XML streaming ────────────────────────────────────────────

def iter_pages(dump_path: Path):
    """Stream pages from a bz2-compressed MediaWiki XML dump."""
    open_func = bz2.open if dump_path.suffix == ".bz2" else open
    with open_func(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        try:
            for event, elem in context:
                if elem.tag == f"{MW_NS}page":
                    title_el = elem.find(f"{MW_NS}title")
                    ns_el = elem.find(f"{MW_NS}ns")
                    rev_el = elem.find(f"{MW_NS}revision")
                    text_el = (
                        rev_el.find(f"{MW_NS}text")
                        if rev_el is not None else None
                    )
                    title = title_el.text if title_el is not None else ""
                    ns = ns_el.text if ns_el is not None else "0"
                    wikitext = text_el.text if text_el is not None else ""
                    yield title, ns, wikitext or ""
                    elem.clear()
        except ET.ParseError as e:
            logger.warning("XML parse error (truncated dump?): %s — stopping", e)


# ── Download ─────────────────────────────────────────────────

def download_dump(output_dir: Path) -> Path:
    import httpx

    dump_path = output_dir / DUMP_FILENAME
    if dump_path.exists():
        size_gb = dump_path.stat().st_size / 1e9
        logger.info("Dump already exists: %s (%.2f GB)", dump_path, size_gb)
        return dump_path

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = dump_path.with_suffix(".bz2.part")
    logger.info("Downloading dump from %s (~2 GB)...", DUMP_URL)

    with httpx.stream(
        "GET", DUMP_URL,
        headers={"User-Agent": "MentionMap/0.1 (https://github.com/matyushkin/mention-map)"},
        timeout=600.0, follow_redirects=True,
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                    logger.info("  %.1f%%", downloaded / total * 100)

    tmp_path.rename(dump_path)
    logger.info("Download complete: %s", dump_path)
    return dump_path


# ── Parquet writer ───────────────────────────────────────────

class BatchParquetWriter:
    def __init__(self, output_dir: Path, prefix: str, schema: pa.Schema, batch_size: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.schema = schema
        self.batch_size = batch_size
        self.records: list[dict] = []
        self.batch_num = 0
        self.total_records = 0
        self.total_bytes = 0

    def add(self, record: dict) -> bool:
        self.records.append(record)
        if len(self.records) >= self.batch_size:
            self._flush()
            if self.total_bytes >= MAX_OUTPUT_BYTES:
                logger.warning("%s: output size limit reached (%.1f GB)", self.prefix, self.total_bytes / 1e9)
                return False
        return True

    def _flush(self):
        if not self.records:
            return
        table = pa.table(
            {f.name: [r.get(f.name) for r in self.records] for f in self.schema},
            schema=self.schema,
        )
        out_path = self.output_dir / f"{self.prefix}-{self.batch_num:04d}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        size = out_path.stat().st_size
        self.total_bytes += size
        logger.info(
            "%s batch %d: %d records → %s (%.1f MB)",
            self.prefix, self.batch_num, len(self.records), out_path.name, size / 1e6,
        )
        self.total_records += len(self.records)
        self.records = []
        self.batch_num += 1

    def finalize(self):
        self._flush()
        logger.info("%s: %d total records, %.2f GB", self.prefix, self.total_records, self.total_bytes / 1e9)


# ── Main processing ──────────────────────────────────────────

def process_dump(dump_path: Path, output_dir: Path, batch_size: int = 5000, literary_only: bool = True):
    """Single-pass: extract both authors and works from the dump."""
    output_dir.mkdir(parents=True, exist_ok=True)

    authors_writer = BatchParquetWriter(output_dir, "authors", AUTHOR_SCHEMA, batch_size=2000)
    works_writer = BatchParquetWriter(output_dir, "works", WORK_SCHEMA, batch_size=batch_size)

    # Track author_id → death_year for PD validation
    author_deaths: dict[str, int | None] = {}

    skipped = {
        "redirect": 0, "ns": 0, "short": 0,
        "no_template": 0, "nonliterary": 0,
    }

    for i, (title, ns, wikitext) in enumerate(iter_pages(dump_path)):
        if i % 50000 == 0 and i > 0:
            logger.info(
                "Processed %d pages | authors: %d | works: %d",
                i,
                authors_writer.total_records + len(authors_writer.records),
                works_writer.total_records + len(works_writer.records),
            )

        # Skip non-content namespaces
        if ns in SKIP_NS:
            skipped["ns"] += 1
            continue

        # Skip redirects
        if wikitext.strip().lower().startswith("#"):
            skipped["redirect"] += 1
            continue

        # ── Author pages ─────────────────────────────────
        if title.startswith("Автор:"):
            author = parse_author_page(title, wikitext)
            if author:
                authors_writer.add(author)
                author_deaths[title] = author["death_year"]
            continue

        # ── Work pages ───────────────────────────────────
        if _is_nonliterary(title):
            skipped["nonliterary"] += 1
            continue

        params = parse_template_params(wikitext, "Отексте")
        if not params:
            has_poem = "{{f1|" in wikitext or "<poem>" in wikitext
            has_prose = len(wikitext) > 2000 and ns == "0"
            if not (has_poem or has_prose):
                skipped["no_template"] += 1
                continue

        body = extract_clean_body(wikitext)
        if len(body) < 50:
            skipped["short"] += 1
            continue

        categories = extract_categories(wikitext)
        genre = detect_genre(categories)

        author_raw = params.get("АВТОР", "")
        if literary_only and genre == "other" and not author_raw:
            skipped["nonliterary"] += 1
            continue

        birth, death = parse_author_years(author_raw)

        # Resolve author_id
        author_name = clean_author_name(author_raw)
        author_id = ""
        # Try to find author link in wikitext
        author_link = re.search(r"\[\[Автор:([^\]|]+)", wikitext)
        if author_link:
            author_id = f"Автор:{author_link.group(1).strip()}"
            # Use death year from author catalogue if we have it
            if not death and author_id in author_deaths:
                death = author_deaths[author_id]

        license_template = detect_license(categories)
        license_val, license_reason = compute_license_reason(death, license_template)

        raw_title = params.get("НАЗВАНИЕ", title)
        clean_title = clean_wikitext(raw_title) if "[[" in raw_title else raw_title

        record = {
            "id": title,
            "title": clean_title,
            "author": author_name,
            "author_id": author_id,
            "author_birth_year": birth,
            "author_death_year": death,
            "year_written": parse_year(params.get("ДАТАСОЗДАНИЯ", "")),
            "year_published": parse_year(params.get("ДАТАПУБЛИКАЦИИ", "")),
            "genre": genre,
            "text": body,
            "text_length": len(body),
            "word_count": len(body.split()),
            "source": clean_wikitext(params.get("ИСТОЧНИК", "")),
            "categories": filter_categories(categories),
            "interwiki": extract_interwiki(wikitext),
            "quality": detect_quality(categories),
            "license": license_val,
            "license_reason": license_reason,
            "wikisource_page": title,
        }

        if not works_writer.add(record):
            break

    authors_writer.finalize()
    works_writer.finalize()

    logger.info("Skipped: %s", json.dumps(skipped))

    # Update author works_count
    _update_author_works_count(output_dir)


def _update_author_works_count(output_dir: Path):
    """Count works per author and update authors parquet."""
    # Count from works
    counts: dict[str, int] = {}
    for f in sorted(output_dir.glob("works-*.parquet")):
        t = pq.read_table(f, columns=["author_id"])
        for aid in t.column("author_id").to_pylist():
            if aid:
                counts[aid] = counts.get(aid, 0) + 1

    # Update authors
    author_files = sorted(output_dir.glob("authors-*.parquet"))
    if not author_files:
        return

    tables = [pq.read_table(f) for f in author_files]
    combined = pa.concat_tables(tables)

    # Update works_count column
    author_ids = combined.column("author_id").to_pylist()
    works_counts = [counts.get(aid, 0) for aid in author_ids]

    # Replace column
    idx = combined.schema.get_field_index("works_count")
    combined = combined.set_column(idx, "works_count", pa.array(works_counts, type=pa.int32()))

    # Write single authors file
    out_path = output_dir / "authors.parquet"
    pq.write_table(combined, out_path, compression="zstd")
    logger.info(
        "Authors index: %d authors, %d with works → %s",
        len(author_ids), sum(1 for c in works_counts if c > 0), out_path,
    )

    # Remove batch files
    for f in author_files:
        f.unlink()


# ── Wikidata enrichment ──────────────────────────────────────

def enrich_wikidata(output_dir: Path):
    """Enrich authors.parquet with Wikidata IDs and metadata via SPARQL."""
    import httpx

    authors_path = output_dir / "authors.parquet"
    if not authors_path.exists():
        logger.error("authors.parquet not found — run dump processing first")
        return

    table = pq.read_table(authors_path)
    author_names = table.column("name").to_pylist()
    author_ids = table.column("author_id").to_pylist()
    existing_wd = table.column("wikidata_id").to_pylist()

    logger.info("Enriching %d authors from Wikidata...", len(author_names))

    # Query Wikidata for all Wikisource-ru authors in one SPARQL query
    sparql_url = "https://query.wikidata.org/sparql"
    query = """
    SELECT ?item ?itemLabel ?ruwikisource ?birthYear ?deathYear WHERE {
      ?item wdt:P31 wd:Q5 .
      ?ruwikisource schema:about ?item ;
                    schema:isPartOf <https://ru.wikisource.org/> .
      OPTIONAL { ?item wdt:P569 ?birth . BIND(YEAR(?birth) AS ?birthYear) }
      OPTIONAL { ?item wdt:P570 ?death . BIND(YEAR(?death) AS ?deathYear) }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en" . }
    }
    """

    # Fetch in pages to avoid truncated JSON responses
    results = []
    page_size = 5000
    offset = 0
    while True:
        paged_query = query.rstrip() + f"\nLIMIT {page_size} OFFSET {offset}"
        logger.info("Wikidata SPARQL: offset=%d ...", offset)
        try:
            resp = httpx.get(
                sparql_url,
                params={"query": paged_query, "format": "json"},
                headers={
                    "User-Agent": "MentionMap/0.1 (https://github.com/matyushkin/mention-map)",
                    "Accept": "application/sparql-results+json",
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            batch = resp.json().get("results", {}).get("bindings", [])
        except (httpx.HTTPStatusError, Exception) as e:
            logger.warning("Wikidata query failed at offset %d: %s — using %d results so far", offset, e, len(results))
            break
        results.extend(batch)
        logger.info("  got %d results (total: %d)", len(batch), len(results))
        if len(batch) < page_size:
            break
        offset += page_size

    logger.info("Wikidata: %d total results", len(results))

    # Build mapping: wikisource page title → wikidata info
    wd_map: dict[str, dict] = {}
    for r in results:
        ws_url = r.get("ruwikisource", {}).get("value", "")
        # Extract page title from URL
        if "/wiki/" in ws_url:
            page = ws_url.split("/wiki/")[-1]
            from urllib.parse import unquote
            page = unquote(page).replace("_", " ")
            wd_id = r["item"]["value"].split("/")[-1]
            birth = r.get("birthYear", {}).get("value")
            death = r.get("deathYear", {}).get("value")
            wd_map[page] = {
                "wikidata_id": wd_id,
                "birth_year": int(birth) if birth else None,
                "death_year": int(death) if death else None,
            }

    # Match and update
    new_wd_ids = list(existing_wd)
    new_births = table.column("birth_year").to_pylist()
    new_deaths = table.column("death_year").to_pylist()
    matched = 0

    for i, aid in enumerate(author_ids):
        # Try both with and without "Автор:" prefix, and with underscores
        if aid in wd_map:
            info = wd_map[aid]
            new_wd_ids[i] = info["wikidata_id"]
            if not new_births[i] and info["birth_year"]:
                new_births[i] = info["birth_year"]
            if not new_deaths[i] and info["death_year"]:
                new_deaths[i] = info["death_year"]
            matched += 1

    # Write updated table
    idx_wd = table.schema.get_field_index("wikidata_id")
    idx_b = table.schema.get_field_index("birth_year")
    idx_d = table.schema.get_field_index("death_year")

    table = table.set_column(idx_wd, "wikidata_id", pa.array(new_wd_ids, type=pa.string()))
    table = table.set_column(idx_b, "birth_year", pa.array(new_births, type=pa.int16()))
    table = table.set_column(idx_d, "death_year", pa.array(new_deaths, type=pa.int16()))

    pq.write_table(table, authors_path, compression="zstd")
    logger.info("Wikidata enrichment: %d/%d authors matched", matched, len(author_ids))


# ── Push to HF Hub ───────────────────────────────────────────

def push_to_hub(output_dir: Path, repo_id: str):
    from datasets import Dataset, DatasetDict

    works_files = sorted(output_dir.glob("works-*.parquet"))
    authors_path = output_dir / "authors.parquet"

    if not works_files:
        logger.error("No works parquet files found in %s", output_dir)
        return

    works_tables = [pq.read_table(f) for f in works_files]
    works = pa.concat_tables(works_tables)

    ds_dict = {"works": Dataset(works)}

    if authors_path.exists():
        authors = pq.read_table(authors_path)
        ds_dict["authors"] = Dataset(authors)

    ds = DatasetDict(ds_dict)
    logger.info("Pushing %d works + %d authors to %s",
                len(ds["works"]), len(ds.get("authors", [])), repo_id)
    ds.push_to_hub(repo_id, private=False)
    logger.info("Done! https://huggingface.co/datasets/%s", repo_id)


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build HF dataset from ru.wikisource.org XML dump",
    )
    parser.add_argument("--dump-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--push", type=str, metavar="REPO_ID")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-gb", type=float, default=100.0)
    parser.add_argument("--all-content", action="store_true")
    parser.add_argument("--enrich-wikidata", action="store_true")
    args = parser.parse_args()

    global MAX_OUTPUT_BYTES
    MAX_OUTPUT_BYTES = int(args.max_gb * 1024 * 1024 * 1024)

    if args.push:
        push_to_hub(args.output_dir, args.push)
        return

    if args.enrich_wikidata:
        enrich_wikidata(args.output_dir)
        return

    dump_path = args.dump_file
    if not dump_path:
        dump_path = download_dump(args.output_dir)
    elif not dump_path.exists():
        logger.error("Dump file not found: %s", dump_path)
        return

    process_dump(dump_path, args.output_dir, args.batch_size, literary_only=not args.all_content)


if __name__ == "__main__":
    main()
