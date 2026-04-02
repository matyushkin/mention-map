"""Build tagged text and offset annotations for dramatic texts.

Parses wiki templates ({{Реплика}}, {{ремарка}}, {{rem}}) from dump,
generates text_tagged field with XML-like tags, then derives offset
annotations from tag positions in the clean text.

Adds `text_tagged` field to works parquet for drama texts.
Creates annotations.parquet with character-offset spans.

Usage:
    uv run --extra hf python scripts/build_annotations.py --output-dir ../hf_dataset
"""

import argparse
import bz2
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"

ANNOTATION_SCHEMA = pa.schema([
    ("work_id", pa.string()),
    ("start", pa.int32()),
    ("end", pa.int32()),
    ("type", pa.string()),
    ("value", pa.string()),
    ("speaker", pa.string()),
])


def wikitext_to_tagged(wikitext: str) -> str | None:
    """Convert drama wikitext to tagged text.

    Returns None if text doesn't contain drama templates.
    """
    if "{{Реплика" not in wikitext and "{{реплика" not in wikitext:
        return None

    # Remove header templates
    wt = re.sub(
        r"\{\{(?:Отексте|Обавторе|Об авторе|header|Header)[^{}]*(?:\{\{[^{}]*\}\}[^{}]*)*\}\}",
        "", wikitext, flags=re.DOTALL | re.IGNORECASE,
    )
    wt = re.sub(r"\{\{(?:PD|pd|ОД|Public domain|Лицензия|License|simple|аудиостатья)[^{}]*\}\}", "", wt, flags=re.IGNORECASE)
    wt = re.sub(r"__[A-Z]+__", "", wt)
    wt = re.sub(r"^==+\s*(Примечания|Комментарии|Источники|Ссылки)\s*==+.*", "", wt, flags=re.MULTILINE | re.DOTALL)
    wt = re.sub(r"\[\[(?:Категория|Category):[^\]]+\]\]", "", wt)
    wt = re.sub(r"\[\[[a-z]{2,3}:[^\]]+\]\]", "", wt)

    lines_out = []

    for line in wt.split("\n"):
        line = line.strip()
        if not line:
            lines_out.append("")
            continue

        # == Scene heading ==
        hm = re.match(r"^=+\s*(.+?)\s*=+$", line)
        if hm:
            lines_out.append(f'<scene>{hm.group(1)}</scene>')
            continue

        # Unwrap razr first (visual emphasis → plain)
        line = re.sub(r"\{\{razr\|([^}]+)\}\}", r"\1", line)

        out = ""
        rest = line

        while rest:
            # {{Реплика|Name}} or {{Реплика|Name|dir}}
            m = re.match(r"\{\{[Рр]еплика\|([^|}]+)(?:\|([^}]*))?\}\}\s*(.*)", rest, re.DOTALL)
            if m:
                name = m.group(1).strip()
                direction = (m.group(2) or "").strip()
                speech = m.group(3)

                if direction:
                    out += f'<speaker name="{name}" dir="{direction}"/>'
                else:
                    out += f'<speaker name="{name}"/>'

                # Process inline ремарки in speech
                speech = re.sub(r"\{\{ремарка\|([^}]+)\}\}", r"<stage>\1</stage>", speech)
                speech = re.sub(r"\{\{rem\|([^}]+)\}\}", r"<stage>\1</stage>", speech)
                speech = re.sub(r"\{\{razr\|([^}]+)\}\}", r"\1", speech)
                speech = re.sub(r"\{\{[^}]*\}\}", "", speech)

                out += speech
                rest = ""
                continue

            # {{rem|...}} or {{rem2|...}} standalone
            m = re.match(r"\{\{rem2?\|(.+?)\}\}\s*(.*)", rest, re.DOTALL)
            if m:
                content = re.sub(r"\{\{razr\|([^}]+)\}\}", r"\1", m.group(1))
                out += f"<stage>{content}</stage>"
                rest = m.group(2)
                continue

            # {{ремарка|...}} standalone
            m = re.match(r"\{\{ремарка\|(.+?)\}\}\s*(.*)", rest, re.DOTALL)
            if m:
                out += f"<stage>{m.group(1)}</stage>"
                rest = m.group(2)
                continue

            # No more templates
            out += rest
            rest = ""

        # Clean leftover templates and wiki markup
        out = re.sub(r"\{\{[^}]*\}\}", "", out)
        out = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", out)
        out = re.sub(r"\[\[([^\]]+)\]\]", r"\1", out)
        out = re.sub(r"<ref[^>]*>.*?</ref>", "", out, flags=re.DOTALL)
        out = re.sub(r"<ref[^/]*/?>", "", out)
        out = re.sub(r"</?(?!speaker|stage|scene)[a-zA-Z][^>]*>", "", out)
        out = re.sub(r"&nbsp;", " ", out)
        out = re.sub(r"(?<=\S)[ \t]+", " ", out)  # preserve leading whitespace
        out = out.rstrip()

        lines_out.append(out)

    result = "\n".join(lines_out).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def strip_tags(tagged_text: str) -> str:
    """Remove all tags from tagged text, producing clean text."""
    text = re.sub(r"<speaker[^/]*/?>", "", tagged_text)
    text = re.sub(r"</?(?:stage|scene)>", "", text)
    return text


def tagged_to_annotations(work_id: str, tagged_text: str) -> list[dict]:
    """Extract offset annotations from tagged text.

    Offsets reference positions in the CLEAN (untagged) text.
    """
    annotations = []
    clean_pos = 0
    tag_pos = 0
    clean_text = strip_tags(tagged_text)

    for m in re.finditer(r"<(speaker|stage|scene)([^>]*)>([^<]*)</\1>|<speaker([^/]*?)/>", tagged_text):
        # Calculate clean text position for this match
        # Everything before this match in tagged text
        before_tagged = tagged_text[tag_pos:m.start()]
        before_clean = strip_tags(before_tagged)
        clean_pos_at_match = clean_pos + len(before_clean)

        if m.group(4) is not None:
            # Self-closing <speaker name="..." dir="..."/>
            attrs = m.group(4)
            name_m = re.search(r'name="([^"]+)"', attrs)
            dir_m = re.search(r'dir="([^"]+)"', attrs)
            if name_m:
                annotations.append({
                    "work_id": work_id,
                    "start": clean_pos_at_match,
                    "end": clean_pos_at_match,  # zero-width marker
                    "type": "speaker",
                    "value": name_m.group(1),
                    "speaker": "",
                })
                if dir_m:
                    annotations.append({
                        "work_id": work_id,
                        "start": clean_pos_at_match,
                        "end": clean_pos_at_match,
                        "type": "speaker_direction",
                        "value": dir_m.group(1),
                        "speaker": name_m.group(1),
                    })
        else:
            tag_type = m.group(1)
            content = m.group(3)
            content_start = clean_pos_at_match
            content_end = content_start + len(content)

            ann_type = {
                "stage": "stage_direction",
                "scene": "scene_heading",
            }.get(tag_type, tag_type)

            annotations.append({
                "work_id": work_id,
                "start": content_start,
                "end": content_end,
                "type": ann_type,
                "value": content if ann_type == "scene_heading" else "",
                "speaker": "",
            })

        tag_pos = m.end()
        clean_pos = clean_pos_at_match + len(strip_tags(m.group(0)))

    annotations.sort(key=lambda a: (a["start"], a["end"]))
    return annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir
    dump = out / "ruwikisource-latest-pages-articles.xml.bz2"

    if not dump.exists():
        logger.error("Dump not found: %s", dump)
        return

    # Collect all work IDs from dataset
    all_ids = set()
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f, columns=["id"])
        for i in range(t.num_rows):
            all_ids.add(t.column("id")[i].as_py())

    logger.info("Works in dataset: %d", len(all_ids))

    # Scan dump for drama pages
    tagged_texts = {}  # work_id → tagged_text
    with bz2.open(str(dump), "rb") as f:
        try:
            for event, elem in ET.iterparse(f, events=("end",)):
                if elem.tag == f"{MW_NS}page":
                    title_el = elem.find(f"{MW_NS}title")
                    ns_el = elem.find(f"{MW_NS}ns")
                    title = title_el.text if title_el is not None else ""
                    ns = ns_el.text if ns_el is not None else ""
                    if ns != "0" or title not in all_ids:
                        elem.clear()
                        continue
                    rev = elem.find(f"{MW_NS}revision")
                    text_el = rev.find(f"{MW_NS}text") if rev is not None else None
                    wt = text_el.text if text_el is not None else ""
                    if wt:
                        tagged = wikitext_to_tagged(wt)
                        if tagged:
                            tagged_texts[title] = tagged
                    elem.clear()
        except ET.ParseError as e:
            logger.warning("XML parse error: %s", e)

    logger.info("Drama pages with tags: %d", len(tagged_texts))

    # Add text_tagged to works parquet
    updated_works = 0
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f)
        ids = t.column("id").to_pylist()

        # Check if any IDs match
        has_tags = any(wid in tagged_texts for wid in ids)
        if not has_tags:
            # Still need to add empty text_tagged column if not present
            if "text_tagged" not in t.column_names:
                t = t.append_column("text_tagged", pa.array([""] * t.num_rows, type=pa.string()))
                pq.write_table(t, f, compression="zstd")
            continue

        tagged_col = []
        for wid in ids:
            tagged_col.append(tagged_texts.get(wid, ""))
            if wid in tagged_texts:
                updated_works += 1

        if "text_tagged" in t.column_names:
            idx = t.schema.get_field_index("text_tagged")
            t = t.set_column(idx, "text_tagged", pa.array(tagged_col, type=pa.string()))
        else:
            t = t.append_column("text_tagged", pa.array(tagged_col, type=pa.string()))

        pq.write_table(t, f, compression="zstd")

    logger.info("Works with text_tagged: %d", updated_works)

    # Build annotations from tagged texts
    all_annotations = []
    for work_id, tagged in tagged_texts.items():
        annots = tagged_to_annotations(work_id, tagged)
        all_annotations.extend(annots)

    logger.info("Total annotations: %d", len(all_annotations))

    if all_annotations:
        table = pa.table(
            {f.name: [a[f.name] for a in all_annotations] for f in ANNOTATION_SCHEMA},
            schema=ANNOTATION_SCHEMA,
        )
        out_path = out / "annotations.parquet"
        pq.write_table(table, out_path, compression="zstd")
        logger.info("Written → %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)

        from collections import Counter
        types = Counter(a["type"] for a in all_annotations)
        for tp, c in types.most_common():
            logger.info("  %s: %d", tp, c)
        speakers = Counter(a["value"] for a in all_annotations if a["type"] == "speaker")
        logger.info("Unique speakers: %d", len(speakers))
        for s, c in speakers.most_common(10):
            logger.info("  %4d  %s", c, s)

    # Verify: spot-check one play
    if "Вишнёвый сад (Чехов)/Действие первое" in tagged_texts:
        tagged = tagged_texts["Вишнёвый сад (Чехов)/Действие первое"]
        logger.info("\n=== Sample tagged text (Вишнёвый сад) ===")
        logger.info(tagged[:600])


if __name__ == "__main__":
    main()
