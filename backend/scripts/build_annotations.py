"""Build NER-style annotations for dramatic texts.

Parses {{Реплика}}, {{ремарка}}, {{rem}} templates from Wikisource dump,
generates character-offset annotations aligned with clean text in works parquet.

Output: annotations.parquet with columns:
  work_id, start, end, type, value, speaker

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
    ("type", pa.string()),     # speaker, speech, stage_direction, scene_heading
    ("value", pa.string()),    # character name for speaker, heading text for scene_heading
    ("speaker", pa.string()),  # who is speaking (for speech spans)
])


def parse_drama_annotations(wikitext: str) -> tuple[str, list[dict]]:
    """Parse wikitext with drama templates into clean text + annotations.

    Returns (clean_text, annotations_list).
    """
    # Remove header templates
    wt = re.sub(
        r"\{\{(?:Отексте|Обавторе|Об авторе|header|Header)[^{}]*(?:\{\{[^{}]*\}\}[^{}]*)*\}\}",
        "", wikitext, flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove license/PD templates
    wt = re.sub(r"\{\{(?:PD|pd|ОД|Public domain|Лицензия|License|simple|аудиостатья)[^{}]*\}\}", "", wt, flags=re.IGNORECASE)
    # Remove magic words
    wt = re.sub(r"__[A-Z]+__", "", wt)
    # Remove notes section
    wt = re.sub(r"^==+\s*(Примечания|Комментарии|Источники|Ссылки)\s*==+.*", "", wt, flags=re.MULTILINE | re.DOTALL)
    # Remove categories and interwiki
    wt = re.sub(r"\[\[(?:Категория|Category):[^\]]+\]\]", "", wt)
    wt = re.sub(r"\[\[[a-z]{2,3}:[^\]]+\]\]", "", wt)

    annotations = []
    lines_out = []
    pos = 0

    for line in wt.split("\n"):
        line = line.strip()
        if not line:
            lines_out.append("")
            pos += 1
            continue

        # == Scene heading ==
        hm = re.match(r"^=+\s*(.+?)\s*=+$", line)
        if hm:
            val = hm.group(1)
            annotations.append({"start": pos, "end": pos + len(val), "type": "scene_heading", "value": val, "speaker": ""})
            lines_out.append(val)
            pos += len(val) + 1
            continue

        # Unwrap razr (visual emphasis) before processing
        line = re.sub(r"\{\{razr\|([^}]+)\}\}", r"\1", line)

        out = ""
        rest = line

        while rest:
            # {{Реплика|Name}} or {{Реплика|Name|direction}}
            m = re.match(r"\{\{[Рр]еплика\|([^|}]+)(?:\|([^}]*))?\}\}\s*(.*)", rest, re.DOTALL)
            if m:
                name = m.group(1).strip()
                direction = (m.group(2) or "").strip()
                speech = m.group(3)

                # Build speaker text
                if direction:
                    speaker_str = f"{name} ({direction})."
                else:
                    speaker_str = f"{name}."

                s_start = pos + len(out)
                annotations.append({"start": s_start, "end": s_start + len(name), "type": "speaker", "value": name, "speaker": ""})
                out += speaker_str + " "

                # Process inline ремарки in speech, recording their positions
                processed_speech = ""
                speech_rest = speech
                while speech_rest:
                    rm = re.search(r"\{\{ремарка\|([^}]+)\}\}", speech_rest)
                    if not rm:
                        rm = re.search(r"\{\{rem\|([^}]+)\}\}", speech_rest)
                    if rm:
                        # Text before ремарка
                        processed_speech += speech_rest[:rm.start()]
                        # Ремарка
                        rem_text = f"({rm.group(1)})"
                        rem_start = pos + len(out) + len(processed_speech)
                        annotations.append({"start": rem_start, "end": rem_start + len(rem_text), "type": "stage_direction", "value": "", "speaker": ""})
                        processed_speech += rem_text
                        speech_rest = speech_rest[rm.end():]
                    else:
                        processed_speech += speech_rest
                        speech_rest = ""

                # Clean any remaining templates
                processed_speech = re.sub(r"\{\{[^}]*\}\}", "", processed_speech)

                sp_start = pos + len(out)
                annotations.append({"start": sp_start, "end": sp_start + len(processed_speech), "type": "speech", "value": "", "speaker": name})
                out += processed_speech
                rest = ""
                continue

            # {{rem|...}} or {{rem2|...}} standalone
            m = re.match(r"\{\{rem2?\|(.+?)\}\}\s*(.*)", rest, re.DOTALL)
            if m:
                content = re.sub(r"\{\{razr\|([^}]+)\}\}", r"\1", m.group(1))
                rem_str = f"({content})"
                r_start = pos + len(out)
                annotations.append({"start": r_start, "end": r_start + len(rem_str), "type": "stage_direction", "value": "", "speaker": ""})
                out += rem_str
                rest = m.group(2)
                continue

            # {{ремарка|...}} standalone
            m = re.match(r"\{\{ремарка\|(.+?)\}\}\s*(.*)", rest, re.DOTALL)
            if m:
                rem_str = f"({m.group(1)})"
                r_start = pos + len(out)
                annotations.append({"start": r_start, "end": r_start + len(rem_str), "type": "stage_direction", "value": "", "speaker": ""})
                out += rem_str
                rest = m.group(2)
                continue

            # No more templates
            out += rest
            rest = ""

        # Clean leftover templates
        out = re.sub(r"\{\{[^}]*\}\}", "", out).strip()
        # Clean wiki links
        out = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", out)
        out = re.sub(r"\[\[([^\]]+)\]\]", r"\1", out)
        # Clean HTML
        out = re.sub(r"<[^>]+>", "", out)
        out = re.sub(r"&nbsp;", " ", out)

        lines_out.append(out)
        pos += len(out) + 1

    text = "\n".join(lines_out).strip()

    # Remove leading empty lines
    while text.startswith("\n"):
        offset = 1
        # Shift all annotations
        for a in annotations:
            a["start"] -= offset
            a["end"] -= offset
        text = text[offset:]

    return text, annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir
    dump = out / "ruwikisource-latest-pages-articles.xml.bz2"

    if not dump.exists():
        logger.error("Dump not found: %s", dump)
        return

    # Collect IDs of drama works
    drama_ids = set()
    for f in sorted(out.glob("works-*.parquet")):
        t = pq.read_table(f, columns=["id", "text"])
        for i in range(t.num_rows):
            wid = t.column("id")[i].as_py()
            drama_ids.add(wid)

    logger.info("Works in dataset: %d", len(drama_ids))

    # Scan dump for pages with drama templates
    all_annotations = []
    pages_with_drama = 0

    with bz2.open(str(dump), "rb") as f:
        try:
            for event, elem in ET.iterparse(f, events=("end",)):
                if elem.tag == f"{MW_NS}page":
                    title_el = elem.find(f"{MW_NS}title")
                    ns_el = elem.find(f"{MW_NS}ns")
                    title = title_el.text if title_el is not None else ""
                    ns = ns_el.text if ns_el is not None else ""

                    if ns != "0" or title not in drama_ids:
                        elem.clear()
                        continue

                    rev = elem.find(f"{MW_NS}revision")
                    text_el = rev.find(f"{MW_NS}text") if rev is not None else None
                    wt = text_el.text if text_el is not None else ""

                    if wt and ("{{Реплика" in wt or "{{реплика" in wt):
                        _, annots = parse_drama_annotations(wt)
                        if annots:
                            for a in annots:
                                a["work_id"] = title
                            all_annotations.extend(annots)
                            pages_with_drama += 1

                    elem.clear()
        except ET.ParseError as e:
            logger.warning("XML parse error: %s", e)

    logger.info("Drama pages found: %d, annotations: %d", pages_with_drama, len(all_annotations))

    if not all_annotations:
        logger.warning("No annotations generated")
        return

    # Write annotations.parquet
    table = pa.table(
        {f.name: [a.get(f.name, "") for a in all_annotations] for f in ANNOTATION_SCHEMA},
        schema=ANNOTATION_SCHEMA,
    )
    out_path = out / "annotations.parquet"
    pq.write_table(table, out_path, compression="zstd")
    logger.info("Written %d annotations → %s (%.1f MB)",
                len(all_annotations), out_path.name, out_path.stat().st_size / 1e6)

    # Stats
    from collections import Counter
    types = Counter(a["type"] for a in all_annotations)
    for t, c in types.most_common():
        logger.info("  %s: %d", t, c)

    speakers = Counter(a["value"] for a in all_annotations if a["type"] == "speaker")
    logger.info("Unique speakers: %d", len(speakers))
    logger.info("Top speakers:")
    for s, c in speakers.most_common(10):
        logger.info("  %4d  %s", c, s)


if __name__ == "__main__":
    main()
