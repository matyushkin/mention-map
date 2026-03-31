def extract_mentions(
    text: str, characters: list[dict], lang: str
) -> list[dict]:
    """Extract mentions — detect who mentions/references whom within the text.

    A mention is when character A talks about or references character B.
    This requires understanding dialogue attribution and narrative context.

    TODO: Implement proper mention extraction using:
    - Dialogue detection and speaker attribution
    - Proximity-based co-occurrence in sentences/paragraphs
    - Dependency parsing for subject-object relationships

    Current implementation uses simple sentence co-occurrence as a baseline.
    """
    sentences = _split_sentences(text)
    mentions = []

    char_names = {c["id"]: c["name"] for c in characters}

    for sentence in sentences:
        present = []
        for char in characters:
            if char["name"].lower() in sentence.lower():
                present.append(char["id"])

        # If two characters co-occur in a sentence, record a mention
        for i, source_id in enumerate(present):
            for target_id in present[i + 1 :]:
                mentions.append({
                    "source": char_names[source_id],
                    "target": char_names[target_id],
                    "context": sentence.strip()[:200],
                    "chapter": None,
                    "date": None,
                })

    return mentions


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    import re

    return re.split(r"(?<=[.!?])\s+", text)
