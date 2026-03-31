from nlp.ner import extract_entities
from nlp.coreference import resolve_coreferences
from nlp.mentions import extract_mentions
from nlp.disambiguation import disambiguate_entities
from api.schemas import AnalysisResponse, GraphData, Character, Mention


class MentionPipeline:
    """Main NLP pipeline for extracting character mention graphs."""

    def __init__(self, default_language: str = "ru"):
        self.default_language = default_language

    def process(self, text: str, language: str | None = None) -> AnalysisResponse:
        lang = language or self.default_language

        # Step 1: Extract named entities (characters)
        entities = extract_entities(text, lang)

        # Step 2: Resolve coreferences (pronouns -> characters)
        resolved = resolve_coreferences(text, entities, lang)

        # Step 3: Disambiguate entities with same/similar names
        unique_characters = disambiguate_entities(resolved)

        # Step 4: Extract mentions (who mentions whom)
        mentions = extract_mentions(text, unique_characters, lang)

        # Build response
        characters = [
            Character(
                id=char["id"],
                name=char["name"],
                aliases=char.get("aliases", []),
                mention_count=char.get("mention_count", 0),
            )
            for char in unique_characters
        ]

        mention_objects = [
            Mention(
                source=m["source"],
                target=m["target"],
                context=m["context"],
                chapter=m.get("chapter"),
                date=m.get("date"),
            )
            for m in mentions
        ]

        return AnalysisResponse(
            graph=GraphData(characters=characters, mentions=mention_objects),
            timeline=mention_objects,
            metadata={"language": lang, "text_length": len(text)},
        )
