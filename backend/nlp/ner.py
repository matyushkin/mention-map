import spacy


_models: dict[str, spacy.Language] = {}

LANG_MODELS = {
    "ru": "ru_core_news_lg",
    "en": "en_core_web_trf",
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
}


def _get_model(lang: str) -> spacy.Language:
    if lang not in _models:
        model_name = LANG_MODELS.get(lang)
        if model_name is None:
            raise ValueError(f"Unsupported language: {lang}")
        _models[lang] = spacy.load(model_name)
    return _models[lang]


def extract_entities(text: str, lang: str) -> list[dict]:
    """Extract person entities from text using spaCy NER."""
    nlp = _get_model(lang)
    doc = nlp(text)

    entities = []
    seen_names: set[str] = set()

    for ent in doc.ents:
        if ent.label_ == "PER" and ent.text not in seen_names:
            seen_names.add(ent.text)
            entities.append({
                "name": ent.text,
                "spans": [(ent.start_char, ent.end_char)],
                "label": ent.label_,
            })
        elif ent.label_ == "PER" and ent.text in seen_names:
            for e in entities:
                if e["name"] == ent.text:
                    e["spans"].append((ent.start_char, ent.end_char))
                    break

    return entities
