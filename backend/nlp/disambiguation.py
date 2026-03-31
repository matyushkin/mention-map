import uuid


def disambiguate_entities(entities: list[dict]) -> list[dict]:
    """Disambiguate entities that may refer to the same or different people.

    Handles cases like:
    - "Александр" could be Alexander I or Alexander Pushkin
    - "Наташа" and "Наталья Ростова" are the same character

    TODO: Implement clustering based on context similarity,
    co-occurrence patterns, and knowledge base lookup.
    Current implementation assigns unique IDs and counts mentions.
    """
    result = []
    for entity in entities:
        result.append({
            "id": str(uuid.uuid4())[:8],
            "name": entity["name"],
            "aliases": [],
            "mention_count": len(entity.get("spans", [])),
        })
    return result
