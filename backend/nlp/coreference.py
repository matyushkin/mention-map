def resolve_coreferences(
    text: str, entities: list[dict], lang: str
) -> list[dict]:
    """Resolve coreferences — map pronouns to character entities.

    TODO: Integrate a coreference resolution model.
    Current implementation returns entities unchanged as a placeholder.
    Candidates:
    - For English: neuralcoref, coreferee, or LLM-based
    - For Russian: DeepPavlov coref, or LLM-based
    """
    return entities
