from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    text: str
    language: str | None = None


class Character(BaseModel):
    id: str
    name: str
    aliases: list[str] = []
    mention_count: int = 0


class Mention(BaseModel):
    source: str
    target: str
    context: str
    chapter: str | None = None
    date: str | None = None


class GraphData(BaseModel):
    characters: list[Character]
    mentions: list[Mention]


class AnalysisResponse(BaseModel):
    graph: GraphData
    timeline: list[Mention] = []
    metadata: dict = {}
