from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AnalysisRequest, AnalysisResponse, GraphData
from nlp.pipeline import MentionPipeline
from sources.wikisource import WikisourceClient

app = FastAPI(
    title="Mention Map API",
    description="NLP-powered character mention graph extraction",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = MentionPipeline()
wikisource = WikisourceClient()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze text and extract character mention graph."""
    result = pipeline.process(request.text, language=request.language)
    return result


@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload a text file for analysis."""
    content = await file.read()
    text = content.decode("utf-8")
    result = pipeline.process(text)
    return result


# ── Wikisource endpoints ────────────────────────────────────


@app.get("/wikisource/search")
async def wikisource_search(query: str, limit: int = 10):
    """Search for texts on ru.wikisource.org."""
    return wikisource.search(query, limit=limit)


@app.get("/wikisource/metadata")
async def wikisource_metadata(page: str):
    """Get metadata for a Wikisource page."""
    meta = wikisource.get_metadata(page)
    return {
        "title": meta.title,
        "author": meta.author,
        "created": meta.created,
        "published": meta.published,
        "source": meta.source,
        "categories": meta.categories,
    }


@app.get("/wikisource/text")
async def wikisource_text(page: str):
    """Get plain text from a Wikisource page."""
    text = wikisource.get_page_text(page)
    return {"page": page, "text": text, "length": len(text)}


@app.post("/wikisource/analyze")
async def wikisource_analyze(page: str, language: str = "ru"):
    """Fetch a Wikisource page and run the NLP mention pipeline on it."""
    text = wikisource.get_page_text(page)
    result = pipeline.process(text, language=language)
    return result
