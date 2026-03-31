from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AnalysisRequest, AnalysisResponse, GraphData
from nlp.pipeline import MentionPipeline
from sources.corpus import list_works, load_work
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


# ── Corpus endpoints ────────────────────────────────────────


@app.get("/corpus")
async def corpus_list():
    """List all available works in the local corpus."""
    return list_works()


@app.get("/corpus/{slug}")
async def corpus_work(slug: str):
    """Get metadata and chapter list for a work."""
    work = load_work(slug)
    return {
        "slug": work.slug,
        "title": work.title,
        "author": work.author,
        "year": work.year,
        "genre": work.genre,
        "language": work.language,
        "stats": work.stats,
        "chapters": [
            {"title": ch.title, "number": ch.number, "char_count": ch.char_count}
            for ch in work.chapters
        ],
    }


@app.get("/corpus/{slug}/text")
async def corpus_text(slug: str, chapter: int | None = None):
    """Get text of a work or a specific chapter."""
    work = load_work(slug)
    if chapter is not None:
        for ch in work.chapters:
            if ch.number == chapter:
                return {"chapter": ch.title, "text": ch.text}
        return {"error": f"Chapter {chapter} not found"}
    return {"text": work.full_text, "length": work.total_chars}


@app.post("/corpus/{slug}/analyze")
async def corpus_analyze(slug: str, chapter: int | None = None):
    """Run NLP mention pipeline on a corpus work."""
    work = load_work(slug)
    if chapter is not None:
        for ch in work.chapters:
            if ch.number == chapter:
                return pipeline.process(ch.text, language=work.language)
        return {"error": f"Chapter {chapter} not found"}
    return pipeline.process(work.full_text, language=work.language)
