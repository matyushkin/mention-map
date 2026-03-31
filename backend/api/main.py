from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AnalysisRequest, AnalysisResponse, GraphData
from nlp.pipeline import MentionPipeline

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
