# Mention Map

NLP-powered graph of character mentions in literary and historical texts.

Mention Map analyzes texts (diaries, books, memoirs), extracts characters and their mentions of each other, and builds an interactive visual graph of connections — with chronology, multilingual support, and disambiguation of ambiguous references.

## Architecture

```
mention-map/
├── backend/           # Python (FastAPI)
│   ├── api/           # REST API endpoints
│   ├── nlp/           # NER, coreference resolution, entity linking
│   ├── timeline/      # chronological analysis
│   └── tests/
├── frontend/          # React + graph visualization
│   └── src/
│       ├── components/
│       ├── graph/     # D3.js / Cytoscape.js rendering
│       ├── timeline/  # timeline of mentions
│       └── upload/    # text upload UI
└── examples/          # sample texts and pre-built graphs
```

## NLP Pipeline

1. **Text ingestion** — upload and segment into chapters/entries (with dates if available)
2. **NER** — extract character names across multiple languages
3. **Coreference resolution** — resolve pronouns ("he", "she") to specific characters
4. **Disambiguation** — handle cases where the same name refers to different people
5. **Mention extraction** — detect who mentions whom, in what context
6. **Graph construction** — build weighted, chronological graph of connections

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, spaCy, Stanza
- **Frontend**: React, TypeScript, D3.js
- **Deployment**: Docker, configurable for Vercel/Railway

## Development

### Backend

```bash
cd backend
uv sync
uv run fastapi dev api/main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## License

MIT
