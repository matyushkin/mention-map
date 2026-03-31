"""Local corpus loader.

Loads pre-extracted texts from the data/ directory.
Each work is a JSON file with metadata and chapter structure.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass
class Chapter:
    title: str
    number: int
    text: str
    char_count: int


@dataclass
class Work:
    slug: str
    title: str
    author: str
    year: int
    genre: str
    language: str
    chapters: list[Chapter]
    stats: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(ch.text for ch in self.chapters)

    @property
    def total_chars(self) -> int:
        return self.stats.get("total_chars", sum(ch.char_count for ch in self.chapters))


def list_works(data_dir: Path = DATA_DIR) -> list[dict]:
    """List all available works in the corpus."""
    index_file = data_dir / "index.json"
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            return json.load(f)

    # Fallback: scan directory
    works = []
    for path in sorted(data_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        works.append({
            "slug": data["slug"],
            "title": data["title"],
            "author": data["author"],
            "year": data["year"],
            "genre": data["genre"],
            "file": path.name,
        })
    return works


def load_work(slug: str, data_dir: Path = DATA_DIR) -> Work:
    """Load a single work by slug."""
    path = data_dir / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"Work '{slug}' not found in {data_dir}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chapters = [
        Chapter(
            title=ch["title"],
            number=ch["number"],
            text=ch["text"],
            char_count=ch["char_count"],
        )
        for ch in data["chapters"]
    ]

    return Work(
        slug=data["slug"],
        title=data["title"],
        author=data["author"],
        year=data["year"],
        genre=data["genre"],
        language=data["language"],
        chapters=chapters,
        stats=data.get("stats", {}),
    )


def load_all(data_dir: Path = DATA_DIR) -> list[Work]:
    """Load all works in the corpus."""
    works = []
    for entry in list_works(data_dir):
        works.append(load_work(entry["slug"], data_dir))
    return works
