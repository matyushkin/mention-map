"""Client for ru.wikisource.org MediaWiki API.

Extracts structured literary texts with metadata from Russian Wikisource.
Handles pagination, table of contents navigation, and wiki markup cleanup.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

API_URL = "https://ru.wikisource.org/w/api.php"
USER_AGENT = "MentionMap/0.1 (https://github.com/matyushkin/mention-map)"
REQUEST_DELAY = 1.0  # seconds between requests, per Wikimedia policy


@dataclass
class TextMetadata:
    title: str
    author: str = ""
    created: str = ""
    published: str = ""
    source: str = ""
    categories: list[str] = field(default_factory=list)


@dataclass
class TextChapter:
    title: str
    number: int | None = None
    text: str = ""
    page_title: str = ""


@dataclass
class WikisourceText:
    metadata: TextMetadata
    chapters: list[TextChapter] = field(default_factory=list)
    full_text: str = ""

    @property
    def total_length(self) -> int:
        return len(self.full_text) or sum(len(ch.text) for ch in self.chapters)


class WikisourceClient:
    """Client for fetching texts from ru.wikisource.org."""

    def __init__(self, delay: float = REQUEST_DELAY, maxlag: int = 5):
        self.delay = delay
        self.maxlag = maxlag
        self._last_request = 0.0
        self._client = httpx.Client(
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "gzip",
            },
            timeout=30.0,
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _throttle(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    def _api_get(self, **params) -> dict:
        self._throttle()
        params.setdefault("format", "json")
        params.setdefault("maxlag", self.maxlag)
        for attempt in range(5):
            resp = self._client.get(API_URL, params=params)
            # Respect maxlag: server asks us to retry later
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data and data["error"].get("code") == "maxlag":
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning(
                        "Server lagged, waiting %ds (attempt %d/5)",
                        retry_after, attempt + 1,
                    )
                    time.sleep(retry_after)
                    continue
                return data
            resp.raise_for_status()
        raise RuntimeError("maxlag retries exhausted")

    # ── Search ──────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search for texts on Wikisource."""
        data = self._api_get(
            action="query",
            list="search",
            srsearch=query,
            srlimit=limit,
        )
        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item["title"],
                "snippet": _strip_html(item.get("snippet", "")),
                "size": item.get("size", 0),
                "word_count": item.get("wordcount", 0),
            })
        return results

    # ── Metadata ────────────────────────────────────────────

    def get_metadata(self, page_title: str) -> TextMetadata:
        """Extract metadata from a Wikisource page's wikitext templates."""
        data = self._api_get(
            action="parse",
            page=page_title,
            prop="wikitext|categories",
        )
        parse = data.get("parse", {})
        wikitext = parse.get("wikitext", {}).get("*", "")
        categories = [
            cat["*"] for cat in parse.get("categories", [])
        ]

        metadata = TextMetadata(title=page_title, categories=categories)

        # Parse common Wikisource templates (ТолстойПСС, Отексте, etc.)
        templates = {
            "НАЗВАНИЕ": "title",
            "АВТОР": "author",
            "ДАТАСОЗДАНИЯ": "created",
            "ДАТАПУБЛИКАЦИИ": "published",
            "ИСТОЧНИК": "source",
        }
        for key, attr in templates.items():
            match = re.search(
                rf"\|\s*{key}\s*=\s*(.+?)(?:\n|\||\}})", wikitext
            )
            if match:
                setattr(metadata, attr, match.group(1).strip())

        # Try to extract author from "Автор:" link pattern
        if not metadata.author:
            author_match = re.search(
                r"Автор:([^\]|]+)", wikitext
            )
            if author_match:
                metadata.author = author_match.group(1).strip()

        return metadata

    # ── Page text ───────────────────────────────────────────

    def get_page_text(self, page_title: str) -> str:
        """Get clean plain text from a Wikisource page (rendered HTML → text)."""
        data = self._api_get(
            action="parse",
            page=page_title,
            prop="text",
        )
        html = data.get("parse", {}).get("text", {}).get("*", "")
        return _html_to_text(html)

    # ── Subpages / Table of Contents ────────────────────────

    def get_subpages(self, page_title: str, limit: int = 500) -> list[str]:
        """Get all subpages of a given page (e.g., chapters of a book)."""
        data = self._api_get(
            action="query",
            list="allpages",
            apprefix=page_title + "/",
            apnamespace=0,
            aplimit=limit,
        )
        return [
            p["title"]
            for p in data.get("query", {}).get("allpages", [])
        ]

    def get_links(self, page_title: str) -> list[str]:
        """Get internal links from a page (useful for TOC navigation)."""
        data = self._api_get(
            action="parse",
            page=page_title,
            prop="links",
        )
        return [
            link["*"]
            for link in data.get("parse", {}).get("links", [])
            if link.get("ns", 0) == 0
        ]

    # ── Full text extraction ────────────────────────────────

    def fetch_work(
        self,
        page_title: str,
        with_chapters: bool = True,
    ) -> WikisourceText:
        """Fetch a complete literary work with metadata and chapter structure.

        Args:
            page_title: Main page title (e.g., "Война и мир (Толстой)/Том 1")
            with_chapters: If True, fetch individual chapters separately.
                If False, fetch the full text from the main page.
        """
        logger.info("Fetching metadata for %s", page_title)
        metadata = self.get_metadata(page_title)
        result = WikisourceText(metadata=metadata)

        if with_chapters:
            subpages = self.get_subpages(page_title)
            if not subpages:
                # No subpages — try links from the page itself
                subpages = [
                    link for link in self.get_links(page_title)
                    if link.startswith(page_title + "/")
                ]

            logger.info("Found %d chapters for %s", len(subpages), page_title)

            for i, subpage in enumerate(subpages):
                logger.info("Fetching chapter %d/%d: %s", i + 1, len(subpages), subpage)
                text = self.get_page_text(subpage)
                chapter_title = subpage.rsplit("/", 1)[-1]
                result.chapters.append(TextChapter(
                    title=chapter_title,
                    number=i + 1,
                    text=text,
                    page_title=subpage,
                ))

            result.full_text = "\n\n".join(ch.text for ch in result.chapters)
        else:
            logger.info("Fetching full text for %s", page_title)
            result.full_text = self.get_page_text(page_title)

        return result

    # ── Author works ────────────────────────────────────────

    def get_author_works(self, author_page: str) -> list[str]:
        """Get list of works linked from an author's page.

        Args:
            author_page: e.g., "Автор:Лев Николаевич Толстой"
        """
        links = self.get_links(author_page)
        # Filter out meta pages, keeping actual works
        return [
            link for link in links
            if not link.startswith(("Автор:", "Категория:", "Обсуждение:"))
            and ":" not in link
        ]

    # ── Category browsing ───────────────────────────────────

    def get_category_members(
        self, category: str, limit: int = 50
    ) -> list[dict]:
        """List pages in a Wikisource category.

        Args:
            category: Category name without "Категория:" prefix
        """
        data = self._api_get(
            action="query",
            list="categorymembers",
            cmtitle=f"Категория:{category}",
            cmlimit=limit,
            cmprop="title|sortkeyprefix",
        )
        return data.get("query", {}).get("categorymembers", [])


# ── HTML / wikitext cleanup ─────────────────────────────────


def _strip_html(text: str) -> str:
    """Remove HTML tags."""
    return re.sub(r"<[^>]+>", "", text)


def _remove_tagged_blocks(html: str, tag: str, attr_pattern: str) -> str:
    """Remove HTML blocks by tag and attribute pattern, handling nested tags."""
    pattern = re.compile(
        rf"<{tag}[^>]*{attr_pattern}[^>]*>", re.DOTALL
    )
    open_tag = re.compile(rf"<{tag}[\s>]", re.IGNORECASE)
    close_tag = re.compile(rf"</{tag}\s*>", re.IGNORECASE)

    result = html
    while True:
        match = pattern.search(result)
        if not match:
            break
        start = match.start()
        depth = 1
        pos = match.end()
        while depth > 0 and pos < len(result):
            next_open = open_tag.search(result, pos)
            next_close = close_tag.search(result, pos)
            if next_close is None:
                break
            if next_open and next_open.start() < next_close.start():
                depth += 1
                pos = next_open.end()
            else:
                depth -= 1
                pos = next_close.end()
        result = result[:start] + result[pos:]
    return result


def _html_to_text(html: str) -> str:
    """Convert rendered Wikisource HTML to clean plain text."""
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL)

    # FIRST: remove all ws-noexport blocks (license, headers, nav)
    # These contain nested tags, so use balanced-tag removal
    text = _remove_tagged_blocks(text, "table", r"\bws-noexport\b")
    text = _remove_tagged_blocks(text, "div", r"\bws-noexport\b")
    # Remove license containers
    text = _remove_tagged_blocks(text, "div", r"\blicenseContainer\b")

    # Remove Wikisource header/navigation template (contains title, author, nav arrows)
    text = re.sub(r'<div\s+id="headertemplate">.*?</div>\s*(?:</div>\s*)*<div\s+class="mw-parser-output-content">', "", text, flags=re.DOTALL)
    text = re.sub(r'<div\s+id="headertemplate">.*?(?=<div\s+class="poem">|<div\s+class="mw-parser-output-content">|<p>)', "", text, flags=re.DOTALL)
    text = re.sub(r'<div[^>]*\bid="headertemplate"[^>]*>.*?</table>\s*</div>', "", text, flags=re.DOTALL)
    # Remove header notes (source info, dates)
    text = re.sub(r'<table[^>]*\bheader_notes\b[^>]*>.*?</table>', "", text, flags=re.DOTALL)
    text = re.sub(r'<div[^>]*\bid="extra_nav"[^>]*>.*?</div>\s*</div>', "", text, flags=re.DOTALL)
    # Remove navigation elements
    text = re.sub(r'<div[^>]*class="[^"]*mw-heading[^"]*"[^>]*>.*?</div>', "", text, flags=re.DOTALL)
    # Remove category links at the bottom
    text = re.sub(r'<div[^>]*class="[^"]*catlinks[^"]*"[^>]*>.*?</div>', "", text, flags=re.DOTALL)

    # Replace <br> and <p> with newlines
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</p>", "\n", text)
    text = re.sub(r"<p[^>]*>", "", text)

    # Remove footnote references [1], [2], etc.
    text = re.sub(r"\[(\d+)\]", "", text)

    # Remove remaining HTML tags
    text = _strip_html(text)

    # Decode HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&#160;", " ")
    text = text.replace("&mdash;", "—")
    text = text.replace("&laquo;", "«")
    text = text.replace("&raquo;", "»")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&#91;", "[")
    text = text.replace("&#93;", "]")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text
