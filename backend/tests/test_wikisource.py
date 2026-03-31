"""Tests for Wikisource client.

Tests marked with @pytest.mark.network require internet access
and are skipped by default. Run with: pytest -m network
"""

import pytest

from sources.wikisource import (
    WikisourceClient,
    _html_to_text,
    _strip_html,
)


# ── Unit tests (no network) ────────────────────────────────


def test_strip_html():
    assert _strip_html("<b>bold</b> text") == "bold text"
    assert _strip_html("no tags") == "no tags"


def test_html_to_text_basic():
    html = "<p>First paragraph.</p><p>Second paragraph.</p>"
    text = _html_to_text(html)
    assert "First paragraph." in text
    assert "Second paragraph." in text


def test_html_to_text_entities():
    html = "Толстой&nbsp;&mdash; великий писатель"
    text = _html_to_text(html)
    assert text == "Толстой — великий писатель"


def test_html_to_text_footnotes():
    html = "Текст с примечанием[1] и ещё одним[2]."
    text = _html_to_text(html)
    assert "[1]" not in text
    assert "[2]" not in text
    assert "Текст с примечанием" in text


def test_html_to_text_br():
    html = "Строка 1<br/>Строка 2<br>Строка 3"
    text = _html_to_text(html)
    assert "Строка 1\nСтрока 2\nСтрока 3" == text


def test_html_to_text_noexport():
    html = (
        '<div class="ws-noexport">Navigation</div>'
        '<p>Actual content here.</p>'
    )
    text = _html_to_text(html)
    assert "Navigation" not in text
    assert "Actual content" in text


# ── Integration tests (network required) ────────────────────


@pytest.mark.network
def test_search():
    with WikisourceClient() as client:
        results = client.search("Война и мир Толстой", limit=3)
        assert len(results) > 0
        assert any("Война и мир" in r["title"] for r in results)


@pytest.mark.network
def test_get_metadata():
    with WikisourceClient() as client:
        meta = client.get_metadata("Война и мир (Толстой)/Том 1")
        assert "Война и мир" in meta.title
        assert len(meta.categories) > 0


@pytest.mark.network
def test_get_page_text():
    with WikisourceClient() as client:
        text = client.get_page_text("Война и мир (Толстой)/Том 1")
        assert len(text) > 1000
        assert "князь" in text.lower() or "Болконский" in text


@pytest.mark.network
def test_get_subpages():
    with WikisourceClient() as client:
        subpages = client.get_subpages("Евгений Онегин (Пушкин)")
        assert len(subpages) > 0


@pytest.mark.network
def test_get_category_members():
    with WikisourceClient() as client:
        members = client.get_category_members(
            "Война и мир (Толстой)", limit=5
        )
        assert len(members) > 0
