"""Unit tests for the tool logic. No API key needed: pytest tests/"""

import json

from writer_agent import logic
from writer_agent.resources import load_resources

BRIEF = {
    "objective": "Test topic",
    "findings": [
        {"claim": "A", "source_url": "https://a.example", "source_title": "Source A"},
        {"claim": "B", "source_url": "https://b.example"},
        {"claim": "A again", "source_url": "https://a.example"},
    ],
}
RESOURCES = {
    "brief.json": {"description": "brief", "text": json.dumps(BRIEF)},
    "notes.md": {"description": "notes", "text": "# Just notes"},
}


def test_extract_sources_dedupes_and_numbers():
    sources = logic.extract_sources(RESOURCES)
    assert [s["number"] for s in sources] == [1, 2]
    assert sources[0]["title"] == "Source A"
    assert sources[1]["title"] == "https://b.example"  # falls back to URL


def test_check_draft_catches_missing_and_invalid_citations():
    sources = logic.extract_sources(RESOURCES)
    result = logic.check_draft(
        ["Intro", "Details"], {"Intro": "Claim [1] and [9]."}, sources
    )
    assert any("Details" in e for e in result["errors"])  # not written
    assert any("[9]" in e for e in result["errors"])  # invalid citation


def test_check_draft_passes_good_draft_and_warns_on_length():
    sources = logic.extract_sources(RESOURCES)
    sections = {"Intro": "Claim one [1].", "Details": "Claim two [2]."}
    result = logic.check_draft(["Intro", "Details"], sections, sources, target_words=500)
    assert result["errors"] == []
    assert any("target" in w for w in result["warnings"])


def test_render_lists_only_cited_sources():
    sources = logic.extract_sources(RESOURCES)
    md = logic.render_markdown("T", ["Intro"], {"Intro": "Text [2]."}, sources)
    assert "## References" in md
    assert "https://b.example" in md and "https://a.example" not in md


def test_safe_filename():
    assert logic.safe_filename("What is MCP? A guide!") == "what-is-mcp-a-guide"
    assert logic.safe_filename("../../etc") == "etc"


def test_load_sample_resources():
    resources = load_resources("sample_data")
    assert "research_brief.json" in resources
    assert resources["research_brief.json"]["description"].startswith("Research brief")
