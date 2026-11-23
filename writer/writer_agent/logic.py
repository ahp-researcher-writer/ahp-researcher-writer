"""Plain Python helpers used by the tools.

Keeping the logic here (no LLM, no LangChain) means every piece can be
unit-tested on its own: "check each tool works" before wiring the agent.
"""

from __future__ import annotations

import json
import re

CITATION_RE = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------- sources
def extract_sources(resources: dict) -> list[dict]:
    """Collect the unique sources from every research-brief resource.

    A research brief is JSON with a `findings` list, where each finding has
    `source_url` (and optionally `source_title`). Sources are numbered in
    the order they first appear, so the writer can cite them as [1], [2]...
    Resources that are not JSON briefs are ignored.
    """
    sources: list[dict] = []
    seen: set[str] = set()
    for resource in resources.values():
        try:
            brief = json.loads(resource.get("text", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(brief, dict):
            continue
        for finding in brief.get("findings", []):
            url = (finding.get("source_url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append(
                {
                    "number": len(sources) + 1,
                    "title": finding.get("source_title") or url,
                    "url": url,
                }
            )
    return sources


def format_source_list(sources: list[dict]) -> str:
    return "\n".join(f"[{s['number']}] {s['title']} - {s['url']}" for s in sources)


def find_citations(text: str) -> set[int]:
    return {int(n) for n in CITATION_RE.findall(text)}


# ------------------------------------------------------------------ draft
def render_markdown(
    title: str, outline: list[str], sections: dict[str, str], sources: list[dict]
) -> str:
    """Assemble the draft in outline order, with a References section
    listing only the sources that were actually cited."""
    parts = [f"# {title or 'Untitled draft'}"]
    for heading in outline:
        body = sections.get(heading, "").strip()
        parts.append(f"## {heading}\n\n{body or '_(not written yet)_'}")

    cited = set()
    for body in sections.values():
        cited |= find_citations(body)
    used = [s for s in sources if s["number"] in cited]
    if used:
        parts.append("## References\n\n" + "\n".join(
            f"{s['number']}. [{s['title']}]({s['url']})" for s in used
        ))
    return "\n\n".join(parts) + "\n"


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def check_draft(
    outline: list[str],
    sections: dict[str, str],
    sources: list[dict],
    target_words: int | None = None,
) -> dict:
    """Deterministic quality checks. Returns errors (must fix before saving)
    and warnings (worth fixing)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not outline:
        errors.append("There is no outline yet. Call create_outline first.")

    for heading in outline:
        if not sections.get(heading, "").strip():
            errors.append(f"Section '{heading}' is not written yet.")

    valid_numbers = {s["number"] for s in sources}
    all_cited: set[int] = set()
    for heading, body in sections.items():
        cited = find_citations(body)
        all_cited |= cited
        invalid = sorted(cited - valid_numbers)
        if invalid:
            errors.append(
                f"Section '{heading}' cites {invalid}, which are not in the source list."
            )
        if sources and not cited:
            warnings.append(f"Section '{heading}' has no citations.")

    if sources and not all_cited:
        errors.append("The draft cites none of the sources.")

    total = sum(word_count(b) for b in sections.values())
    if target_words:
        low, high = int(target_words * 0.85), int(target_words * 1.15)
        if not low <= total <= high:
            warnings.append(
                f"Draft is {total} words; target is about {target_words} ({low}-{high})."
            )

    return {"errors": errors, "warnings": warnings, "word_count": total}


def safe_filename(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "draft").lower()).strip("-")
    return (slug or "draft")[:60]
