"""The writer agent's tools.

Tools READ the agent state through `runtime.state` and WRITE to it by
returning a LangGraph `Command(update=...)`. That keeps the outline and
draft in the graph's state (saved per thread by the checkpointer) instead
of only inside the chat history.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from . import logic


def _output_dir() -> Path:
    path = Path(os.getenv("WRITER_OUTPUT_DIR", "output"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reply(runtime: ToolRuntime, text: str, **updates) -> Command:
    """Update state and answer the tool call in one step."""
    return Command(
        update={
            **updates,
            "messages": [ToolMessage(text, tool_call_id=runtime.tool_call_id)],
        }
    )


def _objective(state: dict) -> str:
    """The task the writer was given: the first user message."""
    for message in state.get("messages", []):
        if getattr(message, "type", "") == "human":
            return message.text if hasattr(message, "text") else str(message.content)
    return ""


# ------------------------------------------------------------ 1. inputs
@tool
def list_resources(runtime: ToolRuntime) -> str:
    """List the research materials you were handed, with a short description
    of each. Call this first, before planning anything."""
    resources = runtime.state.get("resources") or {}
    if not resources:
        return (
            "No resources were provided. You have no research to write from: "
            "call request_more_research."
        )
    return "\n".join(f"- {name}: {r['description']}" for name, r in resources.items())


@tool
def read_resource(name: str, runtime: ToolRuntime) -> str:
    """Read the full text of one resource by its name (from list_resources)."""
    resources = runtime.state.get("resources") or {}
    if name not in resources:
        return f"No resource named '{name}'. Available: {', '.join(resources) or 'none'}."
    return resources[name]["text"]


@tool
def get_sources(runtime: ToolRuntime) -> str:
    """Get the numbered list of sources from the research. Cite them in your
    text as [1], [2], ... using exactly these numbers. Never cite anything else."""
    sources = logic.extract_sources(runtime.state.get("resources") or {})
    if not sources:
        return "No sources found in the resources. Do not cite anything."
    return logic.format_source_list(sources)


# ------------------------------------------------------- 2. plan & write
@tool
def create_outline(title: str, sections: list[str], runtime: ToolRuntime) -> Command:
    """Plan the document: set its title and the ordered list of section
    headings. Call again to change the plan. Sections already written are kept
    if their heading stays in the outline."""
    clean = [s.strip() for s in sections if s.strip()]
    if not clean:
        return _reply(runtime, "The outline needs at least one section.")
    if len(set(clean)) != len(clean):
        return _reply(runtime, "Section headings must be unique.")
    kept = {h: b for h, b in (runtime.state.get("sections") or {}).items() if h in clean}
    return _reply(
        runtime,
        f"Outline set: '{title}' with sections: {', '.join(clean)}.",
        title=title.strip(),
        outline=clean,
        sections=kept,
    )


@tool
def write_section(heading: str, content: str, runtime: ToolRuntime) -> Command:
    """Write or rewrite ONE section of the draft. `heading` must match a heading
    from the outline exactly. `content` is markdown, without the heading line.
    Support factual claims with citations like [1]."""
    outline = runtime.state.get("outline") or []
    if heading not in outline:
        return _reply(
            runtime,
            f"'{heading}' is not in the outline {outline}. Use an outline heading, "
            "or update the outline with create_outline first.",
        )
    sections = dict(runtime.state.get("sections") or {})
    action = "Rewrote" if heading in sections else "Wrote"
    sections[heading] = content.strip()
    return _reply(
        runtime,
        f"{action} section '{heading}' ({logic.word_count(content)} words).",
        sections=sections,
    )


@tool
def read_draft(runtime: ToolRuntime) -> str:
    """Read the whole draft as it stands, to check flow and consistency."""
    state = runtime.state
    if not state.get("outline"):
        return "There is no draft yet. Call create_outline first."
    sources = logic.extract_sources(state.get("resources") or {})
    return logic.render_markdown(
        state.get("title", ""), state["outline"], state.get("sections") or {}, sources
    )


# ---------------------------------------------------- 3. check & deliver
@tool
def check_draft(runtime: ToolRuntime, target_words: int | None = None) -> str:
    """Run quality checks on the draft: missing or empty sections, citations
    that don't match a source, sections without citations, and length against
    `target_words` if the task gave one. Fix every error before saving."""
    state = runtime.state
    result = logic.check_draft(
        state.get("outline") or [],
        state.get("sections") or {},
        logic.extract_sources(state.get("resources") or {}),
        target_words,
    )
    lines = [f"Word count: {result['word_count']}"]
    lines += [f"ERROR: {e}" for e in result["errors"]]
    lines += [f"WARNING: {w}" for w in result["warnings"]]
    if not result["errors"] and not result["warnings"]:
        lines.append("All checks passed.")
    return "\n".join(lines)


@tool
def save_draft(runtime: ToolRuntime) -> Command:
    """Save the finished draft as a markdown file. Refuses if check_draft would
    report errors. Call this once the draft is complete."""
    state = runtime.state
    outline = state.get("outline") or []
    sections = state.get("sections") or {}
    sources = logic.extract_sources(state.get("resources") or {})

    errors = logic.check_draft(outline, sections, sources)["errors"]
    if errors:
        return _reply(runtime, "Not saved. Fix these first:\n" + "\n".join(errors))

    title = state.get("title", "")
    path = _output_dir() / f"{logic.safe_filename(title)}.md"
    path.write_text(logic.render_markdown(title, outline, sections, sources), encoding="utf-8")
    return _reply(runtime, f"Draft saved to {path}", saved_path=str(path))


@tool(return_direct=True)
def request_more_research(gaps: list[str], runtime: ToolRuntime) -> Command:
    """Use when the resources are not enough to meet the objective without
    guessing. List each missing piece as a specific research question. This
    ends your turn and sends the task back to the researcher."""
    gaps = [g.strip() for g in gaps if g.strip()]
    if not gaps:
        return _reply(runtime, "List at least one specific question.")

    # Standalone for now: write the hand-back to a file. Later this becomes
    # the AHP handoff (objective + gaps + draft so far) to the researcher.
    state = runtime.state
    handback = {
        "objective": _objective(state),
        "gaps": gaps,
        "draft_so_far": state.get("sections") or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _output_dir() / "handback.json"
    path.write_text(json.dumps(handback, indent=2, ensure_ascii=False), encoding="utf-8")
    text = "Sent back to the researcher. Open questions:\n" + "\n".join(f"- {g}" for g in gaps)
    return _reply(runtime, text, gaps=gaps)


WRITER_TOOLS = [
    list_resources,
    read_resource,
    get_sources,
    create_outline,
    write_section,
    read_draft,
    check_draft,
    save_draft,
    request_more_research,
]
