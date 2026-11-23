# Writer Agent

The Writer agent of our Agent Handoff Protocol (AHP) project. It turns the
Researcher's brief into a structured, cited markdown document. It is built with
LangChain's `create_agent`, which runs on LangGraph. Its working state (outline,
sections, gaps) lives in LangGraph state and is saved per thread.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your OPENAI_API_KEY in .env
```

## Run

```bash
python run_writer.py \
  --objective "Write a 400-word explainer on what MCP is, for CS students." \
  --resources sample_data \
  --thread demo-1
```

You will see each tool call as it happens. The draft is saved in `output/`.
After the first draft you can type follow-ups ("make it shorter") on the same
thread. If the research is not enough, the agent writes `output/handback.json`
with its questions for the Researcher.

`sample_data/research_brief.json` is test data. Replace it with real Researcher output.

## Test (no API key needed)

```bash
pytest
```

- `tests/test_logic.py` checks each tool's logic on its own.
- `tests/test_agent_smoke.py` runs the whole agent loop with a scripted fake model.

## Tools

| Tool | What it does |
| --- | --- |
| `list_resources` | Lists the materials handed over |
| `read_resource` | Reads one resource in full |
| `get_sources` | Numbered source list to cite as [1], [2] |
| `create_outline` | Sets the title and section plan |
| `write_section` | Writes or rewrites one section |
| `read_draft` | Shows the whole draft as it stands |
| `check_draft` | Checks sections, citations and length |
| `save_draft` | Saves the markdown file (refuses if errors remain) |
| `request_more_research` | Ends the turn and hands questions back to the Researcher |

## Input contract (shared with the Researcher team)

The writer reads a research brief in this JSON shape:

```json
{
  "objective": "the question that was researched",
  "findings": [
    {"claim": "...", "source_url": "https://...", "source_title": "...", "note": "..."}
  ],
  "open_questions": ["..."]
}
```

## Layout

```
writer_agent/
  state.py      # WriterState: resources, outline, sections, gaps
  logic.py      # plain Python helpers (unit-tested)
  tools.py      # the 9 tools
  prompts.py    # system prompt
  agent.py      # build_writer_agent()
  resources.py  # load resources from a folder
run_writer.py   # terminal runner
```
