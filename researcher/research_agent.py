"""Full Single-Agent Research System (one file).

clarify -> research brief -> ONE researcher agent (search loop + compression) -> final report
No multi-agent supervisor, no agent-to-agent handoff, no AHP.

Needs alongside it: prompt.py, state_research.py, utils.py (your existing files).
"""

import os
import operator
from typing import Optional
from pydantic import BaseModel, Field
from typing_extensions import Literal, Annotated, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, filter_messages, get_buffer_string,
)
from langchain.chat_models import init_chat_model

from state_research import (
    ResearcherState, ResearcherOutputState, ClarifyWithUser, ResearchQuestion,
)
from utils import tavily_search, get_today_str, think_tool
from prompt import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    research_agent_prompt,
    compress_research_system_prompt,
    compress_research_human_message,
    final_report_generation_prompt,
)

# ===== CONFIGURATION =====
load_dotenv()

MAX_TOOL_ITERATIONS = 10  # search/think rounds before the researcher is forced to answer


def make_model(**kwargs):
    return init_chat_model(
        model="openrouter/free",
        model_provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        max_retries=3,
        **kwargs,
    )


tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

model = make_model()
model_with_tools = model.bind_tools(tools)
compress_model = make_model()
writer_model = make_model()  # model="anthropic:claude-sonnet-4-6", max_tokens=64000


# ===== RESEARCHER -> WRITER CONTRACT =====

class Finding(BaseModel):
    claim: str = Field(description="One self-contained factual claim.")
    source_url: str = Field(description="URL of the source. Must appear in the research text; empty if unknown.")
    source_title: str = Field(description="Title of that source.")
    note: str = Field(default="", description="Caveat such as a conflict, weak source or date limit. Empty if none.")


class ResearchOutput(BaseModel):
    objective: str
    findings: List[Finding]
    open_questions: List[str] = Field(default_factory=list)


STRUCTURE_PROMPT = """Convert the research text below into structured findings for a report writer.

OBJECTIVE:
{brief}

RESEARCH TEXT:
{text}

Rules:
- One finding per factual claim; keep numbers, names and dates exactly as written.
- Attach each claim to the source it came from. Only use URLs that appear in the research text; never invent one.
- Do not add facts that are not in the text.
- note: a short caveat (conflicting sources, weak source, outdated) or "" if none.
- open_questions: parts of the objective the research did not answer."""


def format_findings(out: dict) -> str:
    """Render the structured findings as text for the writer prompt."""
    lines = []
    for i, f in enumerate(out["findings"], 1):
        src = f["source_title"] or "unknown source"
        if f["source_url"]:
            src += f" ({f['source_url']})"
        lines.append(f"[{i}] {f['claim']}\n    Source: {src}")
        if f.get("note"):
            lines.append(f"    Researcher note: {f['note']}")
    if out.get("open_questions"):
        lines.append("\nOPEN QUESTIONS (research did not answer these; say so in the report, do not fill them from memory):")
        lines += [f"- {q}" for q in out["open_questions"]]
    return "\n".join(lines)


# ===== FULL-PIPELINE STATE =====

class AgentInputState(MessagesState):
    """Input is just the user's messages."""


class AgentState(MessagesState):
    research_brief: Optional[str]
    notes: Annotated[List[str], operator.add]
    raw_notes: Annotated[List[str], operator.add]
    research_output: dict
    final_report: str


# ===== SCOPING NODES =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """Ask a clarifying question if the request is too vague; otherwise move on."""
    prompt = clarify_with_user_instructions.format(
        messages=get_buffer_string(state["messages"]), date=get_today_str()
    )
    try:
        resp = model.with_structured_output(ClarifyWithUser).invoke([HumanMessage(content=prompt)])
    except Exception as e:  # free models can fail structured output; don't block research
        print(f"Clarification step failed, continuing without it: {e}")
        return Command(goto="write_research_brief")

    if resp.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=resp.question)]})
    return Command(goto="write_research_brief", update={"messages": [AIMessage(content=resp.verification)]})


def write_research_brief(state: AgentState):
    """Turn the conversation into one focused research brief."""
    prompt = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state["messages"]), date=get_today_str()
    )
    try:
        brief = model.with_structured_output(ResearchQuestion).invoke(
            [HumanMessage(content=prompt)]
        ).research_brief
    except Exception as e:  # fall back to the user's own words
        print(f"Brief generation failed, using the user's message: {e}")
        brief = next(
            (str(m.content) for m in reversed(state["messages"]) if m.type == "human"), ""
        )
    return {"research_brief": brief}


# ===== RESEARCHER AGENT NODES =====

def llm_call(state: ResearcherState):
    """Decide whether to call search tools or give a final answer."""
    # After the iteration cap, drop the tools so the model has to write its answer.
    llm = model_with_tools if state.get("tool_call_iterations", 0) < MAX_TOOL_ITERATIONS else model
    return {
        "researcher_messages": [
            llm.invoke(
                [SystemMessage(content=research_agent_prompt.format(date=get_today_str()))]
                + state["researcher_messages"]
            )
        ]
    }


def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response."""
    tool_calls = state["researcher_messages"][-1].tool_calls

    outputs = []
    for tc in tool_calls:
        try:
            obs = tools_by_name[tc["name"]].invoke(tc["args"])
        except Exception as e:  # a failed search shouldn't kill the run
            obs = f"Tool '{tc['name']}' failed: {e}"
        outputs.append(ToolMessage(content=obs, name=tc["name"], tool_call_id=tc["id"]))

    return {
        "researcher_messages": outputs,
        "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
    }


def compress_research(state: ResearcherState) -> dict:
    """Compress research messages and tool outputs into a concise summary."""
    # research_topic is required by the human prompt; fall back to the first message
    # so calling researcher_agent directly (as run.py does) still works.
    topic = state.get("research_topic") or str(state["researcher_messages"][0].content)

    messages = (
        [SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))]
        + list(state.get("researcher_messages", []))
        + [HumanMessage(content=compress_research_human_message.format(research_topic=topic))]
    )
    response = compress_model.invoke(messages)

    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], include_types=["tool", "ai"]
        )
    ]
    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)],
    }


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Continue the search loop while the LLM keeps calling tools."""
    if state["researcher_messages"][-1].tool_calls:
        return "tool_node"
    return "compress_research"


# ===== RESEARCHER SUBGRAPH =====

researcher_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
researcher_builder.add_node("llm_call", llm_call)
researcher_builder.add_node("tool_node", tool_node)
researcher_builder.add_node("compress_research", compress_research)

researcher_builder.add_edge(START, "llm_call")
researcher_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {"tool_node": "tool_node", "compress_research": "compress_research"},
)
researcher_builder.add_edge("tool_node", "llm_call")
researcher_builder.add_edge("compress_research", END)

researcher_agent = researcher_builder.compile()


# ===== PIPELINE NODES =====

def research(state: AgentState):
    """Run the single researcher on the brief and store its findings."""
    brief = state.get("research_brief", "")
    result = researcher_agent.invoke({
        "researcher_messages": [HumanMessage(content=brief)],
        "research_topic": brief,
    })
    return {
        "notes": [result.get("compressed_research", "")],
        "raw_notes": result.get("raw_notes", []),
    }


def structure_findings(state: AgentState):
    """Turn the researcher's compressed text into the structured findings the writer takes."""
    brief = state.get("research_brief", "")
    text = "\n".join(state.get("notes", []))
    try:
        out = model.with_structured_output(ResearchOutput).invoke(
            [HumanMessage(content=STRUCTURE_PROMPT.format(brief=brief, text=text))]
        )
    except Exception as e:  # keep the text rather than lose it
        print(f"Structuring failed, passing text through as one finding: {e}")
        out = ResearchOutput(objective=brief, findings=[Finding(claim=text, source_url="", source_title="")])
    out.objective = brief
    known = text + "\n".join(state.get("raw_notes", []))
    for f in out.findings:  # drop any URL the model made up
        if f.source_url and f.source_url not in known:
            f.source_url = ""
    return {"research_output": out.model_dump()}


def final_report_generation(state: AgentState):
    """Synthesize the findings into the final report."""
    prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=format_findings(state["research_output"]),
        date=get_today_str(),
    )
    report = writer_model.invoke([HumanMessage(content=prompt)])
    return {
        "final_report": report.content,
        "messages": [AIMessage(content="Here is the final report: " + report.content)],
    }


# ===== FULL GRAPH =====

builder = StateGraph(AgentState, input_schema=AgentInputState)
builder.add_node("clarify_with_user", clarify_with_user)
builder.add_node("write_research_brief", write_research_brief)
builder.add_node("research", research)
builder.add_node("structure_findings", structure_findings)
builder.add_node("final_report_generation", final_report_generation)

builder.add_edge(START, "clarify_with_user")  # clarify_with_user routes itself via Command
builder.add_edge("write_research_brief", "research")
builder.add_edge("research", "structure_findings")
builder.add_edge("structure_findings", "final_report_generation")
builder.add_edge("final_report_generation", END)

agent = builder.compile()
