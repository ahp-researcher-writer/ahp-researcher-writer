"""Checks the whole agent loop (graph + tools + state) with a scripted fake
model, so it runs without an API key and costs nothing."""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from writer_agent import build_writer_agent, load_resources


class ScriptedModel(FakeMessagesListChatModel):
    """Returns the scripted messages in order; ignores the tools it's given."""

    def bind_tools(self, tools, **kwargs):
        return self


def call(tool_name, i, **args):
    return {"name": tool_name, "args": args, "id": f"call_{i}", "type": "tool_call"}


def run(script, tmp_path, monkeypatch, thread="t1"):
    monkeypatch.setenv("WRITER_OUTPUT_DIR", str(tmp_path))
    agent = build_writer_agent(model=ScriptedModel(responses=script))
    config = {"configurable": {"thread_id": thread}}
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "Write a short explainer on MCP."}],
            "resources": load_resources("sample_data"),
        },
        config,
    )
    return agent.get_state(config).values


def test_full_writing_flow(tmp_path, monkeypatch):
    script = [
        AIMessage("", tool_calls=[call("list_resources", 1)]),
        AIMessage("", tool_calls=[call("read_resource", 2, name="research_brief.json")]),
        AIMessage("", tool_calls=[call("get_sources", 3)]),
        AIMessage("", tool_calls=[call("create_outline", 4, title="What is MCP?",
                                       sections=["Overview", "How it works"])]),
        AIMessage("", tool_calls=[call("write_section", 5, heading="Overview",
                                       content="MCP links LLM apps to tools [1].")]),
        AIMessage("", tool_calls=[call("write_section", 6, heading="How it works",
                                       content="Servers use JSON-RPC [1]. Sampling was deprecated [2].")]),
        AIMessage("", tool_calls=[call("check_draft", 7)]),
        AIMessage("", tool_calls=[call("save_draft", 8)]),
        AIMessage("Done. I wrote a two-section explainer on MCP."),
    ]
    state = run(script, tmp_path, monkeypatch)

    assert state["outline"] == ["Overview", "How it works"]
    assert set(state["sections"]) == {"Overview", "How it works"}
    saved = (tmp_path / "what-is-mcp.md").read_text()
    assert saved.startswith("# What is MCP?")
    assert "## References" in saved
    assert state["messages"][-1].text.startswith("Done")


def test_save_refuses_incomplete_draft(tmp_path, monkeypatch):
    script = [
        AIMessage("", tool_calls=[call("create_outline", 1, title="X", sections=["A", "B"])]),
        AIMessage("", tool_calls=[call("write_section", 2, heading="A", content="Text [1].")]),
        AIMessage("", tool_calls=[call("save_draft", 3)]),
        AIMessage("Stopping."),
    ]
    state = run(script, tmp_path, monkeypatch)
    assert "saved_path" not in state
    tool_replies = [m.text for m in state["messages"] if m.type == "tool"]
    assert any(r.startswith("Not saved") for r in tool_replies)


def test_request_more_research_ends_the_turn(tmp_path, monkeypatch):
    script = [
        AIMessage("", tool_calls=[call("request_more_research", 1,
                                       gaps=["How many MCP servers exist?"])]),
        AIMessage("This should never be reached."),
    ]
    state = run(script, tmp_path, monkeypatch)
    assert state["gaps"] == ["How many MCP servers exist?"]
    assert (tmp_path / "handback.json").exists()
    assert state["messages"][-1].type == "tool"  # stopped right after the tool
