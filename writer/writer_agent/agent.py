"""Build the writer agent.

`create_agent` (LangChain) builds a LangGraph graph: a model node that
decides which tool to call, and a tool node that runs it, looping until the
model answers without calling a tool. The checkpointer (LangGraph) saves
the state per `thread_id`, so a later message on the same thread continues
the same document. That thread_id is where the AHP thread ID will plug in.
"""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from .prompts import SYSTEM_PROMPT
from .state import WriterState
from .tools import WRITER_TOOLS

DEFAULT_MODEL = "openai:gpt-5-nano"


def build_writer_agent(model: str | BaseChatModel | None = None, checkpointer=None):
    return create_agent(
        model=model or os.getenv("WRITER_MODEL", DEFAULT_MODEL),
        tools=WRITER_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        state_schema=WriterState,
        checkpointer=checkpointer or InMemorySaver(),
        name="writer",
    )
