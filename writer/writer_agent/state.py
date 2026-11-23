"""The writer agent's state.

LangGraph keeps this state for each thread (conversation). We extend
LangChain's AgentState (which already holds `messages`) with the
writer's own working memory: the resources it was handed, the outline,
the sections written so far, and any research gaps it found.
"""

from typing import NotRequired

from langchain.agents import AgentState
from typing_extensions import TypedDict


class Resource(TypedDict):
    """One piece of material handed to the writer (later: an AHP resource)."""

    description: str
    text: str


class WriterState(AgentState):
    # Material handed over by the researcher, keyed by resource name.
    resources: NotRequired[dict[str, Resource]]
    # Document plan.
    title: NotRequired[str]
    outline: NotRequired[list[str]]
    # Section heading -> markdown body.
    sections: NotRequired[dict[str, str]]
    # Questions the research did not answer (triggers the hand-back).
    gaps: NotRequired[list[str]]
    # Where the last saved draft was written.
    saved_path: NotRequired[str]
