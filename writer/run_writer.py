"""Run the writer agent from the terminal.

Example:
    python run_writer.py \
        --objective "Write a 400-word explainer on what MCP is, for CS students." \
        --resources sample_data \
        --thread demo-1

After the first draft you can type follow-up requests ("make it shorter",
"add a conclusion") on the same thread. Press Enter on an empty line to quit.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv

from writer_agent import build_writer_agent, load_resources


def run_turn(agent, payload: dict, config: dict) -> None:
    """Stream one turn, printing each tool call so you can see the agent work."""
    for update in agent.stream(payload, config, stream_mode="updates"):
        for node, data in update.items():
            for message in (data or {}).get("messages", []):
                for call in getattr(message, "tool_calls", None) or []:
                    args = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in call["args"].items())
                    print(f"  -> {call['name']}({args})")
                if node == "tools":
                    print(f"     {message.text[:200]}")

    state = agent.get_state(config).values
    last = state["messages"][-1]
    print("\nWriter:", last.text)
    if state.get("saved_path"):
        print(f"Draft file: {state['saved_path']}")
    if state.get("gaps"):
        print("Gaps for the researcher:", *state["gaps"], sep="\n  - ")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the writer agent.")
    parser.add_argument("--objective", required=True, help="What to write.")
    parser.add_argument("--resources", default="sample_data", help="Folder with research files.")
    parser.add_argument("--thread", default=None, help="Thread ID (reuse to continue).")
    args = parser.parse_args()

    agent = build_writer_agent()
    config = {
        "configurable": {"thread_id": args.thread or str(uuid.uuid4())},
        "recursion_limit": 60,
    }
    print(f"Thread: {config['configurable']['thread_id']}\n")

    first = {
        "messages": [{"role": "user", "content": args.objective}],
        "resources": load_resources(args.resources),
    }
    run_turn(agent, first, config)

    while True:
        follow_up = input("\nYou (Enter to quit): ").strip()
        if not follow_up:
            break
        run_turn(agent, {"messages": [{"role": "user", "content": follow_up}]}, config)


if __name__ == "__main__":
    main()
