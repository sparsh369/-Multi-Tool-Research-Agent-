"""Command-line entry point.

Usage:
    python run_cli.py "your research question"
    python run_cli.py            # then type your question at the prompt
"""
import sys

from langchain_core.messages import HumanMessage

from src.agent import build_agent
from src.config import MAX_AGENT_STEPS


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Research question: ").strip()
    if not question:
        print("No question provided.")
        return

    agent = build_agent()
    final = ""

    print("\n--- Agent trace ---")
    for step in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": MAX_AGENT_STEPS * 2},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                args = tc.get("args", {})
                arg_str = args.get("query") or args.get("url") or ""
                print(f"  [tool] {tc['name']} -> {arg_str}")
        elif msg.type == "tool":
            preview = (msg.content or "")[:160].replace("\n", " ")
            print(f"  [obs]  {preview}...")
        elif msg.type == "ai" and msg.content:
            final = msg.content

    print("\n--- Final report ---\n")
    print(final)


if __name__ == "__main__":
    main()
