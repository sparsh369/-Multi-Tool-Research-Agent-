"""LangGraph ReAct agent wiring: LLM + tools + system prompt + guardrails."""
from __future__ import annotations

from typing import Iterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .config import OPENAI_MODEL, MAX_AGENT_STEPS, require_api_key
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS


def build_agent():
    """Construct the compiled LangGraph agent."""
    require_api_key()
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    # create_react_agent returns a compiled graph that loops: LLM -> tools -> LLM ...
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def run_research(question: str) -> str:
    """Run the agent to completion and return the final report (Markdown string)."""
    agent = build_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": MAX_AGENT_STEPS * 2},  # guardrail
    )
    return result["messages"][-1].content


def stream_research(question: str) -> Iterator[str]:
    """Yield human-readable progress lines as the agent works (for live UIs)."""
    agent = build_agent()
    for step in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": MAX_AGENT_STEPS * 2},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        # Tool calls -> show what the agent decided to do
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                args = tc.get("args", {})
                arg_str = args.get("query") or args.get("url") or ""
                yield f"🔧 **{tc['name']}** → `{arg_str}`"
        elif msg.type == "tool":
            preview = (msg.content or "")[:160].replace("\n", " ")
            yield f"📄 observed: {preview}..."
        elif msg.type == "ai" and msg.content:
            yield f"🧠 {msg.content[:200]}"
