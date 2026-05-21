"""Streamlit UI for the Multi-Tool Research Agent.

Run with:  streamlit run app.py
"""
import streamlit as st

from src.agent import build_agent
from src.config import OPENAI_MODEL, MAX_AGENT_STEPS
from src.prompts import SYSTEM_PROMPT
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Multi-Tool Research Agent", page_icon="🔎", layout="wide")

st.title("🔎 Multi-Tool Research Agent")
st.caption(
    f"LangGraph ReAct agent · {OPENAI_MODEL} · DuckDuckGo search + web reader · "
    f"step cap = {MAX_AGENT_STEPS}"
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Plans** sub-questions\n"
        "2. **Searches** the web (DuckDuckGo)\n"
        "3. **Reads** the best sources\n"
        "4. **Writes** a cited report\n\n"
        "Set `OPENAI_API_KEY` in your `.env` file before running."
    )

question = st.text_area(
    "Research question",
    placeholder="e.g. What are the latest advances in solid-state batteries, and which companies lead?",
    height=90,
)

if st.button("Run research", type="primary") and question.strip():
    agent = build_agent()
    progress = st.status("Agent working…", expanded=True)
    final_msg = None

    try:
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
                    progress.write(f"🔧 **{tc['name']}** → `{arg_str}`")
            elif msg.type == "tool":
                preview = (msg.content or "")[:200].replace("\n", " ")
                progress.write(f"📄 _observed:_ {preview}…")
            elif msg.type == "ai" and msg.content:
                final_msg = msg.content
        progress.update(label="Done ✅", state="complete", expanded=False)
    except Exception as exc:
        progress.update(label="Error", state="error")
        st.error(f"Agent failed: {exc}")
        final_msg = None

    if final_msg:
        st.markdown("---")
        st.markdown(final_msg)
        st.download_button("⬇️ Download report (.md)", final_msg, file_name="report.md")
