# 🔎 Multi-Tool Research Agent

An autonomous AI research agent that **plans sub-questions, searches the web, reads sources, and writes a cited report** — built with **LangGraph**, **OpenAI GPT-4o**, **DuckDuckGo** search, and a **Streamlit** UI.

It uses a ReAct loop (*think → act → observe → repeat*): the LLM decides which tool to call next, observes the result, and continues until it has enough grounded evidence to answer.

## Features
- 🧠 **Agentic ReAct loop** via LangGraph (`create_react_agent`)
- 🔧 **Two tools**: `web_search` (DuckDuckGo, free) and `read_url` (clean page extraction)
- 📚 **Cited reports** — every claim grounded in a source the agent actually read
- 🛡️ **Guardrails** — step cap + page-size truncation to bound cost and prevent infinite loops
- 🖥️ **Streamlit UI** with a live agent trace, plus a CLI

## Architecture
```
question → [Planner/LLM] → ReAct loop ⇄ tools (web_search, read_url) → [Synthesizer] → cited report
```

## Setup
```bash
cd research-agent
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt

copy .env.example .env        # then add your OPENAI_API_KEY
```

## Run
**Web UI:**
```bash
streamlit run app.py
```
**Command line:**
```bash
python run_cli.py "What are the latest advances in solid-state batteries, and which companies lead?"
```

## Project structure
```
research-agent/
├── app.py            # Streamlit UI
├── run_cli.py        # CLI entry point
├── requirements.txt
├── .env.example
└── src/
    ├── agent.py      # LangGraph ReAct agent + guardrails
    ├── tools.py      # web_search + read_url tools
    ├── prompts.py    # system prompt (research behaviour)
    └── config.py     # settings loaded from .env
```

## Configuration (`.env`)
| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_MODEL` | `gpt-4o` | LLM model |
| `MAX_AGENT_STEPS` | `12` | guardrail on agent loop length |

## Resume bullet
> Built an autonomous multi-tool research agent (LangGraph + GPT-4o) that plans sub-tasks, searches and reads web sources, and generates cited reports — with a step-capped ReAct loop and source-grounding to control cost and reduce hallucinated citations.

## Possible extensions
- Add a re-ranker to pick the best sources before reading
- Add vector memory so findings persist across queries
- Add an eval harness (citation accuracy, groundedness) with LLM-as-judge
- Add cost/latency tracking per query
