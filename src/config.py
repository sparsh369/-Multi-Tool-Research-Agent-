"""Central configuration loaded from environment / .env file."""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Guardrail: cap the number of reasoning/tool steps so the agent can't loop forever
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "12"))

# Tool behaviour
SEARCH_RESULTS_PER_QUERY = 5
MAX_PAGE_CHARS = 6000  # truncate fetched pages to control token cost


def require_api_key() -> None:
    """Fail fast with a friendly message if the key is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
