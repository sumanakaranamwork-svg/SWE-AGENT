import os
from pathlib import Path

from dotenv import load_dotenv

# The folder that contains the "app" folder, i.e. C:\Projects\swe-agent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read the .env file in the project root and load its values
# into the program's environment variables.
load_dotenv(PROJECT_ROOT / ".env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
LLM_API_KEY = os.getenv("LLM_API_KEY")


def key_status(value):
    """Describe a key without ever revealing it."""
    if not value:
        return "missing"
    if value.startswith("your_"):
        return "placeholder"
    return "set"


if __name__ == "__main__":
    print("GITHUB_TOKEN:", key_status(GITHUB_TOKEN))
    print("LLM_API_KEY:", key_status(LLM_API_KEY))