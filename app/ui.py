import sys
from pathlib import Path

# Make "from app.x import ..." work however Streamlit starts this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.agent import run_agent  # noqa: E402
from app.github_loader import GitHubError  # noqa: E402
from app.retriever import build_repo_index  # noqa: E402

# This must be the first Streamlit call in the script.
st.set_page_config(page_title="SWE Agent", page_icon="🧭", layout="wide")

# st.session_state survives re-runs. Set up our three values once.
for key, default in {"repo_index": None, "repo_url": "", "last": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def index_repository(url):
    """Build the index for a GitHub URL, showing live progress."""
    with st.status("Indexing repository...", expanded=True) as status:
        try:
            # build_repo_index calls st.write("...") for every progress message.
            repo_index = build_repo_index(url, progress=st.write)
        except (GitHubError, ValueError) as error:
            status.update(label="Indexing failed", state="error")
            st.error(str(error))
            return
        except Exception as error:  # anything unexpected: show the type only
            status.update(label="Indexing failed", state="error")
            st.error("Unexpected error: " + type(error).__name__)
            return
        status.update(label="Repository indexed", state="complete", expanded=False)

    # Only replace the old index after the new one succeeded.
    st.session_state.repo_index = repo_index
    st.session_state.repo_url = url
    st.session_state.last = None


# ---------- page header ----------

st.title("AI-Powered Software Engineering Agent")
st.caption(
    "Ask questions about a public GitHub repository. Answers are grounded in the "
    "retrieved code and cite path:start-end lines. Only Python (.py) files are indexed."
)

# ---------- step 1: choose a repository ----------

st.subheader("1. Choose a repository")
with st.form("index_form"):
    url_input = st.text_input(
        "Public GitHub repository URL",
        placeholder="https://github.com/pallets/flask",
    )
    index_clicked = st.form_submit_button("Index repository")

if index_clicked:
    url = url_input.strip()
    if not url:
        st.error("Please paste a GitHub repository URL.")
    elif url == st.session_state.repo_url and st.session_state.repo_index is not None:
        st.info("This repository is already indexed. You can ask questions below.")
    else:
        index_repository(url)

repo_index = st.session_state.repo_index
if repo_index is None:
    st.info("Index a repository to start asking questions.")
    st.stop()  # nothing below runs until a repository is indexed

with st.sidebar:
    st.header("Indexed repository")
    st.write("**" + repo_index.owner + "/" + repo_index.repo + "**")
    st.write("Python files: " + str(repo_index.file_count))
    st.write("Code chunks: " + str(repo_index.chunk_count))
    if repo_index.skipped:
        st.write("Files skipped: " + str(len(repo_index.skipped)))

# ---------- step 2: ask a question ----------

st.subheader("2. Ask a question")
with st.form("question_form"):
    question_input = st.text_input(
        "Your question",
        placeholder="How does Flask register a URL route?",
    )
    ask_clicked = st.form_submit_button("Ask")

if ask_clicked:
    question = question_input.strip()
    if not question:
        st.error("Please type a question.")
    else:
        with st.spinner("Searching the code and writing an answer..."):
            try:
                result = run_agent(repo_index, question)
            except Exception as error:
                result = None
                st.error("Something went wrong: " + type(error).__name__)
        if result is not None:
            st.session_state.last = {"question": question, "result": result}

# ---------- show the latest answer ----------

last = st.session_state.last
if last is not None:
    result = last["result"]

    st.subheader("Answer")
    st.caption("Question: " + last["question"])
    st.markdown(result["answer"])

    if result["bad_citations"]:
        st.warning(
            "These citations do not match the retrieved code, so double-check them: "
            + ", ".join(result["bad_citations"])
        )

    if result["sources"]:
        with st.expander("Sources (" + str(len(result["sources"])) + " retrieved chunks)"):
            for source in result["sources"]:
                st.text(source)

    with st.expander("Agent trace (what the agent did)"):
        st.caption(
            "Attempts: " + str(result["attempts"])
            + " | Top score: " + str(round(result["top_score"], 2))
            + " | LLM used: " + str(result["used_llm"])
        )
        st.text("\n".join(result["trace"]))