# AI-Powered Software Engineering Agent

Paste the URL of a public GitHub repository and ask questions about its code. The agent retrieves the relevant code, answers using only that code, and cites the exact lines it used, like `src/flask/sansio/scaffold.py:343-346`. If the repository doesn't contain the answer, it says so instead of guessing.

Stack: Python, LangGraph, Google Gemini, FAISS, sentence-transformers, GitHub API, Streamlit.

## How it works

### Indexing (once per repository)

1. Parse the GitHub URL and list the repository's files through the GitHub API (with error and rate-limit handling).
2. Keep only Python (`.py`) files and download them.
3. Split each file into chunks using Python's `ast` module, so chunks are whole functions and classes that keep their file path and line numbers.
4. Embed each chunk with `all-MiniLM-L6-v2` and store the vectors in a FAISS index.

### Answering (per question), as a LangGraph agent

```
analyze_query -> retrieve -> [score >= 0.25?] -> generate -> verify_citations
                                  |
                  weak, 1st try:  rewrite_query -> retrieve (once more)
                  weak, 2nd try:  refuse
```

- `analyze_query`: cleans the question and stops early on empty input.
- `retrieve`: finds the top 5 most similar chunks.
- Check and retry: if the best similarity score is below 0.25, the agent asks the LLM to rewrite the question as a short code-style search query and retrieves once more. A retry cap of 2 attempts prevents loops.
- `generate`: sends the chunks to Gemini with a strict prompt: answer only from the chunks and cite exact lines.
- `verify_citations`: checks in code that every cited `path:start-end` lies inside a chunk that was actually retrieved, and flags any that don't.
- `refuse`: returns a fixed "I could not find this in the repository." without asking the model to answer.

### Reducing wrong citations

The model used to miscount line numbers, so each chunk's lines are numbered in the prompt, and a code-level checker validates every citation afterwards.

## Setup (Windows)

Tested with Python 3.14.3.

```powershell
git clone https://github.com/sumanakaranamwork-svg/SWE-AGENT.git
cd SWE-AGENT
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a file named `.env` in the project root with your own keys (this file is git-ignored; never commit it):

```
GITHUB_TOKEN=your_github_token_here
LLM_API_KEY=your_gemini_api_key_here
```

Check that the keys load (this prints only whether they are set):

```powershell
python -m app.config
```

## Run the app

```powershell
python -m streamlit run app/ui.py
```

Open `http://localhost:8501`, paste a public repository URL, click Index repository, then ask a question. The first run downloads the embedding model, so it can take a little while to start.

The answer page shows the cited answer, the retrieved sources, a warning if any citation doesn't match the retrieved code, and an Agent trace panel listing each step the agent took.

## Run the tests

These scripts index this project's own files and run a few checks:

```powershell
python -m app.answerer   # citation checker and grounded answers
python -m app.agent      # routing, empty input, answer and refusal paths
```

## Project structure

| File | Purpose |
|---|---|
| `app/config.py` | Loads keys from `.env` |
| `app/github_loader.py` | URL parsing, GitHub API calls, file listing |
| `app/file_filter.py` | Keeps `.py` files only |
| `app/repo_downloader.py` | Downloads the selected files |
| `app/chunker.py` | AST-based chunking with line numbers |
| `app/embedder.py` | Sentence embeddings (`all-MiniLM-L6-v2`) |
| `app/vector_store.py` | FAISS index and search |
| `app/retriever.py` | `build_repo_index()` and `retrieve()` |
| `app/llm.py` | Gemini calls with retries |
| `app/answerer.py` | Prompt, line numbering, citation checker |
| `app/agent.py` | LangGraph agent |
| `app/ui.py` | Streamlit interface |
| `app/graph_demo.py` | Small LangGraph example without an LLM |

## Tested on

`pallets/flask`: 80 Python files, 976 code chunks, with cited answers and refusals for unrelated questions.

## Limitations

- Public repositories and Python files only.
- Retrieval uses embeddings only (no keyword search), so exact-name lookups can miss.
- Test files can appear among the retrieved sources and compete with real code.
- The sources list shows all retrieved chunks, not only the ones the answer cited.
- The citation checker confirms a citation points inside retrieved code. It cannot confirm the cited lines truly support the claim.
- The relevance threshold (0.25) was chosen by hand from a small number of tests, and there is no formal evaluation set yet.
- Each browser session builds its own index, so indexing a large repository takes a while.

## Possible improvements

- Show only the sources the answer actually cites.
- Exclude test files from retrieval.
- Add a small evaluation set and tune the threshold.
- Add keyword search alongside embeddings.
