"""LangGraph agent: analyze -> retrieve -> check (retry once) -> generate -> verify."""

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.answerer import (
    NOT_FOUND_PHRASE,
    SYSTEM_INSTRUCTION,
    TOP_K,
    build_context,
    find_bad_citations,
    has_enough_context,
)
from app.llm import LLMError, generate
from app.retriever import format_source, retrieve

MAX_ATTEMPTS = 2           # first search + one retry
MAX_QUESTION_CHARS = 500
EMPTY_QUESTION_MESSAGE = "Please type a question about the repository."

REWRITE_INSTRUCTION = (
    "You rewrite a user's question into a short search query for finding code "
    "in a repository. Output ONLY the query: 3 to 10 words, using likely "
    "function, class, variable or file names and key technical terms. "
    "No explanation, no quotes."
)


class AgentState(TypedDict):
    """The shared dictionary that flows through the graph."""
    repo_index: Any        # the index built by build_repo_index()
    question: str          # cleaned user question
    search_query: str      # what we search with (changes on retry)
    results: list          # retrieved chunks with scores
    attempts: int          # how many retrievals so far
    top_score: float       # best score of the latest retrieval
    answer: str
    sources: list
    used_llm: bool         # True if the answer was written by the model
    bad_citations: list
    error: str             # non-empty if the model failed
    trace: Annotated[list, operator.add]   # reducer: entries are appended


# ---------- nodes: each takes the state and returns only what it changes ----------

def analyze_query_node(state):
    question = " ".join(state["question"].split())[:MAX_QUESTION_CHARS]
    if not question:
        return {
            "question": "",
            "answer": EMPTY_QUESTION_MESSAGE,
            "trace": ["analyze_query: empty question, stopping"],
        }
    return {
        "question": question,
        "search_query": question,
        "trace": ["analyze_query: ok (" + str(len(question)) + " characters)"],
    }


def retrieve_node(state):
    results = retrieve(state["repo_index"], state["search_query"], k=TOP_K)
    top_score = results[0]["score"] if results else 0.0
    attempt = state["attempts"] + 1
    return {
        "results": results,
        "top_score": top_score,
        "attempts": attempt,
        "trace": [
            f"retrieve: attempt {attempt}, query '{state['search_query']}', "
            f"top score {top_score:.2f}"
        ],
    }


def rewrite_query_node(state):
    original = state["question"]
    try:
        raw = generate("Question: " + original, system_instruction=REWRITE_INSTRUCTION)
    except LLMError as error:
        return {"trace": ["rewrite_query: model failed (" + str(error) + "), keeping old query"]}

    lines = raw.strip().splitlines()
    new_query = lines[0].strip().strip("\"'`")[:200] if lines else ""
    if not new_query:
        new_query = original
    return {
        "search_query": new_query,
        "trace": ["rewrite_query: new query '" + new_query + "'"],
    }


def generate_node(state):
    repo_index = state["repo_index"]
    prompt = (
        "Repository: " + repo_index.owner + "/" + repo_index.repo + "\n\n"
        "Retrieved code chunks:\n\n" + build_context(state["results"]) + "\n\n"
        "Question: " + state["question"] + "\n\n"
        "Answer using only the chunks above, and cite exact line numbers."
    )
    try:
        answer = generate(prompt, system_instruction=SYSTEM_INSTRUCTION)
    except LLMError as error:
        return {
            "answer": "The language model could not answer right now: " + str(error),
            "used_llm": False,
            "error": str(error),
            "trace": ["generate: model error"],
        }
    return {
        "answer": answer,
        "used_llm": True,
        "trace": ["generate: answer written by the model"],
    }


def verify_citations_node(state):
    answer = state["answer"]
    if state["error"] or NOT_FOUND_PHRASE.lower() in answer.lower():
        return {
            "sources": [],
            "bad_citations": [],
            "trace": ["verify_citations: nothing to check"],
        }
    bad = find_bad_citations(answer, state["results"])
    sources = [format_source(r["chunk"]) for r in state["results"]]
    return {
        "sources": sources,
        "bad_citations": bad,
        "trace": ["verify_citations: " + str(len(bad)) + " bad citation(s)"],
    }


def refuse_node(state):
    return {
        "answer": NOT_FOUND_PHRASE,
        "sources": [],
        "used_llm": False,
        "bad_citations": [],
        "trace": ["refuse: context still weak after retry"],
    }


# ---------- conditional edges: look at the state, return the next step's label ----------

def route_after_analyze(state):
    return "stop" if not state["question"] else "retrieve"


def route_after_retrieve(state):
    if has_enough_context(state["results"]):
        return "generate"
    if state["attempts"] < MAX_ATTEMPTS:
        return "rewrite"
    return "refuse"


# ---------- build and run ----------

def build_agent():
    builder = StateGraph(AgentState)
    builder.add_node("analyze_query", analyze_query_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify_citations", verify_citations_node)
    builder.add_node("refuse", refuse_node)

    builder.add_edge(START, "analyze_query")
    builder.add_conditional_edges(
        "analyze_query", route_after_analyze, {"retrieve": "retrieve", "stop": END}
    )
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "rewrite": "rewrite_query", "refuse": "refuse"},
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", "verify_citations")
    builder.add_edge("verify_citations", END)
    builder.add_edge("refuse", END)
    return builder.compile()


_AGENT = None


def run_agent(repo_index, question):
    """Run the agent. Returns answer, sources, used_llm, top_score,
    bad_citations, attempts, search_query and trace."""
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()

    final = _AGENT.invoke(
        {
            "repo_index": repo_index,
            "question": question,
            "search_query": "",
            "results": [],
            "attempts": 0,
            "top_score": 0.0,
            "answer": "",
            "sources": [],
            "used_llm": False,
            "bad_citations": [],
            "error": "",
            "trace": [],
        }
    )
    return {
        "answer": final["answer"],
        "sources": final["sources"],
        "used_llm": final["used_llm"],
        "top_score": final["top_score"],
        "bad_citations": final["bad_citations"],
        "attempts": final["attempts"],
        "search_query": final["search_query"],
        "trace": final["trace"],
    }


if __name__ == "__main__":
    from pathlib import Path

    from app.chunker import chunk_documents
    from app.retriever import RepoIndex
    from app.vector_store import VectorStore

    # Index this project's own files (no GitHub calls). We skip files that
    # contain the test questions, so they cannot "find" their own text.
    skip = {"answerer.py", "vector_store.py", "agent.py", "graph_demo.py"}
    documents = []
    for file_path in sorted(Path(__file__).parent.glob("*.py")):
        if file_path.name in skip:
            continue
        documents.append(
            {"path": "app/" + file_path.name, "content": file_path.read_text(encoding="utf-8")}
        )

    store = VectorStore(chunk_documents(documents))
    repo_index = RepoIndex("local", "swe-agent", {}, store, len(documents), [])
    print("Indexed", len(documents), "files,", repo_index.chunk_count, "chunks")

    # Test 1 (offline): the routing function picks the right next step.
    strong = [{"score": 0.5}]
    weak = [{"score": 0.1}]
    routing_ok = (
        route_after_retrieve({"results": strong, "attempts": 1}) == "generate"
        and route_after_retrieve({"results": weak, "attempts": 1}) == "rewrite"
        and route_after_retrieve({"results": weak, "attempts": 2}) == "refuse"
        and route_after_retrieve({"results": [], "attempts": 2}) == "refuse"
    )
    print("PASS" if routing_ok else "FAIL", "- routing: generate / retry once / refuse")

    # Test 2 (offline): an empty question stops before any search.
    empty = run_agent(repo_index, "   ")
    empty_ok = (
        empty["answer"] == EMPTY_QUESTION_MESSAGE
        and empty["attempts"] == 0
        and not empty["used_llm"]
    )
    print("PASS" if empty_ok else "FAIL", "- empty question stops early")

    # Tests 3-5 (use the LLM): (question, check that decides PASS or FAIL)
    tests = [
        (
            "What does the code do when GitHub returns a 404 status?",
            lambda r: r["used_llm"]
            and r["attempts"] == 1
            and not r["bad_citations"]
            and any("github_loader.py" in s for s in r["sources"]),
        ),
        (
            "What is the recipe for chocolate cake?",
            lambda r: r["attempts"] == 2
            and NOT_FOUND_PHRASE.lower() in r["answer"].lower(),
        ),
        (
            "Does this project store user accounts in a PostgreSQL database?",
            lambda r: r["attempts"] == 2
            and NOT_FOUND_PHRASE.lower() in r["answer"].lower(),
        ),
    ]

    for question, check in tests:
        try:
            result = run_agent(repo_index, question)
        except Exception as error:   # show any problem instead of crashing the test run
            print()
            print("ERROR:", type(error).__name__, error)
            continue

        status = "PASS" if check(result) else "FAIL"
        print()
        print(status, "-", question)
        print("   attempts:", result["attempts"], "| top score:", round(result["top_score"], 2),
              "| LLM used:", result["used_llm"])
        print("   bad citations:", result["bad_citations"])
        print("   trace:")
        for step in result["trace"]:
            print("     -", step.encode("ascii", "replace").decode())
        print("   answer:")
        for line in result["answer"].splitlines():
            print("     ", line.encode("ascii", "replace").decode())