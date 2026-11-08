"""Toy LangGraph demo: shows state, nodes, edges and a conditional retry.
No LLM, no repo. Scores are fake so we can watch the routing."""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

MIN_SCORE = 0.25
MAX_ATTEMPTS = 2


class State(TypedDict):
    scores: list[float]                    # fake score handed out per attempt
    attempts: int                          # how many retrievals so far
    score: float                           # score of the latest retrieval
    answer: str                            # final result
    log: Annotated[list[str], operator.add]  # reducer: new entries are appended


def retrieve(state: State) -> dict:
    attempt = state["attempts"]            # 0 on the first run
    score = state["scores"][attempt]
    return {
        "attempts": attempt + 1,
        "score": score,
        "log": [f"retrieve: attempt {attempt + 1}, score {score:.2f}"],
    }


def generate(state: State) -> dict:
    return {"answer": "ANSWERED", "log": ["generate: context was good enough"]}


def refuse(state: State) -> dict:
    return {
        "answer": "I could not find this in the repository.",
        "log": ["refuse: context still weak after retry"],
    }


def route_after_retrieve(state: State) -> str:
    """Conditional edge: looks at the state and returns the NAME of the next step."""
    if state["score"] >= MIN_SCORE:
        return "generate"
    if state["attempts"] < MAX_ATTEMPTS:
        return "retry"
    return "refuse"


def build_graph():
    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("refuse", refuse)

    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "retry": "retrieve", "refuse": "refuse"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("refuse", END)
    return builder.compile()


def run_case(name: str, scores: list[float]) -> None:
    graph = build_graph()
    result = graph.invoke(
        {"scores": scores, "attempts": 0, "score": 0.0, "answer": "", "log": []}
    )
    print(f"\n=== {name} ===")
    for line in result["log"]:
        print("  ", line)
    print("   final answer:", result["answer"])


if __name__ == "__main__":
    run_case("A: good on first try", [0.60, 0.60])
    run_case("B: weak, then good on retry", [0.10, 0.60])
    run_case("C: weak both times", [0.10, 0.10])