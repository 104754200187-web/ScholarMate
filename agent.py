"""Literature agent CLI. Orchestration lives in graph.py (LangGraph).

Usage:
    python3 agent.py "your research question here"
    python3 agent.py --search "paper keywords"     # search-only mode
"""

import argparse

import graph


def ask(question, mode="ask"):
    """Compatibility shim: execute the LangGraph pipeline."""
    return graph.run(question, mode=mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Literature agent")
    parser.add_argument("question", nargs="*", help="research question or paper keywords")
    parser.add_argument("--search", action="store_true",
                        help="search-only mode: list papers without generating an answer")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        parser.exit("Usage: python3 agent.py [--search] \"your question or keywords\"")
    print()
    print(graph.run(question, mode="search" if args.search else "ask"))
