"""Live end-to-end /ask check: real Chroma index + real configured LLM.

Usage:
    .venv/bin/python scripts/test_ask_live.py "query one" "query two"

Text-input mode on purpose (no mic needed); uses the same ASGI app uvicorn
would serve, via Starlette's TestClient transport.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def main() -> int:
    queries = sys.argv[1:]
    if not queries:
        print("pass one or more queries as arguments")
        return 1

    client = TestClient(app)
    for q in queries:
        response = client.post("/ask", data={"query": q})
        print(f"\n=== QUERY: {q!r} -> HTTP {response.status_code} ===")
        if response.status_code != 200:
            print(response.text[:500])
            continue
        body = response.json()
        print(f"transcript : {body['transcript']}")
        print(f"answer     : {body['answer']}")
        top = body["chunks"][0] if body["chunks"] else None
        if top:
            print(f"top source : {top['doc_id']} score={top['score']:.3f} "
                  f"text={top['text'][:80]!r}")
        print(f"llm        : {body['llm_provider']}/{body['llm_model']} "
              f"is_mock={body['llm_is_mock']}")
        print(f"trace_ms   : {json.dumps(body['latency_trace_ms'])}")
        print(f"retrieval_ms={body['retrieval_ms']} total_ms={body['total_ms']}")
        if body["warnings"]:
            print(f"warnings   : {body['warnings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
