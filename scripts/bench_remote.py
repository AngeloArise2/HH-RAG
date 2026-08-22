#!/usr/bin/env python
"""Measure retrieval-only latency on a DEPLOYED backend (e.g. Render).

Why this exists: the phase 8 benchmark ran the pipeline in-process locally.
After the ONNX swap AND a move to Render's hardware, server-side stage
timings differ — and every /ask response already carries its own
latency_trace_ms measured where the code actually runs. This script replays
real dataset queries against a live URL, collects those traces, and reports
P50/P70/P100 for the retrieval-only budget stages plus context stages.

Refused queries are excluded from retrieval percentiles (off-topic refusals
short-circuit before embedding — no retrieval happened), but counted.

Usage:
    .venv/bin/python scripts/bench_remote.py --url https://<app>.onrender.com \
        --queries 20 --sleep 15 --output backend/data/remote_benchmark_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import httpx

from app.benchmarking.latency import summarize
from app.benchmarking.queries import load_real_queries

STAGES = ["embed_query", "vector_search", "retrieval_ms", "generation", "guardrail_check", "total_ms"]


def fetch(client: httpx.Client, base_url: str, query: str) -> dict:
    """One /ask round-trip. Isolated for testability."""
    resp = client.post(f"{base_url}/ask", data={"query": query})
    resp.raise_for_status()
    return resp.json()


def run(client: httpx.Client, base_url: str, queries: list[str], sleep_s: float) -> dict:
    rows: list[dict] = []
    refused = 0
    for i, q in enumerate(queries, 1):
        data = fetch(client, base_url, q)
        if data.get("refused"):
            refused += 1
            print(f"[{i}/{len(queries)}] REFUSED ({data.get('refusal_reason')}): {q[:50]}")
        else:
            t = data.get("latency_trace_ms") or {}
            row = {**t, "retrieval_ms": data.get("retrieval_ms"), "total_ms": data.get("total_ms")}
            rows.append(row)
            print(f"[{i}/{len(queries)}] ok retrieval={row['retrieval_ms']:.1f}ms total={row['total_ms']:.0f}ms")
        if i < len(queries) and sleep_s > 0:
            time.sleep(sleep_s)
    return {"n_sent": len(queries), "n_refused": refused, "rows": rows}


def report(result: dict) -> dict[str, dict]:
    table = {}
    for stage in STAGES:
        vals = [r[stage] for r in result["rows"] if r.get(stage) is not None]
        if vals:
            table[stage] = summarize(vals)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="deployed base URL, no trailing slash")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=15.0, help="seconds between queries (Groq TPM pacing)")
    ap.add_argument("--output", default="backend/data/remote_benchmark_results.json")
    args = ap.parse_args()

    health = httpx.get(f"{args.url}/health", timeout=30)
    health.raise_for_status()
    print(f"health: {health.json()}")

    queries = load_real_queries(args.queries)
    result = run(httpx.Client(timeout=120), args.url.rstrip("/"), queries, args.sleep)
    table = report(result)

    print(f"\nn={result['n_sent']} sent, {result['n_refused']} refused "
          f"→ {len(result['rows'])} retrieval samples\n")
    print(f"{'stage':<18}{'n':>4}{'p50':>9}{'p70':>9}{'p100':>9}")
    for stage, s in table.items():
        print(f"{stage:<18}{s['n']:>4}{s['p50']:>9.1f}{s['p70']:>9.1f}{s['p100']:>9.1f}")

    out = Path(args.output)
    out.write_text(json.dumps({"url": args.url, "table": table, **result}, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
