#!/usr/bin/env python
"""Phase 8 latency benchmark: P50/P70/P100 over real dataset queries.

Runs the REAL orchestrator (guards + retrieval + generation, live providers
unless --mock) and reports two honestly-separated buckets per the task spec:

  1. retrieval-only = embed_query + vector_search + chunk_assembly.
     This is the ONLY number checked against the sub-200ms target.
  2. full end-to-end = every stage including mandated external STT/LLM calls.
     Reported as-is; no 200ms claim is attached to it anywhere.

Honesty rules baked in:
- Warm-up (embedder load + one discarded pipeline call) happens BEFORE any
  timed run and is reported but never averaged into results.
- Refused queries are kept in end-to-end stats (real user-perceived latency)
  but EXCLUDED from retrieval-only stats — a guard short-circuit runs no
  retrieval, so a 0ms would fake a better percentile. Refusal counts are
  printed either way.

Usage:
    .venv/bin/python scripts/run_benchmark.py --queries 50 \
        --output docs/latency_report.md [--voice 5] [--strategies all] [--mock]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.benchmarking.latency import CANONICAL_STAGES, summarize
from app.benchmarking.queries import load_real_queries
from app.config import Settings, get_settings
from app.generation.llm_client import MockLLM, get_llm_provider
from app.harness.orchestrator import AskResponse, run_pipeline
from app.retrieval.retriever import Retriever, warm_retrieval
from app.retrieval.vector_store import collection_counts
from app.stt.factory import get_stt_provider
from app.stt.mock import MockSTTProvider

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = REPO_ROOT / "backend" / "data" / "audio" / "sample.webm"
RAW_JSON_PATH = REPO_ROOT / "backend" / "data" / "benchmark_results.json"

# Everything below this marker in the output file is hand-authored narrative
# (interpretation, patterns-considered). Reruns regenerate only the part above.
AUTHORED_MARKER = "<!-- authored-sections: preserved verbatim on rerun -->"

TARGET_MS = 200.0


def _fmt_row(label: str, s: dict) -> str:
    return (
        f"| {label} | {s['n']} | {s['p50']:.1f} | {s['p70']:.1f} "
        f"| {s['p100']:.1f} | {s['mean']:.1f} |"
    )


def _table(header_label: str, series: dict[str, dict]) -> str:
    lines = [
        f"| metric | n | p50 | p70 | p100 (max) | mean |",
        "|---|---|---|---|---|---|",
    ]
    lines += [_fmt_row(k, v) for k, v in series.items()]
    return "\n".join(lines)


def run_text_queries(queries: list[str], settings: Settings, llm,
                     sleep_s: float = 0.0) -> list[dict]:
    """Full pipeline per query; captures wall clock + structured response."""
    rows = []
    for i, q in enumerate(queries):
        if i and sleep_s:
            time.sleep(sleep_s)  # pace under provider TPM limits
        t0 = time.perf_counter()
        resp = run_pipeline(text=q, settings=settings, llm_provider=llm)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(
            {
                "mode": "text",
                "query": q,
                "wall_ms": round(wall_ms, 3),
                "retrieval_ms": resp.retrieval_ms,
                "stages": dict(resp.latency_trace_ms),
                "refused": resp.refused,
                "refusal_reason": resp.refusal_reason,
                "answer_chars": len(resp.answer),
                "warnings": resp.warnings,
            }
        )
        print(
            f"  [{i + 1:>2}/{len(queries)}] retrieval={resp.retrieval_ms:6.1f}ms "
            f"e2e={wall_ms:7.1f}ms {'REFUSED(' + str(resp.refusal_reason) + ')' if resp.refused else 'ok':>18}"
            f"  {q[:44]!r}"
        )
    return rows


def run_voice_queries(n: int, audio_path: Path, settings: Settings, llm,
                      sleep_s: float = 0.0) -> list[dict]:
    """Same pipeline with audio input through the configured STT provider."""
    audio = audio_path.read_bytes()
    rows = []
    for i in range(n):
        if i and sleep_s:
            time.sleep(sleep_s)  # pace under provider TPM limits
        t0 = time.perf_counter()
        resp = run_pipeline(audio_bytes=audio, mime_type="audio/webm",
                            settings=settings, llm_provider=llm)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(
            {
                "mode": "voice",
                "query": f"<audio:{audio_path.name}> transcript={resp.transcript[:60]!r}",
                "wall_ms": round(wall_ms, 3),
                "retrieval_ms": resp.retrieval_ms,
                "stages": dict(resp.latency_trace_ms),
                "refused": resp.refused,
                "refusal_reason": resp.refusal_reason,
                "answer_chars": len(resp.answer),
                "warnings": resp.warnings,
            }
        )
        print(
            f"  [{i + 1:>2}/{n}] stt={resp.latency_trace_ms.get('stt', 0):6.0f}ms "
            f"retrieval={resp.retrieval_ms:5.1f}ms e2e={wall_ms:7.1f}ms"
        )
    return rows


def run_strategy_comparison(queries: list[str], strategies: list[str],
                            settings: Settings) -> dict[str, list[float]]:
    """Retrieval-only latencies per indexed strategy (embed + search only)."""
    out: dict[str, list[float]] = {}
    for name in strategies:
        retriever = Retriever(strategy_name=name, settings=settings)
        times = []
        for q in queries:
            result = retriever.retrieve(q)
            times.append(result.trace.retrieval_ms())
        out[name] = times
    return out


def build_report(args, settings: Settings, text_rows: list[dict],
                 voice_rows: list[dict], strategy_stats: dict[str, dict],
                 counts: dict[str, int], warm_discard_ms: float) -> str:
    """Markdown report — auto-generated numbers only; narrative interpretation
    sections are maintained separately in docs/latency_report.md."""
    llm_desc = "mock (offline dry-run)" if args.mock else (
        f"{settings.llm_provider} / model from env")
    try:
        stt_name = get_stt_provider(settings).__class__.__name__
    except Exception:
        stt_name = "unresolved"

    lines = [
        "# Latency Report",
        "",
        f"_Auto-generated by `scripts/run_benchmark.py` at "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        f"Raw per-query rows: `backend/data/benchmark_results.json`._",
        "",
        "## Run configuration",
        "",
        "| setting | value |",
        "|---|---|",
        f"| text-mode queries | {len(text_rows)} |",
        f"| voice-mode runs | {len(voice_rows)}"
        + (f" (same clip: {args.audio})" if voice_rows else "")
        + " |",
        f"| LLM | {llm_desc} |",
        f"| STT provider | {stt_name} |",
        f"| index sizes | "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + " |",
        f"| warmup | embedder warm + 1 discarded pipeline call "
        f"({warm_discard_ms:.0f}ms), excluded from all stats below |",
        "",
    ]

    # --- bucket 1: retrieval-only vs target ---------------------------------
    ret_vals = [r["retrieval_ms"] for r in text_rows if not r["refused"]]
    refused_n = sum(1 for r in text_rows if r["refused"])
    lines += ["## Retrieval-only vs the 200ms target", ""]
    if ret_vals:
        rs = summarize(ret_vals)
        lines += [
            _table("metric", {"retrieval_only_ms": rs}),
            "",
            f"{rs['n']} of {len(text_rows)} queries measured"
            + (f" ({refused_n} refused by the input guard before retrieval ran — "
               f"excluded here, counted in end-to-end)" if refused_n else "")
            + ".",
            "",
            f"**Verdict:** the retrieval-only path "
            f"({'MEETS' if rs['p50'] < TARGET_MS else 'DOES NOT MEET'}) the "
            f"sub-{TARGET_MS:.0f}ms target — P50 {rs['p50']:.1f}ms, "
            f"P70 {rs['p70']:.1f}ms, P100 {rs['p100']:.1f}ms.",
        ]
    else:
        lines.append("_No retrieval measurements (all queries refused or none run)._")
    lines.append("")

    # --- bucket 2: full end-to-end, no target claimed ------------------------
    lines += [
        "## Full end-to-end — honest numbers, NO 200ms claim attached",
        "",
        "Includes guards + generation (+ STT for voice). These stages cross "
        "hosted-API boundaries that requirement #1 mandates, so they cannot "
        "and do not satisfy the retrieval-side latency budget.",
        "",
    ]
    e2e_series: dict[str, dict] = {}
    if text_rows:
        e2e_series["text_e2e_ms"] = summarize([r["wall_ms"] for r in text_rows])
    if voice_rows:
        e2e_series["voice_e2e_ms"] = summarize([r["wall_ms"] for r in voice_rows])
    lines += [_table("metric", e2e_series), ""]

    # --- per-stage breakdown --------------------------------------------------
    stage_vals: dict[str, list[float]] = {s: [] for s in CANONICAL_STAGES}
    for r in text_rows + voice_rows:
        for stage, ms in r["stages"].items():
            stage_vals.setdefault(stage, []).append(ms)
    per_stage = {
        k: summarize(v) for k, v in stage_vals.items() if v
    }
    if per_stage:
        lines += ["## Per-stage breakdown (text + voice runs)", "",
                  _table("stage", per_stage), ""]

    # --- strategy comparison --------------------------------------------------
    if strategy_stats:
        lines += [
            "## Chunking strategy comparison (retrieval-only)",
            "",
            "| strategy | vectors | n | p50 | p70 | p100 | mean |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, vals in strategy_stats.items():
            s = summarize(vals)
            lines.append(
                f"| {name} | {counts.get(name, '?')} | {s['n']} | {s['p50']:.1f} "
                f"| {s['p70']:.1f} | {s['p100']:.1f} | {s['mean']:.1f} |"
            )
        fastest = min(strategy_stats, key=lambda k: summarize(strategy_stats[k])["p50"])
        slowest = max(strategy_stats, key=lambda k: summarize(strategy_stats[k])["p50"])
        lines += [
            "",
            f"Fastest by P50: **{fastest}**; slowest: **{slowest}**. "
            "Latency comparison only — answer-quality differences between "
            "strategies are not evaluated here.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--voice", type=int, default=0,
                    help="also run N audio-mode requests")
    ap.add_argument("--audio", type=str, default=str(DEFAULT_AUDIO))
    ap.add_argument("--strategies", choices=["default", "all"], default="all")
    ap.add_argument("--output", type=str, default=str(REPO_ROOT / "docs" / "latency_report.md"))
    ap.add_argument("--mock", action="store_true",
                    help="offline dry-run: mock LLM, mock STT (machinery check only)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to wait BETWEEN queries (stay under provider "
                         "TPM limits; wall-clock timing per query is unaffected)")
    args = ap.parse_args()

    settings = get_settings()
    counts = collection_counts(settings)
    if counts.get(settings.default_chunk_strategy, 0) == 0:
        raise SystemExit(
            f"no index for {settings.default_chunk_strategy!r} (have: {counts}) "
            "— run scripts/build_index.py first"
        )

    llm = MockLLM() if args.mock else get_llm_provider(settings)

    # warm-up FIRST: embedder load (~9-11s cold) + connection pools, then one
    # full discarded pipeline call. Reported, never averaged in.
    warm_embed_ms = warm_retrieval(settings)
    t0 = time.perf_counter()
    run_pipeline(text="warmup discard", settings=settings, llm_provider=llm)
    warm_discard_ms = (time.perf_counter() - t0) * 1000.0
    mode = "MOCK" if args.mock else "LIVE"
    print(f"[{mode}] warmed embedder in {warm_embed_ms:.0f}ms; "
          f"discarded first pipeline call took {warm_discard_ms:.0f}ms\n")

    text_rows: list[dict] = []
    if args.queries > 0:
        queries = load_real_queries(args.queries, settings)
        print(f"running {len(queries)} real dataset queries through the orchestrator:")
        text_rows = run_text_queries(queries, settings, llm, sleep_s=args.sleep)

    voice_rows: list[dict] = []
    if args.voice > 0:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise SystemExit(f"--audio file not found: {audio_path}")
        print(f"\nrunning {args.voice} voice-mode requests via {audio_path.name}:")
        voice_rows = run_voice_queries(args.voice, audio_path, settings, llm,
                                       sleep_s=args.sleep)

    strategy_stats: dict[str, list[float]] = {}
    if text_rows:
        strat_names = (
            [settings.default_chunk_strategy] if args.strategies == "default"
            else [k for k, v in sorted(counts.items()) if v > 0]
        )
        print("\nretrieval-only per chunking strategy:")
        comp_queries = [r["query"] for r in text_rows]
        strategy_stats = run_strategy_comparison(comp_queries, strat_names, settings)
        for name, vals in strategy_stats.items():
            s = summarize(vals)
            print(f"  {name:<20} p50={s['p50']:6.1f}ms p70={s['p70']:6.1f}ms "
                  f"p100={s['p100']:6.1f}ms")

    report = build_report(args, settings, text_rows, voice_rows, strategy_stats,
                          counts, warm_discard_ms)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        old = out_path.read_text(encoding="utf-8")
        if AUTHORED_MARKER in old:
            authored = old.split(AUTHORED_MARKER, 1)[1]
            report = report.rstrip("\n") + "\n\n" + AUTHORED_MARKER + authored
    out_path.write_text(report, encoding="utf-8")

    RAW_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_JSON_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mock": args.mock,
        "text_rows": text_rows,
        "voice_rows": voice_rows,
        "strategy_stats_ms": strategy_stats,
    }, indent=2), encoding="utf-8")

    print(f"\nreport written to {out_path}")
    print(f"raw rows written to {RAW_JSON_PATH}")

    # final verdict block last — /latency-check pipes tail -60
    ret_vals = [r["retrieval_ms"] for r in text_rows if not r["refused"]]
    if ret_vals:
        rs = summarize(ret_vals)
        print(f"\nRETRIEVAL-ONLY vs sub-{TARGET_MS:.0f}ms target: "
              f"P50 {rs['p50']:.1f}ms / P70 {rs['p70']:.1f}ms / "
              f"P100 {rs['p100']:.1f}ms -> "
              f"{'MEETS TARGET' if rs['p50'] < TARGET_MS else 'MISSES TARGET'}")
    for label, rows in (("TEXT", text_rows), ("VOICE", voice_rows)):
        if rows:
            es = summarize([r["wall_ms"] for r in rows])
            print(f"FULL END-TO-END ({label}): P50 {es['p50']:.1f}ms / "
                  f"P70 {es['p70']:.1f}ms / P100 {es['p100']:.1f}ms "
                  f"(no target claimed)")
    if text_rows:
        print(f"refusals during run: {sum(1 for r in text_rows if r['refused'])}/{len(text_rows)}")


if __name__ == "__main__":
    main()
