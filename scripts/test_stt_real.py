"""One-shot live STT check against whatever provider config resolves.

Usage:
    .venv/bin/python scripts/test_stt_real.py [path/to/audio.(webm|wav|mp3|ogg|m4a)]

With no argument it looks for a sample in backend/data/audio/. Prints the
transcript verbatim and whether the result came from the mock provider —
never let this be ambiguous.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.benchmarking.latency import LatencyTrace, STT as STT_STAGE, stage_timer  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.stt.base import STTError  # noqa: E402
from app.stt.factory import get_stt_provider  # noqa: E402

AUDIO_DIR = REPO_ROOT / "backend" / "data" / "audio"
EXTENSIONS = ("*.wav", "*.mp3", "*.webm", "*.ogg", "*.m4a", "*.flac")


def find_sample(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for pattern in EXTENSIONS:
        matches = sorted(AUDIO_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def main() -> int:
    sample = find_sample(sys.argv[1] if len(sys.argv) > 1 else None)
    if sample is None:
        print(
            f"No audio sample found. Record ~5s of speech and save it under "
            f"{AUDIO_DIR}/ (or pass a path)."
        )
        return 1

    settings = get_settings()
    print(f"STT_PROVIDER={settings.stt_provider}")
    provider = get_stt_provider(settings)
    if getattr(provider, "__class__", type).__name__ == "MockSTTProvider":
        print("!! Resolved provider is MOCK — no live API call will happen.")

    audio_bytes = sample.read_bytes()
    mime = {
        ".webm": "audio/webm",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
    }.get(sample.suffix.lower(), "application/octet-stream")
    print(f"file={sample.name} ({len(audio_bytes)} bytes, {mime})")

    trace = LatencyTrace()
    try:
        with stage_timer(STT_STAGE, trace):
            result = provider.transcribe(audio_bytes, mime)
    except STTError as exc:
        print(f"LIVE CALL FAILED after retries: {exc}")
        return 2

    print(f"is_mock={result.is_mock} language={result.language}")
    print(f"stt_ms={trace.stage_ms.get(STT_STAGE):.1f}")
    print("--- transcript ---")
    print(result.text or "(empty)")
    print("------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
