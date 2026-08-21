import json

from pydantic import ValidationError
import pytest

from app.config import Settings
from app.ingestion import download_dataset, preprocess
from app.ingestion.models import RawDocument


# --- RawDocument model ---

def test_raw_document_defaults_and_roundtrip():
    doc = RawDocument(text="some passage", metadata={"query_id": 1})
    assert doc.id is None
    assert doc.metadata == {"query_id": 1}
    reparsed = RawDocument.model_validate_json(doc.model_dump_json())
    assert reparsed == doc


def test_raw_document_rejects_missing_text():
    with pytest.raises(ValidationError):
        RawDocument(metadata={})


# --- row flattening (download side, pure logic) ---

def test_row_to_documents_flattens_english_only():
    row = {
        "query_id": 42,
        "query_type": "description",
        "passages": {
            "English_passages": ["alpha passage text", "beta passage text"],
            "Translated_passages": ["ignored-a", "ignored-b"],
            "is_selected": [1, 0],
        },
    }
    docs = download_dataset.row_to_documents(row, language="hin", split="validation")
    assert [d.text for d in docs] == ["alpha passage text", "beta passage text"]
    assert all("ignored" not in d.text for d in docs)
    assert docs[0].metadata["is_selected"] is True
    assert docs[1].metadata["is_selected"] is False
    assert all(d.metadata["query_id"] == 42 for d in docs)
    assert all(d.metadata["query_type"] == "description" for d in docs)
    assert [d.metadata["passage_index"] for d in docs] == [0, 1]


def test_row_to_documents_survives_missing_flags_and_passages():
    no_flags = {"query_id": 7, "passages": {"English_passages": ["solo"], "is_selected": []}}
    (doc,) = download_dataset.row_to_documents(no_flags, language="hin", split="validation")
    assert doc.metadata["is_selected"] is False

    empty = {"query_id": 8, "passages": {}}
    assert download_dataset.row_to_documents(empty, language="hin", split="validation") == []


def test_shard_filename_layout_matches_hf_repo():
    assert download_dataset.shard_filename("train", "hin") == "train/hintrain.parquet"
    assert download_dataset.shard_filename("validation", "tam") == "validation/tamval.parquet"


# --- preprocessing (clean / filter / dedupe / stable ids) ---

def test_clean_text_strips_html_entities_and_whitespace():
    dirty = "  <b>Hello</b> &amp; <i>world</i>\n  more   text&nbsp;!  "
    assert preprocess.clean_text(dirty) == "Hello & world more text !"


def _write_raw(path, docs: list[RawDocument]) -> None:
    path.write_text("\n".join(d.model_dump_json() for d in docs) + "\n", encoding="utf-8")


def test_preprocess_end_to_end_drops_empty_and_dedupes(tmp_path):
    settings = Settings(raw_data_dir=tmp_path / "raw", processed_data_dir=tmp_path / "proc")
    real = "<p>Real passage about dogs.</p> &nbsp;Dogs are nice animals."
    raw_path = settings.raw_data_dir
    raw_path.mkdir(parents=True)
    _write_raw(
        raw_path / download_dataset.RAW_FILENAME,
        [
            RawDocument(text=real, metadata={"query_id": 1, "is_selected": False}),
            RawDocument(text="   ", metadata={"query_id": 2}),          # whitespace-only
            RawDocument(text="", metadata={"query_id": 3}),             # empty
            RawDocument(text="too short", metadata={"query_id": 4}),    # near-empty
            # duplicate of the first passage after cleaning, but selected:
            RawDocument(
                text="<br>Real passage about dogs. Dogs are nice animals.",
                metadata={"query_id": 5, "is_selected": True},
            ),
            RawDocument(text="Another distinct passage that is long enough.", metadata={"query_id": 6}),
        ],
    )

    stats = preprocess.run(settings)

    out_path = settings.processed_data_dir / preprocess.PROCESSED_FILENAME
    lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    texts = [l["text"] for l in lines]

    assert stats["dropped_empty"] == 2          # "" and "   "
    assert stats["dropped_near_empty"] == 1     # "too short"
    assert stats["duplicates_merged"] == 1      # same content after cleaning
    assert stats["kept"] == 2
    assert len(lines) == 2

    ids = [l["id"] for l in lines]
    assert all(ids) and len(set(ids)) == len(ids)   # unique, non-null ids
    assert texts[0] == "Real passage about dogs. Dogs are nice animals."  # HTML stripped
    # is_selected OR-merged onto the surviving duplicate:
    assert lines[0]["metadata"]["is_selected"] is True


def test_doc_ids_stable_across_runs(tmp_path):
    docs = [
        RawDocument(text=f"<p>Passage number {i} with plenty of content.</p>")
        for i in range(5)
    ]
    kept_a, _ = preprocess.preprocess_documents(docs, min_chars=10)
    kept_b, _ = preprocess.preprocess_documents(
        [RawDocument(text=d.text) for d in docs], min_chars=10
    )
    assert [d.id for d in kept_a] == [d.id for d in kept_b]


def test_preprocess_empty_input_writes_nothing_but_valid_file(tmp_path):
    settings = Settings(raw_data_dir=tmp_path / "raw", processed_data_dir=tmp_path / "proc")
    settings.raw_data_dir.mkdir(parents=True)
    _write_raw(settings.raw_data_dir / download_dataset.RAW_FILENAME, [])
    stats = preprocess.run(settings)
    out_path = settings.processed_data_dir / preprocess.PROCESSED_FILENAME
    assert stats["kept"] == 0
    assert out_path.exists() and out_path.read_text() == ""
