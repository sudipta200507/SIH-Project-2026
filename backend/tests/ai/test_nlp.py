"""NLP tests (Phase E): tokenizer, text prep, load rules, mocked inference."""

from __future__ import annotations

import pytest

from app.ai.nlp.model import (
    DEFAULT_CLASS_LABELS,
    FORENTISAI_NLP_MARKER,
    NLPModelLoadError,
    load_nlp_model,
)
from app.ai.nlp.inference import load_error_to_result, predict_nlp
from app.ai.nlp.tokenizer import build_model_input_text, html_to_text, prepare_model_text

from ai_fixtures.helpers import sample_evidence


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------


def test_html_to_text_strips_scripts_and_tags():
    html = "<html><head><style>body{color:red}</style></head><body><p>Hello <b>world</b></p><script>alert(1)</script></body></html>"
    text = html_to_text(html)
    assert "alert" not in text
    assert "color:red" not in text
    assert "Hello" in text and "world" in text
    assert "<" not in text


def test_html_to_text_empty():
    assert html_to_text("") == ""
    assert html_to_text(None) == ""  # type: ignore[arg-type]


def test_build_model_input_text_preference_order():
    assert build_model_input_text("Subject", "Body", None) == "Subject\nBody"
    assert build_model_input_text(None, "Body", "<p>html</p>") == "Body"
    assert build_model_input_text("Subject", None, None) == "Subject"
    html_only = build_model_input_text(None, None, "<p>Fall back to html</p>")
    assert html_only == "Fall back to html"


def test_build_model_input_text_empty_returns_none():
    assert build_model_input_text(None, None, None) is None
    assert build_model_input_text("", "   ", "") is None


def test_prepare_model_text_truncates():
    evidence = sample_evidence()
    text = prepare_model_text(evidence, max_chars=10)
    assert text is not None
    assert len(text) <= 10


def test_prepare_model_text_empty_body():
    evidence = sample_evidence().model_copy(deep=True)
    evidence.message.subject = None
    evidence.body.plain_text = None
    evidence.body.html = None
    assert prepare_model_text(evidence, max_chars=100) is None


# ---------------------------------------------------------------------------
# Model loading rules (no real model needed)
# ---------------------------------------------------------------------------


def test_load_missing_nlp_model(tmp_path):
    with pytest.raises(NLPModelLoadError) as excinfo:
        load_nlp_model(tmp_path / "missing")
    assert excinfo.value.reason == "model_not_found"


def test_load_base_unmarked_model_is_refused(tmp_path):
    """An artifact WITHOUT the ForentisAI marker must never be used."""

    model_dir = tmp_path / "base_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "deberta-v2", "id2label": {"0": "benign", "1": "suspicious"}}',
        encoding="utf-8",
    )
    with pytest.raises(NLPModelLoadError) as excinfo:
        load_nlp_model(model_dir)
    assert excinfo.value.reason == "model_not_trained"


def test_load_wrong_marker_is_refused(tmp_path):
    model_dir = tmp_path / "foreign_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"_forentisai_finetune_marker": "some_other_project_v9"}',
        encoding="utf-8",
    )
    with pytest.raises(NLPModelLoadError) as excinfo:
        load_nlp_model(model_dir)
    assert excinfo.value.reason == "incompatible_model"


def test_load_marker_with_wrong_labels_is_refused(tmp_path):
    model_dir = tmp_path / "wrong_labels"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"_forentisai_finetune_marker": "%s", "id2label": {"0": "ham", "1": "spam"}}'
        % FORENTISAI_NLP_MARKER,
        encoding="utf-8",
    )
    with pytest.raises(NLPModelLoadError) as excinfo:
        load_nlp_model(model_dir)
    assert excinfo.value.reason == "incompatible_model"


def test_default_labels_are_benign_suspicious():
    assert DEFAULT_CLASS_LABELS == ("benign", "suspicious")


# ---------------------------------------------------------------------------
# Inference with a fake bundle (no transformers weights involved)
# ---------------------------------------------------------------------------


class FakeTokenizer:
    def __call__(self, text, return_tensors, truncation, max_length):
        assert return_tensors == "pt"
        assert truncation is True
        assert max_length == 512
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


class FakeTensor(list):
    """Minimal tensor stand-in: indexing re-wraps rows so .tolist() works."""

    def tolist(self):
        return [list(row) if isinstance(row, (list, FakeTensor)) else row for row in self]

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if isinstance(item, list) and not isinstance(item, FakeTensor):
            return FakeTensor(item)
        return item


FakeLogits = FakeTensor


class FakeModel:
    def eval(self):
        return self

    def __call__(self, **_inputs):
        class _Out:
            logits = FakeLogits([[2.0, -2.0]])  # strongly benign after softmax

        return _Out()


@pytest.fixture()
def fake_torch(monkeypatch):
    """Provide a deterministic torch stand-in for softmax/no_grad."""

    import contextlib
    import math
    import sys

    fake = type("torch", (), {})()
    fake.no_grad = contextlib.nullcontext

    def softmax(logits, dim=-1):
        rows = logits.tolist() if hasattr(logits, "tolist") else logits
        out_rows = []
        for row in rows:
            exps = [math.exp(float(v)) for v in row]
            total = sum(exps)
            out_rows.append([e / total for e in exps])
        return FakeTensor(out_rows)

    fake.softmax = softmax
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def test_predict_nlp_without_bundle_is_unavailable():
    result = predict_nlp("some text", None)
    assert result.available is False
    assert result.reason == "model_not_found"
    assert result.predicted_class == "unknown"


def test_predict_nlp_empty_text_is_unavailable():
    class _Unused:
        pass

    result = predict_nlp(None, _Unused())
    assert result.available is False
    assert result.reason == "empty_text"
    result = predict_nlp("   ", _Unused())
    assert result.available is False
    assert result.reason == "empty_text"


def test_predict_nlp_mocked_inference(fake_torch):
    bundle = type(
        "Bundle",
        (),
        {
            "tokenizer": FakeTokenizer(),
            "model": FakeModel(),
            "model_name": "fake-nlp",
            "class_labels": ("benign", "suspicious"),
            "max_length_tokens": 512,
        },
    )()
    result = predict_nlp("Please review the attached report.", bundle)
    assert result.available is True
    assert result.model_kind == "fine_tuned"
    assert result.model_name == "fake-nlp"
    assert set(result.probabilities) == {"benign", "suspicious"}
    assert result.probabilities["benign"] > result.probabilities["suspicious"]
    assert result.predicted_class == "benign"


def test_predict_nlp_indicator_detection(fake_torch):
    bundle = type(
        "Bundle",
        (),
        {
            "tokenizer": FakeTokenizer(),
            "model": FakeModel(),
            "model_name": "fake-nlp",
            "class_labels": ("benign", "suspicious"),
            "max_length_tokens": 512,
        },
    )()
    text = "URGENT: verify your account immediately or your account will be closed. Wire transfer the invoice amount."
    result = predict_nlp(text, bundle)
    names = {indicator.name for indicator in result.indicators}
    assert "urgency_language" in names
    assert "credential_request_pattern" in names
    assert "payment_request_pattern" in names
    assert "threat_pressure_pattern" in names
    # Indicators are signals, not verdicts — wording check:
    for indicator in result.indicators:
        assert "confirmed" not in indicator.detail.lower()


def test_predict_nlp_inference_failure_is_error_state(fake_torch):
    class BrokenModel:
        def eval(self):
            return self

        def __call__(self, **_inputs):
            raise RuntimeError("model exploded")

    bundle = type(
        "Bundle",
        (),
        {
            "tokenizer": FakeTokenizer(),
            "model": BrokenModel(),
            "model_name": "fake-nlp",
            "class_labels": ("benign", "suspicious"),
            "max_length_tokens": 512,
        },
    )()
    result = predict_nlp("text", bundle)
    assert result.available is False
    assert result.status == "error"
    assert result.predicted_class == "unknown"


def test_load_error_to_result_states():
    assert load_error_to_result(NLPModelLoadError("model_not_trained")).reason == "model_not_trained"
    assert load_error_to_result(NLPModelLoadError("model_not_found")).reason == "model_not_found"
    assert (
        load_error_to_result(NLPModelLoadError("dependency_unavailable")).reason
        == "dependency_unavailable"
    )
