"""Automated coverage for the Step 1 EML extraction engine."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.extractor.email_parser import (
    FileValidationError,
    UnsupportedEmailFormatError,
    extract_email_from_bytes,
    extract_email_from_file,
)
from app.schemas.email import EmailEvidence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = PROJECT_ROOT / "samples" / "safe" / "step1_synthetic.eml"


@pytest.fixture(scope="module")
def evidence() -> EmailEvidence:
    return extract_email_from_file(SAMPLE_PATH)


def test_valid_eml_parsing_returns_normalized_evidence(evidence: EmailEvidence) -> None:
    assert isinstance(evidence, EmailEvidence)
    assert evidence.file.filename == "step1_synthetic.eml"
    assert evidence.file.size == len(SAMPLE_PATH.read_bytes())


def test_subject_is_extracted(evidence: EmailEvidence) -> None:
    assert evidence.message.subject == "Step 1 extraction fixture"


def test_sender_is_extracted(evidence: EmailEvidence) -> None:
    assert evidence.sender is not None
    assert evidence.sender.address == "sender@example.test"


def test_recipients_are_extracted(evidence: EmailEvidence) -> None:
    assert [recipient.address for recipient in evidence.recipients.to] == [
        "recipient@example.test"
    ]
    assert [recipient.address for recipient in evidence.recipients.cc] == [
        "copy@example.test"
    ]
    assert [recipient.address for recipient in evidence.recipients.bcc] == [
        "blind@example.test"
    ]


def test_plain_and_html_bodies_are_extracted(evidence: EmailEvidence) -> None:
    assert evidence.body.plain_text is not None
    assert "safe synthetic message" in evidence.body.plain_text
    assert evidence.body.html is not None
    assert "HTML indicator" in evidence.body.html


def test_headers_are_preserved(evidence: EmailEvidence) -> None:
    assert evidence.headers.raw["From"] == [
        "Forentis Test Sender <sender@example.test>"
    ]
    assert evidence.headers.reply_to == ["Reply Desk <reply@example.test>"]
    assert evidence.headers.return_path == ["<bounce@example.test>"]
    assert evidence.headers.authentication_results == []


def test_received_headers_are_preserved(evidence: EmailEvidence) -> None:
    assert len(evidence.headers.received) == 2
    assert evidence.received_chain == evidence.headers.received


def test_received_chain_preserves_four_hops_and_order_across_header_casing() -> None:
    received_values = [
        "from hop-one.example.test by relay.example.test; 1",
        "from hop-two.example.test by hop-one.example.test; 2",
        "from hop-three.example.test by hop-two.example.test; 3",
        "from hop-four.example.test by hop-three.example.test; 4",
    ]
    raw_email = (
        f"Received: {received_values[0]}\r\n"
        f"received: {received_values[1]}\r\n"
        f"Received: {received_values[2]}\r\n"
        f"RECEIVED: {received_values[3]}\r\n"
        "From: sender@example.test\r\n"
        "To: recipient@example.test\r\n"
        "Subject: Received chain fixture\r\n"
        "\r\n"
        "Body"
    ).encode("ascii")

    result = extract_email_from_bytes(raw_email, filename="received-chain.eml")

    assert result.received_chain == received_values
    assert result.headers.received == received_values
    assert result.headers.raw["Received"] == [received_values[0], received_values[2]]
    assert result.headers.raw["received"] == [received_values[1]]
    assert result.headers.raw["RECEIVED"] == [received_values[3]]


def test_plain_text_only_body_is_extracted_without_changing_input_bytes() -> None:
    raw_email = (
        b"From: sender@example.test\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"Plain text only"
    )
    original_bytes = bytes(raw_email)

    result = extract_email_from_bytes(raw_email, filename="plain-only.eml")

    assert result.body.plain_text == "Plain text only"
    assert result.body.html is None
    assert raw_email == original_bytes


def test_html_only_body_is_extracted() -> None:
    raw_email = (
        b"From: sender@example.test\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"\r\n"
        b"<p>HTML only</p>"
    )

    result = extract_email_from_bytes(raw_email, filename="html-only.eml")

    assert result.body.plain_text is None
    assert result.body.html == "<p>HTML only</p>"


def test_urls_are_extracted_locally_from_plain_and_html(evidence: EmailEvidence) -> None:
    assert {(item.url, item.source, item.hostname) for item in evidence.urls} == {
        ("https://example.test/plain-path?source=plain", "plain", "example.test"),
        ("https://example.test/html-path?source=html", "html", "example.test"),
    }


def test_attachment_metadata_and_hash_are_extracted(evidence: EmailEvidence) -> None:
    assert len(evidence.attachments) == 1
    attachment = evidence.attachments[0]
    payload = b"safe fixture attachment.\n"
    assert attachment.filename == "evidence.txt"
    assert attachment.mime_type == "text/plain"
    assert attachment.size == len(payload)
    assert attachment.content_id == "<evidence-attachment@example.test>"
    assert attachment.sha256 == hashlib.sha256(payload).hexdigest()
    assert attachment.errors == []


def test_sha256_is_for_the_original_file_bytes(evidence: EmailEvidence) -> None:
    assert evidence.file.sha256 == hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest()


def test_normalized_schema_validates_and_serializes_predictably(
    evidence: EmailEvidence,
) -> None:
    serialized = evidence.model_dump(mode="json", by_alias=True)
    assert serialized["headers"]["from"] == [
        "Forentis Test Sender <sender@example.test>"
    ]
    assert EmailEvidence.model_validate(serialized) == evidence


def test_malformed_email_returns_structured_parser_defect() -> None:
    malformed = (
        b"From: sender@example.test\r\n"
        b"Content-Type: multipart/mixed\r\n"
        b"\r\n"
        b"Missing MIME boundary"
    )
    result = extract_email_from_bytes(malformed, filename="malformed.eml")
    assert any(defect.source == "python-email" for defect in result.parser_defects)


def test_missing_optional_fields_do_not_crash() -> None:
    result = extract_email_from_bytes(b"Subject: Minimal\r\n\r\n", filename="minimal.eml")
    assert result.sender is None
    assert result.recipients.to == []
    assert result.message.date is None
    assert result.body.plain_text in {None, ""}


def test_unsupported_file_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not-an-email.txt"
    path.write_bytes(b"This is not an EML file.")
    with pytest.raises(UnsupportedEmailFormatError):
        extract_email_from_file(path)


def test_msg_requires_uninstalled_optional_backend(tmp_path: Path) -> None:
    path = tmp_path / "outlook.msg"
    path.write_bytes(b"not a real MSG file")
    with pytest.raises(UnsupportedEmailFormatError, match="MSG parsing is not enabled"):
        extract_email_from_file(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.eml"
    path.write_bytes(b"")
    with pytest.raises(FileValidationError, match="empty"):
        extract_email_from_file(path)


def test_oversized_email_is_rejected_before_parsing() -> None:
    raw_email = b"Subject: Oversized fixture\r\n\r\nBody"
    with pytest.raises(FileValidationError, match="exceeds"):
        extract_email_from_bytes(
            raw_email,
            filename="oversized.eml",
            max_size_bytes=len(raw_email) - 1,
        )


def test_random_binary_eml_is_handled_without_an_uncontrolled_crash() -> None:
    random_binary = bytes(range(256))

    result = extract_email_from_bytes(random_binary, filename="random.eml")

    assert isinstance(result, EmailEvidence)
    assert result.file.sha256 == hashlib.sha256(random_binary).hexdigest()
    assert result.parser_defects
