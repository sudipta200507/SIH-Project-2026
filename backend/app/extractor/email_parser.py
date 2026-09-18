"""Safe Step 1 EML extraction entry points and a minimal command-line demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import mailparser

from app.extractor.normalizer import build_file_metadata, normalize_email
from app.schemas.email import EmailEvidence, ParserDefect


DEFAULT_MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024


class EmailExtractionError(Exception):
    """Base exception that carries a safe, machine-readable error code."""

    code = "email_extraction_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class FileValidationError(EmailExtractionError):
    """Raised when an upload is invalid before the parser consumes it."""

    code = "file_validation_error"


class UnsupportedEmailFormatError(FileValidationError):
    """Raised for formats that this installed Step 1 configuration cannot parse."""

    code = "unsupported_email_format"


class EmailParsingError(EmailExtractionError):
    """Raised only when no structured EML parsing result can be produced."""

    code = "email_parsing_error"


@dataclass(frozen=True, slots=True)
class ParsedEmail:
    """Internal parser output kept separate from normalized evidence."""

    message: Message
    mail_message: Any | None
    defects: list[ParserDefect]


def _safe_filename(filename: str) -> str:
    """Use a filename only as display metadata, never as an executable command."""

    return filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1].strip() or "uploaded.eml"


def _validate_filename(filename: str) -> None:
    suffix = Path(_safe_filename(filename)).suffix.casefold()
    if suffix == ".msg":
        raise UnsupportedEmailFormatError(
            "MSG parsing is not enabled: mail-parser requires the optional "
            "'mail-parser[outlook]' extract-msg backend (or deprecated msgconvert), "
            "which is not installed for Step 1."
        )
    if suffix != ".eml":
        raise UnsupportedEmailFormatError(
            "Unsupported email file type. Only .eml files are supported in Step 1."
        )


def _validate_bytes(raw_bytes: bytes, max_size_bytes: int) -> None:
    if not raw_bytes:
        raise FileValidationError("The uploaded email file is empty.")
    if len(raw_bytes) > max_size_bytes:
        raise FileValidationError(
            f"The uploaded email exceeds the {max_size_bytes} byte size limit."
        )


def parse_raw_email(raw_bytes: bytes) -> ParsedEmail:
    """Parse raw EML bytes with mail-parser and the standard MIME parser.

    mail-parser is invoked for its enhanced parsing and defect information.
    Python's ``BytesParser`` supplies the MIME tree used for duplicate-preserving
    headers and exact decoded attachment bytes. Both operate only in memory.
    """

    defects: list[ParserDefect] = []
    try:
        mail_message = mailparser.parse_from_bytes(raw_bytes)
    except Exception as error:  # mail-parser should not prevent safe fallback parsing
        mail_message = None
        defects.append(
            ParserDefect(
                source="mail-parser",
                message=f"Unable to parse input: {type(error).__name__}",
            )
        )

    try:
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    except Exception as error:
        raise EmailParsingError(
            "The uploaded data could not be parsed as an RFC 5322 email message."
        ) from error
    return ParsedEmail(message=message, mail_message=mail_message, defects=defects)


def extract_email_from_bytes(
    raw_bytes: bytes | bytearray | memoryview,
    *,
    filename: str = "uploaded.eml",
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> EmailEvidence:
    """Extract normalized evidence from original EML bytes without modifying them."""

    if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
        raise FileValidationError("Email content must be supplied as bytes.")
    if max_size_bytes <= 0:
        raise ValueError("max_size_bytes must be greater than zero")

    original_bytes = bytes(raw_bytes)
    safe_filename = _safe_filename(filename)
    _validate_filename(safe_filename)
    _validate_bytes(original_bytes, max_size_bytes)

    # This metadata is deliberately calculated before any parsing occurs.
    file_metadata = build_file_metadata(original_bytes, safe_filename)
    parsed = parse_raw_email(original_bytes)
    return normalize_email(
        file_metadata=file_metadata,
        message=parsed.message,
        mail_message=parsed.mail_message,
        parser_defects=parsed.defects,
    )


def extract_email_from_file(
    file_path: str | Path,
    *,
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> EmailEvidence:
    """Read a validated EML file and return its derived normalized evidence."""

    path = Path(file_path)
    if not path.exists():
        raise FileValidationError("The uploaded email file does not exist.")
    if not path.is_file():
        raise FileValidationError("The uploaded email path is not a file.")
    _validate_filename(path.name)

    try:
        file_size = path.stat().st_size
    except OSError as error:
        raise FileValidationError("The uploaded email file could not be inspected.") from error
    if file_size > max_size_bytes:
        raise FileValidationError(
            f"The uploaded email exceeds the {max_size_bytes} byte size limit."
        )

    try:
        original_bytes = path.read_bytes()
    except OSError as error:
        raise FileValidationError("The uploaded email file could not be read.") from error

    return extract_email_from_bytes(
        original_bytes,
        filename=path.name,
        max_size_bytes=max_size_bytes,
    )


def extract_email(
    source: bytes | bytearray | memoryview | str | Path,
    *,
    filename: str | None = None,
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> EmailEvidence:
    """Accept either raw EML bytes or a filesystem path in one explicit API."""

    if isinstance(source, (bytes, bytearray, memoryview)):
        return extract_email_from_bytes(
            source,
            filename=filename or "uploaded.eml",
            max_size_bytes=max_size_bytes,
        )
    if isinstance(source, (str, Path)):
        if filename is not None:
            raise ValueError("filename is only used when source is raw bytes")
        return extract_email_from_file(source, max_size_bytes=max_size_bytes)
    raise FileValidationError("Email source must be raw bytes or a file path.")


# Clear aliases for callers that prefer parser-oriented naming.
parse_email_bytes = extract_email_from_bytes
parse_email_file = extract_email_from_file


def _safe_summary(evidence: EmailEvidence) -> dict[str, object]:
    """Return only the requested metadata summary; bodies and headers stay private."""

    return {
        "Filename": evidence.file.filename,
        "SHA-256": evidence.file.sha256,
        "Sender": evidence.sender.address if evidence.sender else None,
        "Recipients": [
            recipient.address for recipient in evidence.recipients.to if recipient.address
        ],
        "Subject": evidence.message.subject,
        "Date": evidence.message.date,
        "Number of headers": sum(len(values) for values in evidence.headers.raw.values()),
        "Number of Received headers": len(evidence.received_chain),
        "Number of URLs": len(evidence.urls),
        "Number of attachments": len(evidence.attachments),
    }


def main(argv: list[str] | None = None) -> int:
    """Run a safe local extraction summary for one EML file."""

    parser = argparse.ArgumentParser(description="Extract safe metadata from an EML file")
    parser.add_argument("eml_file", type=Path, help="Path to a .eml file")
    arguments = parser.parse_args(argv)
    try:
        evidence = extract_email_from_file(arguments.eml_file)
    except EmailExtractionError as error:
        parser.exit(status=2, message=f"error: {error.message}\n")

    print(json.dumps(_safe_summary(evidence), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
