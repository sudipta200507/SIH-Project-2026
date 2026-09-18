"""Pydantic models for the Step 1 normalized email evidence format."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceModel(BaseModel):
    """Base model that keeps the exported evidence shape predictable."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FileMetadata(EvidenceModel):
    """Metadata calculated from the original uploaded bytes before parsing."""

    filename: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_format: Literal["eml"] = "eml"


class EmailAddress(EvidenceModel):
    """A parsed mailbox address while raw header values remain in ``headers``."""

    display_name: str | None = None
    address: str | None = None


class RecipientGroups(EvidenceModel):
    """Recipient lists parsed from the message headers."""

    to: list[EmailAddress] = Field(default_factory=list)
    cc: list[EmailAddress] = Field(default_factory=list)
    bcc: list[EmailAddress] = Field(default_factory=list)


class MessageMetadata(EvidenceModel):
    """Message-level values that are useful without inspecting every header."""

    subject: str | None = None
    date: str | None = None
    message_id: str | None = None
    reply_to: list[EmailAddress] = Field(default_factory=list)
    return_path: str | None = None


class EmailBody(EvidenceModel):
    """Decoded body representations; the original email bytes are not altered."""

    plain_text: str | None = None
    html: str | None = None
    plain_text_parts: list[str] = Field(default_factory=list)
    html_parts: list[str] = Field(default_factory=list)


class HeaderSummary(EvidenceModel):
    """Important headers plus a complete, duplicate-preserving header map.

    Authentication-related values are extraction-only data. Their presence does
    not indicate that SPF, DKIM, DMARC, or ARC has been verified.
    """

    raw: dict[str, list[str]] = Field(default_factory=dict)
    from_header: list[str] = Field(
        default_factory=list,
        validation_alias="from",
        serialization_alias="from",
    )
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    bcc: list[str] = Field(default_factory=list)
    reply_to: list[str] = Field(default_factory=list)
    return_path: list[str] = Field(default_factory=list)
    message_id: list[str] = Field(default_factory=list)
    date: list[str] = Field(default_factory=list)
    received: list[str] = Field(default_factory=list)
    authentication_results: list[str] = Field(default_factory=list)
    received_spf: list[str] = Field(default_factory=list)
    dkim_signature: list[str] = Field(default_factory=list)
    arc_headers: dict[str, list[str]] = Field(default_factory=dict)
    mime_headers: dict[str, list[str]] = Field(default_factory=dict)
    content_type: str | None = None
    mime_version: str | None = None
    content_transfer_encoding: str | None = None
    user_agent: str | None = None
    x_mailer: str | None = None


class MimePart(EvidenceModel):
    """A leaf MIME part observed in the message."""

    content_type: str | None = None
    content_disposition: str | None = None
    charset: str | None = None
    filename: str | None = None
    content_id: str | None = None


class MimeInformation(EvidenceModel):
    """Top-level and leaf-part MIME information."""

    content_type: str | None = None
    mime_version: str | None = None
    content_transfer_encoding: str | None = None
    is_multipart: bool = False
    parts: list[MimePart] = Field(default_factory=list)


class URLIndicator(EvidenceModel):
    """A URL extracted locally from a plain-text or HTML body."""

    url: str
    source: Literal["plain", "html"]
    hostname: str | None = None


class AttachmentMetadata(EvidenceModel):
    """Metadata for an attachment without executing or exporting its content."""

    filename: str | None = None
    mime_type: str | None = None
    size: int = Field(ge=0)
    content_id: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    errors: list[str] = Field(default_factory=list)


class ParserDefect(EvidenceModel):
    """A safely reported issue from one of the local parsing stages."""

    source: str
    message: str


class EmailEvidence(EvidenceModel):
    """Derived, normalized forensic evidence for one original EML file."""

    schema_version: Literal["1.0"] = "1.0"
    file: FileMetadata
    message: MessageMetadata
    sender: EmailAddress | None = None
    recipients: RecipientGroups = Field(default_factory=RecipientGroups)
    body: EmailBody = Field(default_factory=EmailBody)
    headers: HeaderSummary = Field(default_factory=HeaderSummary)
    received_chain: list[str] = Field(default_factory=list)
    mime: MimeInformation = Field(default_factory=MimeInformation)
    urls: list[URLIndicator] = Field(default_factory=list)
    attachments: list[AttachmentMetadata] = Field(default_factory=list)
    parser_defects: list[ParserDefect] = Field(default_factory=list)
