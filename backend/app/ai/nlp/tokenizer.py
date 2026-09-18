"""NLP text preparation (Phase E).

Builds the model input text from ALREADY-EXTRACTED EmailEvidence fields.
Everything is local string handling: no network requests, no HTML
rendering, no URL fetching. Raw email text never leaves the process and
never enters logs — only the prepared string is passed to the local model.

The base DeBERTa-v3 tokenizer is a Hugging Face artifact; this module only
prepares plain text so inference stays testable without the model.
"""

from __future__ import annotations

import re

from app.schemas.email import EmailEvidence

_SCRIPT_BLOCK = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_TAG_BLOCK = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Very conservative local HTML-to-text normalization.

    Removes script/style blocks and then remaining tags. This is text
    normalization for model input only — it is NOT a rendering engine, it
    never fetches remote resources referenced by the HTML, and it never
    executes anything.
    """

    if not html:
        return ""
    without_scripts = _SCRIPT_BLOCK.sub(" ", html)
    without_tags = _TAG_BLOCK.sub(" ", without_scripts)
    return _WHITESPACE.sub(" ", without_tags).strip()


def build_model_input_text(
    subject: str | None,
    plain_text: str | None,
    html: str | None,
) -> str | None:
    """Compose the classifier input text; None when there is no text at all.

    Preference order: subject + plain text; subject + normalized
    HTML-derived text when no plain part exists; subject alone otherwise.
    """

    subject = (subject or "").strip()
    plain = (plain_text or "").strip()

    if subject and plain:
        combined = f"{subject}\n{plain}"
    elif plain:
        combined = plain
    elif subject:
        combined = subject
    else:
        html_text = html_to_text(html or "")
        if not html_text:
            return None
        combined = html_text
        return combined or None

    return combined.strip() or None


def prepare_model_text(evidence: EmailEvidence, max_chars: int) -> str | None:
    """Prepared, length-bounded model input for one email; None if empty."""

    text = build_model_input_text(
        evidence.message.subject,
        evidence.body.plain_text,
        evidence.body.html,
    )
    if text is None:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text or None


__all__ = ["build_model_input_text", "html_to_text", "prepare_model_text"]
