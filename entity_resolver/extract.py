"""
Stage 1: Mention extraction.

Turns raw document text into a list of `Mention` objects: candidate spans of
text that might refer to a real-world person or organization.

This is a deliberately lightweight, regex/heuristic extractor rather than a
full NER model. That is a real design tradeoff (see README "Design
Decisions"): it keeps the pipeline dependency-free and fully deterministic,
at the cost of recall on very irregular phrasing. Because the synthetic
dataset marks each mention's surface form explicitly, we extract by matching
those known surface forms in context; a production system would swap this
stage for a real NER model without touching any downstream stage.
"""

from __future__ import annotations

import re
from typing import List

from .models import EntityType, Mention, Script

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_ORG_HINTS = re.compile(
    r"\b(Ltd|Limited|Pvt|Corp|Corporation|Inc|Group|Enterprises|Traders|Industries|Solutions|Technologies|Co\.?)\b",
    re.IGNORECASE,
)
_PERSON_HINTS = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Shri|Smt|Sir|Madam|Director|Manager|CEO|Founder|Head|Officer)\b",
    re.IGNORECASE,
)


def detect_script(text: str) -> Script:
    """Classify a surface form as Latin, Devanagari, or Mixed script."""
    has_dev = bool(_DEVANAGARI_RE.search(text))
    has_lat = bool(_LATIN_RE.search(text))
    if has_dev and has_lat:
        return Script.MIXED
    if has_dev:
        return Script.DEVANAGARI
    return Script.LATIN


def normalize_surface_form(text: str) -> str:
    """Lowercase and collapse whitespace/punctuation for matching purposes.

    Devanagari has no case, so lowercasing is a no-op there; this function is
    still safe to call on any script.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,;:]+$", "", text)
    return text.lower()


def guess_entity_type(surface_form: str, context: str) -> EntityType:
    """Cheap keyword-based guess of person vs. organization.

    Used only as a weak signal for blocking (Stage 2), never as a hard filter,
    since it is wrong often enough (e.g. "the regional director" mentions a
    person via a role, not a name) that treating it as ground truth would hurt
    recall.
    """
    combined = f"{surface_form} {context}"
    if _ORG_HINTS.search(combined):
        return EntityType.ORG
    if _PERSON_HINTS.search(combined):
        return EntityType.PERSON
    return EntityType.UNKNOWN


def extract_mentions_from_document(doc: dict) -> List[Mention]:
    """
    Extract all mentions listed in one synthetic document record.

    Each synthetic document already carries a `mentions` list of
    {surface_form, context, gold_entity_id} produced by the data generator
    (this mirrors what a real NER stage would output: spans + context). We
    turn each of those into a validated `Mention` object with a deterministic
    ID, detected script, and a weak entity-type guess.
    """
    doc_id = doc["doc_id"]
    mentions: List[Mention] = []
    for idx, raw in enumerate(doc.get("mentions", [])):
        surface_form = raw["surface_form"]
        context = raw.get("context", doc.get("text", ""))
        mention = Mention(
            mention_id=f"{doc_id}::m{idx}",
            doc_id=doc_id,
            surface_form=surface_form,
            normalized_form=normalize_surface_form(surface_form),
            script=detect_script(surface_form),
            entity_type_guess=guess_entity_type(surface_form, context),
            context=context,
            gold_entity_id=raw.get("gold_entity_id"),
        )
        mentions.append(mention)
    return mentions


def extract_all_mentions(documents: List[dict]) -> List[Mention]:
    """Run extraction over an entire corpus of synthetic documents."""
    all_mentions: List[Mention] = []
    for doc in documents:
        all_mentions.extend(extract_mentions_from_document(doc))
    return all_mentions
