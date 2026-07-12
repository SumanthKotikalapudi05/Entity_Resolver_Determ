"""
Shared data models for the entity resolution pipeline.

Every stage of the pipeline (extract -> block -> score -> merge -> graph_store
-> audit) passes data around using these pydantic models so that the shape of
"a mention", "a score", and "a merge decision" is defined in exactly one
place. Keeping these as pydantic models (rather than dicts) gives us free
validation and makes the pipeline self-documenting.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Script(str, Enum):
    """Detected writing script of a mention's surface form."""

    LATIN = "latin"
    DEVANAGARI = "devanagari"
    MIXED = "mixed"


class EntityType(str, Enum):
    """Coarse type of the real-world entity a mention refers to."""

    PERSON = "person"
    ORG = "org"
    UNKNOWN = "unknown"


class Mention(BaseModel):
    """
    A single occurrence of an entity's name (or description) inside one
    document, produced by the `extract` stage.

    `mention_id` is a stable, deterministic ID (doc_id + position) so the
    audit log and graph can reference a mention without ambiguity.
    """

    mention_id: str
    doc_id: str
    surface_form: str = Field(..., description="The raw text span as it appeared in the document")
    normalized_form: str = Field(..., description="Lowercased, whitespace-collapsed surface form")
    script: Script
    entity_type_guess: EntityType = EntityType.UNKNOWN
    context: str = Field(..., description="Surrounding sentence/snippet the mention was found in")
    gold_entity_id: Optional[str] = Field(
        default=None,
        description="Ground-truth entity ID from the synthetic dataset, used only for evaluation/tests, never by the pipeline itself",
    )


class ScoreBreakdown(BaseModel):
    """
    Explainable, per-signal scores for one candidate mention pair, plus the
    combined confidence. Every number here is independently inspectable so a
    human reviewer never has to trust a single opaque float.
    """

    mention_id_a: str
    mention_id_b: str
    name_similarity: float = Field(..., ge=0.0, le=1.0)
    context_overlap: float = Field(..., ge=0.0, le=1.0)
    cooccurrence_overlap: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Human-readable explanation of the score")


class MergeTier(str, Enum):
    AUTO_MERGE = "auto_merge"
    FLAGGED_REVIEW = "flagged_review"
    NOT_MERGED = "not_merged"


class MergeDecision(BaseModel):
    """Outcome of applying the tiered-threshold policy to one ScoreBreakdown."""

    mention_id_a: str
    mention_id_b: str
    tier: MergeTier
    score: ScoreBreakdown
