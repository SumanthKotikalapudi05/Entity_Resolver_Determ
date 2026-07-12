"""
Stage 4: Tiered merge decision.

Turns a `ScoreBreakdown` (Stage 3 output) into a `MergeDecision` using three
named, easily-tunable confidence thresholds.

Policy (from the build spec, intentionally biased toward false negatives
over false positives -- see README "Design Decisions"):
  confidence >= AUTO_MERGE_THRESHOLD   -> auto-merge, no review needed
  REVIEW_THRESHOLD <= confidence < AUTO_MERGE_THRESHOLD
                                        -> merge, but flag for human review
  confidence <  REVIEW_THRESHOLD       -> keep separate
"""

from __future__ import annotations

from typing import List

from .models import MergeDecision, MergeTier, ScoreBreakdown

# Named, tunable constants -- change these two numbers to retune the whole
# pipeline's precision/recall tradeoff without touching any other file.
AUTO_MERGE_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.50


def decide_tier(score: ScoreBreakdown) -> MergeTier:
    """Map one confidence score to a MergeTier using the named thresholds.

    NOT_MERGED is the default outcome for anything below REVIEW_THRESHOLD --
    this is the "default to NOT merging when uncertain" requirement made
    concrete: the code path for "we're not sure" is the same as "no evidence
    at all", not a coin flip.
    """
    if score.confidence >= AUTO_MERGE_THRESHOLD:
        return MergeTier.AUTO_MERGE
    if score.confidence >= REVIEW_THRESHOLD:
        return MergeTier.FLAGGED_REVIEW
    return MergeTier.NOT_MERGED


def decide_all(scores: List[ScoreBreakdown]) -> List[MergeDecision]:
    """Apply `decide_tier` to every scored candidate pair."""
    decisions: List[MergeDecision] = []
    for score in scores:
        tier = decide_tier(score)
        decisions.append(
            MergeDecision(
                mention_id_a=score.mention_id_a,
                mention_id_b=score.mention_id_b,
                tier=tier,
                score=score,
            )
        )
    return decisions
