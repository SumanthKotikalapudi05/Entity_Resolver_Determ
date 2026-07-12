"""
Unit tests for Stage 3 (scoring) and Stage 4 (merge tiering) in isolation,
using small hand-built Mention objects rather than the full synthetic
corpus, so each test is unambiguous about what it's checking.
"""

from entity_resolver.extract import detect_script, normalize_surface_form
from entity_resolver.merge import decide_tier
from entity_resolver.models import EntityType, Mention, MergeTier, Script
from entity_resolver.score import (
    build_context_idf,
    build_doc_comention_index,
    score_pair,
)


def make_mention(mention_id, doc_id, surface_form, context, script=None):
    return Mention(
        mention_id=mention_id,
        doc_id=doc_id,
        surface_form=surface_form,
        normalized_form=normalize_surface_form(surface_form),
        script=script or detect_script(surface_form),
        entity_type_guess=EntityType.UNKNOWN,
        context=context,
    )


def score(mentions, a_id, b_id):
    idf = build_context_idf(mentions)
    comention = build_doc_comention_index(mentions)
    by_id = {m.mention_id: m for m in mentions}
    return score_pair(by_id[a_id], by_id[b_id], comention, idf)


def test_same_entity_cross_script_should_merge():
    """Same person, English vs. Devanagari full name, same supporting
    context -> should score high enough to at least merge (auto or flagged),
    never NOT_MERGED."""
    mentions = [
        make_mention(
            "m1", "docA", "Rajesh Kumar Sharma",
            "Rajesh Kumar Sharma leads the North zone sales team alongside Suresh Chandra Gupta.",
        ),
        make_mention(
            "m2", "docB", "राजेश कुमार शर्मा",
            "राजेश कुमार शर्मा उत्तर क्षेत्र की बिक्री टीम का नेतृत्व करते हैं।",
        ),
    ]
    result = score(mentions, "m1", "m2")
    tier = decide_tier(result)
    assert result.name_similarity > 0.8, "romanized names should be near-identical"
    assert tier in (MergeTier.AUTO_MERGE, MergeTier.FLAGGED_REVIEW)


def test_different_entities_should_not_merge():
    """Two clearly different people with different names and unrelated
    context must NOT be merged -- this is the required 'different entity'
    edge case."""
    mentions = [
        make_mention(
            "m1", "docA", "Rajesh Kumar Sharma",
            "Rajesh Kumar Sharma leads the North zone sales team.",
        ),
        make_mention(
            "m2", "docB", "Priya Singh Chauhan",
            "Priya Singh Chauhan runs the marketing campaign for new sellers.",
        ),
    ]
    result = score(mentions, "m1", "m2")
    tier = decide_tier(result)
    assert tier == MergeTier.NOT_MERGED
    assert result.confidence < 0.5


def test_abbreviation_partial_match_is_flagged_not_auto_merged():
    """An initials/abbreviation match ('A. Patel' vs 'Amit Patel') without
    strong supporting context should land as a FLAGGED_REVIEW, not silently
    auto-merged -- this is the 'low confidence -> flagged' edge case."""
    mentions = [
        make_mention("m1", "docA", "Amit Patel", "Amit Patel runs the Gujarat franchise."),
        make_mention("m2", "docB", "A. Patel", "A. Patel requested additional inventory support."),
    ]
    result = score(mentions, "m1", "m2")
    tier = decide_tier(result)
    assert tier == MergeTier.FLAGGED_REVIEW
    assert 0.5 <= result.confidence < 0.85


def test_exact_name_match_auto_merges_even_with_no_context_overlap():
    """A distinctive full org name repeated verbatim across two otherwise
    unrelated documents should auto-merge on name evidence alone -- this is
    the regression test for the earlier bug where exact matches never
    reached the auto-merge threshold."""
    mentions = [
        make_mention("m1", "docA", "KCD Traders Pvt Ltd", "KCD Traders Pvt Ltd renewed its subscription."),
        make_mention("m2", "docB", "KCD Traders Pvt Ltd", "Invoice 4471 was raised for KCD Traders Pvt Ltd."),
    ]
    result = score(mentions, "m1", "m2")
    assert decide_tier(result) == MergeTier.AUTO_MERGE


def test_decide_tier_boundaries():
    """Sanity-check the threshold boundaries directly against the named
    constants rather than magic numbers, so this test stays correct if the
    thresholds are retuned."""
    from entity_resolver.merge import AUTO_MERGE_THRESHOLD, REVIEW_THRESHOLD
    from entity_resolver.models import ScoreBreakdown

    def sb(confidence):
        return ScoreBreakdown(
            mention_id_a="a", mention_id_b="b",
            name_similarity=confidence, context_overlap=0.0, cooccurrence_overlap=0.0,
            confidence=confidence, reason="test",
        )

    assert decide_tier(sb(AUTO_MERGE_THRESHOLD)) == MergeTier.AUTO_MERGE
    assert decide_tier(sb(AUTO_MERGE_THRESHOLD - 0.01)) == MergeTier.FLAGGED_REVIEW
    assert decide_tier(sb(REVIEW_THRESHOLD)) == MergeTier.FLAGGED_REVIEW
    assert decide_tier(sb(REVIEW_THRESHOLD - 0.01)) == MergeTier.NOT_MERGED
