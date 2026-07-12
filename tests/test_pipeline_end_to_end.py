"""End-to-end smoke tests: run the full pipeline against the actual
synthetic dataset shipped in data/synthetic_documents.json."""

from entity_resolver.models import MergeTier
from entity_resolver.pipeline import run_pipeline


def test_pipeline_runs_and_produces_entities(synthetic_documents):
    result = run_pipeline(synthetic_documents)
    assert len(result.mentions) > 0
    entities = result.entities
    assert len(entities) > 0
    # every mention must end up in exactly one entity
    all_member_ids = [mid for e in entities for mid in e["mention_ids"]]
    assert len(all_member_ids) == len(set(all_member_ids)) == len(result.mentions)


def test_pipeline_does_not_collapse_everything_into_one_entity(synthetic_documents):
    """Regression test for the earlier bug where noisy context/co-occurrence
    signals chained unrelated entities together into a single giant
    component. The corpus has 15 underlying entities, so a healthy result
    should have well more than one resolved entity."""
    result = run_pipeline(synthetic_documents)
    assert len(result.entities) >= 10


def test_auto_merges_only_use_high_confidence_pairs(synthetic_documents):
    result = run_pipeline(synthetic_documents)
    for decision in result.decisions:
        if decision.tier == MergeTier.AUTO_MERGE:
            assert decision.score.confidence >= 0.85


def test_flagged_pairs_carry_full_evidence(synthetic_documents):
    """Every flagged-for-review decision must expose enough evidence for a
    human reviewer to make a call, per the build spec."""
    result = run_pipeline(synthetic_documents)
    flagged = [d for d in result.decisions if d.tier == MergeTier.FLAGGED_REVIEW]
    assert len(flagged) > 0
    for decision in flagged:
        assert decision.score.reason
        assert 0.5 <= decision.score.confidence < 0.85


def test_audit_log_has_one_record_per_graph_edge(synthetic_documents):
    result = run_pipeline(synthetic_documents)
    merge_records = [r for r in result.audit_log.records if r["action"] == "merge"]
    assert len(merge_records) == result.graph.graph.number_of_edges()
