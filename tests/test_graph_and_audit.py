"""Tests for Stage 5 (graph_store) and Stage 6 (audit) -- specifically the
revert_merge round trip, since that's a hard requirement of the build spec."""

from entity_resolver.audit import AuditLog
from entity_resolver.extract import detect_script, normalize_surface_form
from entity_resolver.graph_store import EntityGraph
from entity_resolver.models import Mention, MergeDecision, MergeTier, ScoreBreakdown


def make_mention(mention_id, doc_id, surface_form):
    return Mention(
        mention_id=mention_id,
        doc_id=doc_id,
        surface_form=surface_form,
        normalized_form=normalize_surface_form(surface_form),
        script=detect_script(surface_form),
        context="test context",
    )


def make_decision(a_id, b_id, tier, confidence=0.9):
    score = ScoreBreakdown(
        mention_id_a=a_id, mention_id_b=b_id,
        name_similarity=confidence, context_overlap=0.0, cooccurrence_overlap=0.0,
        confidence=confidence, reason="test merge",
    )
    return MergeDecision(mention_id_a=a_id, mention_id_b=b_id, tier=tier, score=score)


def test_merge_then_revert_splits_entity_back_apart():
    mentions = [make_mention("m1", "d1", "Foo Corp"), make_mention("m2", "d2", "Foo Corp")]
    graph = EntityGraph()
    graph.add_mentions(mentions)
    audit = AuditLog()

    decision = make_decision("m1", "m2", MergeTier.AUTO_MERGE)
    merge_id = graph.apply_merge_decision(decision)
    audit.log_merge(decision, merge_id)

    assert graph.entity_count() == 1, "the two mentions should now be one entity"
    assert audit.find_merge_record(merge_id) is not None

    reverted = audit.revert_merge(merge_id, graph)
    assert reverted is True
    assert graph.entity_count() == 2, "reverting should split them back into two entities"

    # append-only log: the original merge record must still be present
    assert audit.find_merge_record(merge_id) is not None
    assert audit.records[-1]["action"] == "revert"
    assert audit.records[-1]["merge_id"] == merge_id


def test_revert_unknown_merge_id_returns_false():
    graph = EntityGraph()
    graph.add_mentions([make_mention("m1", "d1", "Foo Corp")])
    audit = AuditLog()
    assert audit.revert_merge("not-a-real-id", graph) is False


def test_revert_one_edge_does_not_break_other_merges_in_a_chain():
    """A -- B -- C chain: reverting A-B should split A off, but B and C
    should remain merged since their own edge is untouched."""
    mentions = [
        make_mention("m1", "d1", "A"),
        make_mention("m2", "d2", "B"),
        make_mention("m3", "d3", "C"),
    ]
    graph = EntityGraph()
    graph.add_mentions(mentions)
    audit = AuditLog()

    d_ab = make_decision("m1", "m2", MergeTier.AUTO_MERGE)
    merge_id_ab = graph.apply_merge_decision(d_ab)
    audit.log_merge(d_ab, merge_id_ab)

    d_bc = make_decision("m2", "m3", MergeTier.AUTO_MERGE)
    merge_id_bc = graph.apply_merge_decision(d_bc)
    audit.log_merge(d_bc, merge_id_bc)

    assert graph.entity_count() == 1

    audit.revert_merge(merge_id_ab, graph)

    entities = graph.get_entities()
    assert len(entities) == 2
    sizes = sorted(len(e["mention_ids"]) for e in entities)
    assert sizes == [1, 2]


def test_not_merged_decision_never_touches_the_graph():
    mentions = [make_mention("m1", "d1", "A"), make_mention("m2", "d2", "B")]
    graph = EntityGraph()
    graph.add_mentions(mentions)
    decision = make_decision("m1", "m2", MergeTier.NOT_MERGED, confidence=0.1)
    merge_id = graph.apply_merge_decision(decision)
    assert merge_id is None
    assert graph.entity_count() == 2
