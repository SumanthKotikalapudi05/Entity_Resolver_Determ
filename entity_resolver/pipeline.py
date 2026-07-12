"""
Orchestrates all six pipeline stages end-to-end:

    documents -> extract -> block -> score -> merge -> graph_store -> audit

`run_pipeline()` is the single entry point most callers (including
`run_demo.py` and the tests) should use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .audit import AuditLog
from .block import candidate_pairs
from .extract import extract_all_mentions
from .graph_store import EntityGraph
from .merge import decide_all
from .models import Mention, MergeDecision, MergeTier, ScoreBreakdown
from .score import build_context_idf, build_doc_comention_index, score_pair


@dataclass
class PipelineResult:
    mentions: List[Mention]
    scores: List[ScoreBreakdown]
    decisions: List[MergeDecision]
    graph: EntityGraph
    audit_log: AuditLog

    @property
    def entities(self) -> List[dict]:
        return self.graph.get_entities()


def run_pipeline(documents: List[dict]) -> PipelineResult:
    """
    Run the full entity resolution pipeline over a list of synthetic document
    dicts (the shape produced by `data/synthetic_documents.json`) and return
    a `PipelineResult` bundling every intermediate artifact, so callers can
    inspect any stage's output, not just the final entity list.
    """
    # Stage 1: extraction
    mentions = extract_all_mentions(documents)

    # Stage 2: blocking -- cheap candidate generation, avoids O(n^2)
    pairs = candidate_pairs(mentions)
    mentions_by_id = {m.mention_id: m for m in mentions}

    # Stage 3: scoring
    comention_index = build_doc_comention_index(mentions)
    context_idf = build_context_idf(mentions)
    scores: List[ScoreBreakdown] = []
    for pair in pairs:
        id_a, id_b = tuple(pair)
        scores.append(
            score_pair(
                mentions_by_id[id_a], mentions_by_id[id_b], comention_index, context_idf
            )
        )

    # Stage 4: tiered merge decisions
    decisions = decide_all(scores)

    # Stage 5 + 6: build the graph and write the audit log as merges are applied
    graph = EntityGraph()
    graph.add_mentions(mentions)
    audit_log = AuditLog()
    for decision in decisions:
        if decision.tier == MergeTier.NOT_MERGED:
            continue
        merge_id = graph.apply_merge_decision(decision)
        if merge_id is not None:
            audit_log.log_merge(decision, merge_id)

    return PipelineResult(
        mentions=mentions,
        scores=scores,
        decisions=decisions,
        graph=graph,
        audit_log=audit_log,
    )
