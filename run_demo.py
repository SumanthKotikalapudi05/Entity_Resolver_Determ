"""
Single entry point: runs the full entity resolution pipeline over
data/synthetic_documents.json and prints a human-readable summary.

Usage:
    python run_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from entity_resolver.merge import AUTO_MERGE_THRESHOLD, REVIEW_THRESHOLD
from entity_resolver.models import MergeTier
from entity_resolver.pipeline import run_pipeline

DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_documents.json"
AUDIT_LOG_PATH = Path(__file__).resolve().parent / "data" / "audit_log.json"


def main() -> None:
    documents = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(documents)} synthetic documents.\n")

    result = run_pipeline(documents)
    entities = result.entities

    auto_merges = [d for d in result.decisions if d.tier == MergeTier.AUTO_MERGE]
    flagged = [d for d in result.decisions if d.tier == MergeTier.FLAGGED_REVIEW]
    not_merged = [d for d in result.decisions if d.tier == MergeTier.NOT_MERGED]

    print("=" * 72)
    print("PIPELINE CONFIGURATION")
    print("=" * 72)
    print(f"  AUTO_MERGE_THRESHOLD = {AUTO_MERGE_THRESHOLD}")
    print(f"  REVIEW_THRESHOLD     = {REVIEW_THRESHOLD}")
    print()

    print("=" * 72)
    print(f"AUTO-MERGES ({len(auto_merges)})")
    print("=" * 72)
    for d in sorted(auto_merges, key=lambda x: -x.score.confidence):
        a = result.graph.graph.nodes[d.mention_id_a]["surface_form"]
        b = result.graph.graph.nodes[d.mention_id_b]["surface_form"]
        print(f"  [{d.score.confidence:.2f}] '{a}'  <->  '{b}'")
        print(f"         reason: {d.score.reason}")

    print()
    print("=" * 72)
    print(f"FLAGGED FOR HUMAN REVIEW ({len(flagged)})")
    print("=" * 72)
    for d in sorted(flagged, key=lambda x: -x.score.confidence):
        a = result.graph.graph.nodes[d.mention_id_a]["surface_form"]
        b = result.graph.graph.nodes[d.mention_id_b]["surface_form"]
        print(f"  [{d.score.confidence:.2f}] '{a}'  <->  '{b}'")
        print(f"         reason: {d.score.reason}")
        print(
            f"         evidence: name_sim={d.score.name_similarity:.2f} "
            f"context={d.score.context_overlap:.2f} "
            f"cooccurrence={d.score.cooccurrence_overlap:.2f}"
        )

    print()
    print("=" * 72)
    print("RESOLVED ENTITIES")
    print("=" * 72)
    for entity in entities:
        flag = "  [NEEDS REVIEW]" if entity["needs_review"] else ""
        print(f"  {entity['entity_id']}{flag}")
        print(f"    surface forms : {entity['surface_forms']}")
        print(f"    source docs   : {entity['source_docs']}")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Documents processed        : {len(documents)}")
    print(f"  Mentions extracted         : {len(result.mentions)}")
    print(f"  Candidate pairs scored     : {len(result.scores)}")
    print(f"  Auto-merges                : {len(auto_merges)}")
    print(f"  Flagged for review         : {len(flagged)}")
    print(f"  Kept separate (not merged) : {len(not_merged)}")
    print(f"  Entities resolved          : {len(entities)}")
    print(f"  Entities needing review    : {sum(1 for e in entities if e['needs_review'])}")

    result.audit_log.save(str(AUDIT_LOG_PATH))
    print(f"\n  Audit log written to: {AUDIT_LOG_PATH}")


if __name__ == "__main__":
    main()
