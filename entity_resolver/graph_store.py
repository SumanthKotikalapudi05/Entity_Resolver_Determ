"""
Stage 5: Graph construction.

Design decision worth calling out explicitly: mentions are represented as
individual GRAPH NODES that stay separate forever; a "merge" is modeled as an
EDGE between two mention nodes carrying provenance (source docs, confidence,
tier, reason, timestamp, a unique merge_id). "Entities" are not a stored
object at all -- they are simply the CONNECTED COMPONENTS of this graph,
computed on demand.

Why not literally collapse merged mentions into one node (as a first reading
of "merged mentions collapse into one entity node" might suggest)? Because a
graph where merging is modeled as node-collapse is not cleanly reversible:
once two nodes are fused you have destroyed the information needed to split
them apart again if a merge turns out to be wrong. An edge-based graph gives
us that for free -- reverting a bad merge (a hard requirement of Stage 6) is
just "remove one edge and recompute connected components", which can never
corrupt any other merge decision. The external behavior -- "merged mentions
present as one entity" -- is identical either way; `get_entities()` below is
what does that collapsing, non-destructively, at read time.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

import networkx as nx

from .merge import AUTO_MERGE_THRESHOLD
from .models import Mention, MergeDecision, MergeTier


class EntityGraph:
    """Wraps a networkx.Graph of mention-level nodes and merge-decision edges."""

    def __init__(self) -> None:
        self.graph = nx.Graph()

    def add_mentions(self, mentions: List[Mention]) -> None:
        """Register every mention as a node, carrying its metadata as attributes."""
        for m in mentions:
            self.graph.add_node(
                m.mention_id,
                doc_id=m.doc_id,
                surface_form=m.surface_form,
                normalized_form=m.normalized_form,
                script=m.script.value,
                entity_type_guess=m.entity_type_guess.value,
                gold_entity_id=m.gold_entity_id,
            )

    def apply_merge_decision(self, decision: MergeDecision) -> Optional[str]:
        """
        Add one edge for a merge decision (AUTO_MERGE or FLAGGED_REVIEW only;
        NOT_MERGED decisions never touch the graph). Returns the generated
        merge_id, or None if the decision was NOT_MERGED.
        """
        if decision.tier == MergeTier.NOT_MERGED:
            return None

        merge_id = str(uuid.uuid4())
        self.graph.add_edge(
            decision.mention_id_a,
            decision.mention_id_b,
            merge_id=merge_id,
            confidence=decision.score.confidence,
            tier=decision.tier.value,
            reason=decision.score.reason,
            name_similarity=decision.score.name_similarity,
            context_overlap=decision.score.context_overlap,
            cooccurrence_overlap=decision.score.cooccurrence_overlap,
            source_doc_a=self.graph.nodes[decision.mention_id_a]["doc_id"],
            source_doc_b=self.graph.nodes[decision.mention_id_b]["doc_id"],
            timestamp=time.time(),
        )
        return merge_id

    def revert_merge(self, merge_id: str) -> bool:
        """
        Remove the edge carrying the given merge_id, undoing exactly that one
        merge. If the two mentions are still connected via some OTHER merge
        path, they remain in the same entity (correctly) -- only the direct
        evidence for merge_id is removed. Returns True if an edge was found
        and removed, False otherwise.
        """
        for u, v, data in list(self.graph.edges(data=True)):
            if data.get("merge_id") == merge_id:
                self.graph.remove_edge(u, v)
                return True
        return False

    def get_entities(self) -> List[Dict]:
        """
        Collapse the graph into entities at read time: one entity per
        connected component (isolated mentions -- no merge edges at all --
        become singleton entities). Entity ID is deterministic (derived from
        the lexicographically smallest mention_id in the component) so it
        stays stable across repeated reads as long as membership doesn't
        change.
        """
        entities = []
        for component in nx.connected_components(self.graph):
            member_ids = sorted(component)
            entity_id = f"ENTITY::{member_ids[0]}"
            surface_forms = sorted(
                {self.graph.nodes[mid]["surface_form"] for mid in member_ids}
            )
            source_docs = sorted(
                {self.graph.nodes[mid]["doc_id"] for mid in member_ids}
            )
            internal_edges = [
                {
                    "mention_a": u,
                    "mention_b": v,
                    "merge_id": data["merge_id"],
                    "confidence": data["confidence"],
                    "tier": data["tier"],
                    "reason": data["reason"],
                }
                for u, v, data in self.graph.subgraph(member_ids).edges(data=True)
            ]
            needs_review = any(e["tier"] == MergeTier.FLAGGED_REVIEW.value for e in internal_edges)
            entities.append(
                {
                    "entity_id": entity_id,
                    "mention_ids": member_ids,
                    "surface_forms": surface_forms,
                    "source_docs": source_docs,
                    "merge_edges": internal_edges,
                    "needs_review": needs_review,
                }
            )
        entities.sort(key=lambda e: e["entity_id"])
        return entities

    def entity_count(self) -> int:
        return nx.number_connected_components(self.graph)
