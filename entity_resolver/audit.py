"""
Stage 6: Audit log.

Every merge decision that actually touches the graph (AUTO_MERGE or
FLAGGED_REVIEW; NOT_MERGED is not logged since nothing happened) is recorded
as one JSON-serializable record: entity/mention IDs involved, full score
breakdown, reason string, tier, and a timestamp. `revert_merge()` undoes a
specific merge by merge_id, removing the corresponding edge from the
EntityGraph and appending a matching "revert" record -- the log is
append-only, so a revert never deletes history, it adds to it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from .graph_store import EntityGraph
from .models import MergeDecision


class AuditLog:
    """In-memory audit log with JSON load/save and revert support."""

    def __init__(self) -> None:
        self.records: List[dict] = []

    def log_merge(self, decision: MergeDecision, merge_id: str) -> None:
        """Append one merge record. Called right after `EntityGraph.apply_merge_decision`."""
        self.records.append(
            {
                "action": "merge",
                "merge_id": merge_id,
                "mention_id_a": decision.mention_id_a,
                "mention_id_b": decision.mention_id_b,
                "tier": decision.tier.value,
                "score": decision.score.model_dump(),
                "timestamp": time.time(),
            }
        )

    def revert_merge(self, merge_id: str, graph: EntityGraph) -> bool:
        """
        Undo the merge identified by `merge_id`: remove its edge from `graph`
        and append a "revert" record referencing the original merge_id. The
        original "merge" record is left untouched (append-only log), so the
        full history -- merge, then revert -- remains inspectable.

        Returns True if the merge was found and reverted, False if no such
        merge_id exists in the graph (e.g. it was already reverted).
        """
        removed = graph.revert_merge(merge_id)
        if not removed:
            return False
        self.records.append(
            {
                "action": "revert",
                "merge_id": merge_id,
                "timestamp": time.time(),
            }
        )
        return True

    def find_merge_record(self, merge_id: str) -> Optional[dict]:
        for record in self.records:
            if record.get("merge_id") == merge_id and record["action"] == "merge":
                return record
        return None

    def save(self, path: str) -> None:
        """Write the full audit log to a JSON file."""
        Path(path).write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        """Replace in-memory records with the contents of a JSON log file."""
        self.records = json.loads(Path(path).read_text(encoding="utf-8"))
