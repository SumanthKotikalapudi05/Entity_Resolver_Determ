# entity-resolver

A standalone, deterministic, multilingual entity resolution engine. It merges
mentions of the same real-world person or organization across documents,
even when the name changes script (English / Devanagari), gets abbreviated,
picks up an honorific, or disappears into a vague description ("the regional
director").

No cloud services, no paid APIs, no LLM calls by default. Everything runs
locally against a synthetic dataset with `rapidfuzz`, `indic-transliteration`,
`networkx`, and `pydantic`.

## Problem statement

Real organizations accumulate mentions of the same people and companies
scattered across documents in inconsistent forms: full names, initials,
honorifics, a different script, or no name at all. Naively treating every
distinct surface form as a distinct entity fragments your data; naively
merging anything that looks similar corrupts it by fusing different people
together. This project builds a small, fully explainable pipeline that:

1. Extracts candidate name mentions from documents.
2. Cheaply narrows down which mentions are even worth comparing.
3. Scores each candidate pair on three independent, inspectable signals.
4. Applies a tiered decision policy: confident matches auto-merge, uncertain
   matches get merged but flagged for a human to check, and weak matches are
   left alone.
5. Represents the result as a graph so merges are additive and reversible.
6. Logs every merge (and every reversal) to an append-only audit trail.

## Architecture

```
                     ┌─────────────────────┐
                     │  synthetic_documents │
                     │       .json          │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌─────────────────────┐
  Stage 1            │      extract.py      │  regex/heuristic mention
                      │  Mention extraction  │  spans + script detection
                     └──────────┬───────────┘
                                │  List[Mention]
                                ▼
                     ┌─────────────────────┐
  Stage 2            │       block.py        │  romanize + phonetic hash
                      │ Candidate blocking    │  + context keywords
                     └──────────┬───────────┘
                                │  Set[candidate pairs]  (avoids O(n²))
                                ▼
                     ┌─────────────────────┐
  Stage 3            │       score.py        │  name similarity (60%)
                      │  Pairwise scoring     │  context overlap (20%, IDF)
                      │                       │  co-occurrence overlap (20%)
                     └──────────┬───────────┘
                                │  List[ScoreBreakdown]
                                ▼
                     ┌─────────────────────┐
  Stage 4            │       merge.py         │  >=0.85  -> auto-merge
                      │ Tiered merge decision  │  0.5-0.85 -> flag + merge
                     └──────────┬───────────┘  │  <0.5    -> not merged
                                │  List[MergeDecision]
                                ▼
                     ┌─────────────────────┐
  Stage 5            │    graph_store.py      │  mentions = nodes
                      │  Graph construction    │  merges = provenance-
                     └──────────┬───────────┘  │  carrying edges
                                │
                                ▼
                     ┌─────────────────────┐
  Stage 6            │       audit.py         │  JSON log + revert_merge()
                      │      Audit log         │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  entities (connected  │
                     │  components, read-    │
                     │  time only)           │
                     └─────────────────────┘
```

`pipeline.py` wires all six stages together via `run_pipeline()`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.11+.

## Example run

```bash
python run_demo.py
```

This loads `data/synthetic_documents.json` (60 documents, 120 mentions, 15
underlying entities), runs the full pipeline, and prints every auto-merge
and flagged-review decision with its evidence, the final resolved entity
list, and a summary:

```
========================================================================
SUMMARY
========================================================================
  Documents processed        : 60
  Mentions extracted         : 120
  Candidate pairs scored     : 2247
  Auto-merges                : 170
  Flagged for review         : 60
  Kept separate (not merged) : 2017
  Entities resolved          : 28
  Entities needing review    : 14

  Audit log written to: data/audit_log.json
```

Run tests with:

```bash
pytest tests/ -v
```

## Design decisions

### Why blocking uses union-of-signals, not intersection

`block.py` puts every mention into buckets by three independent cheap keys
(romanized-name prefix, a hand-rolled phonetic hash, and context keywords),
and two mentions become a *candidate pair* if they share **any one** bucket,
not all three. This is a deliberate recall-over-precision choice at the
blocking stage: blocking's only job is to avoid comparing every mention to
every other mention (O(n²)); it is not supposed to make the merge decision.
Under-blocking (missing a true match here) is unrecoverable — Stage 3 never
even sees the pair. Over-blocking just costs a cheap Stage 3 comparison that
Stage 4 can correctly reject. So blocking is tuned to be generous.

### Why the phonetic hash is hand-written instead of a real Soundex/Metaphone

The approved dependency list doesn't include a phonetic-matching library, and
reimplementing Soundex/Metaphone from memory risks silently wrong behavior
that's hard to catch, which is worse than an intentionally simple, readable
set of substitution rules (digraph collapsing, vowel-dropping, doubled-letter
collapsing). It only has to be good enough to *shrink the candidate pool* for
Stage 3's real scoring — it is explicitly not the final similarity measure.

### Why name similarity takes the MAX of fuzzy-match and initials-match

A single distance metric can't handle both "same name, reordered/reworded"
(`token_sort_ratio` handles this well) and "same name, abbreviated"
(`R.K. Sharma` vs `Rajesh Kumar Sharma` — a fuzzy string metric scores this
low, since the strings genuinely are quite different character-for-character,
even though a person would recognize them instantly as the same name). Taking
the max of a fuzzy-string score and an initials-consistency score lets either
kind of evidence carry the match. The initials check requires **at least two
initials on both sides** — this was a real bug caught during development: an
early version compared a single-word surface form's one-letter "initials"
against anything, which trivially prefix-matched almost every other mention
and silently chained unrelated entities together into one giant blob. The
regression test `test_different_entities_should_not_merge` and the
`test_pipeline_does_not_collapse_everything_into_one_entity` end-to-end test
both guard against that class of bug recurring.

### Why context overlap is IDF-weighted instead of plain Jaccard

An early version used plain Jaccard similarity over a small set of
non-stopword keywords from each mention's context sentence. In a corpus of
short, templated business sentences, that failed badly: generic connective
words ("coordinated", "alongside", "mentions", "note") recur across *every*
entity's documents purely because they share a writing style, not because
the entities are related, and with keyword sets this small, a single shared
generic word produces a large, misleading Jaccard score. Rather than
hand-curating an ever-growing stopword list to chase each offending word,
`build_context_idf()` computes an inverse-document-frequency weight for every
keyword across the whole mention pool once per run, so common words are
automatically down-weighted and genuinely distinctive words (city names,
product/domain terms, organization names) count for more. This is a general
fix, not a patch for this specific dataset.

### Why merges are represented as edges, not node-collapse

Stage 5 keeps every mention as its own permanent graph node; a "merge" is
modeled as an edge carrying provenance (source docs, confidence, tier,
reason, timestamp, a unique `merge_id`). "Entities" are never stored — they
are the connected components of the graph, computed on demand by
`get_entities()`. Physically collapsing merged mentions into one node (a
literal reading of "merged mentions collapse into one entity node") would
destroy the information needed to undo a bad merge. With an edge-based
graph, reverting is just "remove one edge, recompute connected components" —
it can never corrupt any other merge's evidence, and if two mentions are
still connected via some *other* valid merge path, they correctly stay
together. `test_revert_one_edge_does_not_break_other_merges_in_a_chain`
covers this directly.

### Why the two merged-but-different thresholds exist, and why they're where they are

`AUTO_MERGE_THRESHOLD = 0.85` and `REVIEW_THRESHOLD = 0.50` are named
constants in `merge.py`, deliberately separated from the scoring logic in
`score.py` so they can be retuned without touching how evidence is computed.
0.85 was chosen so that only near-exact name matches (see below) or
strong multi-signal agreement cross it — auto-merging should be reserved for
cases a human would find boring to double-check. 0.50 is the floor below
which we've decided the evidence isn't even worth a human's time; anything
from 0.50–0.85 is exactly the band where a person, not a heuristic, should
make the call, and the flagged decision carries every underlying signal
(`name_similarity`, `context_overlap`, `cooccurrence_overlap`, `reason`) so
that review is fast.

### Why exact/near-exact name matches use a different formula than everything else

`score.py`'s `score_pair()` branches: if romanized name similarity is
`>= 0.92`, confidence is anchored at `0.90` and only nudged up by secondary
evidence, instead of going through the ordinary weighted sum
(`0.6 * name + 0.2 * context + 0.2 * co-occurrence`). This was a bug found
during development, not an initial design: with only the weighted sum, two
mentions of the *literal same, distinctive full name* (`"KCD Traders Pvt
Ltd"` in two unrelated documents) capped out around 0.65 confidence whenever
context/co-occurrence signal was weak — meaning the system could never
auto-merge even the most obvious cases, which defeats the entire point of
having an auto-merge tier. The fix reflects an actual belief about the
evidence: two mentions of a long, specific, low-collision-probability proper
name that are identical after script normalization are already very strong
evidence on their own; context and co-occurrence exist to matter most when
the name signal is weak, partial, or entirely absent (vague descriptions),
not to gate every case equally. `test_exact_name_match_auto_merges_even_with_no_context_overlap`
is the regression test for this.

### Default-to-no-merge is enforced, not just documented

The build spec requires the system to prefer false negatives over false
positives. Concretely, in this codebase that means:
- `decide_tier()` in `merge.py` returns `NOT_MERGED` for anything below
  `REVIEW_THRESHOLD` — there's no code path that guesses on a coin flip.
- Weak or missing evidence (e.g. an unresolved script, an empty context
  keyword set) makes every individual scoring function return `0.0`, never a
  guessed default.
- `test_different_entities_should_not_merge` and
  `test_pipeline_does_not_collapse_everything_into_one_entity` are permanent
  regression tests against the two concrete false-positive-chaining bugs
  found and fixed while building this (broken initials check; unweighted
  context overlap) — both of which caused unrelated entities to merge
  through transitive chains of borderline `FLAGGED_REVIEW` edges.

### Known limitation: the synthetic dataset's own repetition can still create borderline noise

The synthetic corpus (see `scripts/generate_synthetic_data.py`) gives each
entity a *fixed* associate entity that recurs across its documents, which is
what makes co-occurrence a meaningful signal at all (see the script's own
comments for why a random-per-document associate was tried first and
rejected). But with only 15 entities and short mentions, that fixed
relationship graph occasionally produces coincidental context/co-occurrence
overlap between two genuinely different organizations (in the shipped run,
`Bharat Steel Industries` and `Chandrashekhar Industries` share the word
"Industries" and end up sharing a co-occurring name by construction of the
associate graph). This lands at confidence ~0.60 — correctly a
`FLAGGED_REVIEW`, never an auto-merge — and is a reasonable stand-in for a
real-world failure mode (generic corporate suffixes inflating fuzzy name
similarity), not a bug to hide. It's flagged in the demo output like any
other uncertain case, evidence included, for a human to resolve either way.

## Repository layout

```
entity-resolver/
├── README.md
├── requirements.txt
├── run_demo.py
├── data/
│   └── synthetic_documents.json
├── scripts/
│   └── generate_synthetic_data.py
├── entity_resolver/
│   ├── models.py       # shared pydantic models
│   ├── extract.py       # Stage 1
│   ├── block.py          # Stage 2
│   ├── score.py           # Stage 3
│   ├── merge.py            # Stage 4
│   ├── graph_store.py       # Stage 5
│   ├── audit.py              # Stage 6
│   └── pipeline.py            # orchestrator
└── tests/
    ├── test_score_and_merge.py
    ├── test_graph_and_audit.py
    └── test_pipeline_end_to_end.py
```

## Retuning the pipeline

- Merge thresholds: `entity_resolver/merge.py` → `AUTO_MERGE_THRESHOLD`,
  `REVIEW_THRESHOLD`.
- Signal weights: `entity_resolver/score.py` → `WEIGHT_NAME_SIMILARITY`,
  `WEIGHT_CONTEXT_OVERLAP`, `WEIGHT_COOCCURRENCE_OVERLAP` (must sum to 1.0),
  plus `NAME_MATCH_FLOOR_THRESHOLD` / `NAME_MATCH_FLOOR_BASE` for the
  exact-match fast path.
- Blocking bucket cap (protects against a pathologically generic keyword
  bucket): `entity_resolver/block.py` → the `capped = ids[:50]` line in
  `candidate_pairs()`.
