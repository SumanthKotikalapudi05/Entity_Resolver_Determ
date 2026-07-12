"""
Stage 3: Pairwise scoring.

For every candidate pair produced by Stage 2, compute three independent
signals and combine them into one confidence score plus a plain-English
reason string. Keeping the three signals separate (rather than one black-box
similarity number) is the whole point: a human reviewing a "flagged" merge in
Stage 4 can see exactly *why* the pipeline thinks two mentions might be the
same entity.

Weights are named constants (see WEIGHTS below) so they are easy to find and
tune -- this is called out explicitly in the build spec.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List

from rapidfuzz import fuzz

from .block import context_keywords, romanize
from .models import Mention, Script, ScoreBreakdown

# Combination weights for the three independent signals, used for the
# "general" (non-exact-name) case. Must sum to 1.0. Name similarity is
# weighted highest because it is the strongest single signal when a name is
# present at all; context and co-occurrence exist mainly to (a) disambiguate
# two different people/orgs who happen to share a similar name and (b)
# rescue name-free mentions ("the regional director") that name similarity
# alone cannot resolve. See README Design Decisions for the reasoning behind
# these specific numbers, and for why exact-name matches use a different
# formula (NAME_MATCH_FLOOR below) rather than this weighted sum.
WEIGHT_NAME_SIMILARITY = 0.60
WEIGHT_CONTEXT_OVERLAP = 0.20
WEIGHT_COOCCURRENCE_OVERLAP = 0.20

assert abs(WEIGHT_NAME_SIMILARITY + WEIGHT_CONTEXT_OVERLAP + WEIGHT_COOCCURRENCE_OVERLAP - 1.0) < 1e-9

# When two mentions have a near-exact name match (>= this threshold) AFTER
# script normalization, that equality is treated as near-conclusive evidence
# on its own -- two mentions of a full, distinctive proper name like "Sunita
# Devi Verma" or "KCD Traders Pvt Ltd" being byte-for-byte (post-
# romanization) identical is extremely unlikely to be a coincidence, even
# with zero supporting context. Below this threshold we fall back to the
# ordinary weighted-sum formula, where context/co-occurrence matter more.
# See README Design Decisions.
NAME_MATCH_FLOOR_THRESHOLD = 0.92
NAME_MATCH_FLOOR_BASE = 0.90
NAME_MATCH_FLOOR_BONUS_WEIGHT = 0.10


def _initials(roman_text: str) -> str:
    return "".join(w[0] for w in roman_text.split() if w)


def name_similarity(mention_a: Mention, mention_b: Mention) -> float:
    """
    Script-aware name similarity, normalized to [0, 1].

    Both surface forms are first romanized (Stage 2's `romanize`), so a
    Devanagari mention and a Latin mention of the same name are compared in
    the same alphabet instead of falling back to raw string distance (which
    would score two same-name mentions in different scripts as totally
    dissimilar). We then take the MAX of:
      - rapidfuzz token_sort_ratio  (robust to word order: "Sharma Rajesh"
        vs "Rajesh Sharma")
      - an initials-match bonus     (handles abbreviation: "R.K. Sharma" vs
        "Rajesh Kumar Sharma" -> initials "rks" vs "rks")
    so that either full-name similarity OR abbreviation-consistency can carry
    the score.
    """
    roman_a = romanize(mention_a.surface_form, mention_a.script)
    roman_b = romanize(mention_b.surface_form, mention_b.script)

    if not roman_a or not roman_b:
        return 0.0

    fuzzy_score = fuzz.token_sort_ratio(roman_a, roman_b) / 100.0

    initials_a, initials_b = _initials(roman_a), _initials(roman_b)
    initials_score = 0.0
    # Require both sides to have a real, multi-letter initials string --
    # a single-word surface form (e.g. "KCD", "Rani") produces a one-letter
    # "initials" string that would trivially prefix-match almost anything,
    # so we only apply the abbreviation bonus when there's more than one
    # word (and hence real ambiguity to resolve) on both sides.
    if len(initials_a) >= 2 and len(initials_b) >= 2:
        shorter, longer = sorted([initials_a, initials_b], key=len)
        if longer.startswith(shorter):
            # e.g. "rks" is a prefix of "rksharma"'s multi-letter initials
            # (R.K. Sharma vs Rajesh Kumar Sharma) -- reward abbreviation
            # consistency, but never above a fuzzy full match, since
            # initials alone are weaker evidence than a real name match.
            initials_score = 0.75 if shorter == longer else 0.6

    return max(fuzzy_score, initials_score)


def build_context_idf(mentions: List[Mention]) -> Dict[str, float]:
    """
    Precompute an IDF (inverse document frequency) weight for every keyword
    that appears across all mentions' context snippets, once per pipeline
    run.

    Why IDF and not a plain Jaccard on raw keyword sets: our context
    snippets are short, templated sentences, so generic connective words
    ("coordinated", "alongside", "note", "mentions") recur across many
    UNRELATED entities' documents purely because they share a writing
    style, not because the entities are related. A hand-curated stopword
    list can chase individual offending words forever; IDF instead
    automatically down-weights ANY word that turns out to be common across
    the corpus and up-weights genuinely distinctive ones (city names,
    product/domain terms, organization names) without us having to name
    them. See README Design Decisions.
    """
    doc_freq: Counter = Counter()
    n_mentions = len(mentions)
    for m in mentions:
        for kw in context_keywords(m.context):
            doc_freq[kw] += 1
    return {
        kw: math.log((n_mentions + 1) / (df + 1)) + 1.0
        for kw, df in doc_freq.items()
    }


def context_overlap(mention_a: Mention, mention_b: Mention, idf: Dict[str, float]) -> float:
    """IDF-weighted Jaccard similarity between the two mentions' context
    keyword sets: sum of IDF weights of shared keywords over sum of IDF
    weights of all keywords in either set. Falls back to weight 1.0 for any
    keyword not seen in `idf` (shouldn't happen if idf was built from the
    same mention pool, but keeps this function safe to call standalone).

    Two mentions whose surrounding text shares distinctive keywords
    (organization names, cities, domain terms) are more likely to refer to
    the same entity, especially useful when `name_similarity` is weak or
    zero (vague descriptions like "the regional director").
    """
    kws_a = context_keywords(mention_a.context)
    kws_b = context_keywords(mention_b.context)
    if not kws_a or not kws_b:
        return 0.0
    intersection = kws_a & kws_b
    union = kws_a | kws_b
    if not union:
        return 0.0
    weight = lambda kw: idf.get(kw, 1.0)
    num = sum(weight(kw) for kw in intersection)
    den = sum(weight(kw) for kw in union)
    return num / den if den else 0.0


def cooccurrence_overlap(
    mention_a: Mention, mention_b: Mention, doc_comention_index: Dict[str, set],
) -> float:
    """
    Jaccard similarity between the *other* named entities that co-occur in
    each mention's document. If mention A appears in a document alongside
    normalized names {"amit patel", "kcd traders"} and mention B appears in a
    different document alongside the same two names, that shared social/
    organizational context is evidence they refer to the same entity even if
    the two documents never use the same name for A/B directly.

    `doc_comention_index` maps mention_id -> set of other mentions'
    normalized_form values found in the SAME document, precomputed once per
    pipeline run for efficiency (see pipeline.py).
    """
    set_a = doc_comention_index.get(mention_a.mention_id, set())
    set_b = doc_comention_index.get(mention_b.mention_id, set())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def build_doc_comention_index(mentions: List[Mention]) -> Dict[str, set]:
    """Precompute, for every mention, the set of normalized names of the
    OTHER mentions in the same document. O(n) per document, done once."""
    by_doc: Dict[str, List[Mention]] = {}
    for m in mentions:
        by_doc.setdefault(m.doc_id, []).append(m)

    index: Dict[str, set] = {}
    for doc_id, doc_mentions in by_doc.items():
        for m in doc_mentions:
            others = {
                other.normalized_form
                for other in doc_mentions
                if other.mention_id != m.mention_id
            }
            index[m.mention_id] = others
    return index


def score_pair(
    mention_a: Mention,
    mention_b: Mention,
    doc_comention_index: Dict[str, set],
    context_idf: Dict[str, float],
) -> ScoreBreakdown:
    """
    Compute the full ScoreBreakdown for one candidate pair: three independent
    signals, a combined confidence, and a human-readable reason string built
    from whichever signal(s) actually drove the score.

    Confidence combination uses two formulas depending on name_similarity:
      - Near-exact name match (>= NAME_MATCH_FLOOR_THRESHOLD): confidence is
        anchored at NAME_MATCH_FLOOR_BASE and only nudged upward by
        secondary evidence, since two mentions with (post-normalization)
        identical distinctive names are already strong evidence on their
        own. This is what lets, e.g., "KCD Traders Pvt Ltd" in two different
        documents auto-merge even with weak or zero context/co-occurrence
        overlap -- requiring all three signals to independently clear 0.85
        would make the pipeline unable to auto-merge even the most obvious
        cases, which defeats the point of tiered thresholds.
      - Otherwise: the ordinary WEIGHT_* weighted sum, where context and
        co-occurrence carry proportionally more of the decision -- this is
        the case that matters for vague, name-free mentions and for
        abbreviation/partial matches.
    """
    ns = name_similarity(mention_a, mention_b)
    co = context_overlap(mention_a, mention_b, context_idf)
    cc = cooccurrence_overlap(mention_a, mention_b, doc_comention_index)

    if ns >= NAME_MATCH_FLOOR_THRESHOLD:
        secondary = max(co, cc)
        confidence = NAME_MATCH_FLOOR_BASE + NAME_MATCH_FLOOR_BONUS_WEIGHT * secondary
    else:
        confidence = (
            WEIGHT_NAME_SIMILARITY * ns
            + WEIGHT_CONTEXT_OVERLAP * co
            + WEIGHT_COOCCURRENCE_OVERLAP * cc
        )
    confidence = min(confidence, 1.0)

    reasons = []
    if ns >= 0.85:
        reasons.append(f"name strongly matches ({ns:.2f})")
    elif ns >= 0.5:
        reasons.append(f"name partially matches ({ns:.2f})")
    elif ns > 0:
        reasons.append(f"name weakly matches ({ns:.2f})")
    else:
        reasons.append("no name similarity")

    if co >= 0.3:
        reasons.append(f"shared context keywords ({co:.2f})")
    if cc >= 0.3:
        reasons.append(f"shared co-occurring entities ({cc:.2f})")

    reason = "; ".join(reasons)

    return ScoreBreakdown(
        mention_id_a=mention_a.mention_id,
        mention_id_b=mention_b.mention_id,
        name_similarity=round(ns, 4),
        context_overlap=round(co, 4),
        cooccurrence_overlap=round(cc, 4),
        confidence=round(confidence, 4),
        reason=reason,
    )
