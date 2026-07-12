"""
Stage 2: Script normalization + candidate blocking.

Comparing every mention against every other mention is O(n^2) and mostly
wasted work -- almost all pairs obviously refer to different entities. This
stage cheaply groups mentions into "buckets" of plausible candidates so that
Stage 3 (scoring) only ever looks at pairs within the same bucket.

Three independent, cheap blocking keys are used (a mention can land in
multiple buckets):
  1. Romanized-name prefix key   -- catches same name, different script.
  2. Simplified phonetic key     -- catches transliteration spelling drift
                                     (e.g. "Chandrashekhar" vs "Chandrasekar").
  3. Context-keyword key         -- catches partial/no-name mentions like
                                     "the regional director" that share no
                                     name at all but do share context words
                                     (organization, city, role) with a named
                                     mention of the same entity.

See README "Design Decisions" for why these three specific keys were chosen.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from .models import Mention, Script

_STOPWORDS = {
    "the", "a", "an", "of", "in", "at", "on", "for", "and", "to", "is",
    "was", "with", "by", "our", "their", "his", "her", "he", "she", "mr",
    "mrs", "ms", "dr", "shri", "smt", "who", "that", "this", "regional",
    "senior", "sir", "madam",
}

_HONORIFIC_RE = re.compile(
    r"\b(mr|mrs|ms|dr|shri|smt|sir|madam)\.?\s*", re.IGNORECASE
)


def romanize(text: str, script: Script) -> str:
    """
    Convert a surface form to a normalized Roman-script string so that
    Devanagari and Latin spellings of the same name land in the same
    comparison space. Devanagari is transliterated via the ITRANS scheme;
    Latin text is passed through lowercase/stripped.
    """
    if script == Script.DEVANAGARI:
        roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    else:
        roman = text
    roman = roman.lower()
    roman = _HONORIFIC_RE.sub("", roman)
    roman = re.sub(r"[^a-z\s]", "", roman)
    roman = re.sub(r"\s+", " ", roman).strip()
    return roman


def phonetic_key(roman_text: str) -> str:
    """
    A deliberately simple, hand-written phonetic hash (not a full Soundex/
    Metaphone implementation -- we don't have a phonetic-matching library in
    our approved dependency list, and reimplementing Soundex incorrectly from
    memory would be worse than a small, explicit rule set).

    Rules, applied per word then concatenated:
      1. Collapse common transliteration-variant digraphs to one symbol
         (e.g. "sh"/"sh"-like sounds, "kh"->"k", "chandra"->"chandr").
      2. Drop vowels except a leading one (vowel spelling is the single
         biggest source of transliteration drift, e.g. "Sharma"/"Sharmaa").
      3. Collapse doubled letters.
      4. Keep only the first 6 characters per word, joined.

    This is intentionally coarse: its job is only to shrink the candidate
    pool for Stage 3's real scoring, not to be the final similarity measure.
    """
    if not roman_text:
        return ""
    words = roman_text.split()
    keys = []
    for word in words:
        w = word
        w = w.replace("kh", "k").replace("gh", "g").replace("ph", "f")
        w = w.replace("th", "t").replace("dh", "d").replace("bh", "b")
        w = w.replace("chh", "ch").replace("sh", "s")
        # keep first letter, drop remaining vowels, collapse doubles
        if not w:
            continue
        head, rest = w[0], w[1:]
        rest = re.sub(r"[aeiou]", "", rest)
        rest = re.sub(r"(.)\1+", r"\1", rest)
        key = (head + rest)[:6]
        keys.append(key)
    return "-".join(keys)


def context_keywords(context: str) -> Set[str]:
    """Extract a small set of lowercase, non-stopword tokens from a context
    snippet, used as a fallback blocking signal for name-free mentions."""
    tokens = re.findall(r"[A-Za-z\u0900-\u097F]+", context.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def build_blocks(mentions: List[Mention]) -> Dict[str, List[str]]:
    """
    Assign every mention to one or more blocking buckets and return
    {bucket_key: [mention_id, ...]}. Bucket keys are prefixed by type
    ("name:", "phon:", "ctx:") so the three signals never collide with each
    other.
    """
    buckets: Dict[str, List[str]] = defaultdict(list)
    for m in mentions:
        roman = romanize(m.surface_form, m.script)

        if roman:
            name_key = f"name:{roman[:4]}"
            buckets[name_key].append(m.mention_id)

            phon = phonetic_key(roman)
            if phon:
                buckets[f"phon:{phon}"].append(m.mention_id)

        for kw in context_keywords(m.context):
            buckets[f"ctx:{kw}"].append(m.mention_id)

    return buckets


def candidate_pairs(mentions: List[Mention]) -> Set[frozenset]:
    """
    Turn blocking buckets into a deduplicated set of candidate mention pairs
    to be scored in Stage 3. A pair only needs to co-occur in ONE bucket to
    be considered a candidate (union across signals, not intersection) --
    this favors recall, matching the pipeline-wide "don't merge when
    uncertain" policy: we would rather over-generate candidates for Stage 3
    to correctly reject than miss a true match at the blocking stage.
    """
    buckets = build_blocks(mentions)
    pairs: Set[frozenset] = set()
    for key, ids in buckets.items():
        if len(ids) < 2:
            continue
        # cap bucket size to avoid a pathological O(n^2) blowup from an
        # overly generic keyword bucket (e.g. a very common context word)
        capped = ids[:50]
        for i in range(len(capped)):
            for j in range(i + 1, len(capped)):
                if capped[i] != capped[j]:
                    pairs.add(frozenset((capped[i], capped[j])))
    return pairs
