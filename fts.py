from __future__ import annotations

import re
import unicodedata

# \w keeps Vietnamese letters and digits, drops the FTS5 operators
# (- " * : ^ ( ) and bare AND/OR/NOT) that would be a syntax error in MATCH
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

FUZZY_CANDIDATES = 500   # trigram pre-filter pool before rescoring
FUZZY_MIN_SCORE = 0.6    # share of query trigrams a row must contain


def tokenize(query: str) -> list[str]:
    return _TOKEN_RE.findall(query)


def build_match(query: str) -> str | None:
    """User text -> an FTS5 MATCH expression for the exact phrase: the words adjacent and in order ("sạt lở"), or None if there is nothing to search.

    The phrase is quoted, so the raw query can hold any punctuation.
    """
    tokens = tokenize(query)
    if not tokens:
        return None
    return '"' + " ".join(tokens) + '"'


def normalize(text: str) -> str:
    """Lowercase, strip Vietnamese tones/marks (đ -> d), keep only words.
    Padded with spaces so a word's first/last letters form boundary trigrams."""
    text = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " " + " ".join(tokenize(text)) + " "


def _word_grams(word: str) -> set[str]:
    return {f" {word} "[i:i + 3] for i in range(len(word))}


def trigrams(query: str) -> list[str]:
    """Per-word trigrams (" le", "le "), so word order does not matter."""
    return sorted(set().union(*map(_word_grams, tokenize(normalize(query)))))


def build_fuzzy_match(grams: list[str]) -> str | None:
    if not grams:
        return None
    return " OR ".join(f'"{g}"' for g in grams)


def rank_fuzzy(rows: list, query: str, k: int,
               min_score: float = FUZZY_MIN_SCORE) -> list[tuple]:
    phrase = normalize(query)
    query_grams = [_word_grams(w) for w in tokenize(phrase)]
    word_cache: dict[str, set[str]] = {}
    scored = []
    for row in rows:
        text = normalize(row.text)
        row_grams = [word_cache.setdefault(w, _word_grams(w))
                     for w in set(tokenize(text))]
        score = sum(max((2 * len(q & r) / (len(q) + len(r)) for r in row_grams),
                        default=0.0)
                    for q in query_grams) / len(query_grams)
        if score >= min_score:
            scored.append((row, score, phrase in text))
    scored.sort(key=lambda r: (r[1], r[2]), reverse=True)  # stable: bm25 breaks ties
    return [(row, score) for row, score, _ in scored[:k]]
