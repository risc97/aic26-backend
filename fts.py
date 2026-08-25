from __future__ import annotations

import re

# \w keeps Vietnamese letters and digits, drops the FTS5 operators
# (- " * : ^ ( ) and bare AND/OR/NOT) that would be a syntax error in MATCH
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(query: str) -> list[str]:
    return _TOKEN_RE.findall(query)


def build_match(query: str, phrase: bool = False) -> str | None:
    """User text -> an FTS5 MATCH expression, or None if there is nothing to search.

    Every token is quoted, so the raw query can hold any punctuation.
    phrase=True keeps the words adjacent and in order ("sạt lở"); the default
    requires all of them anywhere in the same row.
    """
    tokens = tokenize(query)
    if not tokens:
        return None
    if phrase:
        return '"' + " ".join(tokens) + '"'
    return " AND ".join(f'"{t}"' for t in tokens)
