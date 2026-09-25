from __future__ import annotations

import sqlite3
from fts import normalize
from .connection import transaction

def fuzzy_is_stale(conn: sqlite3.Connection, table: str, fuzzy_table: str) -> bool:
    """True when fuzzy_table is missing or no longer mirrors table's rowids"""
    sig = "SELECT COUNT(*), MAX(rowid) FROM {}"
    try:
        return (tuple(conn.execute(sig.format(table)).fetchone())
                != tuple(conn.execute(sig.format(fuzzy_table)).fetchone()))
    except sqlite3.OperationalError:  # table not created yet
        return True


def rebuild_fuzzy(conn: sqlite3.Connection, table: str, fuzzy_table: str) -> None:
    """Refill fuzzy_table with table's accent-stripped text, same rowids"""
    conn.create_function("vn_norm", 1, normalize, deterministic=True)
    with transaction(conn):
        conn.execute(f"DELETE FROM {fuzzy_table}")
        conn.execute(f"INSERT INTO {fuzzy_table}(rowid, text) "
                     f"SELECT rowid, vn_norm(text) FROM {table}")
