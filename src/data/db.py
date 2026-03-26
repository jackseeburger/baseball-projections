"""Turso database connection helper.

Usage:
    from src.data.db import get_connection

    conn = get_connection()
    rows = conn.execute("SELECT * FROM batting_stats WHERE season = 2024").fetchall()
"""
import os
import libsql_experimental as libsql

# Default to environment variables; fall back for local dev
TURSO_DATABASE_URL = os.environ.get(
    "TURSO_DATABASE_URL",
    "libsql://baseball-projections-jseeburger4.aws-us-east-1.turso.io",
)
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")

# Local replica path (for embedded replica mode)
_LOCAL_DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "local_replica.db")


def get_connection(local_replica: bool = True):
    """Get a connection to the Turso database.

    Args:
        local_replica: If True (default), use embedded replica mode -
            reads are instant from a local SQLite file, writes sync to cloud.
            If False, connect directly to cloud (slower reads, no local file).

    Returns:
        A libsql connection object (sqlite3-compatible interface).
    """
    if not TURSO_AUTH_TOKEN:
        raise RuntimeError(
            "TURSO_AUTH_TOKEN not set. Get it with: turso db tokens create baseball-projections"
        )

    if local_replica:
        conn = libsql.connect(
            os.path.abspath(_LOCAL_DB),
            sync_url=TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
        conn.sync()
    else:
        conn = libsql.connect(
            TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
    return conn


def sync(conn):
    """Sync local replica with Turso Cloud. Call after writes."""
    conn.sync()
