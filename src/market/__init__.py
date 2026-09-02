"""Station M — market archive (docs/architecture.md §0, §2).

Prices cannot be reconstructed later, so this package's only job is to
pull every open MLB market from the public, keyless exchange APIs (Kalshi,
Polymarket), normalize them to one schema, map game markets to MLB Stats
API `game_pk`s, and write dated snapshots that are never overwritten.

    from src.market import snapshot
    records = snapshot.collect()          # list[dict] in schema.FIELDS order
    snapshot.write(records)               # data/market/snapshots/<ts>.jsonl.gz
"""
