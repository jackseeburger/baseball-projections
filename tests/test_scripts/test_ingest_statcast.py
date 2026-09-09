"""Incremental Statcast ingest: resume date, merge/dedup, and the full-rebuild
escape hatches (`--since`, `--full`). No network — R2 is a fake S3 client and
Savant is a fake session returning fixed rows per window."""
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "ingest_statcast", ROOT / "scripts/ingest_statcast.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load()


def _season_frame(dates, key_start=0):
    """A pitch-level frame with `game_date` and the merge keys populated."""
    n = len(dates)
    return pd.DataFrame({
        "game_pk": [key_start + i for i in range(n)],
        "at_bat_number": [1] * n,
        "pitch_number": [1] * n,
        "game_date": dates,
    })


class FakeS3:
    """Stands in for the R2 client: local files keyed by (bucket, key)."""

    def __init__(self, objects: dict[str, Path]):
        self.objects = objects          # key -> local path
        self.downloaded = []
        self.uploaded = []

    def download_file(self, bucket, key, path):
        self.downloaded.append(key)
        if key not in self.objects:
            raise FileNotFoundError(f"no such key: {key}")
        import shutil
        shutil.copy(self.objects[key], path)

    def upload_file(self, path, bucket, key):
        self.uploaded.append(key)
        self.objects[key] = Path(path)


# ── resolve_resume_since ────────────────────────────────────────────────

def test_resume_since_is_max_game_date_minus_overlap(tmp_path):
    existing = tmp_path / "existing.parquet"
    _season_frame(["2026-06-01", "2026-06-02", "2026-06-05"]).to_parquet(existing, index=False)
    s3 = FakeS3({"statcast/statcast_2026.parquet": existing})

    since, existing_path = ingest.resolve_resume_since(2026, tmp_path / "work", s3=s3)

    assert since == date(2026, 6, 5) - pd_timedelta(ingest.RESUME_OVERLAP_DAYS)
    assert existing_path is not None and existing_path.exists()


def pd_timedelta(days):
    from datetime import timedelta
    return timedelta(days=days)


def test_missing_object_falls_back_to_march_1(tmp_path):
    s3 = FakeS3({})  # nothing in R2

    since, existing_path = ingest.resolve_resume_since(2026, tmp_path / "work", s3=s3)

    assert since == date(2026, 3, 1)
    assert existing_path is None


# ── merge_with_existing ─────────────────────────────────────────────────

def test_merge_dedups_overlap_and_keeps_whole_season(tmp_path):
    existing_path = tmp_path / "existing.parquet"
    fetched_path = tmp_path / "fetched.parquet"
    out_path = tmp_path / "out.parquet"

    # Existing season file: March through the overlap window (June 2-5).
    existing = _season_frame(
        ["2026-03-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"], key_start=0)
    existing.to_parquet(existing_path, index=False)

    # Freshly fetched tail re-covers June 2-5 (the overlap) plus new June 6-7.
    fetched = _season_frame(
        ["2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-06", "2026-06-07"],
        key_start=1)  # same game_pk values as the overlapping rows in `existing`
    fetched.to_parquet(fetched_path, index=False)

    out = ingest.merge_with_existing(existing_path, fetched_path, out_path)
    merged = pd.read_parquet(out)

    # March 1 (only in existing) + the 6 fetched dates = 7 rows, not 5 + 6 = 11.
    assert len(merged) == 7
    assert sorted(merged["game_date"]) == [
        "2026-03-01", "2026-06-02", "2026-06-03", "2026-06-04",
        "2026-06-05", "2026-06-06", "2026-06-07",
    ]
    # sorted by game_date
    assert list(merged["game_date"]) == sorted(merged["game_date"])


def test_merge_keeps_fetched_copy_on_overlap(tmp_path):
    """The freshly fetched row wins over the stale existing one for the same key."""
    existing_path = tmp_path / "existing.parquet"
    fetched_path = tmp_path / "fetched.parquet"
    out_path = tmp_path / "out.parquet"

    existing = pd.DataFrame({"game_pk": [1], "at_bat_number": [1], "pitch_number": [1],
                             "game_date": ["2026-06-02"], "estimated_woba": [0.300]})
    existing.to_parquet(existing_path, index=False)
    fetched = pd.DataFrame({"game_pk": [1], "at_bat_number": [1], "pitch_number": [1],
                            "game_date": ["2026-06-02"], "estimated_woba": [0.410]})
    fetched.to_parquet(fetched_path, index=False)

    out = ingest.merge_with_existing(existing_path, fetched_path, out_path)
    merged = pd.read_parquet(out)

    assert len(merged) == 1
    assert merged["estimated_woba"].iloc[0] == pytest.approx(0.410)


# ── main(): explicit --since / --full bypass R2 ─────────────────────────

def test_explicit_since_skips_r2_resume(monkeypatch, tmp_path):
    calls = {"resolve": 0, "fetch": []}

    def fake_resolve(season, raw_dir, s3=None):
        calls["resolve"] += 1
        return date(season, 3, 1), None

    def fake_fetch(season, start, end, raw_dir, chunk_days):
        calls["fetch"].append(start)
        raw_dir.mkdir(parents=True, exist_ok=True)
        out = raw_dir / f"statcast_{season}.parquet"
        _season_frame(["2026-08-01"]).to_parquet(out, index=False)
        return out

    def fake_process_year(season, data_dir):
        return pd.DataFrame({
            "game_pk": [1], "is_k": [0], "is_bb": [0], "is_hr": [0],
        })

    monkeypatch.setattr(ingest, "resolve_resume_since", fake_resolve)
    monkeypatch.setattr(ingest, "fetch_to_parquet", fake_fetch)
    monkeypatch.setattr(ingest, "process_year", fake_process_year)

    argv = ["ingest_statcast.py", "--season", "2026", "--since", "2026-08-01",
            "--work-dir", str(tmp_path), "--pa-dir", str(tmp_path / "pa"), "--no-upload"]
    monkeypatch.setattr(sys, "argv", argv)

    ingest.main()

    assert calls["resolve"] == 0, "--since must not touch R2 at all"
    assert calls["fetch"] == [date(2026, 8, 1)]


def test_full_flag_skips_r2_resume(monkeypatch, tmp_path):
    calls = {"resolve": 0, "fetch": []}

    def fake_resolve(season, raw_dir, s3=None):
        calls["resolve"] += 1
        return date(season, 3, 1), None

    def fake_fetch(season, start, end, raw_dir, chunk_days):
        calls["fetch"].append(start)
        raw_dir.mkdir(parents=True, exist_ok=True)
        out = raw_dir / f"statcast_{season}.parquet"
        _season_frame(["2026-03-01"]).to_parquet(out, index=False)
        return out

    def fake_process_year(season, data_dir):
        return pd.DataFrame({"game_pk": [1], "is_k": [0], "is_bb": [0], "is_hr": [0]})

    monkeypatch.setattr(ingest, "resolve_resume_since", fake_resolve)
    monkeypatch.setattr(ingest, "fetch_to_parquet", fake_fetch)
    monkeypatch.setattr(ingest, "process_year", fake_process_year)

    argv = ["ingest_statcast.py", "--season", "2026", "--full",
            "--work-dir", str(tmp_path), "--pa-dir", str(tmp_path / "pa"), "--no-upload"]
    monkeypatch.setattr(sys, "argv", argv)

    ingest.main()

    assert calls["resolve"] == 0, "--full must not touch R2's resume logic"
    assert calls["fetch"] == [None], "--full with no --since fetches the default March-1 window"


def test_default_mode_resumes_and_merges(monkeypatch, tmp_path):
    """No --since / --full: reads R2's resume date and merges the fetched tail."""
    calls = {"resolve": [], "fetch": [], "merge": []}

    def fake_resolve(season, raw_dir, s3=None):
        existing_path = raw_dir / "existing.parquet"
        raw_dir.mkdir(parents=True, exist_ok=True)
        _season_frame(["2026-03-01", "2026-06-02"]).to_parquet(existing_path, index=False)
        calls["resolve"].append(season)
        return date(2026, 6, 2), existing_path

    def fake_fetch(season, start, end, raw_dir, chunk_days):
        calls["fetch"].append(start)
        out = raw_dir / f"statcast_{season}_fetched.parquet"
        _season_frame(["2026-06-02", "2026-06-03"], key_start=5).to_parquet(out, index=False)
        return out

    def fake_merge(existing_path, fetched_path, out_path):
        calls["merge"].append((existing_path, fetched_path))
        merged = pd.concat([pd.read_parquet(existing_path), pd.read_parquet(fetched_path)])
        merged = merged.drop_duplicates(subset=["game_pk"]).sort_values("game_date")
        merged.to_parquet(out_path, index=False)
        return out_path

    def fake_process_year(season, data_dir):
        return pd.DataFrame({"game_pk": [1], "is_k": [0], "is_bb": [0], "is_hr": [0]})

    monkeypatch.setattr(ingest, "resolve_resume_since", fake_resolve)
    monkeypatch.setattr(ingest, "fetch_to_parquet", fake_fetch)
    monkeypatch.setattr(ingest, "merge_with_existing", fake_merge)
    monkeypatch.setattr(ingest, "process_year", fake_process_year)

    argv = ["ingest_statcast.py", "--season", "2026", "--work-dir", str(tmp_path), "--pa-dir", str(tmp_path / "pa"), "--no-upload"]
    monkeypatch.setattr(sys, "argv", argv)

    ingest.main()

    assert calls["resolve"] == [2026]
    assert calls["fetch"] == [date(2026, 6, 2)]
    assert len(calls["merge"]) == 1
