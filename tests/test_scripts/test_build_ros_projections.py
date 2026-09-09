"""The nightly build fetches the seasons the live engines train on.

The contact and stuff engines fit walk-forward on 2017-2025 cells that the
nightly runner does not have in its checkout. The first night after BAS-72
wired the contact engine, every hitter component fell back to tuned Marcel
without a word and the committed document said so only in its `engine`
field. These tests pin the two halves of the fix: the build script asks R2
for every training season it lacks, and a fallback is never silent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts import build_ros_projections as b  # noqa: E402


def test_training_seasons_are_the_union_of_both_engines_below_the_served_one():
    from src.eval import contact, stuff

    years = b.training_pa_seasons(2026)
    assert years == tuple(sorted(
        (set(contact.LIVE_CELL_SEASONS) | set(stuff.LIVE_CELL_SEASONS)) - {2026}))
    assert 2017 in years and 2025 in years and 2026 not in years
    assert 2020 not in years


def test_ensure_downloads_only_the_missing_seasons(tmp_path):
    (tmp_path / "pa_outcomes_2024.parquet").write_bytes(b"x")
    fetched = []

    def fake_download(year, data_dir):
        fetched.append((year, Path(data_dir)))
        (Path(data_dir) / f"pa_outcomes_{year}.parquet").write_bytes(b"x")

    missing = b.ensure_training_pa_outcomes((2023, 2024, 2025), tmp_path,
                                            download=fake_download)
    assert missing == []
    assert [y for y, _ in fetched] == [2023, 2025]
    assert all(d == tmp_path for _, d in fetched)


def test_ensure_never_raises_and_reports_what_it_could_not_fetch(tmp_path, caplog):
    def broken_download(year, data_dir):
        raise OSError("no R2 here")

    with caplog.at_level("WARNING"):
        missing = b.ensure_training_pa_outcomes((2018, 2019), tmp_path,
                                                download=broken_download)
    assert missing == [2018, 2019]
    assert "could not fetch" in caplog.text
    assert "fall back" in caplog.text
