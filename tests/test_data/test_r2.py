import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import r2


def test_bucket_path_is_stripped_from_endpoint():
    raw = "https://abc123.r2.cloudflarestorage.com/baseball-data"
    assert r2.endpoint_url(raw) == "https://abc123.r2.cloudflarestorage.com"


def test_clean_endpoint_unchanged():
    raw = "https://abc123.r2.cloudflarestorage.com"
    assert r2.endpoint_url(raw) == raw


def test_bad_endpoint_rejected():
    with pytest.raises(ValueError):
        r2.endpoint_url("abc123.r2.cloudflarestorage.com")


def test_missing_variable_names_itself(monkeypatch):
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)
    with pytest.raises(KeyError, match="R2_ENDPOINT_URL"):
        r2.endpoint_url()
