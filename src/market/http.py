"""Tiny GET helper shared by the venue clients: retries, timeouts, JSON."""
from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "baseball-projections/0.1 (+https://github.com/jackseeburger/baseball-projections)"


def get_json(url: str, params: dict | None = None, *, retries: int = 3,
             timeout: float = 30.0, session: requests.Session | None = None):
    sess = session or requests
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} from {url}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:  # ValueError: bad JSON
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("GET %s failed (%s); retry in %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {retries} tries") from last_exc
