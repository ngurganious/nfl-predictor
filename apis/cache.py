"""
Shared file-based cache with TTL.

All API modules use this to avoid redundant calls and stay inside
free-tier rate limits. Cache files are stored as JSON under cache/.
"""

import hashlib
import json
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"


def _path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def get(key: str):
    """Return cached value, or None if missing or expired."""
    p = _path(key)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            record = json.load(f)
        if time.time() > record["expires_at"]:
            p.unlink(missing_ok=True)
            return None
        return record["value"]
    except Exception:
        return None


def set(key: str, value, ttl: int = 3600):
    """Store value with a TTL in seconds."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_path(key), "w") as f:
            json.dump({"expires_at": time.time() + ttl, "value": value}, f)
    except Exception:
        pass


def clear():
    """Delete every cache file."""
    if not CACHE_DIR.exists():
        return
    for p in CACHE_DIR.glob("*.json"):
        p.unlink(missing_ok=True)
