"""Download Citi Bike archives and record what was retrieved."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config
from .months import archive_for_month

DOWNLOAD_LOG = "downloads.json"
CHUNK = 8 * 1024 * 1024


def archive_url(archive: str) -> str:
    return f"{config.TRIP_BUCKET_URL}/{archive}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_size(url: str) -> int:
    response = requests.head(url, timeout=60)
    response.raise_for_status()
    return int(response.headers["Content-Length"])


def fetch(url: str, target: Path, expected_size: int, attempts: int = 4) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(CHUNK):
                        handle.write(block)
            if partial.stat().st_size != expected_size:
                raise OSError(f"size mismatch for {url}: {partial.stat().st_size} != {expected_size}")
            partial.replace(target)
            return
        except (requests.RequestException, OSError) as error:
            if attempt == attempts:
                raise
            print(f"retry {attempt} for {url}: {error}")
            time.sleep(10 * attempt)


def load_log() -> dict:
    path = config.RAW_DIR / DOWNLOAD_LOG
    return json.loads(path.read_text()) if path.exists() else {}


def save_log(log: dict) -> None:
    (config.RAW_DIR / DOWNLOAD_LOG).write_text(json.dumps(log, indent=2, sort_keys=True))


def download(months: list[str]) -> None:
    config.ensure_dirs()
    log = load_log()
    for archive in sorted({archive_for_month(m) for m in months}):
        url = archive_url(archive)
        target = config.RAW_DIR / archive
        size = remote_size(url)
        entry = log.get(archive)
        if target.exists() and target.stat().st_size == size and entry and entry["size_bytes"] == size:
            print(f"skip {archive} (present)")
            continue
        print(f"download {archive} ({size / 1e9:.2f} GB)")
        fetch(url, target, size)
        log[archive] = {
            "url": url,
            "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "size_bytes": size,
            "sha256": sha256_of(target),
        }
        save_log(log)
    for archive, reason in config.EXCLUDED_ARCHIVES.items():
        url = archive_url(archive)
        log.setdefault(archive, {"url": url, "size_bytes": remote_size(url), "excluded": reason})
    save_log(log)
