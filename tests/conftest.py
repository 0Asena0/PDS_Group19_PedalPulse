"""Shared fixtures. Data tests skip when the pipeline outputs are not present."""

import pandas as pd
import pytest

from pedalpulse import config


def require(path):
    if not path.exists():
        pytest.skip(f"{path.name} not built; run the pipeline first")
    return path


@pytest.fixture(scope="session")
def calendar():
    return pd.read_parquet(require(config.PROCESSED_DIR / "calendar.parquet"))


@pytest.fixture(scope="session")
def release_dir():
    return require(config.RELEASE_DIR / "SHA256SUMS").parent
