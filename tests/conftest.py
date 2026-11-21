"""Shared fixtures. The package is installed into .venv by `uv sync`, so tests import it directly."""

import pytest

from askmoma.data.collection import Collection


@pytest.fixture(scope="session")
def moma() -> Collection:
    return Collection()
