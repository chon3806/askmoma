"""Shared test setup. Runs before any test module is imported."""

import os

# askmoma.config reads its settings when first imported, so the test date must be set here.
os.environ.setdefault("ASKMOMA_DATA_AS_OF", "November 20, 2026")
