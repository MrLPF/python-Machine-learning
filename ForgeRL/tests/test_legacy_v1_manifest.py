from __future__ import annotations

import json
from pathlib import Path

from forge_rl.benchmarks import LEGACY_V1_ARCHIVE_SHA256


def test_legacy_v1_manifest_matches_frozen_runtime_source() -> None:
    manifest_path = Path(__file__).parents[1] / "benchmarks" / "legacy_v1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["archive_sha256"] == LEGACY_V1_ARCHIVE_SHA256
    assert manifest["files"]["forge_rl/core/predictor.py"] == (
        "b6165c4123951f5affd1c017ac70298242e3c846e42a3c25f9bbfbeaad123714"
    )
    assert manifest["published_scope"] == "benchmark-only behavioral reference"
