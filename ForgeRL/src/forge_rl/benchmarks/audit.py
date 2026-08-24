from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from forge_rl.runtime.transition import TransitionBatch

TransitionIdentity = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class TransitionAudit:
    collected_valid_rows: int
    trained_valid_rows: int
    unique_collected_rows: int
    unique_trained_rows: int
    duplicate_collected_rows: int
    duplicate_trained_rows: int
    missing_rows: int
    unexpected_rows: int
    invalid_rows_in_loss: int
    useful_sample_ratio: float

    @property
    def passed(self) -> bool:
        return (
            self.duplicate_collected_rows == 0
            and self.duplicate_trained_rows == 0
            and self.missing_rows == 0
            and self.unexpected_rows == 0
            and self.invalid_rows_in_loss == 0
            and self.useful_sample_ratio >= 0.995
        )

    def to_dict(self) -> dict[str, int | float | bool]:
        return {**asdict(self), "passed": self.passed}


def identities_from_batch(
    batch: TransitionBatch,
    *,
    valid_only: bool = True,
) -> list[TransitionIdentity]:
    mask = batch.valid_mask > 0.0 if valid_only else np.ones(batch.size, dtype=np.bool_)
    rows = zip(
        batch.actor_id[mask].tolist(),
        batch.env_id[mask].tolist(),
        batch.episode_id[mask].tolist(),
        batch.step_id[mask].tolist(),
    )
    return [tuple(int(value) for value in row) for row in rows]


def audit_transition_identities(
    collected: Iterable[Sequence[int]],
    trained: Iterable[Sequence[int]],
    *,
    invalid_rows_in_loss: int = 0,
) -> TransitionAudit:
    collected_ids = [tuple(int(value) for value in row) for row in collected]
    trained_ids = [tuple(int(value) for value in row) for row in trained]
    if any(len(row) != 4 for row in [*collected_ids, *trained_ids]):
        raise ValueError("transition identities must contain actor, env, episode and step IDs")
    if invalid_rows_in_loss < 0:
        raise ValueError("invalid_rows_in_loss cannot be negative")

    collected_counts = Counter(collected_ids)
    trained_counts = Counter(trained_ids)
    duplicate_collected = sum(max(0, count - 1) for count in collected_counts.values())
    duplicate_trained = sum(max(0, count - 1) for count in trained_counts.values())
    missing = sum(max(0, count - trained_counts[identity]) for identity, count in collected_counts.items())
    # Duplicate occurrences of an expected identity are reported exclusively by
    # duplicate_trained_rows. unexpected_rows counts rows whose identity was never collected.
    unexpected = sum(
        count for identity, count in trained_counts.items() if identity not in collected_counts
    )
    matched_unique = len(set(collected_counts).intersection(trained_counts))
    useful_ratio = matched_unique / len(collected_ids) if collected_ids else 1.0

    return TransitionAudit(
        collected_valid_rows=len(collected_ids),
        trained_valid_rows=len(trained_ids),
        unique_collected_rows=len(collected_counts),
        unique_trained_rows=len(trained_counts),
        duplicate_collected_rows=duplicate_collected,
        duplicate_trained_rows=duplicate_trained,
        missing_rows=missing,
        unexpected_rows=unexpected,
        invalid_rows_in_loss=int(invalid_rows_in_loss),
        useful_sample_ratio=float(useful_ratio),
    )
