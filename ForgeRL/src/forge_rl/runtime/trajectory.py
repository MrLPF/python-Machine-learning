from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .experience import ExperienceItem
from .transition import TransitionBatch


def _sample(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return array.copy()


def _stack_tree(records: Sequence[Mapping[str, np.ndarray]], name: str) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError(f"{name} records must not be empty")
    keys = set(records[0])
    if not keys:
        raise ValueError(f"{name} tree must not be empty")
    if any(set(record) != keys for record in records):
        raise ValueError(f"{name} keys differ across transitions")
    return {key: np.stack([record[key] for record in records], axis=0) for key in sorted(keys)}


def _stack_optional_tree(
    records: Sequence[Mapping[str, np.ndarray] | None],
    name: str,
) -> dict[str, np.ndarray] | None:
    if all(record is None for record in records):
        return None
    if any(record is None for record in records):
        raise ValueError(f"{name} is missing from some transitions")
    return _stack_tree([record for record in records if record is not None], name)


@dataclass(frozen=True, slots=True)
class TrajectoryFragment:
    transitions: TransitionBatch
    bootstrap_obs: dict[str, np.ndarray]
    bootstrap_value: np.ndarray
    bootstrap_hidden: dict[str, np.ndarray] | None = None
    closed_by_episode: bool = False
    created_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not self.bootstrap_obs:
            raise ValueError("bootstrap_obs must not be empty")
        value = np.asarray(self.bootstrap_value, dtype=np.float32)
        if value.size != 1:
            raise ValueError("bootstrap_value must contain exactly one scalar")
        object.__setattr__(self, "bootstrap_value", value.reshape(1))
        object.__setattr__(
            self,
            "bootstrap_obs",
            {name: np.asarray(array).copy() for name, array in self.bootstrap_obs.items()},
        )
        if self.bootstrap_hidden is not None:
            object.__setattr__(
                self,
                "bootstrap_hidden",
                {
                    name: np.asarray(array).copy()
                    for name, array in self.bootstrap_hidden.items()
                },
            )
        if self.closed_by_episode and bool(self.transitions.terminated[-1]):
            object.__setattr__(self, "bootstrap_value", np.zeros(1, dtype=np.float32))

    @property
    def train_rows(self) -> int:
        return int(np.count_nonzero(self.transitions.valid_mask > 0.0))

    @property
    def policy_version_min(self) -> int:
        valid = self.transitions.policy_version[self.transitions.valid_mask > 0.0]
        return int(valid.min()) if valid.size else 0

    @property
    def policy_version_max(self) -> int:
        valid = self.transitions.policy_version[self.transitions.valid_mask > 0.0]
        return int(valid.max()) if valid.size else 0

    def to_experience_item(self) -> ExperienceItem:
        if self.train_rows <= 0:
            raise ValueError("a fragment without valid rows cannot enter the experience queue")
        actor_ids = np.unique(self.transitions.actor_id[self.transitions.valid_mask > 0.0])
        if actor_ids.size != 1:
            raise ValueError("one trajectory fragment must belong to exactly one actor")
        return ExperienceItem.create(
            self,
            train_rows=self.train_rows,
            policy_version_min=self.policy_version_min,
            policy_version_max=self.policy_version_max,
            actor_id=int(actor_ids[0]),
        )


@dataclass(slots=True)
class _TransitionRecord:
    obs: dict[str, np.ndarray]
    next_obs: dict[str, np.ndarray]
    action: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    valid_mask: float
    behavior_log_prob: float
    behavior_value: float
    policy_version: int
    episode_id: int
    step_id: int
    hidden_in: dict[str, np.ndarray] | None
    hidden_out: dict[str, np.ndarray] | None
    extras: dict[str, np.ndarray]


class TrajectoryBuilder:
    """Collects real transitions and stores bootstrap state separately.

    A fragment contains exactly the transitions supplied to :meth:`append`. The final real
    transition is never repurposed as a bootstrap-only row and therefore cannot be silently lost.
    """

    def __init__(self, *, actor_id: int, env_id: int, fragment_length: int) -> None:
        if fragment_length <= 0:
            raise ValueError("fragment_length must be positive")
        self.actor_id = int(actor_id)
        self.env_id = int(env_id)
        self.fragment_length = int(fragment_length)
        self._records: list[_TransitionRecord] = []
        self._last_identity: tuple[int, int] | None = None
        self._episode_closed = False

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def needs_flush(self) -> bool:
        return bool(
            self._records
            and (
                len(self._records) >= self.fragment_length
                or self._records[-1].terminated
                or self._records[-1].truncated
            )
        )

    def append(
        self,
        *,
        obs: Mapping[str, Any],
        next_obs: Mapping[str, Any],
        action: Mapping[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        valid_mask: float,
        behavior_log_prob: float,
        behavior_value: float,
        policy_version: int,
        episode_id: int,
        step_id: int,
        hidden_in: Mapping[str, Any] | None = None,
        hidden_out: Mapping[str, Any] | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        if self._episode_closed:
            raise RuntimeError("flush the closed episode before appending another transition")
        if len(self._records) >= self.fragment_length:
            raise RuntimeError("fragment is full; flush before appending")
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        if valid_mask < 0.0 or valid_mask > 1.0:
            raise ValueError("valid_mask must lie in [0, 1]")
        identity = (int(episode_id), int(step_id))
        if self._last_identity is not None:
            previous_episode, previous_step = self._last_identity
            valid_sequence = (
                identity[0] == previous_episode and identity[1] == previous_step + 1
            )
            if not valid_sequence:
                raise ValueError(
                    f"non-contiguous transition identity: previous={self._last_identity}, "
                    f"new={identity}"
                )
        record = _TransitionRecord(
            obs={name: _sample(value) for name, value in obs.items()},
            next_obs={name: _sample(value) for name, value in next_obs.items()},
            action={name: _sample(value) for name, value in action.items()},
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            valid_mask=float(valid_mask),
            behavior_log_prob=float(behavior_log_prob),
            behavior_value=float(behavior_value),
            policy_version=int(policy_version),
            episode_id=int(episode_id),
            step_id=int(step_id),
            hidden_in=(
                {name: _sample(value) for name, value in hidden_in.items()}
                if hidden_in is not None
                else None
            ),
            hidden_out=(
                {name: _sample(value) for name, value in hidden_out.items()}
                if hidden_out is not None
                else None
            ),
            extras={name: _sample(value) for name, value in (extras or {}).items()},
        )
        if not record.obs or not record.next_obs or not record.action:
            raise ValueError("obs, next_obs and action trees must not be empty")
        self._records.append(record)
        self._last_identity = identity
        self._episode_closed = record.terminated or record.truncated

    def flush(
        self,
        *,
        bootstrap_obs: Mapping[str, Any],
        bootstrap_value: float,
        bootstrap_hidden: Mapping[str, Any] | None = None,
    ) -> TrajectoryFragment:
        if not self._records:
            raise RuntimeError("cannot flush an empty trajectory")
        records = self._records
        transitions = TransitionBatch(
            obs=_stack_tree([record.obs for record in records], "obs"),
            next_obs=_stack_tree([record.next_obs for record in records], "next_obs"),
            action=_stack_tree([record.action for record in records], "action"),
            reward=np.asarray([record.reward for record in records], dtype=np.float32),
            terminated=np.asarray([record.terminated for record in records], dtype=np.bool_),
            truncated=np.asarray([record.truncated for record in records], dtype=np.bool_),
            valid_mask=np.asarray([record.valid_mask for record in records], dtype=np.float32),
            behavior_log_prob=np.asarray(
                [record.behavior_log_prob for record in records], dtype=np.float32
            ),
            behavior_value=np.asarray(
                [record.behavior_value for record in records], dtype=np.float32
            ),
            policy_version=np.asarray(
                [record.policy_version for record in records], dtype=np.int64
            ),
            episode_id=np.asarray([record.episode_id for record in records], dtype=np.int64),
            step_id=np.asarray([record.step_id for record in records], dtype=np.int64),
            actor_id=np.full(len(records), self.actor_id, dtype=np.int64),
            env_id=np.full(len(records), self.env_id, dtype=np.int64),
            hidden_in=_stack_optional_tree(
                [record.hidden_in for record in records], "hidden_in"
            ),
            hidden_out=_stack_optional_tree(
                [record.hidden_out for record in records], "hidden_out"
            ),
            extras=(
                _stack_tree([record.extras for record in records], "extras")
                if any(record.extras for record in records)
                else {}
            ),
        )
        closed_by_episode = bool(records[-1].terminated or records[-1].truncated)
        fragment = TrajectoryFragment(
            transitions=transitions,
            bootstrap_obs={name: _sample(value) for name, value in bootstrap_obs.items()},
            bootstrap_value=np.asarray([bootstrap_value], dtype=np.float32),
            bootstrap_hidden=(
                {name: _sample(value) for name, value in bootstrap_hidden.items()}
                if bootstrap_hidden is not None
                else None
            ),
            closed_by_episode=closed_by_episode,
        )
        self._records = []
        if closed_by_episode:
            self._last_identity = None
        self._episode_closed = False
        return fragment
