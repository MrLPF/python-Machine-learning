from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

TensorTree = Mapping[str, np.ndarray]


class TransitionIdentityError(ValueError):
    """Raised when transition identities are duplicated or malformed."""


def _as_array(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _leading_size(tree: TensorTree, name: str) -> int:
    if not tree:
        raise ValueError(f"{name} must not be empty")
    sizes = {int(_as_array(value).shape[0]) for value in tree.values()}
    if len(sizes) != 1:
        raise ValueError(f"{name} tensors have inconsistent leading dimensions: {sizes}")
    return sizes.pop()


@dataclass(slots=True)
class TransitionBatch:
    """Canonical ForgeRL v2 transition payload.

    `terminated` controls value bootstrapping. `truncated` ends advantage recursion but still
    permits a value bootstrap from the final observation. `valid_mask=0` marks infrastructure
    faults or padding and must exclude the row from every loss.
    """

    obs: dict[str, np.ndarray]
    next_obs: dict[str, np.ndarray]
    action: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    valid_mask: np.ndarray
    behavior_log_prob: np.ndarray
    behavior_value: np.ndarray
    policy_version: np.ndarray
    episode_id: np.ndarray
    step_id: np.ndarray
    actor_id: np.ndarray
    env_id: np.ndarray
    hidden_in: dict[str, np.ndarray] | None = None
    hidden_out: dict[str, np.ndarray] | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.obs = {key: _as_array(value) for key, value in self.obs.items()}
        self.next_obs = {key: _as_array(value) for key, value in self.next_obs.items()}
        self.action = {key: _as_array(value) for key, value in self.action.items()}
        self.reward = _as_array(self.reward, dtype=np.float32)
        self.terminated = _as_array(self.terminated, dtype=np.bool_)
        self.truncated = _as_array(self.truncated, dtype=np.bool_)
        self.valid_mask = _as_array(self.valid_mask, dtype=np.float32)
        self.behavior_log_prob = _as_array(self.behavior_log_prob, dtype=np.float32)
        self.behavior_value = _as_array(self.behavior_value, dtype=np.float32)
        self.policy_version = _as_array(self.policy_version, dtype=np.int64)
        self.episode_id = _as_array(self.episode_id, dtype=np.int64)
        self.step_id = _as_array(self.step_id, dtype=np.int64)
        self.actor_id = _as_array(self.actor_id, dtype=np.int64)
        self.env_id = _as_array(self.env_id, dtype=np.int64)
        if self.hidden_in is not None:
            self.hidden_in = {key: _as_array(value) for key, value in self.hidden_in.items()}
        if self.hidden_out is not None:
            self.hidden_out = {key: _as_array(value) for key, value in self.hidden_out.items()}
        self.extras = {key: _as_array(value) for key, value in self.extras.items()}
        self.validate()

    @property
    def size(self) -> int:
        return int(self.reward.shape[0])

    @property
    def done(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)

    @property
    def bootstrap_mask(self) -> np.ndarray:
        """Mask for r + gamma * V(next_obs). Truncation still bootstraps."""
        return self.valid_mask * (~self.terminated).astype(np.float32)

    @property
    def trace_mask(self) -> np.ndarray:
        """Mask for recursive GAE/V-trace continuation across the next transition."""
        return self.valid_mask * (~self.done).astype(np.float32)

    def validate(self) -> None:
        size = _leading_size(self.obs, "obs")
        named_trees: list[tuple[str, TensorTree | None]] = [
            ("next_obs", self.next_obs),
            ("action", self.action),
            ("hidden_in", self.hidden_in),
            ("hidden_out", self.hidden_out),
            ("extras", self.extras or None),
        ]
        for name, tree in named_trees:
            if tree is not None and _leading_size(tree, name) != size:
                raise ValueError(f"{name} leading dimension does not match obs")

        arrays = {
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "valid_mask": self.valid_mask,
            "behavior_log_prob": self.behavior_log_prob,
            "behavior_value": self.behavior_value,
            "policy_version": self.policy_version,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "actor_id": self.actor_id,
            "env_id": self.env_id,
        }
        for name, value in arrays.items():
            if int(value.shape[0]) != size:
                raise ValueError(f"{name} leading dimension {value.shape[0]} != {size}")

        if np.any(self.valid_mask < 0.0) or np.any(self.valid_mask > 1.0):
            raise ValueError("valid_mask must lie in [0, 1]")
        if np.any(np.logical_and(self.terminated, self.truncated)):
            raise ValueError("a transition cannot be both terminated and truncated")
        if np.any(self.policy_version < 0):
            raise ValueError("policy_version must be non-negative")
        if np.any(self.step_id < 0) or np.any(self.episode_id < 0):
            raise ValueError("episode_id and step_id must be non-negative")

        identities = np.stack(
            [self.actor_id, self.env_id, self.episode_id, self.step_id], axis=1
        )
        valid_rows = self.valid_mask > 0.0
        if valid_rows.any():
            unique = np.unique(identities[valid_rows], axis=0)
            if unique.shape[0] != int(valid_rows.sum()):
                raise TransitionIdentityError(
                    "duplicate valid transition identity (actor, env, episode, step)"
                )

    def select(self, indices: np.ndarray | list[int]) -> "TransitionBatch":
        index = np.asarray(indices, dtype=np.int64)

        def take_tree(tree: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
            if tree is None:
                return None
            return {key: value[index] for key, value in tree.items()}

        return TransitionBatch(
            obs=take_tree(self.obs) or {},
            next_obs=take_tree(self.next_obs) or {},
            action=take_tree(self.action) or {},
            reward=self.reward[index],
            terminated=self.terminated[index],
            truncated=self.truncated[index],
            valid_mask=self.valid_mask[index],
            behavior_log_prob=self.behavior_log_prob[index],
            behavior_value=self.behavior_value[index],
            policy_version=self.policy_version[index],
            episode_id=self.episode_id[index],
            step_id=self.step_id[index],
            actor_id=self.actor_id[index],
            env_id=self.env_id[index],
            hidden_in=take_tree(self.hidden_in),
            hidden_out=take_tree(self.hidden_out),
            extras=take_tree(self.extras) or {},
        )
