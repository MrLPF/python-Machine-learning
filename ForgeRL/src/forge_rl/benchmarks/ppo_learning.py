from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import os
import platform
import statistics
import sys
import time
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from forge_rl.runtime import (
    DoubleBufferedPolicyReplica,
    NodeLocalInferenceService,
    PolicyRegistry,
    SharedInferenceClient,
    SharedInferenceEndpoint,
)

from .audit import audit_transition_identities
from .legacy_v1 import (
    LegacyV1InferenceRuntime,
    ReferencePPOPolicy,
    run_synthetic_policy,
)

Subject = Literal["v1", "v2"]


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    name: str
    target_return: float
    return_floor: float
    action_kind: Literal["discrete", "continuous"]
    formal_max_updates: int
    formal_eval_interval: int


ENVIRONMENT_SPECS: dict[str, EnvironmentSpec] = {
    "CartPole-v1": EnvironmentSpec(
        name="CartPole-v1",
        target_return=475.0,
        return_floor=0.0,
        action_kind="discrete",
        formal_max_updates=250,
        formal_eval_interval=5,
    ),
    "Pendulum-v1": EnvironmentSpec(
        name="Pendulum-v1",
        target_return=-200.0,
        return_floor=-2000.0,
        action_kind="continuous",
        formal_max_updates=500,
        formal_eval_interval=10,
    ),
}


@dataclass(frozen=True, slots=True)
class PPOHyperparameters:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    continuous_log_std: float = -0.5


@dataclass(frozen=True, slots=True)
class LearningBenchmarkConfig:
    environments: tuple[str, ...] = ("CartPole-v1", "Pendulum-v1")
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    actors: int = 4
    envs_per_actor: int = 4
    rollout_steps: int = 128
    hidden_size: int = 64
    eval_episodes: int = 10
    max_batch_items: int = 128
    v2_min_batch_items: int = 32
    legacy_min_batch_items: int = 1
    max_wait_ms: float = 2.0
    request_slots: int = 8
    device: str = "cpu"
    smoke: bool = False
    cartpole_max_updates: int | None = None
    pendulum_max_updates: int | None = None
    cartpole_eval_interval: int | None = None
    pendulum_eval_interval: int | None = None
    ppo: PPOHyperparameters = PPOHyperparameters()

    def validate(self) -> None:
        if not self.environments:
            raise ValueError("at least one environment is required")
        unknown = set(self.environments) - set(ENVIRONMENT_SPECS)
        if unknown:
            raise ValueError(f"unsupported environments: {sorted(unknown)}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        positive = [
            self.actors,
            self.envs_per_actor,
            self.rollout_steps,
            self.hidden_size,
            self.eval_episodes,
            self.max_batch_items,
            self.request_slots,
            self.ppo.epochs,
            self.ppo.minibatch_size,
        ]
        if any(value <= 0 for value in positive):
            raise ValueError("benchmark dimensions and PPO counts must be positive")
        if self.envs_per_actor > self.max_batch_items:
            raise ValueError("one actor request exceeds max_batch_items")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA learning benchmark requested but CUDA is unavailable")

    @classmethod
    def smoke_config(
        cls,
        *,
        seeds: Sequence[int] = (7,),
        environments: Sequence[str] = ("CartPole-v1", "Pendulum-v1"),
        device: str = "cpu",
    ) -> "LearningBenchmarkConfig":
        return cls(
            environments=tuple(environments),
            seeds=tuple(int(seed) for seed in seeds),
            actors=1,
            envs_per_actor=2,
            rollout_steps=16,
            hidden_size=32,
            eval_episodes=1,
            max_batch_items=4,
            v2_min_batch_items=2,
            legacy_min_batch_items=1,
            max_wait_ms=1.0,
            request_slots=4,
            device=device,
            smoke=True,
            cartpole_max_updates=2,
            pendulum_max_updates=2,
            cartpole_eval_interval=1,
            pendulum_eval_interval=1,
            ppo=PPOHyperparameters(epochs=1, minibatch_size=32),
        )


@dataclass(frozen=True, slots=True)
class EvaluationPoint:
    environment_steps: int
    wall_seconds: float
    mean_return: float


@dataclass(frozen=True, slots=True)
class LearningRun:
    environment: str
    subject: Subject
    seed: int
    target_return: float
    target_reached: bool
    time_to_target_seconds: float | None
    elapsed_seconds: float
    environment_steps: int
    collected_rows: int
    trained_rows: int
    useful_sample_ratio: float
    missing_rows: int
    duplicate_rows: int
    unexpected_rows: int
    invalid_rows_in_loss: int
    runtime_errors: int
    policy_versions: tuple[int, ...]
    normalized_auc: float
    evaluations: tuple[EvaluationPoint, ...]
    runtime_metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["policy_versions"] = list(self.policy_versions)
        result["evaluations"] = [asdict(point) for point in self.evaluations]
        return result


@dataclass(frozen=True, slots=True)
class LearningBenchmarkReport:
    config: LearningBenchmarkConfig
    runs: tuple[LearningRun, ...]
    environments: tuple[dict[str, Any], ...]
    formal_eligible: bool
    learning_gate_passed: bool
    status: str
    environment_fingerprint: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "config": {
                **asdict(self.config),
                "ppo": asdict(self.config.ppo),
            },
            "runs": [run.to_dict() for run in self.runs],
            "environments": [
                {
                    "environment": row["environment"],
                    "seeds": row["seeds"],
                    "v1_time_to_target_median_seconds": row[
                        "v1_time_to_target_median_seconds"
                    ],
                    "v2_time_to_target_median_seconds": row[
                        "v2_time_to_target_median_seconds"
                    ],
                    "v1_normalized_auc": row["v1_normalized_auc"],
                    "v2_normalized_auc": row["v2_normalized_auc"],
                }
                for row in self.environments
            ],
            "environment_details": list(self.environments),
            "formal_eligible": self.formal_eligible,
            "learning_gate_passed": self.learning_gate_passed,
            "status": self.status,
            "environment_fingerprint": self.environment_fingerprint,
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = os.fspath(path)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE with separate bootstrap and recursive-continuation masks."""

    reward = np.asarray(rewards, dtype=np.float32)
    value = np.asarray(values, dtype=np.float32)
    next_value = np.asarray(next_values, dtype=np.float32)
    terminal = np.asarray(terminated, dtype=np.bool_)
    truncation = np.asarray(truncated, dtype=np.bool_)
    if not (
        reward.shape
        == value.shape
        == next_value.shape
        == terminal.shape
        == truncation.shape
    ):
        raise ValueError("GAE arrays must have identical shapes")
    if reward.ndim != 2:
        raise ValueError("GAE arrays must have shape [time, environments]")
    advantages = np.zeros_like(reward, dtype=np.float32)
    carry = np.zeros(reward.shape[1], dtype=np.float32)
    for index in range(reward.shape[0] - 1, -1, -1):
        bootstrap = (~terminal[index]).astype(np.float32)
        continuation = (~(terminal[index] | truncation[index])).astype(np.float32)
        delta = reward[index] + gamma * bootstrap * next_value[index] - value[index]
        carry = delta + gamma * gae_lambda * continuation * carry
        advantages[index] = carry
    return advantages, advantages + value


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _sample_discrete(
    logits: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = _softmax(np.asarray(logits, dtype=np.float64))
    uniforms = rng.random(probabilities.shape[0])
    actions = np.asarray(
        [
            np.searchsorted(np.cumsum(row), value, side="right")
            for row, value in zip(probabilities, uniforms, strict=True)
        ],
        dtype=np.int64,
    )
    actions = np.minimum(actions, probabilities.shape[1] - 1)
    log_probability = np.log(
        probabilities[np.arange(probabilities.shape[0]), actions] + 1e-12
    ).astype(np.float32)
    return actions, actions[:, None].astype(np.float32), log_probability


def _squashed_normal_log_prob(
    raw_action: np.ndarray,
    mean: np.ndarray,
    log_std: float,
    action_scale: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float64)
    location = np.asarray(mean, dtype=np.float64)
    scale = math.exp(float(log_std))
    gaussian = -0.5 * (((raw - location) / scale) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
    correction = np.log(1.0 - np.tanh(raw) ** 2 + 1e-6)
    action_scale_log = np.log(np.asarray(action_scale, dtype=np.float64))
    return (gaussian - correction - action_scale_log).sum(axis=-1).astype(np.float32)


def _sample_continuous(
    mean: np.ndarray,
    rng: np.random.Generator,
    *,
    log_std: float,
    action_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(mean, dtype=np.float32) + math.exp(log_std) * rng.standard_normal(
        np.asarray(mean).shape, dtype=np.float32
    )
    action = np.tanh(raw) * action_scale
    log_probability = _squashed_normal_log_prob(raw, mean, log_std, action_scale)
    return action.astype(np.float32), raw.astype(np.float32), log_probability


class _LegacyLearningRuntime:
    def __init__(
        self,
        *,
        actor_count: int,
        observation_size: int,
        action_size: int,
        hidden_size: int,
        state_dict: Mapping[str, torch.Tensor],
        config: LearningBenchmarkConfig,
    ) -> None:
        self.runtime = LegacyV1InferenceRuntime(
            actor_count=actor_count,
            width=observation_size,
            output_size=action_size,
            hidden_size=hidden_size,
            model_kind="reference_ppo",
            state_dict=state_dict,
            policy_version=0,
            max_batch_items=config.max_batch_items,
            min_batch_items=config.legacy_min_batch_items,
            max_wait_ms=config.max_wait_ms,
            device=config.device,
        )
        self._sequences = [0] * actor_count
        self._request_count = 0
        self.runtime.start()

    def infer(
        self, actor_id: int, observation: np.ndarray, *, min_policy_version: int
    ) -> tuple[dict[str, np.ndarray], int]:
        sequence = self._sequences[actor_id]
        self._sequences[actor_id] += 1
        self._request_count += 1
        outputs, version, returned = self.runtime.infer(
            actor_id,
            sequence,
            observation,
            min_policy_version=min_policy_version,
            timeout=30.0,
        )
        if returned != sequence:
            raise RuntimeError("legacy response sequence mismatch")
        return outputs, version

    def update(self, state_dict: Mapping[str, torch.Tensor], version: int) -> int:
        return self.runtime.update_policy(state_dict, version=version, timeout=30.0)

    def metrics(self) -> dict[str, float | int]:
        metrics = self.runtime.metrics(self._request_count, timeout=5.0)
        return {key: value for key, value in asdict(metrics).items()}

    def close(self) -> None:
        self.runtime.close()


class _SharedLearningRuntime:
    def __init__(
        self,
        *,
        actor_count: int,
        observation_size: int,
        action_size: int,
        hidden_size: int,
        state_dict: Mapping[str, torch.Tensor],
        config: LearningBenchmarkConfig,
    ) -> None:
        self.endpoints = [
            SharedInferenceEndpoint.create(
                actor_id=actor_id,
                slot_count=config.request_slots,
                max_items=config.envs_per_actor,
                request_fields={"obs": ((observation_size,), np.float32)},
                response_fields={
                    "action": ((action_size,), np.float32),
                    "value": ((1,), np.float32),
                },
            )
            for actor_id in range(actor_count)
        ]

        def factory() -> ReferencePPOPolicy:
            return ReferencePPOPolicy(observation_size, action_size, hidden_size)

        self.registry = PolicyRegistry(history_size=2)
        snapshot = self.registry.publish(state_dict, version=0)
        self.service = NodeLocalInferenceService(
            endpoints=self.endpoints,
            replica=DoubleBufferedPolicyReplica(factory, device=config.device),
            infer_fn=run_synthetic_policy,
            max_batch_items=config.max_batch_items,
            min_batch_items=min(
                config.v2_min_batch_items,
                actor_count * config.envs_per_actor,
                config.max_batch_items,
            ),
            max_wait_ms=config.max_wait_ms,
            response_timeout_seconds=30.0,
        )
        self.service.refresh_policy(snapshot)
        self.service.start()
        self.clients = [SharedInferenceClient(endpoint) for endpoint in self.endpoints]

    def infer(
        self, actor_id: int, observation: np.ndarray, *, min_policy_version: int
    ) -> tuple[dict[str, np.ndarray], int]:
        response = self.clients[actor_id].infer(
            {"obs": np.ascontiguousarray(observation, dtype=np.float32)},
            min_policy_version=min_policy_version,
            timeout=30.0,
        )
        return response.outputs, response.policy_version

    def update(self, state_dict: Mapping[str, torch.Tensor], version: int) -> int:
        snapshot = self.registry.publish(state_dict, version=version)
        return self.service.refresh_policy(snapshot)

    def metrics(self) -> dict[str, float | int]:
        return {key: value for key, value in asdict(self.service.metrics()).items()}

    def close(self) -> None:
        try:
            self.service.stop(timeout=10.0)
        finally:
            for endpoint in self.endpoints:
                endpoint.shutdown()
                endpoint.close()
                endpoint.unlink()


def _state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone(memory_format=torch.preserve_format)
        for name, value in module.state_dict().items()
    }


def _evaluate(
    model: ReferencePPOPolicy,
    *,
    environment: str,
    seed: int,
    episodes: int,
    device: torch.device,
    action_kind: str,
    action_scale: np.ndarray,
) -> float:
    try:
        import gymnasium as gym
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install ForgeRL with the gym extra") from error
    was_training = model.training
    model.eval()
    returns: list[float] = []
    try:
        for episode in range(episodes):
            env = gym.make(environment)
            try:
                observation, _ = env.reset(seed=seed + 100_000 + episode)
                total = 0.0
                done = False
                while not done:
                    tensor = torch.as_tensor(
                        np.asarray(observation, dtype=np.float32),
                        device=device,
                    ).unsqueeze(0)
                    with torch.inference_mode():
                        policy, _value = model(tensor)
                    if action_kind == "discrete":
                        action: Any = int(policy.argmax(dim=-1).item())
                    else:
                        action = (
                            torch.tanh(policy).cpu().numpy()[0] * action_scale
                        ).astype(np.float32)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    done = bool(terminated or truncated)
                returns.append(total)
            finally:
                env.close()
    finally:
        model.train(was_training)
    return float(np.mean(returns))


def _normalized_auc(
    points: Sequence[EvaluationPoint], *, floor: float, target: float
) -> float:
    if not points:
        return 0.0
    denominator = max(target - floor, 1e-12)
    scores = np.clip(
        [(point.mean_return - floor) / denominator for point in points],
        0.0,
        1.0,
    )
    steps = np.asarray([point.environment_steps for point in points], dtype=np.float64)
    if len(points) == 1 or steps[-1] <= steps[0]:
        return float(scores[-1])
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy 1.x compatibility
        integrate = np.trapz
    return float(integrate(scores, steps) / (steps[-1] - steps[0]))


def _ppo_update(
    model: ReferencePPOPolicy,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, np.ndarray],
    *,
    action_kind: str,
    action_scale: np.ndarray,
    hyperparameters: PPOHyperparameters,
    device: torch.device,
    generator: torch.Generator,
) -> None:
    observation = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    old_log_probability = torch.as_tensor(
        batch["old_log_probability"], dtype=torch.float32, device=device
    )
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    if action_kind == "discrete":
        actions = torch.as_tensor(batch["action"], dtype=torch.int64, device=device)
        raw_actions = None
    else:
        actions = None
        raw_actions = torch.as_tensor(
            batch["raw_action"], dtype=torch.float32, device=device
        )
        scale_tensor = torch.as_tensor(action_scale, dtype=torch.float32, device=device)
        fixed_std = math.exp(hyperparameters.continuous_log_std)

    total = observation.shape[0]
    minibatch = min(hyperparameters.minibatch_size, total)
    model.train()
    for _ in range(hyperparameters.epochs):
        permutation = torch.randperm(total, generator=generator, device="cpu").to(device)
        for start in range(0, total, minibatch):
            index = permutation[start : start + minibatch]
            policy, value = model(observation[index])
            value = value.squeeze(-1)
            if action_kind == "discrete":
                distribution = Categorical(logits=policy)
                new_log_probability = distribution.log_prob(actions[index])
                entropy = distribution.entropy().mean()
            else:
                distribution = Normal(policy, fixed_std)
                selected_raw = raw_actions[index]
                new_log_probability = distribution.log_prob(selected_raw).sum(dim=-1)
                new_log_probability -= torch.log(
                    1.0 - torch.tanh(selected_raw).pow(2) + 1e-6
                ).sum(dim=-1)
                new_log_probability -= torch.log(scale_tensor).sum()
                entropy = distribution.entropy().sum(dim=-1).mean()
            ratio = torch.exp(new_log_probability - old_log_probability[index])
            unclipped = ratio * advantages[index]
            clipped = torch.clamp(
                ratio,
                1.0 - hyperparameters.clip_ratio,
                1.0 + hyperparameters.clip_ratio,
            ) * advantages[index]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (value - returns[index]).pow(2).mean()
            loss = (
                policy_loss
                + hyperparameters.value_coefficient * value_loss
                - hyperparameters.entropy_coefficient * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyperparameters.max_grad_norm)
            optimizer.step()


def _run_subject(
    *,
    environment: str,
    subject: Subject,
    seed: int,
    config: LearningBenchmarkConfig,
) -> LearningRun:
    try:
        import gymnasium as gym
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install ForgeRL with the gym extra") from error

    spec = ENVIRONMENT_SPECS[environment]
    device = torch.device(config.device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    probe = gym.make(environment)
    try:
        observation_size = int(np.prod(probe.observation_space.shape))
        if spec.action_kind == "discrete":
            action_size = int(probe.action_space.n)
            action_scale = np.ones(1, dtype=np.float32)
        else:
            low = np.asarray(probe.action_space.low, dtype=np.float32)
            high = np.asarray(probe.action_space.high, dtype=np.float32)
            if not np.allclose(low, -high):
                raise ValueError("continuous benchmark requires a symmetric action range")
            action_size = int(np.prod(probe.action_space.shape))
            action_scale = high.reshape(-1)
    finally:
        probe.close()

    model = ReferencePPOPolicy(observation_size, action_size, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
    initial_state = _state_dict_cpu(model)
    runtime: _LegacyLearningRuntime | _SharedLearningRuntime
    runtime_class = _LegacyLearningRuntime if subject == "v1" else _SharedLearningRuntime
    runtime = runtime_class(
        actor_count=config.actors,
        observation_size=observation_size,
        action_size=action_size,
        hidden_size=config.hidden_size,
        state_dict=initial_state,
        config=config,
    )

    envs = [
        [gym.make(environment) for _ in range(config.envs_per_actor)]
        for _ in range(config.actors)
    ]
    observations = np.zeros(
        (config.actors, config.envs_per_actor, observation_size), dtype=np.float32
    )
    episode_ids = np.zeros((config.actors, config.envs_per_actor), dtype=np.int64)
    step_ids = np.zeros_like(episode_ids)
    rngs = [np.random.default_rng(seed + actor * 9973 + 17) for actor in range(config.actors)]
    for actor in range(config.actors):
        for env_id, env in enumerate(envs[actor]):
            observation, _ = env.reset(seed=seed + actor * 1000 + env_id)
            observations[actor, env_id] = np.asarray(observation, dtype=np.float32).reshape(-1)

    max_updates = (
        config.cartpole_max_updates
        if environment == "CartPole-v1" and config.cartpole_max_updates is not None
        else config.pendulum_max_updates
        if environment == "Pendulum-v1" and config.pendulum_max_updates is not None
        else spec.formal_max_updates
    )
    eval_interval = (
        config.cartpole_eval_interval
        if environment == "CartPole-v1" and config.cartpole_eval_interval is not None
        else config.pendulum_eval_interval
        if environment == "Pendulum-v1" and config.pendulum_eval_interval is not None
        else spec.formal_eval_interval
    )
    policy_version = 0
    policy_versions = [0]
    environment_steps = 0
    collected_identities: list[tuple[int, int, int, int]] = []
    trained_identities: list[tuple[int, int, int, int]] = []
    evaluations: list[EvaluationPoint] = []
    time_to_target: float | None = None
    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(seed + 55_000)

    def infer_actor(actor_id: int) -> tuple[dict[str, np.ndarray], int]:
        return runtime.infer(
            actor_id,
            observations[actor_id],
            min_policy_version=policy_version,
        )

    try:
        initial_return = _evaluate(
            model,
            environment=environment,
            seed=seed,
            episodes=config.eval_episodes,
            device=device,
            action_kind=spec.action_kind,
            action_scale=action_scale,
        )
        evaluations.append(EvaluationPoint(0, 0.0, initial_return))
        if initial_return >= spec.target_return:
            time_to_target = 1e-9

        with ThreadPoolExecutor(max_workers=config.actors) as executor:
            for update in range(1, max_updates + 1):
                rollout_obs: list[np.ndarray] = []
                rollout_next_obs: list[np.ndarray] = []
                rollout_action: list[np.ndarray] = []
                rollout_raw_action: list[np.ndarray] = []
                rollout_reward: list[np.ndarray] = []
                rollout_terminated: list[np.ndarray] = []
                rollout_truncated: list[np.ndarray] = []
                rollout_log_probability: list[np.ndarray] = []
                rollout_value: list[np.ndarray] = []
                rollout_identities: list[tuple[int, int, int, int]] = []

                for _step in range(config.rollout_steps):
                    inference = list(executor.map(infer_actor, range(config.actors)))
                    step_observation = observations.copy()
                    next_observation = np.empty_like(observations)
                    rewards = np.zeros((config.actors, config.envs_per_actor), np.float32)
                    terminated = np.zeros_like(rewards, dtype=np.bool_)
                    truncated = np.zeros_like(rewards, dtype=np.bool_)
                    values = np.zeros_like(rewards)
                    old_log_probability = np.zeros_like(rewards)
                    if spec.action_kind == "discrete":
                        actions = np.zeros_like(rewards, dtype=np.int64)
                        raw_actions = actions[..., None].astype(np.float32)
                    else:
                        actions = np.zeros(
                            (config.actors, config.envs_per_actor, action_size), np.float32
                        )
                        raw_actions = np.zeros_like(actions)

                    for actor_id, (outputs, returned_version) in enumerate(inference):
                        if returned_version != policy_version:
                            raise RuntimeError(
                                f"actor {actor_id} used policy {returned_version}, expected {policy_version}"
                            )
                        policy_output = np.asarray(outputs["action"], dtype=np.float32)
                        values[actor_id] = np.asarray(outputs["value"], dtype=np.float32).reshape(-1)
                        if spec.action_kind == "discrete":
                            sampled, raw, log_probability = _sample_discrete(
                                policy_output, rngs[actor_id]
                            )
                            actions[actor_id] = sampled
                            raw_actions[actor_id] = raw
                        else:
                            sampled, raw, log_probability = _sample_continuous(
                                policy_output,
                                rngs[actor_id],
                                log_std=config.ppo.continuous_log_std,
                                action_scale=action_scale,
                            )
                            actions[actor_id] = sampled
                            raw_actions[actor_id] = raw
                        old_log_probability[actor_id] = log_probability

                    for actor_id in range(config.actors):
                        for env_id, env in enumerate(envs[actor_id]):
                            identity = (
                                actor_id,
                                env_id,
                                int(episode_ids[actor_id, env_id]),
                                int(step_ids[actor_id, env_id]),
                            )
                            rollout_identities.append(identity)
                            env_action: Any = (
                                int(actions[actor_id, env_id])
                                if spec.action_kind == "discrete"
                                else actions[actor_id, env_id]
                            )
                            observed, reward, term, trunc, _ = env.step(env_action)
                            terminal_observation = np.asarray(
                                observed, dtype=np.float32
                            ).reshape(-1)
                            next_observation[actor_id, env_id] = terminal_observation
                            rewards[actor_id, env_id] = float(reward)
                            terminated[actor_id, env_id] = bool(term)
                            truncated[actor_id, env_id] = bool(trunc)
                            if term or trunc:
                                reset_observation, _ = env.reset()
                                observations[actor_id, env_id] = np.asarray(
                                    reset_observation, dtype=np.float32
                                ).reshape(-1)
                                episode_ids[actor_id, env_id] += 1
                                step_ids[actor_id, env_id] = 0
                            else:
                                observations[actor_id, env_id] = terminal_observation
                                step_ids[actor_id, env_id] += 1

                    rollout_obs.append(step_observation)
                    rollout_next_obs.append(next_observation)
                    rollout_action.append(actions.copy())
                    rollout_raw_action.append(raw_actions.copy())
                    rollout_reward.append(rewards)
                    rollout_terminated.append(terminated)
                    rollout_truncated.append(truncated)
                    rollout_log_probability.append(old_log_probability)
                    rollout_value.append(values)
                    environment_steps += config.actors * config.envs_per_actor

                obs_array = np.stack(rollout_obs)
                next_obs_array = np.stack(rollout_next_obs)
                reward_array = np.stack(rollout_reward)
                value_array = np.stack(rollout_value)
                terminal_array = np.stack(rollout_terminated)
                truncation_array = np.stack(rollout_truncated)
                flattened_next = next_obs_array.reshape(-1, observation_size)
                with torch.inference_mode():
                    _policy, next_values_tensor = model(
                        torch.as_tensor(flattened_next, dtype=torch.float32, device=device)
                    )
                next_value_array = (
                    next_values_tensor.squeeze(-1)
                    .cpu()
                    .numpy()
                    .reshape(reward_array.shape)
                )
                time_size = reward_array.shape[0]
                environment_count = config.actors * config.envs_per_actor
                advantages, returns = compute_gae(
                    reward_array.reshape(time_size, environment_count),
                    value_array.reshape(time_size, environment_count),
                    next_value_array.reshape(time_size, environment_count),
                    terminal_array.reshape(time_size, environment_count),
                    truncation_array.reshape(time_size, environment_count),
                    gamma=config.ppo.gamma,
                    gae_lambda=config.ppo.gae_lambda,
                )
                batch = {
                    "obs": obs_array.reshape(-1, observation_size),
                    "action": np.stack(rollout_action).reshape(-1),
                    "old_log_probability": np.stack(rollout_log_probability).reshape(-1),
                    "advantages": advantages.reshape(-1),
                    "returns": returns.reshape(-1),
                }
                if spec.action_kind == "continuous":
                    batch["action"] = np.stack(rollout_action).reshape(-1, action_size)
                    batch["raw_action"] = np.stack(rollout_raw_action).reshape(
                        -1, action_size
                    )
                collected_identities.extend(rollout_identities)
                trained_identities.extend(rollout_identities)
                _ppo_update(
                    model,
                    optimizer,
                    batch,
                    action_kind=spec.action_kind,
                    action_scale=action_scale,
                    hyperparameters=config.ppo,
                    device=device,
                    generator=generator,
                )
                policy_version = update
                updated = runtime.update(_state_dict_cpu(model), policy_version)
                if updated != policy_version:
                    raise RuntimeError("runtime policy update acknowledgement mismatch")
                policy_versions.append(policy_version)

                if update % eval_interval == 0 or update == max_updates:
                    score = _evaluate(
                        model,
                        environment=environment,
                        seed=seed + update * 10,
                        episodes=config.eval_episodes,
                        device=device,
                        action_kind=spec.action_kind,
                        action_scale=action_scale,
                    )
                    elapsed = time.perf_counter() - started
                    evaluations.append(
                        EvaluationPoint(environment_steps, elapsed, score)
                    )
                    if time_to_target is None and score >= spec.target_return:
                        time_to_target = elapsed
    finally:
        elapsed = time.perf_counter() - started
        runtime_metrics = runtime.metrics()
        runtime.close()
        for actor_envs in envs:
            for env in actor_envs:
                env.close()

    audit = audit_transition_identities(collected_identities, trained_identities)
    auc = _normalized_auc(
        evaluations,
        floor=spec.return_floor,
        target=spec.target_return,
    )
    runtime_errors = int(runtime_metrics.get("errors", 0))
    return LearningRun(
        environment=environment,
        subject=subject,
        seed=seed,
        target_return=spec.target_return,
        target_reached=time_to_target is not None,
        time_to_target_seconds=time_to_target,
        elapsed_seconds=elapsed,
        environment_steps=environment_steps,
        collected_rows=audit.collected_valid_rows,
        trained_rows=audit.trained_valid_rows,
        useful_sample_ratio=audit.useful_sample_ratio,
        missing_rows=audit.missing_rows,
        duplicate_rows=audit.duplicate_trained_rows,
        unexpected_rows=audit.unexpected_rows,
        invalid_rows_in_loss=audit.invalid_rows_in_loss,
        runtime_errors=runtime_errors,
        policy_versions=tuple(policy_versions),
        normalized_auc=auc,
        evaluations=tuple(evaluations),
        runtime_metrics=runtime_metrics,
    )


def _summary_for_environment(
    environment: str, runs: Sequence[LearningRun], *, required_seeds: int
) -> dict[str, Any]:
    by_subject = {
        subject: [run for run in runs if run.environment == environment and run.subject == subject]
        for subject in ("v1", "v2")
    }
    if any(not rows for rows in by_subject.values()):
        raise ValueError(f"missing subject results for {environment}")

    def median_time(rows: Sequence[LearningRun]) -> float:
        observations = [
            run.time_to_target_seconds
            if run.time_to_target_seconds is not None
            else run.elapsed_seconds
            for run in rows
        ]
        return float(statistics.median(observations))

    v1 = by_subject["v1"]
    v2 = by_subject["v2"]
    seeds = len({run.seed for run in v1} & {run.seed for run in v2})
    v1_reached = sum(run.target_reached for run in v1)
    v2_reached = sum(run.target_reached for run in v2)
    v1_time = median_time(v1)
    v2_time = median_time(v2)
    v1_auc = float(statistics.median(run.normalized_auc for run in v1))
    v2_auc = float(statistics.median(run.normalized_auc for run in v2))
    time_ratio = v2_time / max(v1_time, 1e-12)
    auc_ratio = v2_auc / max(v1_auc, 1e-12)
    passed = (
        seeds >= required_seeds
        and v1_reached >= required_seeds
        and v2_reached >= required_seeds
        and time_ratio <= 1.05
        and auc_ratio >= 0.95
    )
    return {
        "environment": environment,
        "seeds": seeds,
        "v1_target_reached_seeds": v1_reached,
        "v2_target_reached_seeds": v2_reached,
        "required_target_reached_seeds": required_seeds,
        "v1_time_to_target_median_seconds": v1_time,
        "v2_time_to_target_median_seconds": v2_time,
        "v1_normalized_auc": v1_auc,
        "v2_normalized_auc": v2_auc,
        "time_ratio": time_ratio,
        "auc_ratio": auc_ratio,
        "passed": passed,
    }


def run_learning_benchmark(config: LearningBenchmarkConfig) -> LearningBenchmarkReport:
    config.validate()
    runs: list[LearningRun] = []
    for environment in config.environments:
        for seed_index, seed in enumerate(config.seeds):
            order: tuple[Subject, Subject] = (
                ("v1", "v2") if seed_index % 2 == 0 else ("v2", "v1")
            )
            for subject in order:
                runs.append(
                    _run_subject(
                        environment=environment,
                        subject=subject,
                        seed=seed,
                        config=config,
                    )
                )
    required_seeds = 1 if config.smoke else len(config.seeds)
    summaries = tuple(
        _summary_for_environment(environment, runs, required_seeds=required_seeds)
        for environment in config.environments
    )
    correctness = all(
        run.missing_rows == 0
        and run.duplicate_rows == 0
        and run.unexpected_rows == 0
        and run.invalid_rows_in_loss == 0
        and run.runtime_errors == 0
        and run.collected_rows == run.trained_rows
        and run.policy_versions == tuple(range(len(run.policy_versions)))
        for run in runs
    )
    formal_eligible = (
        not config.smoke
        and len(config.seeds) >= 5
        and {"CartPole-v1", "Pendulum-v1"}.issubset(config.environments)
    )
    learning_gate_passed = formal_eligible and correctness and all(
        bool(summary["passed"]) for summary in summaries
    )
    if not correctness:
        status = "FAIL_CORRECTNESS"
    elif config.smoke:
        status = "SMOKE_PASS"
    elif not formal_eligible:
        status = "PENDING_FORMAL_GATE"
    elif learning_gate_passed:
        status = "GO"
    else:
        status = "FAIL_LEARNING_GATE"
    fingerprint = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "github_sha": os.environ.get("GITHUB_SHA", ""),
    }
    try:
        import gymnasium

        fingerprint["gymnasium"] = gymnasium.__version__
    except ImportError:  # pragma: no cover
        fingerprint["gymnasium"] = "unavailable"
    return LearningBenchmarkReport(
        config=config,
        runs=tuple(runs),
        environments=summaries,
        formal_eligible=formal_eligible,
        learning_gate_passed=learning_gate_passed,
        status=status,
        environment_fingerprint=fingerprint,
    )
