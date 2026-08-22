from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from forge_rl.benchmarks.legacy_v1 import ReferencePPOPolicy, run_synthetic_policy
from forge_rl.benchmarks.ppo_learning import PPOHyperparameters, _ppo_update
from forge_rl.envs import CppVectorEnv
from forge_rl.runtime import (
    ExperienceItem,
    OnPolicyExperienceQueue,
    TransitionBatch,
    VectorBatchInferenceRuntime,
)


def _compile_counter_env(project: Path, output: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the vector-env pipeline test")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            "-I",
            str(project / "cpp" / "include"),
            str(project / "cpp" / "src" / "counter_vector_env.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _selected_log_probability(logits: np.ndarray, actions: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probability = shifted - log_normalizer
    return log_probability[np.arange(actions.shape[0]), actions].astype(np.float32)


def test_cpp_vector_env_vector_batch_experience_and_ppo_update(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    library = tmp_path / "libforge_rl_counter_env.so"
    _compile_counter_env(project, library)

    torch.manual_seed(2030)
    policy = ReferencePPOPolicy(2, 2, 16)
    runtime = VectorBatchInferenceRuntime(
        actor_count=2,
        module_type=ReferencePPOPolicy,
        module_args=(2, 2, 16),
        infer_fn=run_synthetic_policy,
        state_dict=_state(policy),
        max_batch_items=4,
        copy_outputs=False,
    )
    runtime.start()
    experience = OnPolicyExperienceQueue(capacity_rows=64)
    actor_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    env_ids = np.asarray([0, 1, 0, 1], dtype=np.int64)
    episode_ids = np.zeros(4, dtype=np.int64)
    step_ids = np.zeros(4, dtype=np.int64)
    fragments: list[TransitionBatch] = []

    try:
        with CppVectorEnv(library, num_envs=4) as env:
            observations = env.reset()
            for _ in range(8):
                responses = runtime.infer_batch(
                    {"obs": observations},
                    actor_item_counts={0: 2, 1: 2},
                    min_policy_version=0,
                )
                logits = np.concatenate(
                    [responses[actor].outputs["action"] for actor in (0, 1)],
                    axis=0,
                )
                values = np.concatenate(
                    [responses[actor].outputs["value"] for actor in (0, 1)],
                    axis=0,
                ).reshape(-1)
                actions = logits.argmax(axis=-1).astype(np.int64)
                log_probability = _selected_log_probability(logits, actions)
                result = env.step(actions)

                fragments.append(
                    TransitionBatch(
                        obs={"obs": observations.copy()},
                        next_obs={"obs": result.observations.copy()},
                        action={"action": actions.copy()},
                        reward=result.rewards.copy(),
                        terminated=result.terminated.copy(),
                        truncated=result.truncated.copy(),
                        valid_mask=np.ones(4, dtype=np.float32),
                        behavior_log_prob=log_probability,
                        behavior_value=values.astype(np.float32),
                        policy_version=np.zeros(4, dtype=np.int64),
                        episode_id=episode_ids.copy(),
                        step_id=step_ids.copy(),
                        actor_id=actor_ids,
                        env_id=env_ids,
                    )
                )

                observations = result.observations.copy()
                done = np.logical_or(result.terminated, result.truncated)
                if done.any():
                    indices = np.flatnonzero(done)
                    observations[indices] = env.reset(indices.tolist())
                    episode_ids[indices] += 1
                    step_ids[indices] = 0
                step_ids[~done] += 1

        merged = TransitionBatch.concat(fragments)
        assert merged.size == 32
        assert merged.valid_mask.sum() == 32
        item = ExperienceItem.create(
            merged,
            train_rows=merged.size,
            policy_version_min=0,
            policy_version_max=0,
            actor_id=-1,
        )
        experience.put(item, timeout=0.0)
        sampled = experience.sample(
            min_rows=merged.size,
            current_policy_version=0,
            max_policy_lag=0,
            timeout=0.0,
        )
        assert len(sampled) == 1
        batch = sampled[0].payload
        assert isinstance(batch, TransitionBatch)

        before = [parameter.detach().clone() for parameter in policy.parameters()]
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        returns = batch.reward + batch.behavior_value
        _ppo_update(
            policy,
            optimizer,
            {
                "obs": batch.obs["obs"],
                "action": batch.action["action"],
                "old_log_probability": batch.behavior_log_prob,
                "advantages": batch.reward.copy(),
                "returns": returns,
            },
            action_kind="discrete",
            action_scale=np.ones(1, dtype=np.float32),
            hyperparameters=PPOHyperparameters(epochs=1, minibatch_size=32),
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu").manual_seed(2031),
        )
        assert any(
            not torch.equal(previous, current.detach())
            for previous, current in zip(before, policy.parameters(), strict=True)
        )
        assert runtime.update_policy(_state(policy), version=1) == 1
        metrics = runtime.metrics()
        assert metrics.batches == 8
        assert metrics.mean_batch_items == 4
        assert metrics.errors == 0
    finally:
        experience.close()
        runtime.close()
