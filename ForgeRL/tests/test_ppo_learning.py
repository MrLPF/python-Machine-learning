from __future__ import annotations

import numpy as np
import torch

from forge_rl.benchmarks.ppo_learning import (
    EvaluationPoint,
    LearningRun,
    _squashed_normal_log_prob,
    _summary_for_environment,
    compute_gae,
)


def test_gae_keeps_truncation_bootstrap_but_stops_recursion() -> None:
    advantages, returns = compute_gae(
        rewards=np.asarray([[1.0], [1.0]], np.float32),
        values=np.zeros((2, 1), np.float32),
        next_values=np.asarray([[10.0], [10.0]], np.float32),
        terminated=np.asarray([[True], [False]]),
        truncated=np.asarray([[False], [True]]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(advantages[:, 0], [1.0, 11.0])
    np.testing.assert_allclose(returns, advantages)


def test_squashed_normal_log_prob_matches_torch_formula() -> None:
    raw = np.asarray([[0.2, -0.4], [0.0, 0.3]], np.float32)
    mean = np.asarray([[0.1, -0.2], [0.2, 0.1]], np.float32)
    scale = np.asarray([2.0, 0.5], np.float32)
    actual = _squashed_normal_log_prob(raw, mean, -0.5, scale)

    raw_tensor = torch.tensor(raw)
    mean_tensor = torch.tensor(mean)
    distribution = torch.distributions.Normal(mean_tensor, np.exp(-0.5))
    expected = distribution.log_prob(raw_tensor).sum(-1)
    expected -= torch.log(1.0 - torch.tanh(raw_tensor).pow(2) + 1e-6).sum(-1)
    expected -= torch.log(torch.tensor(scale)).sum()
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)


def _run(subject: str, seed: int, *, reached: bool, time_value: float, auc: float) -> LearningRun:
    return LearningRun(
        environment="CartPole-v1",
        subject=subject,  # type: ignore[arg-type]
        seed=seed,
        target_return=475.0,
        target_reached=reached,
        time_to_target_seconds=time_value if reached else None,
        elapsed_seconds=time_value,
        environment_steps=100,
        collected_rows=100,
        trained_rows=100,
        useful_sample_ratio=1.0,
        missing_rows=0,
        duplicate_rows=0,
        unexpected_rows=0,
        invalid_rows_in_loss=0,
        runtime_errors=0,
        policy_versions=(0, 1),
        normalized_auc=auc,
        evaluations=(EvaluationPoint(100, time_value, 475.0),),
        runtime_metrics={},
    )


def test_learning_summary_requires_every_formal_seed_to_reach_target() -> None:
    runs = [
        _run("v1", 1, reached=True, time_value=10.0, auc=0.8),
        _run("v1", 2, reached=True, time_value=12.0, auc=0.8),
        _run("v2", 1, reached=True, time_value=10.0, auc=0.8),
        _run("v2", 2, reached=False, time_value=12.0, auc=0.8),
    ]
    summary = _summary_for_environment("CartPole-v1", runs, required_seeds=2)
    assert summary["v2_target_reached_seeds"] == 1
    assert summary["passed"] is False
