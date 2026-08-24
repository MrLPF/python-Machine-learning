from __future__ import annotations

import pytest
import torch

from forge_rl.runtime.policy_registry import PolicyRegistry


def test_policy_registry_versions_and_eviction() -> None:
    registry = PolicyRegistry(history_size=2)
    first = registry.publish({"weight": torch.tensor([1.0])})
    second = registry.publish({"weight": torch.tensor([2.0])})
    third = registry.publish({"weight": torch.tensor([3.0])})

    assert (first.version, second.version, third.version) == (0, 1, 2)
    assert registry.latest_version == 2
    assert registry.lag(0) == 2
    with pytest.raises(LookupError):
        registry.get(0)
    assert registry.get(1).state_dict["weight"].item() == 2.0
