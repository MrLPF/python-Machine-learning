"""Reproducible correctness, throughput and learning acceptance utilities."""

from .audit import TransitionAudit, audit_transition_identities, identities_from_batch
from .legacy_v1 import (
    LEGACY_V1_ARCHIVE_SHA256,
    LegacyRuntimeMetrics,
    LegacyV1InferenceRuntime,
    ReferencePPOPolicy,
    SyntheticPolicy,
    run_synthetic_policy,
)
from .m1_acceptance import (
    LearningGate,
    LearningResult,
    M1AcceptanceConfig,
    M1AcceptanceReport,
    RuntimeReport,
    run_m1_acceptance,
)
from .ppo_learning import (
    ENVIRONMENT_SPECS,
    EvaluationPoint,
    LearningBenchmarkConfig,
    LearningBenchmarkReport,
    LearningRun,
    PPOHyperparameters,
    compute_gae,
    run_learning_benchmark,
)

__all__ = [
    "ENVIRONMENT_SPECS",
    "EvaluationPoint",
    "LEGACY_V1_ARCHIVE_SHA256",
    "LearningBenchmarkConfig",
    "LearningBenchmarkReport",
    "LearningGate",
    "LearningResult",
    "LearningRun",
    "LegacyRuntimeMetrics",
    "LegacyV1InferenceRuntime",
    "M1AcceptanceConfig",
    "M1AcceptanceReport",
    "PPOHyperparameters",
    "ReferencePPOPolicy",
    "RuntimeReport",
    "SyntheticPolicy",
    "TransitionAudit",
    "audit_transition_identities",
    "compute_gae",
    "identities_from_batch",
    "run_learning_benchmark",
    "run_m1_acceptance",
    "run_synthetic_policy",
]
