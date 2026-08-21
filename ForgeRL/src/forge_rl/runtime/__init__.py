"""ForgeRL v2 runtime contracts and local reference implementations."""

from .coordinator import InMemoryCoordinator, NodeLease, NodeRole
from .dynamic_batcher import BatchEnvelope, DeadlineBatcher
from .experience import ExperienceItem, OnPolicyExperienceQueue
from .inference import (
    DoubleBufferedPolicyReplica,
    InferenceMetricsSnapshot,
    InferenceResponse,
    NodeLocalInferenceService,
    SharedInferenceClient,
    SharedInferenceEndpoint,
)
from .policy_registry import PolicyRegistry, PolicySnapshot
from .trajectory import TrajectoryBuilder, TrajectoryFragment
from .transition import TransitionBatch, TransitionIdentityError

__all__ = [
    "BatchEnvelope",
    "DeadlineBatcher",
    "DoubleBufferedPolicyReplica",
    "ExperienceItem",
    "InferenceMetricsSnapshot",
    "InferenceResponse",
    "InMemoryCoordinator",
    "NodeLease",
    "NodeLocalInferenceService",
    "NodeRole",
    "OnPolicyExperienceQueue",
    "PolicyRegistry",
    "PolicySnapshot",
    "SharedInferenceClient",
    "SharedInferenceEndpoint",
    "TrajectoryBuilder",
    "TrajectoryFragment",
    "TransitionBatch",
    "TransitionIdentityError",
]
