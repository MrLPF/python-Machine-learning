"""ForgeRL v2 runtime contracts and local reference implementations."""

from .coordinator import InMemoryCoordinator, NodeLease, NodeRole
from .dynamic_batcher import BatchEnvelope, DeadlineBatcher
from .event_driven_inference import (
    EventDrivenInferenceMetricsSnapshot,
    NodeLocalInferenceService,
)
from .experience import ExperienceItem, OnPolicyExperienceQueue
from .inference import (
    DoubleBufferedPolicyReplica,
    InferenceMetricsSnapshot,
    InferenceResponse,
    SharedInferenceClient,
    SharedInferenceEndpoint,
)
from .optimized_inference import (
    NodeLocalInferenceService as PollingNodeLocalInferenceService,
    OptimizedInferenceMetricsSnapshot,
    ThreadedNodeLocalInferenceService,
)
from .policy_registry import PolicyRegistry, PolicySnapshot
from .trajectory import TrajectoryBuilder, TrajectoryFragment
from .transition import TransitionBatch, TransitionIdentityError

__all__ = [
    "BatchEnvelope",
    "DeadlineBatcher",
    "DoubleBufferedPolicyReplica",
    "EventDrivenInferenceMetricsSnapshot",
    "ExperienceItem",
    "InferenceMetricsSnapshot",
    "InferenceResponse",
    "InMemoryCoordinator",
    "NodeLease",
    "NodeLocalInferenceService",
    "NodeRole",
    "OnPolicyExperienceQueue",
    "OptimizedInferenceMetricsSnapshot",
    "PolicyRegistry",
    "PolicySnapshot",
    "PollingNodeLocalInferenceService",
    "SharedInferenceClient",
    "SharedInferenceEndpoint",
    "ThreadedNodeLocalInferenceService",
    "TrajectoryBuilder",
    "TrajectoryFragment",
    "TransitionBatch",
    "TransitionIdentityError",
]
