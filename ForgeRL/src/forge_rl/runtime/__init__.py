"""ForgeRL v2 runtime contracts and local reference implementations."""

from .actor_local_inference import ActorLocalInferenceRuntime
from .coordinator import InMemoryCoordinator, NodeLease, NodeRole
from .dynamic_batcher import BatchEnvelope, DeadlineBatcher
from .event_driven_inference import (
    EventDrivenInferenceMetricsSnapshot,
    NodeLocalInferenceService,
)
from .experience import ExperienceItem, OnPolicyExperienceQueue
from .fast_mailbox_inference import (
    FastMailboxInferenceClient,
    FastMailboxNodeLocalInferenceService,
)
from .inference import (
    DoubleBufferedPolicyReplica,
    InferenceMetricsSnapshot,
    InferenceResponse,
    SharedInferenceClient,
    SharedInferenceEndpoint,
)
from .mailbox_inference import (
    MailboxClientDescriptor,
    MailboxInferenceClient,
    MailboxInferenceEndpoint,
    MailboxNodeLocalInferenceService,
    MailboxTransportMetricsSnapshot,
)
from .optimized_inference import (
    NodeLocalInferenceService as PollingNodeLocalInferenceService,
    OptimizedInferenceMetricsSnapshot,
    ThreadedNodeLocalInferenceService,
)
from .policy_registry import PolicyRegistry, PolicySnapshot
from .process_mailbox_inference import ProcessMailboxInferenceRuntime
from .trajectory import TrajectoryBuilder, TrajectoryFragment
from .transition import TransitionBatch, TransitionIdentityError

__all__ = [
    "ActorLocalInferenceRuntime",
    "BatchEnvelope",
    "DeadlineBatcher",
    "DoubleBufferedPolicyReplica",
    "EventDrivenInferenceMetricsSnapshot",
    "ExperienceItem",
    "FastMailboxInferenceClient",
    "FastMailboxNodeLocalInferenceService",
    "InferenceMetricsSnapshot",
    "InferenceResponse",
    "InMemoryCoordinator",
    "MailboxClientDescriptor",
    "MailboxInferenceClient",
    "MailboxInferenceEndpoint",
    "MailboxNodeLocalInferenceService",
    "MailboxTransportMetricsSnapshot",
    "NodeLease",
    "NodeLocalInferenceService",
    "NodeRole",
    "OnPolicyExperienceQueue",
    "OptimizedInferenceMetricsSnapshot",
    "PolicyRegistry",
    "PolicySnapshot",
    "PollingNodeLocalInferenceService",
    "ProcessMailboxInferenceRuntime",
    "SharedInferenceClient",
    "SharedInferenceEndpoint",
    "ThreadedNodeLocalInferenceService",
    "TrajectoryBuilder",
    "TrajectoryFragment",
    "TransitionBatch",
    "TransitionIdentityError",
]
