"""ForgeRL v2 runtime contracts and reference implementations."""

from .actor_local_inference import ActorLocalInferenceRuntime
from .coordinator import (
    InMemoryCoordinator,
    NodeLease,
    NodeNotFoundError,
    NodeRole,
    PolicyVersionConflictError,
    StaleGenerationError,
)
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
from .in_process_inference import (
    InProcessBatchingInferenceRuntime as LegacyInProcessBatchingInferenceRuntime,
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
from .network_coordinator import (
    CoordinatorHTTPError,
    NetworkCoordinatorClient,
    NetworkCoordinatorService,
    RemoteNodeLease,
)
from .optimized_inference import (
    NodeLocalInferenceService as PollingNodeLocalInferenceService,
    OptimizedInferenceMetricsSnapshot,
    ThreadedNodeLocalInferenceService,
)
from .policy_registry import PolicyRegistry, PolicySnapshot
from .process_mailbox_inference import ProcessMailboxInferenceRuntime
from .rendezvous_in_process_inference import (
    RendezvousInProcessBatchingInferenceRuntime,
)
from .trajectory import TrajectoryBuilder, TrajectoryFragment
from .transition import TransitionBatch, TransitionIdentityError
from .vector_batch_inference import VectorBatchInferenceRuntime

# The public in-process runtime uses bounded arrival-aware batching. The previous implementation
# remains available under an explicit regression-only name for controlled A/B measurements.
InProcessBatchingInferenceRuntime = RendezvousInProcessBatchingInferenceRuntime

__all__ = [
    "ActorLocalInferenceRuntime",
    "BatchEnvelope",
    "CoordinatorHTTPError",
    "DeadlineBatcher",
    "DoubleBufferedPolicyReplica",
    "EventDrivenInferenceMetricsSnapshot",
    "ExperienceItem",
    "FastMailboxInferenceClient",
    "FastMailboxNodeLocalInferenceService",
    "InferenceMetricsSnapshot",
    "InferenceResponse",
    "InMemoryCoordinator",
    "InProcessBatchingInferenceRuntime",
    "LegacyInProcessBatchingInferenceRuntime",
    "MailboxClientDescriptor",
    "MailboxInferenceClient",
    "MailboxInferenceEndpoint",
    "MailboxNodeLocalInferenceService",
    "MailboxTransportMetricsSnapshot",
    "NetworkCoordinatorClient",
    "NetworkCoordinatorService",
    "NodeLease",
    "NodeLocalInferenceService",
    "NodeNotFoundError",
    "NodeRole",
    "OnPolicyExperienceQueue",
    "OptimizedInferenceMetricsSnapshot",
    "PolicyRegistry",
    "PolicySnapshot",
    "PolicyVersionConflictError",
    "PollingNodeLocalInferenceService",
    "ProcessMailboxInferenceRuntime",
    "RemoteNodeLease",
    "RendezvousInProcessBatchingInferenceRuntime",
    "SharedInferenceClient",
    "SharedInferenceEndpoint",
    "StaleGenerationError",
    "ThreadedNodeLocalInferenceService",
    "TrajectoryBuilder",
    "TrajectoryFragment",
    "TransitionBatch",
    "TransitionIdentityError",
    "VectorBatchInferenceRuntime",
]
