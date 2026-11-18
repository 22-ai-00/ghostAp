"""Employee-scoped, fail-closed context contracts and assembly."""

from .assembler import EmployeeThreadContext
from .group_ledger import (
    CanonicalGroupContext,
    GroupContextLedger,
    GroupEventPayload,
    GroupEventRecord,
    GroupLedgerError,
)
from .models import (
    AssembledContext,
    AuthorizedContextRequest,
    ContextLayer,
    ContextLayerMetrics,
    ContextMessage,
    ContextQuality,
    ContextUnavailableError,
    ContextUnavailableReason,
    ContextWarning,
    EmployeeMessageScope,
    MessageRevision,
    MessageSourceError,
    ThreadContextConfig,
    ThreadWatermark,
    TrimmingRecord,
)
from .service import (
    AuthorizedGroupMemoryReader,
    EmployeeContextService,
)
from .source import (
    CredentialResolver,
    EmployeeMessageSourceFactory,
    EmployeeScopedMessageSource,
    MessagePage,
    ResolvedThread,
)

__all__ = [
    "AssembledContext",
    "AuthorizedContextRequest",
    "AuthorizedGroupMemoryReader",
    "ContextLayer",
    "ContextLayerMetrics",
    "ContextQuality",
    "ContextWarning",
    "ContextMessage",
    "ContextUnavailableError",
    "ContextUnavailableReason",
    "CredentialResolver",
    "EmployeeContextService",
    "EmployeeMessageSourceFactory",
    "EmployeeMessageScope",
    "EmployeeScopedMessageSource",
    "EmployeeThreadContext",
    "CanonicalGroupContext",
    "GroupContextLedger",
    "GroupEventPayload",
    "GroupEventRecord",
    "GroupLedgerError",
    "MessagePage",
    "MessageRevision",
    "MessageSourceError",
    "ResolvedThread",
    "ThreadContextConfig",
    "ThreadWatermark",
    "TrimmingRecord",
]
