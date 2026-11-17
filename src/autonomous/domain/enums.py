"""State and protocol enums for autonomous employees."""

from enum import Enum


class GoalType(str, Enum):
    ONE_SHOT = "one_shot"
    SCHEDULED = "scheduled"
    STANDING = "standing"


class AutonomyMode(str, Enum):
    ASSIST = "assist"
    SUPERVISED = "supervised"
    BOUNDED_AUTONOMOUS = "bounded_autonomous"


class MisfirePolicy(str, Enum):
    RUN_ALL = "run_all"
    SKIP = "skip"
    RUN_LATEST = "run_latest"


class OverlapPolicy(str, Enum):
    FORBID = "forbid"
    QUEUE = "queue"
    ALLOW_PARALLEL = "allow_parallel"


class WorkerType(str, Enum):
    LOGICAL = "logical"
    VISIBLE = "visible"
    EPHEMERAL = "ephemeral"


class EmployeeIdOrigin(str, Enum):
    NATIVE = "native"
    LEGACY_ALIAS = "legacy_alias"


class EmployeeState(str, Enum):
    DRAFT = "draft"
    PROVISIONING_APP = "provisioning_app"
    STORING_CREDENTIAL = "storing_credential"
    CONFIGURING = "configuring"
    VALIDATING = "validating"
    READY_PENDING_VERIFICATION = "ready_pending_verification"
    ACTIVE = "active"
    RETIRING = "retiring"
    ACTION_REQUIRED = "action_required"
    ARCHIVED = "archived"
