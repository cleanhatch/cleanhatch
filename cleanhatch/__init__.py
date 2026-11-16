"""cleanhatch — domain-neutral temporal, causal, invariant, and lineage gates.

The public surface is intentionally small and contains no financial field
names.  Host applications inject their own column roles through
:class:`TemporalSpec`, attach domain rules to :class:`InvarGate`, and configure
:class:`CausalGate` for their transform contract.

This is the Apache-2.0 L0 kernel.  Commercial licensing, audit-report export,
and the three L1 vertical packs live in the separate private ``cleanhatch-l1``
distribution.
"""

from .__about__ import (
    __author__,
    __author_email__,
    __company__,
    __description__,
    __title__,
    __version__,
)
from .ai_summary import MAX_AI_SUMMARY_BYTES, build_ai_summary
from .causal import (
    MAX_TOLERANCE,
    CausalCheck,
    CausalDrift,
    CausalGate,
    CausalMethod,
    CausalResult,
    CausalTransform,
    CheckpointKind,
    compare_causal_values,
    run_causal_analysis,
)
from .errors import (
    CausalContractError,
    CausalExecutionError,
    CausalFailure,
    CleanHatchError,
    InvarViolationError,
    NonIsolatedAblationWarning,
    TemporalContractError,
    VersionMutationError,
)
from .hashing import content_hash, source_ast_hash
from .invar import (
    GateCheck,
    GateReport,
    GateStatus,
    InvarGate,
    InvarRule,
    RuleSeverity,
    Stage,
)
from .lineage import (
    AblationDiffReport,
    GateEvidence,
    InputSnapshot,
    LineageManifest,
    TransformSignature,
)
from .packs import (
    CanaryCheckResult,
    CanarySuiteReport,
    GatePack,
    PackNotInstalledError,
    available_packs,
    get_pack,
    register_pack,
)
from .spec import TemporalSpec
from .temporal import (
    ColumnDiff,
    SourceInput,
    TemporalGate,
    VersionDiffReport,
)
from .tolerance import (
    BPS_TOLERANCE,
    DUST_TOLERANCE,
    EXACT_INT_CENTS,
    ToleranceMode,
    ToleranceProfile,
    at_least,
    at_most,
    scalar_within_tolerance,
    within_tolerance,
)

__all__ = [
    "AblationDiffReport",
    "BPS_TOLERANCE",
    "CanaryCheckResult",
    "CanarySuiteReport",
    "CausalCheck",
    "CausalContractError",
    "CausalDrift",
    "CausalExecutionError",
    "CausalFailure",
    "CausalGate",
    "CausalMethod",
    "CausalResult",
    "CausalTransform",
    "CheckpointKind",
    "CleanHatchError",
    "ColumnDiff",
    "DUST_TOLERANCE",
    "EXACT_INT_CENTS",
    "GateCheck",
    "GateEvidence",
    "GatePack",
    "GateReport",
    "GateStatus",
    "InputSnapshot",
    "InvarGate",
    "InvarRule",
    "InvarViolationError",
    "LineageManifest",
    "MAX_AI_SUMMARY_BYTES",
    "MAX_TOLERANCE",
    "NonIsolatedAblationWarning",
    "PackNotInstalledError",
    "RuleSeverity",
    "SourceInput",
    "Stage",
    "TemporalContractError",
    "TemporalGate",
    "TemporalSpec",
    "ToleranceMode",
    "ToleranceProfile",
    "TransformSignature",
    "VersionDiffReport",
    "VersionMutationError",
    "__author__",
    "__author_email__",
    "__company__",
    "__description__",
    "__title__",
    "__version__",
    "at_least",
    "at_most",
    "available_packs",
    "build_ai_summary",
    "compare_causal_values",
    "content_hash",
    "get_pack",
    "register_pack",
    "run_causal_analysis",
    "scalar_within_tolerance",
    "source_ast_hash",
    "within_tolerance",
]
