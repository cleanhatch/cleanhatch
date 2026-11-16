"""Typed failures for the cleanhatch guard kernel.

These errors deliberately do not inherit from the host application's error
hierarchy.  An adapter may translate them into its own exit-code contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class CleanHatchError(Exception):
    """Base class for every guard-kernel failure."""

    error_code = "CLEANHATCH_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, object] = dict(details) if details is not None else {}

    def to_dict(self) -> dict[str, object]:
        return {
            "error_type": type(self).__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": dict(self.details),
        }


class TemporalContractError(CleanHatchError, ValueError):
    """A table or configuration violates the point-in-time contract."""

    error_code = "TEMPORAL_CONTRACT_ERROR"


class VersionMutationError(TemporalContractError):
    """The same version tag was reused for different content."""

    error_code = "VERSION_MUTATION_ERROR"


class InvarViolationError(CleanHatchError, ValueError):
    """An invariant gate rejected a frame."""

    error_code = "INVAR_VIOLATION_ERROR"


class CausalContractError(CleanHatchError, ValueError):
    """A causal gate input or output does not satisfy the documented contract."""

    error_code = "CAUSAL_CONTRACT_ERROR"


class CausalExecutionError(CleanHatchError, RuntimeError):
    """The user function raised while the causal gate evaluated it."""

    error_code = "CAUSAL_EXECUTION_ERROR"


class CausalFailure(CleanHatchError):
    """The causal gate detected a future-dependent change."""

    error_code = "FAILED_CAUSALITY"

    def __init__(
        self,
        *,
        failed_checks: Sequence[object],
        all_checks: Sequence[object],
        sub_tolerance_drifts: Sequence[object],
        slicing_mode: str,
        checkpoint_kinds: Sequence[str],
        partial_result: Mapping[str, object],
    ) -> None:
        super().__init__(
            "FAILED_CAUSALITY: output changed when future data was removed or perturbed",
            details={
                "failed_checks": _serialize(failed_checks),
                "all_checks": _serialize(all_checks),
                "sub_tolerance_drifts": _serialize(sub_tolerance_drifts),
                "slicing_mode": slicing_mode,
                "checkpoint_kinds": list(checkpoint_kinds),
                "partial_result": dict(partial_result),
            },
        )
        self.failed_checks = tuple(failed_checks)
        self.all_checks = tuple(all_checks)
        self.sub_tolerance_drifts = tuple(sub_tolerance_drifts)
        self.slicing_mode = slicing_mode
        self.checkpoint_kinds = tuple(checkpoint_kinds)


class NonIsolatedAblationWarning(UserWarning):
    """Two experiments changed more than one variable at the same time."""


def _serialize(items: Sequence[object]) -> list[object]:
    rendered: list[object] = []
    for item in items:
        to_dict = getattr(item, "to_dict", None)
        rendered.append(to_dict() if callable(to_dict) else item)
    return rendered
