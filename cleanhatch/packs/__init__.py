"""Industry gate packs: shared protocol, canary evidence, and registry.

A pack owns a domain-specific rule set while reusing the cleanhatch kernel.
The pack API is intentionally narrow so the CLI and Python callers mount every
pack the same way.

The L0 wheel ships one free sample pack (``temporal-core``) and the discovery
protocol.  Commercial L1 packs register through the ``cleanhatch.packs``
entry-point group; listing installed names reads distribution metadata only and
does not import their implementation modules.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import polars as pl

from ..errors import CleanHatchError
from ..invar import GateReport, InvarRule, Stage
from ..spec import TemporalSpec
from ..tolerance import ToleranceProfile

_ENTRY_POINT_GROUP = "cleanhatch.packs"
_KNOWN_L1_PACKS = frozenset(
    {
        "market-bars-pit",
        "sec-fundamentals-pit",
        "ai-rag-ledger-guard",
    }
)


class PackNotInstalledError(CleanHatchError, KeyError):
    """A gate pack is valid but is not installed in this build."""

    error_code = "PACK_NOT_INSTALLED"

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class CanaryCheckResult:
    """One poisoned-input check executed by a pack's canary suite."""

    name: str
    caught: bool
    gate: str
    message: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "caught": self.caught,
            "gate": self.gate,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class CanarySuiteReport:
    """The complete canary evidence package for one pack."""

    pack_name: str
    checks: tuple[CanaryCheckResult, ...]

    @property
    def all_caught(self) -> bool:
        return all(check.caught for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "pack_name": self.pack_name,
            "all_caught": self.all_caught,
            "checks": [check.to_dict() for check in self.checks],
        }


@runtime_checkable
class GatePack(Protocol):
    """Structural interface shared by every packaged gate."""

    name: str

    @property
    def spec(self) -> TemporalSpec: ...

    def rules(self, stage: Stage = Stage.INGEST) -> tuple[InvarRule, ...]: ...

    def validate(
        self,
        frame: pl.DataFrame,
        *,
        stage: Stage | str = Stage.INGEST,
        raise_on_failure: bool = False,
        **kwargs: object,
    ) -> GateReport: ...

    def run_canary_suite(self) -> CanarySuiteReport: ...

    def render_recipe_scaffold(self) -> str: ...

    def render_agents_md(self) -> str: ...


def _temporal_core(**kwargs: object) -> GatePack:
    from .temporal_core import TemporalCoreColMap, TemporalCorePack

    col_map = cast(TemporalCoreColMap | None, kwargs.pop("col_map", None))
    tolerance = cast(ToleranceProfile | None, kwargs.pop("tolerance", None))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported temporal-core pack options: {unexpected}")
    return TemporalCorePack(col_map=col_map, tolerance=tolerance)


_BUILTIN_PACKS: dict[str, Callable[..., GatePack]] = {
    "temporal-core": _temporal_core,
}

_REGISTERED_PACKS: dict[str, Callable[..., GatePack]] = {}


def _entry_points_by_name() -> dict[str, importlib.metadata.EntryPoint]:
    return {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
    }


def _resolve_factory(name: str) -> Callable[..., GatePack] | None:
    factory = _REGISTERED_PACKS.get(name)
    if factory is not None:
        return factory
    factory = _BUILTIN_PACKS.get(name)
    if factory is not None:
        return factory
    entry_point = _entry_points_by_name().get(name)
    if entry_point is None:
        return None
    return entry_point.load()


def register_pack(name: str, factory: Callable[..., GatePack]) -> None:
    """Register a third-party pack factory for the current process."""

    _REGISTERED_PACKS[name] = factory


def available_packs() -> tuple[str, ...]:
    """Return pack names visible through metadata or runtime registration.

    This reads distribution entry-point metadata without importing any plugin
    implementation module, so an L0-only process never loads commercial code
    just to list names.
    """

    names = set(_BUILTIN_PACKS) | set(_REGISTERED_PACKS) | set(_entry_points_by_name())
    return tuple(sorted(names))


def get_pack(name: str, **kwargs: object) -> GatePack:
    """Instantiate an installed, discovered, or registered pack by name."""

    factory = _resolve_factory(name)
    if factory is not None:
        return factory(**kwargs)
    if name in _KNOWN_L1_PACKS:
        raise PackNotInstalledError(
            f"gate pack {name!r} is part of the Cleanhatch L1 Pro Suite and is not "
            "installed in this build; install the private 'cleanhatch-l1' "
            "distribution and retry"
        )
    raise PackNotInstalledError(
        f"unknown gate pack {name!r}; available packs: {list(available_packs())}"
    )


__all__ = [
    "CanaryCheckResult",
    "CanarySuiteReport",
    "GatePack",
    "PackNotInstalledError",
    "available_packs",
    "get_pack",
    "register_pack",
]
