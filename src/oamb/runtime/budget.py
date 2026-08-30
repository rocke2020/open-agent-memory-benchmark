"""Atomic reservation and accounting for approved runtime ceilings."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from threading import Lock


class BudgetExceededError(ValueError):
    """Raised before dispatch when a reservation would cross a ceiling."""


class ReservationStateError(ValueError):
    """Raised when a reservation is missing or already terminal."""


ResourceAmount = tuple[str, Decimal, str]
ProviderAmount = tuple[str, str, str, int]


@dataclass(frozen=True, slots=True)
class BudgetAmount:
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: Decimal = Decimal("0")
    cost: Decimal = Decimal("0")
    resources: tuple[ResourceAmount, ...] = ()
    provider_units: tuple[ProviderAmount, ...] = ()

    def __post_init__(self) -> None:
        scalar_values = (
            self.attempts,
            self.input_tokens,
            self.output_tokens,
            self.wall_seconds,
            self.cost,
        )
        if any(value < 0 for value in scalar_values):
            raise ValueError("budget amounts must be non-negative")
        resources = tuple(sorted(item for item in self.resources if item[1] != 0))
        providers = tuple(sorted(item for item in self.provider_units if item[3] != 0))
        if len({item[0] for item in resources}) != len(resources):
            raise ValueError("resource dimensions must be unique")
        if len({item[:3] for item in providers}) != len(providers):
            raise ValueError("provider unit dimensions must be unique")
        if any(value < 0 or not unit for _, value, unit in resources):
            raise ValueError("resource amounts and units must be valid")
        if any(value < 0 or not all(item[:3]) for item in providers for value in (item[3],)):
            raise ValueError("provider units must be valid")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "provider_units", providers)

    @classmethod
    def zero(cls) -> BudgetAmount:
        return cls()

    def add(self, other: BudgetAmount) -> BudgetAmount:
        return BudgetAmount(
            attempts=self.attempts + other.attempts,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            cost=self.cost + other.cost,
            resources=_combine_resources(self.resources, other.resources, subtract=False),
            provider_units=_combine_providers(
                self.provider_units, other.provider_units, subtract=False
            ),
        )

    def subtract(self, other: BudgetAmount) -> BudgetAmount:
        return BudgetAmount(
            attempts=self.attempts - other.attempts,
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            wall_seconds=self.wall_seconds - other.wall_seconds,
            cost=self.cost - other.cost,
            resources=_combine_resources(self.resources, other.resources, subtract=True),
            provider_units=_combine_providers(
                self.provider_units, other.provider_units, subtract=True
            ),
        )


@dataclass(frozen=True, slots=True)
class BudgetCeiling:
    maximum: BudgetAmount
    currency: str | None

    @classmethod
    def zero(cls, *, currency: str | None = None) -> BudgetCeiling:
        return cls(maximum=BudgetAmount.zero(), currency=currency)


@dataclass(frozen=True, slots=True)
class BudgetOwnerAllocation:
    owner_id: str
    maximum: BudgetAmount

    def __post_init__(self) -> None:
        if not self.owner_id:
            raise ValueError("budget allocation requires an owner identity")


@dataclass(frozen=True, slots=True)
class ReservationRequest:
    reservation_id: str
    maximum: BudgetAmount
    role_binding_id: str | None = None
    provider_operation_ceiling_id: str | None = None
    owner_allocations: tuple[BudgetOwnerAllocation, ...] = ()

    def __post_init__(self) -> None:
        owners = (self.role_binding_id, self.provider_operation_ceiling_id)
        legacy_owner_count = sum(owner is not None for owner in owners)
        if self.owner_allocations:
            if legacy_owner_count:
                raise ValueError("multi-owner reservation cannot carry a legacy owner")
            owner_ids = tuple(allocation.owner_id for allocation in self.owner_allocations)
            if len(set(owner_ids)) != len(owner_ids):
                raise ValueError("multi-owner reservation contains duplicate owners")
        elif legacy_owner_count != 1:
            raise ValueError("reservation requires exactly one discriminated budget owner")

    @property
    def allocations(self) -> tuple[BudgetOwnerAllocation, ...]:
        if self.owner_allocations:
            return self.owner_allocations
        return (
            BudgetOwnerAllocation(
                self.role_binding_id or self.provider_operation_ceiling_id or "",
                self.maximum,
            ),
        )


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    reserved: BudgetAmount
    committed: BudgetAmount


@dataclass(frozen=True, slots=True)
class _Reservation:
    request: ReservationRequest
    state: str


class BudgetLedger:
    """Single-owner ledger whose lock makes parent and role checks indivisible."""

    def __init__(
        self,
        parent_ceiling: BudgetCeiling,
        *,
        role_ceilings: dict[str, BudgetCeiling],
        provider_operation_ceilings: dict[str, BudgetCeiling] | None = None,
    ) -> None:
        operation_ceilings = dict(provider_operation_ceilings or {})
        duplicate_owner_ids = set(role_ceilings) & set(operation_ceilings)
        if duplicate_owner_ids:
            raise ValueError("budget owner identities must be unique across owner kinds")
        owner_ceilings = dict(role_ceilings) | operation_ceilings
        mismatched_currencies = {
            owner: ceiling.currency
            for owner, ceiling in owner_ceilings.items()
            if ceiling.currency != parent_ceiling.currency
        }
        if mismatched_currencies:
            raise ValueError("role budget currency must match the parent currency")
        self._parent_ceiling = parent_ceiling
        self._owner_ceilings = owner_ceilings
        self._reserved = BudgetAmount.zero()
        self._committed = BudgetAmount.zero()
        self._owner_reserved = {owner: BudgetAmount.zero() for owner in owner_ceilings}
        self._owner_committed = {owner: BudgetAmount.zero() for owner in owner_ceilings}
        self._reservations: dict[str, _Reservation] = {}
        self._lock = Lock()

    def reserve(self, request: ReservationRequest) -> None:
        with self._lock:
            if not request.reservation_id or request.reservation_id in self._reservations:
                raise ReservationStateError("reservation identity is empty or already exists")
            parent_candidate = self._committed.add(self._reserved).add(request.maximum)
            _require_fits("parent", parent_candidate, self._parent_ceiling.maximum)
            owner_candidates: list[tuple[str, BudgetAmount]] = []
            for allocation in request.allocations:
                owner = allocation.owner_id
                owner_ceiling = self._owner_ceilings.get(owner)
                if owner_ceiling is None:
                    raise BudgetExceededError(f"missing budget-owner ceiling: {owner}")
                owner_candidate = (
                    self._owner_committed[owner]
                    .add(self._owner_reserved[owner])
                    .add(allocation.maximum)
                )
                _require_fits(owner, owner_candidate, owner_ceiling.maximum)
                owner_candidates.append((owner, owner_candidate))
            self._reserved = self._reserved.add(request.maximum)
            for owner, candidate in owner_candidates:
                self._owner_reserved[owner] = candidate.subtract(self._owner_committed[owner])
            self._reservations[request.reservation_id] = _Reservation(request, "reserved")

    def commit(self, reservation_id: str, *, observed: BudgetAmount) -> None:
        with self._lock:
            reservation = self._active(reservation_id)
            _require_fits("observed", observed, reservation.request.maximum)
            self._reserved = self._reserved.subtract(reservation.request.maximum)
            self._committed = self._committed.add(observed)
            for allocation in reservation.request.allocations:
                owner = allocation.owner_id
                self._owner_reserved[owner] = self._owner_reserved[owner].subtract(
                    allocation.maximum
                )
                self._owner_committed[owner] = self._owner_committed[owner].add(
                    allocation.maximum if reservation.request.owner_allocations else observed
                )
            self._reservations[reservation_id] = _Reservation(reservation.request, "committed")

    def cancel_before_dispatch(self, reservation_id: str) -> None:
        with self._lock:
            reservation = self._active(reservation_id)
            self._reserved = self._reserved.subtract(reservation.request.maximum)
            for allocation in reservation.request.allocations:
                owner = allocation.owner_id
                self._owner_reserved[owner] = self._owner_reserved[owner].subtract(
                    allocation.maximum
                )
            self._reservations[reservation_id] = _Reservation(reservation.request, "cancelled")

    def mark_unknown(self, reservation_id: str) -> None:
        with self._lock:
            reservation = self._active(reservation_id)
            self._reservations[reservation_id] = _Reservation(reservation.request, "unknown")

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(reserved=self._reserved, committed=self._committed)

    def _active(self, reservation_id: str) -> _Reservation:
        try:
            reservation = self._reservations[reservation_id]
        except KeyError as exc:
            raise ReservationStateError(f"unknown reservation: {reservation_id}") from exc
        if reservation.state != "reserved":
            raise ReservationStateError(
                f"reservation is not active: {reservation_id} ({reservation.state})"
            )
        return reservation


def _combine_resources(
    left: tuple[ResourceAmount, ...],
    right: tuple[ResourceAmount, ...],
    *,
    subtract: bool,
) -> tuple[ResourceAmount, ...]:
    values = {dimension: (value, unit) for dimension, value, unit in left}
    for dimension, value, unit in right:
        previous, previous_unit = values.get(dimension, (Decimal("0"), unit))
        if previous_unit != unit:
            raise ValueError(f"resource unit mismatch: {dimension}")
        values[dimension] = (previous - value if subtract else previous + value, unit)
    return tuple((dimension, value, unit) for dimension, (value, unit) in sorted(values.items()))


def _combine_providers(
    left: tuple[ProviderAmount, ...],
    right: tuple[ProviderAmount, ...],
    *,
    subtract: bool,
) -> tuple[ProviderAmount, ...]:
    values = {(provider, operation, unit): value for provider, operation, unit, value in left}
    for provider, operation, unit, value in right:
        key = (provider, operation, unit)
        previous = values.get(key, 0)
        values[key] = previous - value if subtract else previous + value
    return tuple((*key, value) for key, value in sorted(values.items()))


def _require_fits(label: str, value: BudgetAmount, maximum: BudgetAmount) -> None:
    for field_name in (
        "attempts",
        "input_tokens",
        "output_tokens",
        "wall_seconds",
        "cost",
    ):
        if getattr(value, field_name) > getattr(maximum, field_name):
            raise BudgetExceededError(f"{label} {field_name} ceiling exceeded")
    maximum_resources = {dimension: (amount, unit) for dimension, amount, unit in maximum.resources}
    for dimension, amount, unit in value.resources:
        limit, limit_unit = maximum_resources.get(dimension, (Decimal("0"), unit))
        if unit != limit_unit or amount > limit:
            raise BudgetExceededError(f"{label} resource {dimension} ceiling exceeded")
    maximum_providers = {
        (provider, operation, unit): amount
        for provider, operation, unit, amount in maximum.provider_units
    }
    for provider, operation, unit, provider_amount in value.provider_units:
        provider_limit = maximum_providers.get((provider, operation, unit), 0)
        if provider_amount > provider_limit:
            raise BudgetExceededError(
                f"{label} provider {provider}/{operation}/{unit} ceiling exceeded"
            )
