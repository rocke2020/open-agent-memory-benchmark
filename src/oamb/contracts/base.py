"""Shared strict types for public OAMB contracts."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, ClassVar, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    WithJsonSchema,
    field_validator,
    model_validator,
)

CANONICAL_DECIMAL_PATTERN = r"^(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$"


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include an explicit timezone offset")
    return value


UtcDateTime = Annotated[datetime, AfterValidator(_require_timezone)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


def canonical_decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("decimal must be finite")
    if value < 0:
        raise ValueError("decimal must be non-negative")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _validate_decimal(value: object, info: ValidationInfo) -> Decimal:
    if info.mode == "json":
        if not isinstance(value, str):
            raise ValueError("decimal JSON value must be a string")
        try:
            parsed = Decimal(value)
        except Exception as exc:
            raise ValueError("invalid decimal string") from exc
        if value != canonical_decimal_text(parsed):
            raise ValueError("decimal JSON value must use canonical decimal notation")
        return parsed
    if not isinstance(value, Decimal):
        raise ValueError("decimal Python value must be Decimal")
    return Decimal(canonical_decimal_text(value))


NonNegativeDecimal = Annotated[
    Decimal,
    BeforeValidator(_validate_decimal),
    Field(ge=Decimal("0")),
    WithJsonSchema(
        {"type": "string", "pattern": CANONICAL_DECIMAL_PATTERN},
        mode="validation",
    ),
]


class StrictContract(BaseModel):
    """Base for immutable, strict, unknown-field-rejecting public records."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def require_integer_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def require_explicit_json_discriminators(self, info: ValidationInfo) -> Self:
        if info.mode == "json" and not {"schema_name", "schema_version"} <= self.model_fields_set:
            raise ValueError("JSON contract requires explicit schema_name and schema_version")
        return self
