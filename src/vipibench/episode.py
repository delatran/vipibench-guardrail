from __future__ import annotations

import copy
import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _hash_without(value: dict[str, Any], field: str) -> str:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    return _canonical_sha256(payload)


def bind_content_hash(value: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an episode payload bound to its canonical content hash."""

    payload = copy.deepcopy(value)
    payload["content_sha256"] = "0" * 64
    normalized = ExecutableEpisode.model_validate(
        payload,
        context={"skip_integrity_hash": True},
    ).model_dump(mode="json")
    normalized["content_sha256"] = _hash_without(normalized, "content_sha256")
    return normalized


def bind_template_hash(value: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(value)
    payload["template_sha256"] = "0" * 64
    normalized = TypedEpisodeTemplate.model_validate(
        payload,
        context={"skip_integrity_hash": True},
    ).model_dump(mode="json")
    normalized["template_sha256"] = _hash_without(normalized, "template_sha256")
    return normalized


class EpisodeLabel(StrEnum):
    INJECTION = "injection"
    BENIGN = "benign"


class ContextRole(StrEnum):
    USER = "user"
    RETRIEVED = "retrieved"
    TOOL = "tool"


class TrustLevel(StrEnum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class ValueKind(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"
    OBJECT = "object"
    ARRAY = "array"


class ToolOperation(StrEnum):
    READ_STATE = "read_state"
    SET_STATE = "set_state"
    APPEND_STATE = "append_state"
    DELETE_STATE = "delete_state"
    NOOP = "noop"


class PredicateKind(StrEnum):
    STATE_EQUALS = "state_equals"
    STATE_EXISTS = "state_exists"
    STATE_ABSENT = "state_absent"
    ANSWER_CONTAINS = "answer_contains"
    ANSWER_NOT_CONTAINS = "answer_not_contains"
    TOOL_CALLED = "tool_called"
    TOOL_NOT_CALLED = "tool_not_called"


class ArgumentRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ValueKind
    required: bool = True
    allowed_values: list[Any] = Field(default_factory=list)
    pattern: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_rule(self) -> ArgumentRule:
        if self.kind == ValueKind.ENUM and not self.allowed_values:
            raise ValueError("enum rules require allowed_values")
        if self.kind != ValueKind.ENUM and self.allowed_values:
            raise ValueError("allowed_values are valid only for enum rules")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum cannot exceed maximum")
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("min_length cannot exceed max_length")
        if self.pattern is not None:
            re.compile(self.pattern)
        json.dumps(self.allowed_values, ensure_ascii=False)
        return self

    def errors_for(self, value: Any) -> list[str]:
        errors: list[str] = []
        valid_type = {
            ValueKind.STRING: isinstance(value, str),
            ValueKind.INTEGER: isinstance(value, int) and not isinstance(value, bool),
            ValueKind.NUMBER: isinstance(value, int | float) and not isinstance(value, bool),
            ValueKind.BOOLEAN: isinstance(value, bool),
            ValueKind.ENUM: value in self.allowed_values,
            ValueKind.OBJECT: isinstance(value, dict),
            ValueKind.ARRAY: isinstance(value, list),
        }[self.kind]
        if not valid_type:
            return [f"type:{self.kind.value}"]
        if isinstance(value, str):
            if self.pattern is not None and re.fullmatch(self.pattern, value) is None:
                errors.append("pattern")
            if self.min_length is not None and len(value) < self.min_length:
                errors.append("min_length")
            if self.max_length is not None and len(value) > self.max_length:
                errors.append("max_length")
        if isinstance(value, int | float) and not isinstance(value, bool):
            if self.minimum is not None and value < self.minimum:
                errors.append("minimum")
            if self.maximum is not None and value > self.maximum:
                errors.append("maximum")
        return errors


class SandboxToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    description: str = Field(min_length=1)
    operation: ToolOperation
    consequential: bool
    parameters: dict[str, ArgumentRule]
    allow_extra_arguments: Literal[False] = False
    path_argument: str | None = None
    value_argument: str | None = None
    allowed_path_prefixes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_operation_contract(self) -> SandboxToolSpec:
        state_operations = {
            ToolOperation.READ_STATE,
            ToolOperation.SET_STATE,
            ToolOperation.APPEND_STATE,
            ToolOperation.DELETE_STATE,
        }
        if self.operation in state_operations:
            if not self.path_argument or self.path_argument not in self.parameters:
                raise ValueError("state operations require a declared path_argument")
            if not self.allowed_path_prefixes:
                raise ValueError("state operations require allowed_path_prefixes")
        if self.operation in {
            ToolOperation.SET_STATE,
            ToolOperation.APPEND_STATE,
        } and (not self.value_argument or self.value_argument not in self.parameters):
            raise ValueError("write operations require a declared value_argument")
        for prefix in self.allowed_path_prefixes:
            validate_state_path(prefix)
        return self


class AuthorizedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    tool: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    argument_rules: dict[str, ArgumentRule]
    trusted_source: Literal["trusted_user_goal"] = "trusted_user_goal"
    rationale: str = Field(min_length=1)


class ContextChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    role: ContextRole
    trust: TrustLevel
    content: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    attack_intent: bool = False
    source_uri: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    license_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_content_hash(self) -> ContextChunk:
        observed = hashlib.sha256(self.content.encode("utf-8")).hexdigest().upper()
        if observed != self.content_sha256:
            raise ValueError("content_sha256 must bind the exact UTF-8 content")
        if self.attack_intent and self.trust != TrustLevel.UNTRUSTED:
            raise ValueError("attack intent must be in untrusted context")
        return self


class OutcomePredicate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicate_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    kind: PredicateKind
    path: str | None = None
    expected: Any = None
    text: str | None = None
    tool: str | None = None
    case_sensitive: bool = False

    @model_validator(mode="after")
    def validate_shape(self) -> OutcomePredicate:
        state_kinds = {
            PredicateKind.STATE_EQUALS,
            PredicateKind.STATE_EXISTS,
            PredicateKind.STATE_ABSENT,
        }
        answer_kinds = {
            PredicateKind.ANSWER_CONTAINS,
            PredicateKind.ANSWER_NOT_CONTAINS,
        }
        tool_kinds = {PredicateKind.TOOL_CALLED, PredicateKind.TOOL_NOT_CALLED}
        if self.kind in state_kinds:
            if self.path is None:
                raise ValueError("state predicates require path")
            validate_state_path(self.path)
        if self.kind in answer_kinds and not self.text:
            raise ValueError("answer predicates require text")
        if self.kind in tool_kinds and not self.tool:
            raise ValueError("tool predicates require tool")
        json.dumps(self.expected, ensure_ascii=False)
        return self


class SecurityOracleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invariants: list[OutcomePredicate] = Field(default_factory=list)
    canaries: list[str] = Field(default_factory=list)
    attack_objectives: list[OutcomePredicate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_security_predicates(self) -> SecurityOracleSpec:
        invariant_kinds = {
            PredicateKind.STATE_EQUALS,
            PredicateKind.STATE_EXISTS,
            PredicateKind.STATE_ABSENT,
            PredicateKind.ANSWER_NOT_CONTAINS,
            PredicateKind.TOOL_NOT_CALLED,
        }
        if any(item.kind not in invariant_kinds for item in self.invariants):
            raise ValueError("security invariant uses an unsupported predicate kind")
        if len(set(self.canaries)) != len(self.canaries):
            raise ValueError("canaries must be unique")
        if any(not value for value in self.canaries):
            raise ValueError("canaries cannot be empty")
        return self


class UtilityOracleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_outcomes: list[OutcomePredicate] = Field(min_length=1)
    require_all: Literal[True] = True


class EpisodeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    template_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    seed_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    language_form: Literal["native_vi", "translated_vi", "unaccented_vi", "code_mix"]
    native_vietnamese: bool
    hard_negative: bool = False
    matched_pair_id: str | None = None
    generator_id: str = Field(min_length=1)
    generator_revision: str = Field(min_length=1)
    transformation_history: list[str] = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    template_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    split: Literal["unassigned", "train", "dev", "test"] = "unassigned"


class TypedEpisodeTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    template_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    family_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    domain: str = Field(min_length=1)
    variables: dict[str, ArgumentRule]
    blueprint: dict[str, Any]
    template_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")

    @model_validator(mode="after")
    def validate_hash(self, info: ValidationInfo) -> TypedEpisodeTemplate:
        payload = self.model_dump(mode="json")
        observed = _hash_without(payload, "template_sha256")
        skip_hash = bool(info.context and info.context.get("skip_integrity_hash"))
        if not skip_hash and observed != self.template_sha256:
            raise ValueError("template_sha256 mismatch")
        return self


class ExecutableEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    episode_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,95}$")
    label: EpisodeLabel
    trusted_user_goal: str = Field(min_length=1)
    initial_state: dict[str, Any]
    tools: list[SandboxToolSpec] = Field(min_length=1)
    authorized_actions: list[AuthorizedAction]
    context: list[ContextChunk] = Field(min_length=1)
    security: SecurityOracleSpec
    utility: UtilityOracleSpec
    metadata: EpisodeMetadata
    content_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")

    @model_validator(mode="after")
    def validate_episode_contract(self, info: ValidationInfo) -> ExecutableEpisode:
        json.dumps(self.initial_state, ensure_ascii=False)
        _require_unique([item.name for item in self.tools], "tool names")
        _require_unique([item.action_id for item in self.authorized_actions], "action IDs")
        _require_unique([item.chunk_id for item in self.context], "context chunk IDs")
        predicates = [
            *self.security.invariants,
            *self.security.attack_objectives,
            *self.utility.expected_outcomes,
        ]
        _require_unique([item.predicate_id for item in predicates], "predicate IDs")

        tool_map = {item.name: item for item in self.tools}
        for action in self.authorized_actions:
            if action.tool not in tool_map:
                raise ValueError(f"authorized action references unknown tool: {action.tool}")
            unknown = sorted(set(action.argument_rules) - set(tool_map[action.tool].parameters))
            if unknown:
                raise ValueError(f"authorized action has unknown arguments: {unknown}")

        attack_chunks = [item for item in self.context if item.attack_intent]
        if self.label == EpisodeLabel.INJECTION:
            if not attack_chunks or not self.security.attack_objectives:
                raise ValueError("injection episodes require attack context and objectives")
        elif attack_chunks or self.security.attack_objectives:
            raise ValueError("benign episodes cannot contain attack intent or objectives")
        if self.metadata.hard_negative and self.label != EpisodeLabel.BENIGN:
            raise ValueError("hard_negative metadata requires a benign label")

        payload = self.model_dump(mode="json")
        observed = _hash_without(payload, "content_sha256")
        skip_hash = bool(info.context and info.context.get("skip_integrity_hash"))
        if not skip_hash and observed != self.content_sha256:
            raise ValueError("content_sha256 mismatch")
        return self


def _require_unique(values: list[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def validate_state_path(path: str) -> tuple[str, ...]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", path):
        raise ValueError(f"invalid state path: {path}")
    return tuple(path.split("."))


def export_episode_schema(output_path: Path) -> dict[str, object]:
    schema = ExecutableEpisode.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://example.invalid/vipibench/executable-episode.schema.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"status": "PASS", "path": str(output_path), "sha256": _canonical_sha256(schema)}
