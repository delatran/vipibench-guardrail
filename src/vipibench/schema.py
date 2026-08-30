from __future__ import annotations

import hashlib
import unicodedata
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Label(StrEnum):
    INJECTION = "injection"
    BENIGN = "benign"


class SourceRole(StrEnum):
    USER = "user"
    RETRIEVED = "retrieved"
    TOOL = "tool"


class Delivery(StrEnum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    NONE = "none"


class LanguageForm(StrEnum):
    NATIVE_VI = "native_vi"
    TRANSLATED_VI = "translated_vi"
    UNACCENTED_VI = "unaccented_vi"
    CODE_MIX = "code_mix"


class LabelBasis(StrEnum):
    CONSTRUCTION = "construction"
    EXECUTABLE_ORACLE = "executable_oracle"


class DatasetRecord(BaseModel):
    """Detector-context record derived from an executable episode with immutable lineage."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    seed_id: str = Field(min_length=1)
    source_family: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    semantic_cluster: str = Field(min_length=1)
    label: Label
    label_basis: LabelBasis
    is_injection: bool
    source_role: SourceRole
    delivery: Delivery
    language_form: LanguageForm
    raw_text: str = Field(min_length=1)
    raw_text_sha256: str = Field(pattern="^[A-F0-9]{64}$")
    normalized_text: str = Field(min_length=1)
    system_text: str = ""
    user_text: str = ""
    retrieved_text: str = ""
    tool_text: str = ""
    domain: str = Field(min_length=1)
    benign_subtype: str | None = Field(default=None, min_length=1)
    source_uri: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    generator_revision: str = Field(min_length=1)
    oracle_version: str = Field(min_length=1)
    provenance_sha256: str = Field(pattern="^[A-F0-9]{64}$")
    created_at: datetime
    license_id: str = Field(min_length=1)
    license_decision: str = Field(pattern="^(allowed|internal_only|pending|denied)$")
    transformation_tool: str | None = Field(default=None, min_length=1)
    transformation_tool_revision: str | None = Field(default=None, min_length=1)
    transformation_prompt_sha256: str | None = Field(default=None, pattern="^[A-F0-9]{64}$")
    transformation_history: list[str] = Field(min_length=1)
    hard_negative: bool = False
    matched_pair_id: str | None = None
    obfuscation: str | None = None
    fixture_only: bool = False

    @model_validator(mode="after")
    def validate_semantics(self) -> DatasetRecord:
        expected_hash = hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest().upper()
        if self.raw_text_sha256 != expected_hash:
            raise ValueError("raw_text_sha256 must bind the exact UTF-8 raw_text")
        if self.created_at.utcoffset() is None:
            raise ValueError("created_at must include an explicit timezone offset")
        if self.is_injection != (self.label == Label.INJECTION):
            raise ValueError("is_injection must agree with label")
        expected_delivery = {
            SourceRole.USER: Delivery.DIRECT,
            SourceRole.RETRIEVED: Delivery.INDIRECT,
            SourceRole.TOOL: Delivery.INDIRECT,
        }
        if self.label == Label.INJECTION and self.delivery != expected_delivery[self.source_role]:
            raise ValueError("injection delivery must match its source role")
        if self.label == Label.BENIGN and self.delivery != Delivery.NONE:
            raise ValueError("benign records must use delivery=none")
        if self.hard_negative and self.label != Label.BENIGN:
            raise ValueError("hard_negative is valid only for benign records")
        if self.label == Label.BENIGN and not self.benign_subtype:
            raise ValueError("benign records require benign_subtype")
        if self.label == Label.INJECTION and self.benign_subtype is not None:
            raise ValueError("injection records cannot use benign_subtype")
        if self.language_form in {LanguageForm.TRANSLATED_VI, LanguageForm.UNACCENTED_VI} and (
            not self.transformation_tool or not self.transformation_tool_revision
        ):
            raise ValueError("translated/unaccented records require a transformation revision")
        if (
            self.language_form == LanguageForm.TRANSLATED_VI
            and not self.transformation_prompt_sha256
        ):
            raise ValueError("translated records require a transformation prompt hash")
        if not self.fixture_only and self.license_decision not in {"allowed", "internal_only"}:
            raise ValueError("research records require an approved internal-use license decision")
        if unicodedata.normalize("NFC", self.raw_text) != self.normalized_text:
            raise ValueError("normalized_text must be the NFC form of raw_text")
        return self

    def model_text(self, input_mode: str) -> str:
        if input_mode == "role_only":
            return f"[ROLE={self.source_role.value}]"
        if input_mode == "text_only":
            return self.normalized_text
        if input_mode == "text_role":
            return f"[ROLE={self.source_role.value.upper()}] {self.normalized_text}"
        raise ValueError(f"unsupported input mode: {input_mode}")
