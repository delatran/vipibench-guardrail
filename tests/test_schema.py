import hashlib
import unicodedata

import pytest
from pydantic import ValidationError

from vipibench.schema import DatasetRecord


def make_record(**updates: object) -> DatasetRecord:
    raw_text = "Nội dung thử nghiệm."
    value: dict[str, object] = {
        "sample_id": "sample-1",
        "seed_id": "seed-1",
        "source_family": "fixture",
        "template_id": "template-1",
        "semantic_cluster": "cluster-1",
        "label": "benign",
        "label_basis": "construction",
        "is_injection": False,
        "source_role": "user",
        "delivery": "none",
        "language_form": "native_vi",
        "raw_text": raw_text,
        "raw_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest().upper(),
        "normalized_text": unicodedata.normalize("NFC", raw_text),
        "domain": "fixture",
        "benign_subtype": "ordinary_instruction",
        "source_uri": "fixture://test",
        "source_version": "fixture",
        "generator_revision": "fixture-generator",
        "oracle_version": "fixture-oracle",
        "provenance_sha256": "0" * 64,
        "created_at": "2026-01-01T00:00:00+00:00",
        "license_id": "internal",
        "license_decision": "internal_only",
        "transformation_history": ["fixture"],
        "fixture_only": True,
    }
    value.update(updates)
    return DatasetRecord.model_validate(value)


def test_valid_record_has_executable_lineage() -> None:
    record = make_record()
    assert record.label_basis.value == "construction"
    assert record.oracle_version == "fixture-oracle"


def test_injection_delivery_must_match_role() -> None:
    with pytest.raises(ValidationError, match="delivery must match"):
        make_record(
            label="injection",
            is_injection=True,
            benign_subtype=None,
            source_role="retrieved",
            delivery="direct",
        )


def test_hard_negative_cannot_be_injection() -> None:
    with pytest.raises(ValidationError, match="hard_negative"):
        make_record(
            label="injection",
            is_injection=True,
            benign_subtype=None,
            delivery="direct",
            hard_negative=True,
        )


def test_research_record_requires_resolved_license() -> None:
    with pytest.raises(ValidationError, match="approved internal-use"):
        make_record(fixture_only=False, license_decision="pending")


def test_raw_text_hash_and_nfc_are_bound() -> None:
    with pytest.raises(ValidationError, match="raw_text_sha256"):
        make_record(raw_text_sha256="0" * 64)
    with pytest.raises(ValidationError, match="NFC"):
        make_record(normalized_text=unicodedata.normalize("NFD", "Nội dung thử nghiệm."))
