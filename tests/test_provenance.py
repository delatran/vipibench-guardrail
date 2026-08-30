from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import vipibench.provenance as provenance_module
from vipibench.provenance import verify_provenance, verify_training_authorization


def _load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


@pytest.mark.private_integration
def test_live_frozen_benchmark_provenance_passes(tmp_path: Path) -> None:
    output = tmp_path / "provenance.json"
    result = verify_provenance(Path.cwd(), output_path=output)
    assert result["status"] == "PASS", result["errors"]
    assert result["observed_episode_count"] == 2400
    assert result["observed_context_count"] == 4800
    assert result["observed_external_source_count"] == 0
    assert result["observed_confirmatory_contract"] == {
        "test_episode_count": 480,
        "benign_count": 240,
        "injection_count": 240,
        "episode_id_overlap_with_frozen": 0,
        "content_hash_overlap_with_frozen": 0,
        "family_overlap_with_train_or_dev": 0,
    }
    assert output.is_file()


def test_provenance_rejects_unresolved_source(tmp_path: Path) -> None:
    ledger = deepcopy(_load_yaml(Path("data/provenance_ledger.yaml")))
    sources = ledger["sources"]
    assert isinstance(sources, list) and isinstance(sources[0], dict)
    sources[0]["unresolved"] = True
    path = tmp_path / "ledger.yaml"
    _write_yaml(path, ledger)
    result = verify_provenance(Path.cwd(), ledger_path=path)
    assert result["status"] == "FAIL"
    assert any(error.startswith("unresolved_sources:") for error in result["errors"])


def test_provenance_rejects_missing_confirmatory_binding(tmp_path: Path) -> None:
    ledger = deepcopy(_load_yaml(Path("data/provenance_ledger.yaml")))
    bindings = ledger["artifact_bindings"]
    assert isinstance(bindings, dict)
    bindings.pop("confirmatory_holdout_test")
    path = tmp_path / "ledger.yaml"
    _write_yaml(path, ledger)
    result = verify_provenance(Path.cwd(), ledger_path=path)
    assert result["status"] == "FAIL"
    assert "bindings_must_equal_required_set" in result["errors"]


@pytest.mark.private_integration
def test_live_internal_training_authorization_passes(tmp_path: Path) -> None:
    output = tmp_path / "authorization.json"
    result = verify_training_authorization(Path.cwd(), output_path=output)
    assert result["status"] == "PASS", result["errors"]
    assert result["internal_training_use_authorized"] is True
    assert result["paid_compute_authorized"] is False
    assert result["external_mutations_authorized"] is False
    assert output.is_file()


def test_authorization_rejects_paid_compute_or_upload_escalation(tmp_path: Path) -> None:
    decision = deepcopy(_load_yaml(Path("data/release_decision.yaml")))
    decision["paid_compute_authorized"] = True
    decision["upload_authorized"] = True
    path = tmp_path / "decision.yaml"
    _write_yaml(path, decision)
    result = verify_training_authorization(Path.cwd(), decision_path=path)
    assert result["status"] == "FAIL"
    assert any("paid_compute_authorized" in error for error in result["errors"])
    assert any("upload_authorized" in error for error in result["errors"])


def test_authorization_rejects_stale_secret_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        provenance_module,
        "scan_secrets",
        lambda _root: {
            "status": "PASS",
            "scanned_file_set_sha256": "0" * 64,
        },
    )
    result = verify_training_authorization(Path.cwd())
    assert result["status"] == "FAIL"
    assert "secret_scan_not_current" in result["errors"]
