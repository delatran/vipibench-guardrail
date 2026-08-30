import json
from pathlib import Path

from vipibench.compiler import (
    compile_confirmatory_holdout_path,
    load_executable_episodes,
    verify_confirmatory_holdout_package,
)

CONFIG = Path("configs/benchmark/exec_catalog.yaml")
FROZEN = Path("data/splits/frozen")


def test_confirmatory_holdout_is_new_sealed_and_verifiable(tmp_path: Path) -> None:
    output = tmp_path / "confirmatory_final"
    result = compile_confirmatory_holdout_path(CONFIG, FROZEN, output)

    assert result["status"] == "PASS", result["errors"]
    assert result["sealed"] is True
    assert result["episode_counts"] == {"train": 1440, "dev": 480, "test": 480}
    assert result["test_label_counts"] == {"benign": 240, "injection": 240}
    assert result["overlap_counts"] == {
        "episode_id_with_frozen": 0,
        "content_hash_with_frozen": 0,
        "family_with_train_or_dev": 0,
    }
    assert verify_confirmatory_holdout_package(CONFIG, FROZEN, output)["status"] == "PASS"

    original = load_executable_episodes(FROZEN / "test.jsonl")
    final = load_executable_episodes(output / "test.jsonl")
    assert {item.episode_id for item in original}.isdisjoint({item.episode_id for item in final})
    assert {item.content_sha256 for item in original}.isdisjoint(
        {item.content_sha256 for item in final}
    )


def test_confirmatory_holdout_verifier_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "confirmatory_final"
    compile_confirmatory_holdout_path(CONFIG, FROZEN, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_hashes"]["test"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_confirmatory_holdout_package(CONFIG, FROZEN, output)
    assert result["status"] == "FAIL"
    assert "holdout_split_hash_mismatch" in result["errors"]


def test_confirmatory_holdout_compiler_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "confirmatory_final"
    compile_confirmatory_holdout_path(CONFIG, FROZEN, output)

    try:
        compile_confirmatory_holdout_path(CONFIG, FROZEN, output)
    except FileExistsError as exc:
        assert "refusing to overwrite sealed holdout" in str(exc)
    else:
        raise AssertionError("sealed holdout overwrite was not rejected")
