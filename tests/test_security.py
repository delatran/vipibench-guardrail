from pathlib import Path

from vipibench.security import scan_secrets


def test_secret_scan_reports_path_without_secret_value(tmp_path: Path) -> None:
    (tmp_path / "safe.md").write_text("HF_TOKEN=\n", encoding="utf-8")
    assert scan_secrets(tmp_path)["status"] == "PASS"
    (tmp_path / "bad.txt").write_text("token=" + "hf_" + "a" * 30, encoding="utf-8")
    result = scan_secrets(tmp_path)
    assert result["status"] == "FAIL"
    assert result["findings"][0] == {
        "path": "bad.txt",
        "line": 1,
        "pattern": "huggingface_token",
    }


def test_secret_scan_finds_email_before_sentence_period(tmp_path: Path) -> None:
    synthetic_email = "admin" + "@possibly-real.example"
    (tmp_path / "draft.jsonl").write_text(
        f'{{"text":"Send to {synthetic_email}. Then stop."}}\n',
        encoding="utf-8",
    )

    result = scan_secrets(tmp_path)

    assert result["status"] == "FAIL"
    assert [finding["pattern"] for finding in result["findings"]] == ["email_address"]


def test_secret_scan_fingerprint_changes_with_scanned_content(tmp_path: Path) -> None:
    path = tmp_path / "safe.md"
    path.write_text("first\n", encoding="utf-8")
    first = scan_secrets(tmp_path)
    path.write_text("second\n", encoding="utf-8")
    second = scan_secrets(tmp_path)
    assert first["scanned_file_count"] == second["scanned_file_count"] == 1
    assert first["scanned_file_set_sha256"] != second["scanned_file_set_sha256"]


def test_secret_scan_ignores_generated_egg_info(tmp_path: Path) -> None:
    (tmp_path / "safe.md").write_text("stable\n", encoding="utf-8")
    first = scan_secrets(tmp_path)
    metadata = tmp_path / "example.egg-info"
    metadata.mkdir()
    (metadata / "SOURCES.txt").write_text("generated and changing\n", encoding="utf-8")
    second = scan_secrets(tmp_path)
    assert first["status"] == second["status"] == "PASS"
    assert first["scanned_file_count"] == second["scanned_file_count"] == 1
    assert first["scanned_file_set_sha256"] == second["scanned_file_set_sha256"]
