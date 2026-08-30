from pathlib import Path

from vipibench.dataio import write_jsonl
from vipibench.fixture import build_fixture_records
from vipibench.splits import audit_split_directory, split_records, write_splits
from vipibench.validation import validate_path


def test_fixture_is_valid_but_excluded_from_research_gates(tmp_path: Path) -> None:
    path = tmp_path / "fixture.jsonl"
    write_jsonl(path, build_fixture_records())
    result = validate_path(path)
    assert result.status == "PASS"
    assert result.record_count == 40
    assert result.research_gates["applicable"] is False


def test_fixture_fails_when_research_gates_are_requested(tmp_path: Path) -> None:
    path = tmp_path / "fixture.jsonl"
    write_jsonl(path, build_fixture_records())
    result = validate_path(path, require_research_gates=True)
    assert result.status == "FAIL"


def test_group_split_is_nonempty_and_auditable(tmp_path: Path) -> None:
    directory = tmp_path / "splits"
    write_splits(directory, split_records(build_fixture_records()))
    result = audit_split_directory(directory)
    assert result.status == "PASS"
    assert result.counts == {"train": 24, "dev": 8, "test": 8}
