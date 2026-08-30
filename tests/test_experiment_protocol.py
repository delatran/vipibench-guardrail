from copy import deepcopy
from pathlib import Path

import yaml

from vipibench.experiment_protocol import validate_exec_experiment_protocol

CONFIG_PATH = Path("configs/experiments/exec_system.yaml")


def _load_config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_config(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_live_exec_experiment_protocol_passes(tmp_path: Path) -> None:
    output = tmp_path / "protocol.json"
    result = validate_exec_experiment_protocol(Path.cwd(), CONFIG_PATH, output)
    assert result["status"] == "PASS", result["errors"]
    assert result["observed_test_counts"] == {"total": 480, "injection": 240, "benign": 240}
    assert result["static_trajectory_budget"] == 1920
    assert result["attack_search_trajectory_budget"] == 14400
    assert result["maximum_confirmatory_trajectories"] == 16320
    assert output.is_file()


def test_protocol_rejects_reduced_attack_search_budget(tmp_path: Path) -> None:
    config = deepcopy(_load_config())
    attack_search = config["attack_search_evaluation"]
    assert isinstance(attack_search, dict)
    attack_search["queries_per_strategy"] = 9
    attack_search["trajectories"] = 12960
    path = tmp_path / "protocol.yaml"
    _write_config(path, config)
    result = validate_exec_experiment_protocol(Path.cwd(), path)
    assert result["status"] == "FAIL"
    assert "queries_per_strategy_must_equal_10" in result["errors"]


def test_protocol_rejects_source_hash_tampering(tmp_path: Path) -> None:
    config = deepcopy(_load_config())
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    policy = bindings["policy_gate"]
    assert isinstance(policy, dict)
    policy["sha256"] = "0" * 64
    path = tmp_path / "protocol.yaml"
    _write_config(path, config)
    result = validate_exec_experiment_protocol(Path.cwd(), path)
    assert result["status"] == "FAIL"
    assert "binding_mismatch:policy_gate" in result["errors"]


def test_protocol_rejects_path_escape(tmp_path: Path) -> None:
    config = deepcopy(_load_config())
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    policy = bindings["policy_gate"]
    assert isinstance(policy, dict)
    policy["path"] = "../outside.py"
    path = tmp_path / "protocol.yaml"
    _write_config(path, config)
    result = validate_exec_experiment_protocol(Path.cwd(), path)
    assert result["status"] == "FAIL"
    assert "binding_mismatch:policy_gate" in result["errors"]
