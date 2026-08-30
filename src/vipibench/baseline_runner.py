from __future__ import annotations

from pathlib import Path

from vipibench.benchmark_partitions import load_benchmark_partitions
from vipibench.dataio import write_json
from vipibench.metrics import calibrate_thresholds, evaluate_predictions
from vipibench.modeling import load_yaml, predict_dataset, predict_records, train_tfidf


def run_tfidf_baseline(
    config_path: Path,
    *,
    project_root: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    root = project_root.resolve()
    config = load_yaml(config_path)
    train_result = train_tfidf(config_path, project_root=root)
    model_dir = Path(train_result.model_path).parent
    output_root = model_dir.parent
    dev_predictions = output_root / "dev_predictions.jsonl"
    test_predictions = output_root / "test_predictions.jsonl"
    thresholds = output_root / "thresholds.json"
    evaluation = output_root / "evaluation.json"
    source_hashes: dict[str, str] | None = None
    if config.get("contrast_dataset"):
        records, source_hashes = load_benchmark_partitions(
            (root / str(config["train_path"])).parent,
            Path(str(config["contrast_dataset"])),
        )
        dev_result = predict_records(model_dir, records["dev"], dev_predictions, split="dev")
        test_result = predict_records(model_dir, records["test"], test_predictions, split="test")
    else:
        dev_result = predict_dataset(
            model_dir,
            root / str(config["dev_path"]),
            dev_predictions,
            split="dev",
        )
        test_result = predict_dataset(
            model_dir,
            root / str(config["test_path"]),
            test_predictions,
            split="test",
        )
    threshold_result = calibrate_thresholds(dev_predictions, thresholds)
    evaluation_result = evaluate_predictions(test_predictions, thresholds, evaluation)
    result: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "config_path": str(config_path),
        "train": train_result.as_dict(),
        "dev_predictions": dev_result,
        "test_predictions": test_result,
        "thresholds": threshold_result,
        "evaluation": evaluation_result,
        "source_hashes": source_hashes,
        "research_claim_eligible": evaluation_result["research_claim_eligible"],
        "claim_boundary": (
            "This is a deterministic TF-IDF baseline on the frozen benchmark tracks. It does not "
            "substitute for transformer, OOD, system, or adaptive evidence."
        ),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
