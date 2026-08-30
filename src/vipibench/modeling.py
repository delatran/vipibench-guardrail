from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from vipibench.benchmark_partitions import load_benchmark_partitions
from vipibench.dataio import sha256_file, write_json
from vipibench.episode import ExecutableEpisode
from vipibench.exec_detector_data import (
    detector_text,
    load_executable_episodes,
    prediction_row,
)


@dataclass(frozen=True)
class TrainResult:
    status: str
    model_path: str
    manifest_path: str
    input_mode: str
    seed: int
    train_count: int
    fixture_only: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def build_tfidf_pipeline(config: dict[str, object]) -> Pipeline:
    word_range = tuple(int(value) for value in config.get("word_ngram_range", [1, 2]))
    char_range = tuple(int(value) for value in config.get("char_ngram_range", [3, 5]))
    max_features = int(config.get("max_features", 12000))
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=word_range, max_features=max_features // 2
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=char_range, max_features=max_features // 2
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        random_state=int(config.get("seed", 17)),
        class_weight=str(config.get("class_weight", "balanced")),
        max_iter=1000,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def train_tfidf(config_path: Path, *, project_root: Path | None = None) -> TrainResult:
    root = project_root or config_path.resolve().parents[2]
    config = load_yaml(config_path)
    if config.get("model_type") != "tfidf_logistic":
        raise ValueError("the local trainer currently accepts model_type=tfidf_logistic only")
    train_path = root / str(config["train_path"])
    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str]
    if config.get("contrast_dataset"):
        partitions, source_hashes = load_benchmark_partitions(
            train_path.parent,
            Path(str(config["contrast_dataset"])),
        )
        records = partitions["train"]
    else:
        records = load_executable_episodes(train_path)
        source_hashes = {"train": sha256_file(train_path)}
    input_mode = str(config.get("input_mode", "text_role"))
    model = build_tfidf_pipeline(config)
    model.fit(
        [detector_text(record, input_mode) for record in records],
        [1 if record.label.value == "injection" else 0 for record in records],
    )
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    manifest_path = output_dir / "train_manifest.json"
    manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "model_type": "tfidf_logistic",
        "input_mode": input_mode,
        "seed": int(config.get("seed", 17)),
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "source_hashes": source_hashes,
        "train_count": len(records),
        "fixture_only": False,
        "config": config,
    }
    write_json(manifest_path, manifest)
    return TrainResult(
        status="PASS",
        model_path=str(model_path),
        manifest_path=str(manifest_path),
        input_mode=input_mode,
        seed=int(config.get("seed", 17)),
        train_count=len(records),
        fixture_only=bool(manifest["fixture_only"]),
    )


def predict_dataset(
    model_dir: Path, dataset_path: Path, output_path: Path, *, split: str
) -> dict[str, object]:
    return predict_records(
        model_dir,
        load_executable_episodes(dataset_path),
        output_path,
        split=split,
        source_path=dataset_path,
    )


def predict_records(
    model_dir: Path,
    records: list[ExecutableEpisode],
    output_path: Path,
    *,
    split: str,
    source_path: Path | None = None,
) -> dict[str, object]:
    model = joblib.load(model_dir / "model.joblib")
    manifest = json.loads((model_dir / "train_manifest.json").read_text(encoding="utf-8"))
    input_mode = str(manifest["input_mode"])
    texts = [detector_text(record, input_mode) for record in records]
    started = time.perf_counter()
    scores = model.predict_proba(texts)[:, 1]
    elapsed = time.perf_counter() - started
    per_item_ms = (elapsed * 1000 / len(records)) if records else 0.0
    predictions = [
        prediction_row(record, float(score), split=split, latency_ms=per_item_ms)
        for record, score in zip(records, scores, strict=True)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return {
        "status": "PASS",
        "count": len(predictions),
        "predictions_path": str(output_path),
        "predictions_sha256": sha256_file(output_path),
        "source_path": str(source_path) if source_path is not None else None,
        "mean_batch_item_latency_ms": per_item_ms,
    }


def score_text(model_dir: Path, *, role: str, content: str) -> float:
    model = joblib.load(model_dir / "model.joblib")
    manifest = json.loads((model_dir / "train_manifest.json").read_text(encoding="utf-8"))
    input_mode = str(manifest["input_mode"])
    if input_mode == "role_only":
        text = f"[ROLE={role.upper()}]"
    elif input_mode == "text_only":
        text = content
    else:
        text = f"[ROLE={role.upper()}] {content}"
    return float(model.predict_proba([text])[0, 1])
