from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vipibench.compiler import load_compile_config, load_executable_episodes
from vipibench.dataio import canonical_json, sha256_file, write_json
from vipibench.episode import ExecutableEpisode

SPLITS = ("train", "dev", "test")


def freeze_exec_splits(
    dataset_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    near_duplicate_threshold: float = 0.90,
) -> dict[str, Any]:
    _require_empty_output_dir(output_dir)
    episodes = load_executable_episodes(dataset_path)
    by_split = {
        split: sorted(
            [item for item in episodes if item.metadata.split == split],
            key=lambda item: item.episode_id,
        )
        for split in SPLITS
    }
    for split, items in by_split.items():
        _write_episode_jsonl(output_dir / f"{split}.jsonl", items)

    split_hashes = {
        split: sha256_file(output_dir / f"{split}.jsonl") for split in SPLITS
    }
    split_counts = {split: len(items) for split, items in by_split.items()}
    test_episode_ids = [item.episode_id for item in by_split["test"]]
    del episodes
    del by_split

    audit = audit_exec_splits(
        output_dir,
        dataset_path,
        config_path,
        near_duplicate_threshold=near_duplicate_threshold,
        output_path=output_dir / "split_audit.json",
        holdout_output_path=output_dir / "holdout_manifest.json",
    )
    if audit["status"] != "PASS":
        raise ValueError(f"frozen split audit failed: {audit['errors']}")
    manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "frozen": True,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "assignment_source": "episode.metadata.split",
        "family_split_counts": audit["family_counts"],
        "split_episode_counts": split_counts,
        "split_hashes": split_hashes,
        "test_episode_id_sha256": _hash_lines(test_episode_ids),
        "test_split_sha256": split_hashes["test"],
        "split_audit_sha256": sha256_file(output_dir / "split_audit.json"),
        "holdout_manifest_sha256": sha256_file(output_dir / "holdout_manifest.json"),
        "tuning_prohibition": (
            "Development data may tune thresholds; frozen test and alternate holdout folds may not."
        ),
    }
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def audit_exec_splits(
    split_dir: Path,
    dataset_path: Path,
    config_path: Path,
    *,
    near_duplicate_threshold: float = 0.90,
    output_path: Path | None = None,
    holdout_output_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    split_episodes: dict[str, list[ExecutableEpisode]] = {}
    try:
        load_compile_config(config_path)
        for split in SPLITS:
            split_episodes[split] = load_executable_episodes(split_dir / f"{split}.jsonl")
    except Exception as exc:
        result = {
            "schema_version": "1.0.0",
            "status": "FAIL",
            "errors": [f"split_input_error:{type(exc).__name__}:{exc}"],
        }
        if output_path is not None:
            write_json(output_path, result)
        return result

    expected_counts = {"train": 1440, "dev": 480, "test": 480}
    observed_counts = {split: len(items) for split, items in split_episodes.items()}
    _check(errors, observed_counts == expected_counts, "split_episode_counts_mismatch")

    family_owner: dict[str, str] = {}
    template_owner: dict[str, str] = {}
    pair_owner: dict[str, str] = {}
    episode_owner: dict[str, str] = {}
    context_hash_owner: dict[str, tuple[str, str]] = {}
    family_counts: dict[str, int] = {}
    all_episodes: list[ExecutableEpisode] = []
    for split, items in split_episodes.items():
        families = {item.metadata.family_id for item in items}
        family_counts[split] = len(families)
        for episode in items:
            all_episodes.append(episode)
            _bind_owner(errors, family_owner, episode.metadata.family_id, split, "family")
            _bind_owner(errors, template_owner, episode.metadata.template_id, split, "template")
            if episode.metadata.matched_pair_id:
                _bind_owner(
                    errors,
                    pair_owner,
                    episode.metadata.matched_pair_id,
                    split,
                    "matched_pair",
                )
            _bind_owner(errors, episode_owner, episode.episode_id, split, "episode_id")
            retrieved = _retrieved_context(episode)
            previous = context_hash_owner.setdefault(
                retrieved.content_sha256,
                (split, episode.episode_id),
            )
            if previous[0] != split:
                errors.append(
                    "cross_split_exact_context_duplicate:"
                    f"{previous[1]}:{episode.episode_id}"
                )

    _check(
        errors,
        family_counts == {"train": 48, "dev": 16, "test": 16},
        "split_family_counts_mismatch",
    )
    source_ids = _read_source_episode_ids(dataset_path)
    _check(
        errors,
        set(episode_owner) == source_ids and len(episode_owner) == len(source_ids) == 2400,
        "split_source_reconciliation_mismatch",
    )

    near_duplicate = _near_duplicate_audit(
        all_episodes,
        threshold=near_duplicate_threshold,
    )
    if near_duplicate["violation_count"]:
        errors.append("cross_split_near_duplicate_threshold_exceeded")

    holdouts = _build_holdout_manifest(all_episodes, config_path, dataset_path)
    if holdouts["status"] != "PASS":
        errors.extend(f"holdout:{item}" for item in holdouts["errors"])

    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "split_dir": str(split_dir),
        "dataset_sha256": sha256_file(dataset_path),
        "config_sha256": sha256_file(config_path),
        "episode_counts": observed_counts,
        "family_counts": family_counts,
        "unique_episode_count": len(episode_owner),
        "unique_family_count": len(family_owner),
        "unique_template_count": len(template_owner),
        "matched_pair_group_count": len(pair_owner),
        "exact_cross_split_context_duplicate_count": sum(
            item.startswith("cross_split_exact_context_duplicate") for item in errors
        ),
        "near_duplicate_audit": near_duplicate,
        "split_hashes": {
            split: sha256_file(split_dir / f"{split}.jsonl") for split in SPLITS
        },
        "claim_boundary": (
            "PASS proves frozen group isolation, exact reconciliation, and the declared TF-IDF "
            "near-duplicate threshold. It does not prove semantic independence."
        ),
    }
    if output_path is not None:
        write_json(output_path, result)
    if holdout_output_path is not None:
        write_json(holdout_output_path, holdouts)
    return result


def verify_frozen_split_package(
    split_dir: Path,
    dataset_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Verify that frozen records and their evidence envelope are mutually bound."""

    try:
        manifest = _read_json_object(split_dir / "split_manifest.json")
        audit = _read_json_object(split_dir / "split_audit.json")
        holdout = _read_json_object(split_dir / "holdout_manifest.json")
        errors = _frozen_package_errors(
            split_dir,
            dataset_path,
            config_path,
            manifest,
            audit,
            holdout,
        )
    except Exception as exc:
        errors = [f"frozen_package_input_error:{type(exc).__name__}:{exc}"]
    return {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "split_dir": str(split_dir),
        "claim_boundary": (
            "PASS proves byte-level binding among the frozen records, audit, holdout evidence, "
            "dataset, configuration, and split manifest."
        ),
    }


def seal_frozen_split_package(
    split_dir: Path,
    dataset_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Refresh evidence hashes only after every other frozen binding validates."""

    manifest_path = split_dir / "split_manifest.json"
    manifest = _read_json_object(manifest_path)
    audit = _read_json_object(split_dir / "split_audit.json")
    holdout = _read_json_object(split_dir / "holdout_manifest.json")
    candidate = dict(manifest)
    candidate["split_audit_sha256"] = sha256_file(split_dir / "split_audit.json")
    candidate["holdout_manifest_sha256"] = sha256_file(
        split_dir / "holdout_manifest.json"
    )
    errors = _frozen_package_errors(
        split_dir,
        dataset_path,
        config_path,
        candidate,
        audit,
        holdout,
    )
    if errors:
        raise ValueError(f"refusing to seal invalid frozen split package: {errors}")
    write_json(manifest_path, candidate)
    return verify_frozen_split_package(split_dir, dataset_path, config_path)


def _frozen_package_errors(
    split_dir: Path,
    dataset_path: Path,
    config_path: Path,
    manifest: dict[str, Any],
    audit: dict[str, Any],
    holdout: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    split_hashes = {
        split: sha256_file(split_dir / f"{split}.jsonl") for split in SPLITS
    }
    dataset_sha256 = sha256_file(dataset_path)
    config_sha256 = sha256_file(config_path)
    _check(errors, manifest.get("status") == "PASS", "manifest_status_not_pass")
    _check(errors, manifest.get("frozen") is True, "manifest_not_frozen")
    _check(
        errors,
        manifest.get("dataset_sha256") == dataset_sha256,
        "manifest_dataset_binding_mismatch",
    )
    _check(
        errors,
        manifest.get("config_sha256") == config_sha256,
        "manifest_config_binding_mismatch",
    )
    _check(
        errors,
        manifest.get("split_hashes") == split_hashes,
        "manifest_split_hashes_mismatch",
    )
    _check(
        errors,
        manifest.get("test_split_sha256") == split_hashes["test"],
        "manifest_test_binding_mismatch",
    )
    _check(errors, audit.get("status") == "PASS", "split_audit_status_not_pass")
    _check(
        errors,
        audit.get("dataset_sha256") == dataset_sha256,
        "split_audit_dataset_binding_mismatch",
    )
    _check(
        errors,
        audit.get("config_sha256") == config_sha256,
        "split_audit_config_binding_mismatch",
    )
    _check(
        errors,
        audit.get("split_hashes") == split_hashes,
        "split_audit_record_binding_mismatch",
    )
    _check(errors, holdout.get("status") == "PASS", "holdout_status_not_pass")
    _check(
        errors,
        holdout.get("dataset_sha256") == dataset_sha256,
        "holdout_dataset_binding_mismatch",
    )
    _check(
        errors,
        manifest.get("split_audit_sha256")
        == sha256_file(split_dir / "split_audit.json"),
        "manifest_split_audit_binding_mismatch",
    )
    _check(
        errors,
        manifest.get("holdout_manifest_sha256")
        == sha256_file(split_dir / "holdout_manifest.json"),
        "manifest_holdout_binding_mismatch",
    )
    return errors


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _near_duplicate_audit(
    episodes: list[ExecutableEpisode],
    *,
    threshold: float,
) -> dict[str, Any]:
    texts = [_retrieved_context(item).content for item in episodes]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    indices = {
        split: np.array(
            [index for index, item in enumerate(episodes) if item.metadata.split == split],
            dtype=int,
        )
        for split in SPLITS
    }
    top_pairs: list[dict[str, Any]] = []
    violation_count = 0
    maximum = 0.0
    for left_split, right_split in (("train", "dev"), ("train", "test"), ("dev", "test")):
        similarities = cosine_similarity(
            matrix[indices[left_split]],
            matrix[indices[right_split]],
            dense_output=True,
        )
        maximum = max(maximum, float(similarities.max(initial=0.0)))
        violation_positions = np.argwhere(similarities >= threshold)
        violation_count += int(len(violation_positions))
        flat = similarities.ravel()
        top_count = min(3, flat.size)
        top_indices = np.argpartition(flat, -top_count)[-top_count:]
        for flat_index in top_indices[np.argsort(flat[top_indices])[::-1]]:
            left_index, right_index = np.unravel_index(flat_index, similarities.shape)
            left_episode = episodes[indices[left_split][left_index]]
            right_episode = episodes[indices[right_split][right_index]]
            top_pairs.append(
                {
                    "similarity": float(similarities[left_index, right_index]),
                    "left_split": left_split,
                    "left_episode_id": left_episode.episode_id,
                    "right_split": right_split,
                    "right_episode_id": right_episode.episode_id,
                }
            )
    top_pairs.sort(key=lambda item: item["similarity"], reverse=True)
    return {
        "method": "character_tfidf_cosine",
        "analyzer": "char_wb",
        "ngram_range": [3, 5],
        "threshold": threshold,
        "maximum_cross_split_similarity": maximum,
        "violation_count": violation_count,
        "top_cross_split_pairs": top_pairs[:10],
    }


def _build_holdout_manifest(
    episodes: list[ExecutableEpisode],
    config_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    config = load_compile_config(config_path)
    errors: list[str] = []
    mechanisms_by_split = {
        split: sorted(
            item.mechanism_id for item in config.mechanisms if item.split == split
        )
        for split in SPLITS
    }
    _check(
        errors,
        not set(mechanisms_by_split["train"])
        & (set(mechanisms_by_split["dev"]) | set(mechanisms_by_split["test"])),
        "attack_mechanism_overlap",
    )
    family_ids_by_split = {
        split: sorted(
            {item.metadata.family_id for item in episodes if item.metadata.split == split}
        )
        for split in SPLITS
    }
    _check(
        errors,
        not set(family_ids_by_split["train"])
        & (set(family_ids_by_split["dev"]) | set(family_ids_by_split["test"])),
        "family_overlap",
    )

    domain_folds = []
    for domain in sorted({item.metadata.domain for item in episodes}):
        heldout_ids = [
            item.episode_id for item in episodes if item.metadata.domain == domain
        ]
        training_ids = [
            item.episode_id for item in episodes if item.metadata.domain != domain
        ]
        domain_folds.append(
            {
                "fold_id": f"domain_holdout:{domain}",
                "filter": {"metadata.domain": domain},
                "train_count": len(training_ids),
                "heldout_count": len(heldout_ids),
                "train_episode_id_sha256": _hash_lines(training_ids),
                "heldout_episode_id_sha256": _hash_lines(heldout_ids),
            }
        )
    surface_realization_folds = []
    for generator_id in sorted({item.metadata.generator_id for item in episodes}):
        heldout_ids = [
            item.episode_id
            for item in episodes
            if item.metadata.generator_id == generator_id
        ]
        training_ids = [
            item.episode_id
            for item in episodes
            if item.metadata.generator_id != generator_id
        ]
        surface_realization_folds.append(
            {
                "fold_id": f"surface_realization_holdout:{generator_id}",
                "filter": {"metadata.generator_id": generator_id},
                "train_count": len(training_ids),
                "heldout_count": len(heldout_ids),
                "train_episode_id_sha256": _hash_lines(training_ids),
                "heldout_episode_id_sha256": _hash_lines(heldout_ids),
            }
        )
    _check(errors, len(domain_folds) == 4, "domain_holdout_fold_count_not_4")
    _check(
        errors,
        len(surface_realization_folds) == 3,
        "surface_realization_holdout_fold_count_not_3",
    )
    _check(
        errors,
        all(item["heldout_count"] == 600 for item in domain_folds),
        "domain_holdout_size_mismatch",
    )
    _check(
        errors,
        all(item["heldout_count"] == 800 for item in surface_realization_folds),
        "surface_realization_holdout_size_mismatch",
    )
    return {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "dataset_sha256": sha256_file(dataset_path),
        "primary_family_holdout": family_ids_by_split,
        "primary_attack_mechanism_holdout": mechanisms_by_split,
        "domain_leave_one_out_folds": domain_folds,
        "surface_realization_leave_one_out_folds": surface_realization_folds,
        "tuning_policy": (
            "Primary test, domain-held-out, and surface-held-out records are evaluation-only."
        ),
    }


def _retrieved_context(episode: ExecutableEpisode):
    matches = [item for item in episode.context if item.chunk_id == "retrieved-context"]
    if len(matches) != 1:
        raise ValueError(f"episode must have one retrieved-context chunk: {episode.episode_id}")
    return matches[0]


def _read_source_episode_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            episode_id = value.get("episode_id")
            if not isinstance(episode_id, str):
                raise ValueError(f"missing episode_id at {path}:{line_number}")
            if episode_id in ids:
                raise ValueError(f"duplicate episode_id in source: {episode_id}")
            ids.add(episode_id)
    return ids


def _bind_owner(
    errors: list[str],
    owners: dict[str, str],
    key: str,
    split: str,
    label: str,
) -> None:
    previous = owners.setdefault(key, split)
    if previous != split:
        errors.append(f"cross_split_{label}:{key}:{previous}:{split}")


def _require_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty frozen split directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _write_episode_jsonl(path: Path, episodes: list[ExecutableEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            handle.write(canonical_json(episode.model_dump(mode="json")) + "\n")
    temporary.replace(path)


def _hash_lines(values: list[str]) -> str:
    payload = "\n".join(sorted(values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _check(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)
