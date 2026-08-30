from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vipibench.compiler import load_compile_config, load_executable_episodes
from vipibench.dataio import sha256_file, write_json
from vipibench.episode import EpisodeLabel, ExecutableEpisode, TypedEpisodeTemplate


def validate_exec_benchmark(
    dataset_path: Path,
    config_path: Path,
    template_catalog_path: Path,
    *,
    output_path: Path | None = None,
    composition_output_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        config = load_compile_config(config_path)
        episodes = load_executable_episodes(dataset_path)
        templates = _load_templates(template_catalog_path)
    except Exception as exc:
        result = {
            "schema_version": "1.0.0",
            "status": "FAIL",
            "errors": [f"input_validation_error:{type(exc).__name__}:{exc}"],
            "dataset_path": str(dataset_path),
            "config_path": str(config_path),
            "template_catalog_path": str(template_catalog_path),
        }
        if output_path is not None:
            write_json(output_path, result)
        if composition_output_path is not None:
            write_json(composition_output_path, result)
        return result

    config_sha256 = sha256_file(config_path)
    expected_families = {
        f"{domain.domain_id}-{mechanism.mechanism_id}": mechanism.split
        for domain in config.domains
        for mechanism in config.mechanisms
    }
    family_counts = Counter(item.metadata.family_id for item in episodes)
    family_splits: dict[str, set[str]] = defaultdict(set)
    family_template_hashes: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        family_splits[episode.metadata.family_id].add(episode.metadata.split)
        family_template_hashes[episode.metadata.family_id].add(
            episode.metadata.template_sha256
        )

    _check(errors, len(episodes) == 2400, "episode_count_not_2400")
    _check(errors, len(templates) == 80, "template_count_not_80")
    _check(errors, set(family_counts) == set(expected_families), "family_set_mismatch")
    _check(
        errors,
        set(family_counts.values()) == {30},
        "family_episode_count_not_exactly_30",
    )
    _check(
        errors,
        all(len(splits) == 1 for splits in family_splits.values()),
        "family_crosses_split",
    )
    _check(
        errors,
        all(
            family_splits.get(family_id) == {expected_split}
            for family_id, expected_split in expected_families.items()
        ),
        "family_split_assignment_mismatch",
    )
    _check(
        errors,
        all(len(hashes) == 1 for hashes in family_template_hashes.values()),
        "family_has_multiple_template_hashes",
    )

    template_map = {item.family_id: item for item in templates}
    _check(errors, len(template_map) == len(templates), "duplicate_template_family")
    _check(errors, set(template_map) == set(expected_families), "template_family_set_mismatch")
    _check(
        errors,
        all(
            family_template_hashes.get(family_id) == {template.template_sha256}
            for family_id, template in template_map.items()
        ),
        "episode_template_hash_binding_mismatch",
    )

    episode_ids = [item.episode_id for item in episodes]
    episode_hashes = [item.content_sha256 for item in episodes]
    retrieved_hashes = [
        chunk.content_sha256
        for item in episodes
        for chunk in item.context
        if chunk.chunk_id == "retrieved-context"
    ]
    canaries = [canary for item in episodes for canary in item.security.canaries]
    _check(errors, len(set(episode_ids)) == len(episode_ids), "duplicate_episode_id")
    _check(errors, len(set(episode_hashes)) == len(episode_hashes), "duplicate_episode_hash")
    _check(
        errors,
        len(retrieved_hashes) == 2400 and len(set(retrieved_hashes)) == 2400,
        "retrieved_context_not_unique",
    )
    _check(errors, len(canaries) == 2400 and len(set(canaries)) == 2400, "canary_not_unique")
    _check(
        errors,
        all(item.metadata.config_sha256 == config_sha256 for item in episodes),
        "config_hash_binding_mismatch",
    )
    _check(
        errors,
        all(item.metadata.template_id == item.metadata.family_id for item in episodes),
        "template_family_binding_mismatch",
    )
    _check(errors, _labels_match_construction(episodes), "label_construction_mismatch")

    split_episode_counts = Counter(item.metadata.split for item in episodes)
    split_family_counts = {
        split: len(
            {
                item.metadata.family_id
                for item in episodes
                if item.metadata.split == split
            }
        )
        for split in ("train", "dev", "test")
    }
    label_counts = Counter(item.label.value for item in episodes)
    domain_counts = Counter(item.metadata.domain for item in episodes)
    language_counts = Counter(item.metadata.language_form for item in episodes)
    generator_counts = Counter(item.metadata.generator_id for item in episodes)
    generator_label_counts = Counter(
        (item.metadata.generator_id, item.label.value) for item in episodes
    )
    hard_negative_count = sum(item.metadata.hard_negative for item in episodes)
    pair_groups: dict[str, list[ExecutableEpisode]] = defaultdict(list)
    for episode in episodes:
        if episode.metadata.matched_pair_id:
            pair_groups[episode.metadata.matched_pair_id].append(episode)
    complete_pairs = [
        pair_id
        for pair_id, items in pair_groups.items()
        if _is_complete_pair(items)
    ]

    _check(
        errors,
        split_episode_counts == Counter({"train": 1440, "dev": 480, "test": 480}),
        "split_episode_counts_mismatch",
    )
    _check(
        errors,
        split_family_counts == {"train": 48, "dev": 16, "test": 16},
        "split_family_counts_mismatch",
    )
    _check(
        errors,
        label_counts == Counter({"benign": 1200, "injection": 1200}),
        "label_counts_mismatch",
    )
    _check(
        errors,
        domain_counts == Counter({domain.domain_id: 600 for domain in config.domains}),
        "domain_counts_mismatch",
    )
    _check(errors, hard_negative_count == 600, "hard_negative_count_mismatch")
    _check(errors, len(pair_groups) == 200, "matched_pair_count_mismatch")
    _check(errors, len(complete_pairs) == 200, "incomplete_matched_pairs")
    native_count = language_counts["native_vi"]
    _check(errors, native_count >= 1200, "native_vietnamese_below_50_percent")
    _check(
        errors,
        set(generator_counts.values()) == {800} and len(generator_counts) == 3,
        "generator_counts_mismatch",
    )
    _check(
        errors,
        all(
            generator_label_counts[(generator_id, "benign")] == 400
            and generator_label_counts[(generator_id, "injection")] == 400
            for generator_id in generator_counts
        ),
        "generator_label_balance_mismatch",
    )

    composition = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "episode_count": len(episodes),
        "family_count": len(family_counts),
        "template_count": len(templates),
        "split_episode_counts": dict(sorted(split_episode_counts.items())),
        "split_family_counts": split_family_counts,
        "label_counts": dict(sorted(label_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "language_form_counts": dict(sorted(language_counts.items())),
        "generator_counts": dict(sorted(generator_counts.items())),
        "generator_label_counts": {
            f"{generator_id}:{label}": count
            for (generator_id, label), count in sorted(generator_label_counts.items())
        },
        "native_vietnamese_count": native_count,
        "native_vietnamese_ratio": native_count / len(episodes) if episodes else 0.0,
        "hard_negative_count": hard_negative_count,
        "matched_pair_count": len(pair_groups),
        "complete_matched_pair_count": len(complete_pairs),
    }
    result = {
        **composition,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "template_catalog_path": str(template_catalog_path),
        "template_catalog_sha256": sha256_file(template_catalog_path),
        "unique_episode_id_count": len(set(episode_ids)),
        "unique_episode_hash_count": len(set(episode_hashes)),
        "unique_retrieved_context_hash_count": len(set(retrieved_hashes)),
        "unique_canary_count": len(set(canaries)),
        "schema_validated_episode_count": len(episodes),
        "claim_boundary": (
            "PASS proves schema, lineage binding, locked composition, and exact duplicate gates. "
            "Near-duplicate, shortcut, semantic-quality, and runtime outcome gates are separate."
        ),
    }
    if output_path is not None:
        write_json(output_path, result)
    if composition_output_path is not None:
        write_json(composition_output_path, composition)
    return result


def _load_templates(path: Path) -> list[TypedEpisodeTemplate]:
    templates: list[TypedEpisodeTemplate] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                templates.append(TypedEpisodeTemplate.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return templates


def _labels_match_construction(episodes: list[ExecutableEpisode]) -> bool:
    for episode in episodes:
        attack_chunks = [chunk for chunk in episode.context if chunk.attack_intent]
        if episode.label == EpisodeLabel.INJECTION:
            if len(attack_chunks) != 1 or not episode.security.attack_objectives:
                return False
        elif attack_chunks or episode.security.attack_objectives:
            return False
    return True


def _is_complete_pair(items: list[ExecutableEpisode]) -> bool:
    return (
        len(items) == 2
        and {item.label.value for item in items} == {"benign", "injection"}
        and len({item.metadata.family_id for item in items}) == 1
        and len({item.metadata.split for item in items}) == 1
        and len({item.metadata.language_form for item in items}) == 1
    )


def _check(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)
