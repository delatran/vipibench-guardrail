from __future__ import annotations

import hashlib
from pathlib import Path

from vipibench.dataio import canonical_json, sha256_file
from vipibench.episode import ExecutableEpisode
from vipibench.exec_detector_data import load_executable_episodes
from vipibench.provenance_contrast import audit_provenance_contrast_path

CORE_TRACK = "core_stress"
PROVENANCE_TRACK = "provenance_contrast"


def benchmark_track(episode: ExecutableEpisode) -> str:
    if any(
        item.startswith("provenance_contrast:")
        for item in episode.metadata.transformation_history
    ):
        return PROVENANCE_TRACK
    return CORE_TRACK


def partition_by_track(
    episodes: list[ExecutableEpisode],
) -> dict[str, list[ExecutableEpisode]]:
    partitions = {CORE_TRACK: [], PROVENANCE_TRACK: []}
    for episode in episodes:
        partitions[benchmark_track(episode)].append(episode)
    if sum(len(items) for items in partitions.values()) != len(episodes):
        raise ValueError("benchmark track partition did not reconcile")
    return partitions


def load_benchmark_partitions(
    split_dir: Path,
    contrast_dataset: Path,
    *,
    splits: tuple[str, ...] = ("train", "dev", "test"),
) -> tuple[dict[str, list[ExecutableEpisode]], dict[str, str]]:
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("benchmark partitions must be non-empty and unique")
    if not set(splits).issubset({"train", "dev", "test"}):
        raise ValueError("benchmark partitions contain an unsupported split")
    split_root = split_dir.resolve()
    project_root = split_root.parents[2]
    contrast_path = (
        contrast_dataset.resolve()
        if contrast_dataset.is_absolute()
        else (project_root / contrast_dataset).resolve()
    )
    contrast_audit = audit_provenance_contrast_path(contrast_path)
    if contrast_audit["status"] != "PASS":
        raise ValueError(contrast_audit["errors"])
    contrast = load_executable_episodes(contrast_path)
    records = {
        split: [
            *load_executable_episodes(split_root / f"{split}.jsonl"),
            *[episode for episode in contrast if episode.metadata.split == split],
        ]
        for split in splits
    }
    for split, members in records.items():
        ids = [episode.episode_id for episode in members]
        hashes = [episode.content_sha256 for episode in members]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate episode IDs in merged {split} split")
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"duplicate episode hashes in merged {split} split")
        if {episode.metadata.split for episode in members} != {split}:
            raise ValueError(f"split metadata mismatch in merged {split} split")
        tracks = partition_by_track(members)
        if not tracks[CORE_TRACK] or not tracks[PROVENANCE_TRACK]:
            raise ValueError(f"benchmark track missing in merged {split} split")
    source_hashes = {
        **{
            f"core_{split}": sha256_file(split_root / f"{split}.jsonl")
            for split in splits
        },
        "provenance_contrast": sha256_file(contrast_path),
        "merged_episode_set": hashlib.sha256(
            canonical_json(
                {
                    split: [episode.content_sha256 for episode in members]
                    for split, members in records.items()
                }
            ).encode("utf-8")
        ).hexdigest().upper(),
    }
    return records, source_hashes
