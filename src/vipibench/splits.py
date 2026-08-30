from __future__ import annotations

import hashlib
import itertools
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from vipibench.dataio import load_jsonl, sha256_file, write_jsonl
from vipibench.schema import DatasetRecord

SPLITS = ("train", "dev", "test")


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _connected_groups(records: list[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    """Union records sharing any locked lineage key before assigning a split."""

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owner: dict[str, int] = {}
    for index, record in enumerate(records):
        for name, value in (
            ("seed_id", record.seed_id),
            ("template_id", record.template_id),
            ("semantic_cluster", record.semantic_cluster),
            ("matched_pair_id", record.matched_pair_id),
        ):
            if not value:
                continue
            token = f"{name}:{value}"
            if token in owner:
                union(index, owner[token])
            else:
                owner[token] = index

    by_root: dict[int, list[DatasetRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        by_root[find(index)].append(record)
    return {min(member.sample_id for member in members): members for members in by_root.values()}


def split_records(
    records: list[DatasetRecord], *, seed: int = 17
) -> dict[str, list[DatasetRecord]]:
    groups = _connected_groups(records)

    strata: dict[str, list[str]] = defaultdict(list)
    for group_key, members in groups.items():
        signature = ";".join(
            sorted(f"{member.source_role.value}:{member.label.value}" for member in members)
        )
        strata[signature].append(group_key)

    assignments: dict[str, str] = {}
    for signature, group_keys in sorted(strata.items()):
        ordered = sorted(group_keys, key=lambda key: _stable_key(f"{signature}:{key}", seed))
        for index, group_key in enumerate(ordered):
            bucket = index % 10
            assignments[group_key] = "train" if bucket < 6 else "dev" if bucket < 8 else "test"

    output = {split: [] for split in SPLITS}
    for group_key, members in groups.items():
        output[assignments[group_key]].extend(members)
    return output


def write_splits(directory: Path, split_map: dict[str, list[DatasetRecord]]) -> None:
    for split in SPLITS:
        write_jsonl(directory / f"{split}.jsonl", split_map[split])


def _ngrams(text: str, width: int = 5) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if len(normalized) <= width:
        return {normalized}
    return {normalized[index : index + width] for index in range(len(normalized) - width + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class SplitAudit:
    status: str
    directory: str
    split_hashes: dict[str, str]
    counts: dict[str, int]
    errors: list[str]
    warnings: list[str]
    near_duplicate_threshold: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_split_directory(directory: Path, *, near_threshold: float = 0.90) -> SplitAudit:
    records_by_split = {split: load_jsonl(directory / f"{split}.jsonl") for split in SPLITS}
    errors: list[str] = []
    warnings: list[str] = []
    seen_groups: dict[str, str] = {}
    seen_exact: dict[str, tuple[str, str]] = {}

    for split, records in records_by_split.items():
        for record in records:
            for group_name, group_value in (
                ("seed_id", record.seed_id),
                ("template_id", record.template_id),
                ("semantic_cluster", record.semantic_cluster),
                ("matched_pair_id", record.matched_pair_id),
            ):
                if not group_value:
                    continue
                key = f"{group_name}:{group_value}"
                previous = seen_groups.setdefault(key, split)
                if previous != split:
                    errors.append(f"cross-split group leakage: {key} in {previous} and {split}")
            exact_hash = hashlib.sha256(record.normalized_text.encode("utf-8")).hexdigest()
            previous_exact = seen_exact.setdefault(exact_hash, (split, record.sample_id))
            if previous_exact[0] != split:
                errors.append(
                    "cross-split exact duplicate: "
                    f"{previous_exact[1]} ({previous_exact[0]}) vs {record.sample_id} ({split})"
                )

    flattened = [
        (split, record, _ngrams(record.normalized_text))
        for split, records in records_by_split.items()
        for record in records
    ]
    for (left_split, left, left_grams), (right_split, right, right_grams) in itertools.combinations(
        flattened, 2
    ):
        if left_split == right_split or left.seed_id == right.seed_id:
            continue
        similarity = _jaccard(left_grams, right_grams)
        if similarity >= near_threshold:
            message = (
                f"near duplicate {similarity:.3f}: {left.sample_id} ({left_split}) vs "
                f"{right.sample_id} ({right_split})"
            )
            if left.fixture_only or right.fixture_only:
                warnings.append(message)
            else:
                errors.append(message)

    for split, records in records_by_split.items():
        labels = Counter(record.label.value for record in records)
        if not records:
            errors.append(f"empty split: {split}")
        if len(labels) < 2:
            errors.append(f"split lacks both labels: {split} {dict(labels)}")

    unique_errors = sorted(set(errors))
    unique_warnings = sorted(set(warnings))
    return SplitAudit(
        status="PASS" if not unique_errors else "FAIL",
        directory=str(directory),
        split_hashes={split: sha256_file(directory / f"{split}.jsonl") for split in SPLITS},
        counts={split: len(records) for split, records in records_by_split.items()},
        errors=unique_errors,
        warnings=unique_warnings,
        near_duplicate_threshold=near_threshold,
    )
