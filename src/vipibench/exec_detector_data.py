from __future__ import annotations

import hashlib
from pathlib import Path

from vipibench.detector_view import detector_view_from_episode
from vipibench.episode import ExecutableEpisode

INPUT_MODE_ALIASES = {
    "role_only": "role_only",
    "text_only": "text_only",
    "text_role": "text_role_provenance",
    "text_role_provenance": "text_role_provenance",
}


def load_executable_episodes(path: Path) -> list[ExecutableEpisode]:
    episodes: list[ExecutableEpisode] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                episodes.append(ExecutableEpisode.model_validate_json(raw_line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return episodes


def detector_text(episode: ExecutableEpisode, input_mode: str) -> str:
    normalized_mode = INPUT_MODE_ALIASES.get(input_mode)
    if normalized_mode is None:
        raise ValueError(f"unsupported executable detector input mode: {input_mode}")
    return detector_view_from_episode(episode).model_input(normalized_mode)  # type: ignore[arg-type]


def prediction_row(
    episode: ExecutableEpisode,
    score: float,
    *,
    split: str,
    latency_ms: float,
    input_mode: str | None = None,
) -> dict[str, object]:
    conditions = [
        item.removeprefix("condition:")
        for item in episode.metadata.transformation_history
        if item.startswith("condition:")
    ]
    benchmark_track = (
        "provenance_contrast"
        if any(
            item.startswith("provenance_contrast:")
            for item in episode.metadata.transformation_history
        )
        else "core_stress"
    )
    row: dict[str, object] = {
        "sample_id": episode.episode_id,
        "episode_sha256": episode.content_sha256,
        "seed_id": episode.metadata.seed_id,
        "source_family": episode.metadata.family_id,
        "template_family": episode.metadata.family_id,
        "semantic_cluster": episode.metadata.template_id,
        "matched_pair_id": episode.metadata.matched_pair_id,
        "label": episode.label.value,
        "score": float(score),
        "split": split,
        "source_role": "retrieved",
        "trust_level": "untrusted",
        "hard_negative": episode.metadata.hard_negative,
        "benchmark_track": benchmark_track,
        "diagnostic_condition": conditions[0] if len(conditions) == 1 else None,
        "fixture_only": False,
        "latency_ms": float(latency_ms),
    }
    if input_mode is not None:
        model_input = detector_text(episode, input_mode)
        row.update(
            {
                "input_mode": INPUT_MODE_ALIASES[input_mode],
                "model_input_sha256": hashlib.sha256(model_input.encode("utf-8"))
                .hexdigest()
                .upper(),
            }
        )
    return row
