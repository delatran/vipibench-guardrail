"""Fail-closed structural validity checks for adaptive attack candidates."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from collections.abc import Iterable

from vipibench.dataio import canonical_json
from vipibench.episode import EpisodeLabel, ExecutableEpisode

VALIDITY_SCHEMA_VERSION = "1.0.0"
CANDIDATES_PER_STRATEGY = 10
STRATEGIES = ("static_sampling", "feedback_guided")


def normalize_candidate_text(text: str) -> str:
    """Conservative duplicate key: Unicode, case, whitespace, and punctuation normalized."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        " " if character.isspace() or unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(normalized.split())


def validate_adaptive_candidate_set(
    bases: Iterable[ExecutableEpisode],
    candidates: Iterable[ExecutableEpisode],
    *,
    detector_revision: str,
) -> dict[str, object]:
    """Return a retained manifest; any structural drift or duplicate fails closed."""

    candidate_list = list(candidates)
    base_map = {base.episode_id: base for base in bases}
    errors: list[str] = []
    records: list[dict[str, object]] = []
    seen_raw: dict[tuple[str, str], str] = {}
    seen_normalized: dict[tuple[str, str], str] = {}
    identity_counts: Counter[tuple[str, str, int]] = Counter()
    strategy_counts: Counter[str] = Counter()
    for candidate in candidate_list:
        identity = _identity(candidate, errors)
        if identity is None:
            continue
        base_id, strategy, index = identity
        base = base_map.get(base_id)
        if base is None:
            errors.append(f"candidate_base_missing:{candidate.episode_id}")
            continue
        identity_counts[identity] += 1
        strategy_counts[strategy] += 1
        text = _retrieved_text(candidate, errors)
        if text is None:
            continue
        try:
            text.encode("utf-8", "strict")
        except UnicodeError:
            errors.append(f"candidate_invalid_unicode:{candidate.episode_id}")
            continue
        normalized = normalize_candidate_text(text)
        if not normalized:
            errors.append(f"candidate_empty_after_normalization:{candidate.episode_id}")
        raw_sha = _sha256(text)
        normalized_sha = _sha256(normalized)
        base_normalized_sha = _sha256(normalize_candidate_text(_base_retrieved_text(base)))
        raw_key = (base_id, raw_sha)
        normalized_key = (base_id, normalized_sha)
        if raw_key in seen_raw:
            errors.append(f"candidate_exact_duplicate:{candidate.episode_id}:{seen_raw[raw_key]}")
        if normalized_key in seen_normalized:
            errors.append(
                f"candidate_normalized_duplicate:{candidate.episode_id}:{seen_normalized[normalized_key]}"
            )
        if normalized_sha == base_normalized_sha:
            errors.append(f"candidate_duplicates_base_context:{candidate.episode_id}")
        seen_raw[raw_key] = candidate.episode_id
        seen_normalized[normalized_key] = candidate.episode_id
        if candidate.label != EpisodeLabel.INJECTION:
            errors.append(f"candidate_label_changed:{candidate.episode_id}")
        if _objective_signature(candidate) != _objective_signature(base):
            errors.append(f"candidate_immutable_objective_changed:{candidate.episode_id}")
        if _immutable_metadata(candidate) != _immutable_metadata(base):
            errors.append(f"candidate_immutable_provenance_changed:{candidate.episode_id}")
        records.append(
            {
                "episode_id": candidate.episode_id,
                "episode_sha256": candidate.content_sha256,
                "base_episode_id": base_id,
                "strategy": strategy,
                "candidate_index": index,
                "raw_text_sha256": raw_sha,
                "utf8_text_sha256": raw_sha,
                "normalized_text": normalized,
                "normalized_text_sha256": normalized_sha,
                "generator_revision": candidate.metadata.generator_revision,
                "detector_revision": detector_revision,
                "immutable_attack_objective_sha256": _objective_signature(base),
                "semantic_preservation": "UNVERIFIED",
            }
        )
    for base_id in sorted(base_map):
        for strategy in STRATEGIES:
            for index in range(CANDIDATES_PER_STRATEGY):
                if identity_counts[(base_id, strategy, index)] != 1:
                    errors.append(
                        f"candidate_identity_budget_mismatch:{base_id}:{strategy}:{index}"
                    )
    if set(strategy_counts) != set(STRATEGIES) or len(set(strategy_counts.values())) != 1:
        errors.append("candidate_strategy_budget_imbalanced")
    return {
        "schema_version": VALIDITY_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "raw_proposal_count": len(candidate_list),
        "accepted_candidate_count": len(records) if not errors else 0,
        "rejected_candidate_count": len(candidate_list) - len(records)
        if not errors
        else len(candidate_list),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "records": sorted(records, key=lambda row: str(row["episode_id"])),
        "semantic_preservation": "UNVERIFIED",
        "semantic_preservation_verified": False,
        "research_claim_eligible": False,
        "claim_boundary": (
            "PASS proves only structural preservation and duplicate controls. Natural-language "
            "semantic equivalence remains UNVERIFIED and cannot support H4."
        ),
    }


def validate_adaptive_candidate_checkpoint(
    base: ExecutableEpisode,
    candidates: Iterable[ExecutableEpisode],
    *,
    strategy: str,
    detector_revision: str,
) -> dict[str, object]:
    """Validate the resumable prefix for one base/strategy search checkpoint.

    Guided search checkpoints are intentionally saved after every query, so a
    prefix may contain fewer than ten proposals.  This manifest therefore
    verifies the present contiguous prefix, while the final candidate-set
    validator remains responsible for the full two-strategy 20-candidate
    budget.
    """

    candidate_list = list(candidates)
    errors: list[str] = []
    records: list[dict[str, object]] = []
    seen_raw: dict[str, str] = {}
    seen_normalized: dict[str, str] = {}
    base_normalized_sha = _sha256(normalize_candidate_text(_base_retrieved_text(base)))
    if strategy not in STRATEGIES:
        errors.append(f"checkpoint_strategy_invalid:{strategy}")
    if not 1 <= len(candidate_list) <= CANDIDATES_PER_STRATEGY:
        errors.append("checkpoint_candidate_count_out_of_range")

    for expected_index, candidate in enumerate(candidate_list):
        identity = _identity(candidate, errors)
        if identity is None:
            continue
        base_id, observed_strategy, index = identity
        if base_id != base.episode_id:
            errors.append(f"checkpoint_candidate_base_mismatch:{candidate.episode_id}")
        if observed_strategy != strategy:
            errors.append(f"checkpoint_candidate_strategy_mismatch:{candidate.episode_id}")
        if index != expected_index:
            errors.append(f"checkpoint_candidate_index_mismatch:{candidate.episode_id}")
        text = _retrieved_text(candidate, errors)
        if text is None:
            continue
        try:
            text.encode("utf-8", "strict")
        except UnicodeError:
            errors.append(f"candidate_invalid_unicode:{candidate.episode_id}")
            continue
        normalized = normalize_candidate_text(text)
        raw_sha = _sha256(text)
        normalized_sha = _sha256(normalized)
        if not normalized:
            errors.append(f"candidate_empty_after_normalization:{candidate.episode_id}")
        if raw_sha in seen_raw:
            errors.append(f"candidate_exact_duplicate:{candidate.episode_id}:{seen_raw[raw_sha]}")
        if normalized_sha in seen_normalized:
            errors.append(
                f"candidate_normalized_duplicate:{candidate.episode_id}:{seen_normalized[normalized_sha]}"
            )
        if normalized_sha == base_normalized_sha:
            errors.append(f"candidate_duplicates_base_context:{candidate.episode_id}")
        seen_raw[raw_sha] = candidate.episode_id
        seen_normalized[normalized_sha] = candidate.episode_id
        if candidate.label != EpisodeLabel.INJECTION:
            errors.append(f"candidate_label_changed:{candidate.episode_id}")
        if _objective_signature(candidate) != _objective_signature(base):
            errors.append(f"candidate_immutable_objective_changed:{candidate.episode_id}")
        if _immutable_metadata(candidate) != _immutable_metadata(base):
            errors.append(f"candidate_immutable_provenance_changed:{candidate.episode_id}")
        records.append(
            {
                "episode_id": candidate.episode_id,
                "episode_sha256": candidate.content_sha256,
                "base_episode_id": base.episode_id,
                "strategy": strategy,
                "candidate_index": expected_index,
                "raw_text_sha256": raw_sha,
                "utf8_text_sha256": raw_sha,
                "normalized_text": normalized,
                "normalized_text_sha256": normalized_sha,
                "generator_revision": candidate.metadata.generator_revision,
                "detector_revision": detector_revision,
                "immutable_attack_objective_sha256": _objective_signature(base),
                "semantic_preservation": "UNVERIFIED",
            }
        )
    return {
        "schema_version": VALIDITY_SCHEMA_VERSION,
        "scope": "adaptive_checkpoint_prefix",
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "base_episode_id": base.episode_id,
        "base_episode_sha256": base.content_sha256,
        "strategy": strategy,
        "accepted_candidate_count": len(records) if not errors else 0,
        "rejected_candidate_count": len(candidate_list) - len(records)
        if not errors
        else len(candidate_list),
        "records": records,
        "semantic_preservation": "UNVERIFIED",
        "semantic_preservation_verified": False,
        "research_claim_eligible": False,
        "claim_boundary": (
            "PASS proves only structural validity of this resumable candidate prefix. "
            "Semantic equivalence remains UNVERIFIED and cannot support H4."
        ),
    }


def _identity(candidate: ExecutableEpisode, errors: list[str]) -> tuple[str, str, int] | None:
    fields = {
        key: value
        for item in candidate.metadata.transformation_history
        if ":" in item
        for key, value in [item.split(":", maxsplit=1)]
        if key in {"base_episode", "search_strategy", "candidate_index"}
    }
    try:
        strategy = fields["search_strategy"]
        index = int(fields["candidate_index"])
        if strategy not in STRATEGIES or not 0 <= index < CANDIDATES_PER_STRATEGY:
            raise ValueError
        return fields["base_episode"], strategy, index
    except (KeyError, ValueError):
        errors.append(f"candidate_identity_invalid:{candidate.episode_id}")
        return None


def _retrieved_text(episode: ExecutableEpisode, errors: list[str]) -> str | None:
    matches = [chunk.content for chunk in episode.context if chunk.chunk_id == "retrieved-context"]
    if len(matches) != 1:
        errors.append(f"candidate_retrieved_context_invalid:{episode.episode_id}")
        return None
    return matches[0]


def _base_retrieved_text(episode: ExecutableEpisode) -> str:
    matches = [chunk.content for chunk in episode.context if chunk.chunk_id == "retrieved-context"]
    if len(matches) != 1:
        raise ValueError(f"base has no unique retrieved context: {episode.episode_id}")
    return matches[0]


def _objective_signature(episode: ExecutableEpisode) -> str:
    return _sha256(
        canonical_json(
            {
                "label": episode.label.value,
                "trusted_user_goal": episode.trusted_user_goal,
                "initial_state": episode.initial_state,
                "tools": [tool.model_dump(mode="json") for tool in episode.tools],
                "authorized_actions": [
                    action.model_dump(mode="json") for action in episode.authorized_actions
                ],
                "security": episode.security.model_dump(mode="json"),
                "utility": episode.utility.model_dump(mode="json"),
            }
        )
    )


def _immutable_metadata(episode: ExecutableEpisode) -> dict[str, object]:
    metadata = episode.metadata
    return {
        "family_id": metadata.family_id,
        "template_id": metadata.template_id,
        "domain": metadata.domain,
        "language_form": metadata.language_form,
        "native_vietnamese": metadata.native_vietnamese,
        "split": metadata.split,
        "config_sha256": metadata.config_sha256,
        "template_sha256": metadata.template_sha256,
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()
