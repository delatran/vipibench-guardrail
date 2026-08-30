from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import mutual_info_score

from vipibench.compiler import load_executable_episodes
from vipibench.dataio import canonical_json, sha256_file, write_json
from vipibench.detector_view import detector_view_from_episode

FORBIDDEN_MODEL_FIELDS = {
    "attack_intent",
    "hard_negative",
    "matched_pair_id",
    "generator_id",
    "template_id",
    "family_id",
    "content_sha256",
    "label",
}
FORBIDDEN_SYNTHETIC_MARKERS = {
    "Nội dung truy xuất chèn chỉ thị",
    "Biến thể 00",
    "Biến thể 01",
}


def audit_exec_shortcuts(
    dataset_path: Path,
    *,
    role_output_path: Path | None = None,
    template_output_path: Path | None = None,
) -> dict[str, Any]:
    episodes = load_executable_episodes(dataset_path)
    views = [detector_view_from_episode(item) for item in episodes]
    labels = [item.label.value for item in episodes]

    role_features = {
        "source_role": [view.source_role for view in views],
        "trust_level": [view.trust_level for view in views],
    }
    structural_features = {
        "template_id": [item.metadata.template_id for item in episodes],
        "generator_id": [item.metadata.generator_id for item in episodes],
        "domain": [item.metadata.domain for item in episodes],
        "language_form": [item.metadata.language_form for item in episodes],
    }
    role_reports = {
        name: _categorical_shortcut_report(values, labels)
        for name, values in role_features.items()
    }
    structural_reports = {
        name: _categorical_shortcut_report(values, labels)
        for name, values in structural_features.items()
    }

    visible_inputs = [
        view.model_input("text_role_provenance") for view in views
    ]
    forbidden_field_hits = {
        field: sum(field in value for value in visible_inputs)
        for field in sorted(FORBIDDEN_MODEL_FIELDS)
    }
    marker_hits = {
        marker: sum(marker in value for value in visible_inputs)
        for marker in sorted(FORBIDDEN_SYNTHETIC_MARKERS)
    }
    episode_id_hits = sum(
        view.episode_id in visible
        for view, visible in zip(views, visible_inputs, strict=True)
    )
    role_errors = [
        f"role_feature_predictive:{name}"
        for name, report in role_reports.items()
        if report["status"] != "PASS"
    ]
    structural_errors = [
        f"structural_feature_predictive:{name}"
        for name, report in structural_reports.items()
        if report["status"] != "PASS"
    ]
    if any(forbidden_field_hits.values()):
        structural_errors.append("forbidden_metadata_visible_to_model")
    if any(marker_hits.values()):
        structural_errors.append("synthetic_label_marker_visible_to_model")
    if episode_id_hits:
        structural_errors.append("episode_id_visible_to_model")

    model_view_hash = hashlib.sha256(
        "\n".join(
            canonical_json(
                {
                    "episode_id": view.episode_id,
                    "label": view.label.value,
                    "model_input": visible,
                }
            )
            for view, visible in zip(views, visible_inputs, strict=True)
        ).encode("utf-8")
    ).hexdigest().upper()
    role_result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not role_errors else "FAIL",
        "errors": role_errors,
        "dataset_sha256": sha256_file(dataset_path),
        "episode_count": len(episodes),
        "label_counts": dict(sorted(Counter(labels).items())),
        "features": role_reports,
        "model_visible_role_contract": ["source_role", "trust_level"],
        "claim_boundary": (
            "PASS proves categorical role/trust features are at chance on this dataset."
        ),
    }
    structural_result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not structural_errors else "FAIL",
        "errors": structural_errors,
        "dataset_sha256": sha256_file(dataset_path),
        "episode_count": len(episodes),
        "features": structural_reports,
        "author_style_proxy": "generator_id",
        "forbidden_model_field_hits": forbidden_field_hits,
        "forbidden_synthetic_marker_hits": marker_hits,
        "episode_id_model_input_hits": episode_id_hits,
        "model_view_sha256": model_view_hash,
        "model_input_modes": [
            "text_only",
            "role_only",
            "text_role_provenance",
        ],
        "claim_boundary": (
            "PASS rules out the audited categorical and explicit marker shortcuts. It does not "
            "prove that no lexical or semantic shortcut exists."
        ),
    }
    if role_output_path is not None:
        write_json(role_output_path, role_result)
    if template_output_path is not None:
        write_json(template_output_path, structural_result)
    return {
        "schema_version": "1.0.0",
        "status": (
            "PASS"
            if role_result["status"] == structural_result["status"] == "PASS"
            else "FAIL"
        ),
        "role_label": role_result,
        "template_generator_style": structural_result,
    }


def _categorical_shortcut_report(
    values: list[str],
    labels: list[str],
) -> dict[str, Any]:
    cells = Counter(zip(values, labels, strict=True))
    categories = sorted(set(values))
    total = len(labels)
    majority_correct = sum(
        max(cells[(category, "benign")], cells[(category, "injection")])
        for category in categories
    )
    ratios = {}
    complete = True
    for category in categories:
        benign = cells[(category, "benign")]
        injection = cells[(category, "injection")]
        high = max(benign, injection)
        ratios[category] = min(benign, injection) / high if high else 0.0
        complete = complete and benign > 0 and injection > 0
    majority_accuracy = majority_correct / total if total else 1.0
    mutual_information = float(mutual_info_score(values, labels)) if total else 1.0
    passed = (
        complete
        and abs(majority_accuracy - 0.5) <= 1e-12
        and mutual_information <= 1e-12
        and all(abs(value - 1.0) <= 1e-12 for value in ratios.values())
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "category_count": len(categories),
        "label_counts_by_category": {
            f"{category}:{label}": cells[(category, label)]
            for category in categories
            for label in ("benign", "injection")
        },
        "minority_majority_ratio_by_category": ratios,
        "categorical_majority_accuracy": majority_accuracy,
        "mutual_information_nats": mutual_information,
        "expected_chance_accuracy": 0.5,
    }
