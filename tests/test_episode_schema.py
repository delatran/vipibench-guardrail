from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from vipibench.episode import (
    ExecutableEpisode,
    TypedEpisodeTemplate,
    bind_content_hash,
    bind_template_hash,
)
from vipibench.exec_fixture import build_fixture_episode


def test_bind_content_hash_normalizes_defaults_and_validates() -> None:
    episode = build_fixture_episode(injection=False)
    assert episode.schema_version == "1.0.0"
    assert episode.tools[0].allow_extra_arguments is False
    assert len(episode.content_sha256) == 64


def test_episode_hash_tampering_is_rejected() -> None:
    payload = build_fixture_episode(injection=False).model_dump(mode="json")
    payload["trusted_user_goal"] = "Tampered goal"
    with pytest.raises(ValidationError, match="content_sha256 mismatch"):
        ExecutableEpisode.model_validate(payload)


def test_context_hash_tampering_is_rejected_before_episode_hash() -> None:
    payload = build_fixture_episode(injection=False).model_dump(mode="json")
    payload["context"][0]["content"] = "Tampered context"
    with pytest.raises(ValidationError, match="content_sha256 must bind"):
        bind_content_hash(payload)


def test_benign_episode_cannot_hide_attack_intent() -> None:
    payload = build_fixture_episode(injection=False).model_dump(mode="json")
    payload["context"][1]["attack_intent"] = True
    payload["context"][1]["content_sha256"] = payload["context"][1]["content_sha256"]
    with pytest.raises(ValidationError, match="benign episodes cannot contain attack"):
        bind_content_hash(payload)


def test_authorized_action_must_reference_declared_tool() -> None:
    payload = build_fixture_episode(injection=False).model_dump(mode="json")
    payload["authorized_actions"][0]["tool"] = "unknown_tool"
    with pytest.raises(ValidationError, match="references unknown tool"):
        bind_content_hash(payload)


def test_template_hash_binding_and_tamper_detection() -> None:
    payload = {
        "template_id": "fixture-template",
        "family_id": "fixture-family",
        "domain": "document_assistant",
        "variables": {"document": {"kind": "string", "min_length": 1}},
        "blueprint": {"trusted_user_goal": "Tóm tắt {{document}}"},
    }
    template = TypedEpisodeTemplate.model_validate(bind_template_hash(payload))
    tampered = copy.deepcopy(template.model_dump(mode="json"))
    tampered["domain"] = "tampered"
    with pytest.raises(ValidationError, match="template_sha256 mismatch"):
        TypedEpisodeTemplate.model_validate(tampered)
