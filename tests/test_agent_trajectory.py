import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vipibench.agent_trajectory import (
    AGENT_TRAJECTORY_SCHEMA_VERSION,
    MAX_MALFORMED_DIAGNOSTIC_UTF8_BYTES,
    OBSERVED_WALL_LATENCY_DEFINITION,
    TIMING_SCHEMA_VERSION,
    AgentTrajectoryRecord,
    audit_agent_request,
    bind_agent_trajectory_record,
    build_agent_request,
    parse_agent_response,
    render_agent_prompt,
    write_agent_trajectory_records,
    write_malformed_response_diagnostic,
)
from vipibench.exec_detector_data import load_executable_episodes
from vipibench.oracle import (
    TRAJECTORY_EVENT_ID_MAX_LENGTH,
    TRAJECTORY_EVENT_ID_PATTERN,
    TRAJECTORY_SCHEMA_VERSION,
)
from vipibench.system_runner import build_safe_fallback_trajectory


def _episode():
    return load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]


def test_agent_request_excludes_oracle_and_label_fields() -> None:
    episode = _episode()
    request = build_agent_request(episode)
    result = audit_agent_request(episode)
    assert result["status"] == "PASS", result["forbidden_key_leaks"]
    prompt = render_agent_prompt(request)
    request_payload = json.loads(
        prompt.split("--- BEGIN REQUEST DATA ---\n", maxsplit=1)[1].split(
            "\n--- END REQUEST DATA ---", maxsplit=1
        )[0]
    )
    assert request_payload["request_sha256"] == request.request_sha256
    assert "label" not in request_payload
    assert "security" not in request_payload
    assert "utility" not in request_payload
    assert 'only top-level key is "events"' in prompt
    assert '"output_contract"' not in prompt
    assert f"2 to {TRAJECTORY_EVENT_ID_MAX_LENGTH} ASCII characters" in prompt
    assert f"match {TRAJECTORY_EVENT_ID_PATTERN}" in prompt
    assert "unique within events" in prompt
    assert "For an assistant_message, do not include tool" in prompt
    assert "For a tool_call, do not include content" in prompt


def test_observed_response_wrapper_is_rejected() -> None:
    episode = _episode()
    request = build_agent_request(episode)
    observed_invalid_shape = json.dumps(
        {
            "output_contract": {
                "events": [
                    {
                        "event_type": "assistant_message",
                        "event_id": "answer_1",
                        "content": "fixture",
                        "tool": "set_workspace",
                        "arguments": {},
                        "authorization_refs": ["fixture-authorization"],
                        "supporting_context_ids": ["fixture-context"],
                    }
                ]
            }
        }
    )

    with pytest.raises(ValueError) as exc_info:
        parse_agent_response(
            observed_invalid_shape,
            episode=episode,
            request=request,
            model_revision="fixture",
        )

    assert getattr(exc_info.value, "error_class", None) == "response_shape_error"


def test_agent_response_is_strictly_parsed_and_bound() -> None:
    episode = _episode()
    request = build_agent_request(episode)
    response = json.dumps(
        {
            "events": [
                {
                    "event_type": "assistant_message",
                    "event_id": "safe-response",
                    "content": "Tôi không thể thực hiện hành động không được ủy quyền.",
                }
            ]
        },
        ensure_ascii=False,
    )
    record = parse_agent_response(
        response,
        episode=episode,
        request=request,
        model_id="fixture-model",
        model_revision="fixture-model-revision",
        tokenizer_revision="fixture-tokenizer-revision",
    )
    assert record.schema_version == AGENT_TRAJECTORY_SCHEMA_VERSION
    assert record.synthetic_fixture is False
    assert record.episode_sha256 == episode.content_sha256
    assert record.request_sha256 == request.request_sha256
    assert record.prompt_sha256
    assert record.model_id == "fixture-model"
    assert record.tokenizer_revision == "fixture-tokenizer-revision"
    assert record.format_status == "strict_json"
    assert record.fallback_status == "not_used"
    assert record.batch_size == 1
    assert record.timing_schema_version == TIMING_SCHEMA_VERSION
    assert "raw_response" not in record.model_dump(mode="json")
    assert OBSERVED_WALL_LATENCY_DEFINITION.startswith("Per-record model-request wall-clock")


def test_agent_response_accepts_observed_long_event_ids_without_rewriting() -> None:
    episode = _episode()
    request = build_agent_request(episode)
    assistant_event_id = "a" + ("b" * 79)
    tool_event_id = "t" + ("c" * 71)
    response = json.dumps(
        {
            "events": [
                {
                    "event_type": "assistant_message",
                    "event_id": assistant_event_id,
                    "content": "fixture",
                },
                {
                    "event_type": "tool_call",
                    "event_id": tool_event_id,
                    "tool": "fixture_tool",
                    "arguments": {},
                    "authorization_refs": [],
                    "supporting_context_ids": [],
                },
            ]
        }
    )

    record = parse_agent_response(
        response,
        episode=episode,
        request=request,
        model_revision="fixture",
    )

    assert [event.event_id for event in record.trajectory.events] == [
        assistant_event_id,
        tool_event_id,
    ]
    assert record.trajectory.schema_version == TRAJECTORY_SCHEMA_VERSION
    assert record.format_status == "strict_json"
    assert record.generation_attempts == 1


def test_agent_response_rejects_event_id_beyond_registered_envelope() -> None:
    episode = _episode()
    request = build_agent_request(episode)
    oversized_event_id = "a" + ("b" * TRAJECTORY_EVENT_ID_MAX_LENGTH)
    response = json.dumps(
        {
            "events": [
                {
                    "event_type": "assistant_message",
                    "event_id": oversized_event_id,
                    "content": "fixture",
                }
            ]
        }
    )

    with pytest.raises(ValueError) as exc_info:
        parse_agent_response(
            response,
            episode=episode,
            request=request,
            model_revision="fixture",
        )

    assert getattr(exc_info.value, "error_class", None) == "trajectory_validation_error"


def test_agent_response_rejects_markdown_or_extra_fields() -> None:
    episode = _episode()
    request = build_agent_request(episode)
    with pytest.raises(ValueError, match="strict JSON"):
        parse_agent_response(
            "```json\n{}\n```",
            episode=episode,
            request=request,
            model_revision="fixture",
        )
    with pytest.raises(ValueError, match="only the events field"):
        parse_agent_response(
            '{"events": [], "label": "benign"}',
            episode=episode,
            request=request,
            model_revision="fixture",
        )


def test_stale_trajectory_schema_and_missing_provenance_fail_closed() -> None:
    episode = _episode()
    request = build_agent_request(episode)
    record = parse_agent_response(
        '{"events": []}',
        episode=episode,
        request=request,
        model_revision="fixture",
    )
    payload = record.model_dump(mode="json")
    payload["schema_version"] = "1.0.0"
    with pytest.raises(ValidationError, match="schema_version"):
        AgentTrajectoryRecord.model_validate(payload)

    payload = record.model_dump(mode="json")
    payload["trajectory"]["schema_version"] = "1.0.0"
    with pytest.raises(ValidationError, match="schema_version"):
        AgentTrajectoryRecord.model_validate(payload)

    payload = record.model_dump(mode="json")
    payload.pop("prompt_sha256")
    with pytest.raises(ValidationError, match="prompt_sha256"):
        AgentTrajectoryRecord.model_validate(payload)

    payload = record.model_dump(mode="json")
    payload["fallback_status"] = "used"
    payload["format_fallback"] = True
    payload["record_sha256"] = "A" * 64
    with pytest.raises(ValidationError, match="strict JSON"):
        AgentTrajectoryRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("pattern_class", "token"),
    [
        ("huggingface_token", "hf_" + ("A" * 20)),
        ("email_address", "owner" + "@example.com"),
    ],
)
def test_malformed_response_diagnostic_suppresses_sensitive_plaintext(
    tmp_path: Path,
    pattern_class: str,
    token: str,
) -> None:
    episode = _episode()
    request = build_agent_request(episode)
    raw = token + ("x" * (MAX_MALFORMED_DIAGNOSTIC_UTF8_BYTES + 500))
    path = tmp_path / "diagnostic.json"
    digest = write_malformed_response_diagnostic(
        path,
        episode_id=episode.episode_id,
        request_sha256=request.request_sha256,
        prompt_sha256="A" * 64,
        model_id="fixture-model",
        model_revision="fixture-revision",
        tokenizer_revision="fixture-tokenizer",
        attempts=[(1, raw, "json_decode_error")],
    )
    serialized = path.read_text(encoding="utf-8")
    payload = json.loads(serialized)
    assert len(digest) == 64
    assert payload["release_class"] == "NONPUBLIC_DIAGNOSTIC"
    assert payload["public_release_allowed"] is False
    assert payload["include_in_public_manifest"] is False
    assert token not in serialized
    assert payload["plaintext_excerpt_retained"] is False
    assert payload["plaintext_suppression_reason"] == "secret_or_pii_pattern_detected"
    assert payload["retained_utf8_bytes"] == 0
    assert payload["attempts"][0]["truncated"] is True
    assert payload["secret_scan"]["status"] == "FAIL"
    assert payload["secret_scan"]["pattern_classes"] == [pattern_class]
    assert payload["attempts"][0]["matched_pattern_classes"] == [pattern_class]
    assert len(payload["attempts"][0]["raw_response_sha256"]) == 64
    assert payload["attempts"][0]["raw_response_utf8_bytes"] == len(raw.encode("utf-8"))
    assert payload["attempts"][0]["plaintext_excerpt_retained"] is False
    assert "retained_response_excerpt" not in payload["attempts"][0]
    assert "retained_response_sha256" not in payload["attempts"][0]


def test_malformed_response_diagnostic_bounds_non_sensitive_excerpt(tmp_path: Path) -> None:
    episode = _episode()
    request = build_agent_request(episode)
    raw = "x" * (MAX_MALFORMED_DIAGNOSTIC_UTF8_BYTES + 500)
    path = tmp_path / "diagnostic.json"
    write_malformed_response_diagnostic(
        path,
        episode_id=episode.episode_id,
        request_sha256=request.request_sha256,
        prompt_sha256="A" * 64,
        model_id="fixture-model",
        model_revision="fixture-revision",
        tokenizer_revision="fixture-tokenizer",
        attempts=[(1, raw, "json_decode_error")],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    attempt = payload["attempts"][0]
    assert payload["plaintext_excerpt_retained"] is True
    assert payload["plaintext_suppression_reason"] is None
    assert payload["retained_utf8_bytes"] == MAX_MALFORMED_DIAGNOSTIC_UTF8_BYTES
    assert attempt["retained_response_excerpt"] == "x" * MAX_MALFORMED_DIAGNOSTIC_UTF8_BYTES
    assert attempt["retained_response_sha256"]
    assert attempt["matched_pattern_classes"] == []
    assert attempt["truncated"] is True
    assert payload["secret_scan"]["status"] == "PASS"


def test_trajectory_manifest_reports_provenance_prerequisites(tmp_path: Path) -> None:
    episode = _episode()
    request = build_agent_request(episode)
    strict = parse_agent_response(
        '{"events": []}',
        episode=episode,
        request=request,
        model_revision="fixture",
        input_token_count=4,
        output_token_count=2,
    )
    fallback = bind_agent_trajectory_record(
        episode=episode,
        request=request,
        trajectory=build_safe_fallback_trajectory(episode),
        model_revision="fixture",
        synthetic_fixture=False,
        generation_attempts=1,
        format_status="safe_fallback",
        fallback_status="used",
        format_fallback=True,
        malformed_response_sha256="A" * 64,
        parse_error_class="json_decode_error",
        diagnostic_artifact_sha256="B" * 64,
    )
    manifest = write_agent_trajectory_records(
        tmp_path / "trajectories.jsonl",
        [strict, fallback],
    )
    assert manifest["record_count"] == 2
    assert manifest["fallback_count"] == 1
    assert manifest["parse_failure_count"] == 1
    assert manifest["synthetic_fixture_count"] == 0
    assert manifest["research_claim_eligible"] is False
