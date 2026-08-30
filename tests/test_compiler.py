from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vipibench.compiler import (
    CompileFailure,
    RenderRequest,
    compile_catalog,
    compile_single_candidate,
    default_context_renderer,
    load_compile_config,
)
from vipibench.dataio import sha256_file

CONFIG_PATH = Path("configs/benchmark/exec_catalog.yaml")
VIETNAM_PHONE_RE = re.compile(r"(?<![A-Za-z0-9])(?:\+?84|0)(?:3|5|7|8|9)\d{8}(?![A-Za-z0-9])")


@pytest.fixture(scope="module")
def compiled_catalog():
    config = load_compile_config(CONFIG_PATH)
    return compile_catalog(config, config_sha256=sha256_file(CONFIG_PATH))


def test_locked_catalog_compiles_exact_target_composition(compiled_catalog) -> None:
    manifest = compiled_catalog.manifest
    assert manifest["status"] == "PASS"
    assert manifest["template_count"] == 80
    assert manifest["episode_count"] == 2400
    assert manifest["split_family_counts"] == {"train": 48, "dev": 16, "test": 16}
    assert manifest["split_episode_counts"] == {"train": 1440, "dev": 480, "test": 480}
    assert manifest["label_counts"] == {"benign": 1200, "injection": 1200}
    assert manifest["hard_negative_count"] == 600
    assert manifest["complete_matched_pair_count"] == 200
    assert manifest["native_vietnamese_ratio"] == 0.6


def test_catalog_hash_is_deterministic(compiled_catalog) -> None:
    persisted = json.loads(
        Path("outputs/executable_benchmark_compile.json").read_text(encoding="utf-8")
    )
    assert (
        persisted["canonical_episode_set_sha256"]
        == compiled_catalog.manifest["canonical_episode_set_sha256"]
    )


def test_compiled_catalog_has_no_phone_shaped_synthetic_hash(compiled_catalog) -> None:
    for episode in compiled_catalog.episodes:
        assert not VIETNAM_PHONE_RE.search(episode.model_dump_json())


def test_single_candidate_is_deterministic() -> None:
    config = load_compile_config(CONFIG_PATH)
    first = compile_single_candidate(config, config_sha256=sha256_file(CONFIG_PATH))
    second = compile_single_candidate(config, config_sha256=sha256_file(CONFIG_PATH))
    assert first.episode.content_sha256 == second.episode.content_sha256


def test_reject_then_regenerate_is_recorded() -> None:
    config = load_compile_config(CONFIG_PATH)

    def fail_first_attempt(request: RenderRequest) -> str:
        if (
            request.family_id == "document_assistant-direct_override"
            and request.episode_index == 0
            and request.attempt == 0
        ):
            return ""
        return default_context_renderer(request)

    compiled = compile_single_candidate(
        config,
        config_sha256=sha256_file(CONFIG_PATH),
        renderer=fail_first_attempt,
    )
    assert len(compiled.rejections) == 1
    assert compiled.rejections[0]["episode_index"] == 0
    assert compiled.episode.metadata.transformation_history[2] == "renderer_attempt:1"


def test_retry_budget_exhaustion_fails_closed() -> None:
    config = load_compile_config(CONFIG_PATH)
    with pytest.raises(CompileFailure, match="exhausted retry budget"):
        compile_single_candidate(
            config,
            config_sha256=sha256_file(CONFIG_PATH),
            renderer=lambda _request: "",
        )
