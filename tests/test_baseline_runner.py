from pathlib import Path

from vipibench.baseline_runner import run_tfidf_baseline


def test_frozen_tfidf_baseline_runs_end_to_end(tmp_path: Path) -> None:
    config = tmp_path / "tfidf.yaml"
    config.write_text(
        "\n".join(
            [
                "run_name: tfidf_test",
                "model_type: tfidf_logistic",
                "input_mode: text_role",
                "seed: 17",
                "train_path: data/splits/frozen/train.jsonl",
                "dev_path: data/splits/frozen/dev.jsonl",
                "test_path: data/splits/frozen/test.jsonl",
                f"output_dir: {tmp_path.as_posix()}/model",
                "word_ngram_range: [1, 1]",
                "char_ngram_range: [3, 3]",
                "max_features: 2000",
                "class_weight: balanced",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.json"
    result = run_tfidf_baseline(config, project_root=Path.cwd(), output_path=output)
    assert result["status"] == "PASS"
    assert result["research_claim_eligible"] is True
    evaluation = result["evaluation"]
    assert isinstance(evaluation, dict)
    assert evaluation["metrics"]["count"] == 480
    assert evaluation["confidence_intervals"]["method"] == (
        "two_stage_family_then_nested_unit_percentile_bootstrap"
    )
    assert evaluation["confidence_intervals"]["family_weighting"] == "equal"
    assert "small_family_sensitivity" in evaluation["confidence_intervals"]
    assert output.is_file()
