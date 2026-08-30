import gc
import inspect
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from vipibench.checkpoint import StageLedger
from vipibench.dataio import sha256_file, write_json
from vipibench.transformer_runner import (
    _finite_class1_probabilities,
    _finite_optimizer_callback_class,
    _finite_trainer_class,
    _invalid_encoder_capacity_candidate,
    _load_or_measure_capacity,
    _load_trainable_encoder_model,
    _measure_encoder_capacity_candidate,
    _prepare_resume_checkpoint,
    _run_encoder_development_matrix,
    _run_numerics_steps,
    _seed_model_initialization,
    run_encoder_test_analysis_matrix,
    run_encoder_test_matrix,
    run_encoder_test_prediction_matrix,
)

torch = pytest.importorskip("torch")
save_safetensors = pytest.importorskip("safetensors.torch").save_file


def _write_checkpoint(
    root: Path,
    step: int,
    *,
    model_value: float = 1.0,
) -> Path:
    checkpoint = root / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    write_json(
        checkpoint / "trainer_state.json",
        {"global_step": step, "log_history": [{"loss": 0.5}]},
    )
    save_safetensors(
        {"classifier.weight": torch.tensor([[model_value]], dtype=torch.float32)},
        checkpoint / "model.safetensors",
    )
    torch.save(
        {"state": {0: {"exp_avg": torch.tensor([0.0])}}},
        checkpoint / "optimizer.pt",
    )
    torch.save({"last_epoch": step, "_last_lr": [2e-5]}, checkpoint / "scheduler.pt")
    torch.save({"cpu": torch.get_rng_state()}, checkpoint / "rng_state.pth")
    return checkpoint


def test_extreme_finite_logits_produce_finite_fp32_probabilities() -> None:
    logits = torch.tensor([[10000.0, -10000.0], [-10000.0, 10000.0]])

    probabilities = _finite_class1_probabilities(
        torch,
        logits,
        run_id="fixture",
        phase="dev",
    )

    assert probabilities.dtype == torch.float32
    assert torch.isfinite(probabilities).all()
    assert probabilities.tolist() == [0.0, 1.0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_logits_fail_with_run_and_phase(value: float) -> None:
    with pytest.raises(
        FloatingPointError,
        match=r"run_id=mdeberta-text_role-s43 phase=development_eval tensor=logits",
    ):
        _finite_class1_probabilities(
            torch,
            torch.tensor([[0.0, value]]),
            run_id="mdeberta-text_role-s43",
            phase="development_eval",
        )


def test_finite_trainer_rejects_non_finite_training_loss() -> None:
    class ParentTrainer:
        def __init__(self) -> None:
            self.state = SimpleNamespace(global_step=7)

        def compute_loss(self, model, inputs, *args, **kwargs):
            return torch.tensor(float("nan"))

    trainer_class = _finite_trainer_class(ParentTrainer, torch, "mdeberta-text_role-s43")

    with pytest.raises(
        FloatingPointError,
        match=r"phase=training_step_7 tensor=loss",
    ):
        trainer_class().compute_loss(None, {})


def test_finite_optimizer_callback_guards_backward_and_post_step_state() -> None:
    callback_class = _finite_optimizer_callback_class(
        object,
        torch,
        "mdeberta-role_only-s17",
    )
    callback = callback_class()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    state = SimpleNamespace(global_step=0)
    control = object()
    next(model.parameters()).grad = torch.full((2, 2), float("nan"))

    with pytest.raises(
        FloatingPointError,
        match=r"phase=optimizer_step_1_backward tensor=gradient:weight",
    ):
        callback.on_pre_optimizer_step(
            None,
            state,
            control,
            model=model,
            optimizer=optimizer,
        )

    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(
        FloatingPointError,
        match=r"phase=optimizer_step_1 tensor=parameter:weight",
    ):
        callback.on_optimizer_step(
            None,
            state,
            control,
            model=model,
            optimizer=optimizer,
        )


def test_model_initialization_seed_reproduces_new_head_parameters() -> None:
    _seed_model_initialization(torch, 17)
    first = torch.nn.Linear(4, 2)
    _seed_model_initialization(torch, 17)
    second = torch.nn.Linear(4, 2)

    assert torch.equal(first.weight, second.weight)
    assert torch.equal(first.bias, second.bias)


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, 2)

    def forward(self, input_ids, labels):
        logits = self.classifier(input_ids.to(dtype=torch.float32))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


class _TinyEmbeddingClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(8, 4)
        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, input_ids, labels):
        hidden = self.embeddings(input_ids).mean(dim=1)
        logits = self.classifier(hidden)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


def _finite_batches() -> list[list[dict[str, torch.Tensor]]]:
    return [
        [
            {
                "input_ids": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "labels": torch.tensor([0, 1]),
            }
        ],
        [
            {
                "input_ids": torch.tensor([[0.5, 0.5], [1.0, 1.0]]),
                "labels": torch.tensor([0, 1]),
            }
        ],
    ]


def test_two_step_numerics_canary_accepts_finite_training_state() -> None:
    model = _TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)

    result = _run_numerics_steps(
        torch,
        model,
        optimizer,
        _finite_batches(),
        run_id="capacity-canary-role_only",
        max_grad_norm=1.0,
    )

    assert result == {
        "status": "PASS",
        "optimizer_steps": 2,
        "micro_batches": 2,
    }


def test_fp32_protocol_upcasts_half_checkpoint_before_adamw_step() -> None:
    observed_kwargs: dict[str, object] = {}

    class HalfCheckpointModelClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            observed_kwargs.update(kwargs)
            checkpoint_model = _TinyEmbeddingClassifier().to(dtype=torch.float16)
            return checkpoint_model.to(dtype=kwargs.get("dtype", torch.float16))

    model = _load_trainable_encoder_model(
        HalfCheckpointModelClass,
        torch,
        {
            "backbone": "fixture-half-checkpoint",
            "model_revision": "fixture-revision",
            "mixed_precision": "fp32",
        },
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
    batches = [
        [
            {
                "input_ids": torch.tensor([[0, 1], [1, 0]]),
                "labels": torch.tensor([0, 1]),
            }
        ]
    ]

    result = _run_numerics_steps(
        torch,
        model,
        optimizer,
        batches,
        run_id="half-checkpoint-regression",
        max_grad_norm=1.0,
    )

    assert observed_kwargs["dtype"] == torch.float32
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert torch.isfinite(model.embeddings.weight).all()
    assert result == {"status": "PASS", "optimizer_steps": 1, "micro_batches": 1}


def test_bf16_amp_scout_keeps_fp32_master_parameters() -> None:
    observed_kwargs: dict[str, object] = {}

    class HalfCheckpointModelClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            observed_kwargs.update(kwargs)
            return _TinyEmbeddingClassifier().to(dtype=kwargs["dtype"])

    model = _load_trainable_encoder_model(
        HalfCheckpointModelClass,
        torch,
        {
            "backbone": "fixture-half-checkpoint",
            "model_revision": "fixture-revision",
            "mixed_precision": "bf16",
        },
    )

    assert observed_kwargs["dtype"] == torch.float32
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def test_capacity_candidate_frame_releases_post_backward_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    forward_calls = 0

    class TrackedClassifier(_TinyClassifier):
        def gradient_checkpointing_enable(self, **kwargs) -> None:
            raise AssertionError("checkpointing must remain disabled in this fixture")

        def forward(self, input_ids, labels):
            nonlocal forward_calls
            forward_calls += 1
            return super().forward(input_ids, labels)

    class ModelClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = TrackedClassifier()
            parameter_refs.append(weakref.ref(next(model.parameters())))
            return model

    class MovableBatch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return MovableBatch(input_ids=torch.ones((len(texts), 2)))

    class Memory:
        @staticmethod
        def max_memory_reserved() -> int:
            return 1024**3

    interface = SimpleNamespace(memory=Memory(), synchronize=lambda: None)
    records = [SimpleNamespace(label=SimpleNamespace(value="injection"))]
    config = {
        "backbone": "fixture",
        "model_revision": "fixture-revision",
        "seeds": [17],
        "learning_rate": 2e-5,
        "gradient_checkpointing_use_reentrant": False,
        "max_length": 2,
        "max_grad_norm": 1.0,
        "mixed_precision": "fp32",
        "effective_train_batch_size": 2,
        "capacity_probe_input_mode": "text_role",
        "capacity_warmup_optimizer_steps": 1,
        "capacity_measurement_optimizer_steps": 2,
    }
    monkeypatch.setattr(
        "vipibench.transformer_runner.detector_text",
        lambda record, input_mode: "fixture",
    )

    measurement = _measure_encoder_capacity_candidate(
        config,
        records,
        torch,
        ModelClass,
        Tokenizer(),
        torch.device("cpu"),
        interface,
        batch_size=1,
        checkpointing=False,
        total_gib=1.0,
    )
    gc.collect()

    assert measurement.completed is True
    assert measurement.effective_batch_size == 2
    assert measurement.gradient_accumulation_steps == 2
    assert measurement.warmup_optimizer_steps == 1
    assert measurement.measured_optimizer_steps == 2
    assert len(measurement.step_seconds) == 2
    assert len(measurement.step_samples_per_second) == 2
    assert measurement.samples_per_second > 0
    assert forward_calls == 6
    assert len(parameter_refs) == 1
    assert parameter_refs[0]() is None


def test_invalid_capacity_candidate_is_not_misreported_as_oom() -> None:
    measurement = _invalid_encoder_capacity_candidate(
        batch_size=128,
        checkpointing=True,
        effective_batch=64,
        total_gib=79.25,
        config={
            "capacity_warmup_optimizer_steps": 2,
            "capacity_probe_input_mode": "text_role",
        },
    )

    assert measurement.completed is False
    assert measurement.peak_reserved_gib == 0.0
    assert measurement.rejection_reason == "batch_size_must_divide_effective_batch"
    assert measurement.failure_type is None


def test_development_matrix_releases_trainer_before_loading_next_run() -> None:
    source = inspect.getsource(_run_encoder_development_matrix)
    completion = source.index("completed.append(run_id)")
    release = source.index("del trainer, model", completion)
    cache_release = source.index("torch.accelerator.memory.empty_cache()", release)

    assert completion < release < cache_release


def test_test_prediction_releases_accelerator_before_cpu_calibration() -> None:
    source = inspect.getsource(run_encoder_test_prediction_matrix)
    trainer_release = source.index("del trainer")
    model_release = source.index("del model", trainer_release)
    cache_release = source.index("torch.accelerator.memory.empty_cache()", model_release)
    calibration = source.index("calibrate_thresholds", cache_release)

    assert trainer_release < model_release < cache_release < calibration


def test_test_analysis_is_cpu_eligible_and_requires_prediction_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VIPIBENCH_CONFIRMATORY_RUN_APPROVED", "YES")
    selection_path = tmp_path / "model_selection.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    context = {
        "protocol": {"run_matrix": [{"run_id": "fixture"}]},
        "ledger": StageLedger(tmp_path / "stage_ledger"),
        "selection_path": selection_path,
    }
    monkeypatch.setattr(
        "vipibench.transformer_runner._prepare_encoder_test_context",
        lambda *args, **kwargs: context,
    )
    monkeypatch.setattr(
        "vipibench.transformer_runner._encoder_test_run_context",
        lambda *args, **kwargs: (
            "fixture",
            tmp_path / "fixture",
            tmp_path / "fixture" / "model",
            {"runner_sha256": "A" * 64},
        ),
    )

    with pytest.raises(FileNotFoundError, match="verified test prediction receipt missing"):
        run_encoder_test_analysis_matrix(Path("config"), Path("splits"), tmp_path)

    source = inspect.getsource(run_encoder_test_analysis_matrix)
    assert "_experiment_imports" not in source
    assert "from_pretrained" not in source
    assert "require_model_artifact=False" in source


def test_test_matrix_wrapper_orders_prediction_before_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_prediction(*args):
        calls.append("prediction")
        return {"status": "PASS"}

    def fake_analysis(*args):
        calls.append("analysis")
        return {
            "status": "PASS",
            "completed_runs": ["fixture"],
            "model_selection_sha256": "A" * 64,
        }

    monkeypatch.setattr(
        "vipibench.transformer_runner.run_encoder_test_prediction_matrix",
        fake_prediction,
    )
    monkeypatch.setattr(
        "vipibench.transformer_runner.run_encoder_test_analysis_matrix",
        fake_analysis,
    )

    result = run_encoder_test_matrix(Path("config"), Path("splits"), Path("outputs"))

    assert calls == ["prediction", "analysis"]
    assert result["status"] == "PASS"


def test_two_step_numerics_canary_rejects_non_finite_post_step_parameter() -> None:
    model = _TinyClassifier()

    class PoisoningOptimizer:
        def __init__(self) -> None:
            self.base = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
            self.state = self.base.state

        def zero_grad(self, *, set_to_none: bool) -> None:
            self.base.zero_grad(set_to_none=set_to_none)

        def step(self) -> None:
            self.base.step()
            with torch.no_grad():
                next(model.parameters()).fill_(float("nan"))

    with pytest.raises(
        FloatingPointError,
        match=r"phase=optimizer_step_1 tensor=parameter:classifier.weight",
    ):
        _run_numerics_steps(
            torch,
            model,
            PoisoningOptimizer(),
            _finite_batches(),
            run_id="capacity-canary-role_only",
            max_grad_norm=1.0,
        )


def test_two_step_numerics_canary_rejects_non_finite_gradient() -> None:
    class FiniteForwardNanBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.sum()

        @staticmethod
        def backward(ctx, grad_output):
            return torch.full((1,), float("nan"))

    class NanGradientModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, input_ids, labels):
            logits = torch.zeros((input_ids.shape[0], 2)) + self.weight * 0
            loss = FiniteForwardNanBackward.apply(self.weight)
            return SimpleNamespace(logits=logits, loss=loss)

    model = NanGradientModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)

    with pytest.raises(
        FloatingPointError,
        match=r"phase=optimizer_step_1_backward tensor=gradient:weight",
    ):
        _run_numerics_steps(
            torch,
            model,
            optimizer,
            [_finite_batches()[0]],
            run_id="capacity-canary-role_only",
            max_grad_norm=1.0,
        )


def test_numerics_canary_classifies_overflowed_gradient_norm() -> None:
    class FiniteLossHugeBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.new_tensor(1.0)

        @staticmethod
        def backward(ctx, grad_output):
            return torch.full((2,), 3.0e38)

    class GradientNormOverflowModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))

        def forward(self, input_ids, labels):
            logits = torch.zeros((input_ids.shape[0], 2)) + self.weight * 0
            loss = FiniteLossHugeBackward.apply(self.weight)
            return SimpleNamespace(logits=logits, loss=loss)

    model = GradientNormOverflowModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)

    with pytest.raises(
        FloatingPointError,
        match=r"phase=optimizer_step_1_backward tensor=gradient_norm",
    ):
        _run_numerics_steps(
            torch,
            model,
            optimizer,
            [_finite_batches()[0]],
            run_id="capacity-canary-role_only",
            max_grad_norm=1.0,
        )


def test_capacity_plan_reuse_rejects_runner_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "mdeberta_core.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    output_path = tmp_path / "capacity_plan.json"
    write_json(
        output_path,
        {
            "status": "PASS",
            "config_sha256": sha256_file(config_path),
            "runner_sha256": "0" * 64,
            "train_set_sha256": "unused-after-runner-mismatch",
            "selected": {"candidate_id": "batch-8-checkpoint-off"},
            "numerics_canary": {
                "status": "PASS",
                "candidate_id": "batch-8-checkpoint-off",
            },
        },
    )

    with pytest.raises(ValueError, match="stale source bindings"):
        _load_or_measure_capacity(
            output_path,
            config_path,
            {},
            [SimpleNamespace(content_sha256="A" * 64)],
            None,
            None,
            None,
            None,
        )


def test_resume_uses_numeric_latest_checkpoint_after_full_validation(tmp_path: Path) -> None:
    run_dir = tmp_path / "mdeberta-text_role-s43"
    binding = {"config_sha256": "A" * 64, "runner_sha256": "B" * 64}
    assert _prepare_resume_checkpoint(run_dir, binding, torch) is None
    _write_checkpoint(run_dir, 9)
    latest = _write_checkpoint(run_dir, 10)

    assert _prepare_resume_checkpoint(run_dir, binding, torch) == latest


def test_resume_rejects_non_finite_checkpoint_tensor(tmp_path: Path) -> None:
    run_dir = tmp_path / "mdeberta-text_role-s43"
    binding = {"config_sha256": "A" * 64}
    assert _prepare_resume_checkpoint(run_dir, binding, torch) is None
    _write_checkpoint(run_dir, 1, model_value=float("nan"))

    with pytest.raises(FloatingPointError, match="checkpoint contains non-finite tensor"):
        _prepare_resume_checkpoint(run_dir, binding, torch)


def test_resume_rejects_unbound_existing_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "mdeberta-text_role-s43"
    run_dir.mkdir()
    (run_dir / "legacy.txt").write_text("old lineage", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unbound existing run state"):
        _prepare_resume_checkpoint(run_dir, {"config_sha256": "A" * 64}, torch)


def test_resume_rejects_binding_drift(tmp_path: Path) -> None:
    run_dir = tmp_path / "mdeberta-text_role-s43"
    assert _prepare_resume_checkpoint(run_dir, {"runner_sha256": "A" * 64}, torch) is None

    with pytest.raises(RuntimeError, match="resume binding mismatch"):
        _prepare_resume_checkpoint(run_dir, {"runner_sha256": "B" * 64}, torch)


def test_resume_rejects_recorded_numerical_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "mdeberta-text_role-s43"
    binding = {"config_sha256": "A" * 64}
    assert _prepare_resume_checkpoint(run_dir, binding, torch) is None
    (run_dir / "numerical_failure.json").write_text(
        json.dumps({"status": "FAIL"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="cannot be resumed automatically"):
        _prepare_resume_checkpoint(run_dir, binding, torch)
