from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from sklearn.metrics import average_precision_score

from vipibench.artifact_binding import directory_fingerprint
from vipibench.benchmark_partitions import (
    CORE_TRACK,
    PROVENANCE_TRACK,
    load_benchmark_partitions,
    partition_by_track,
)
from vipibench.checkpoint import StageLedger
from vipibench.dataio import canonical_json, sha256_file, write_json
from vipibench.exec_detector_data import (
    detector_text,
    load_executable_episodes,
    prediction_row,
)
from vipibench.metrics import calibrate_thresholds, evaluate_predictions
from vipibench.modeling import load_yaml
from vipibench.run_protocol import validate_encoder_protocol, validate_public_detector_protocol
from vipibench.runtime_capacity import (
    CapacityMeasurement,
    check_runtime_profile_path,
    is_capacity_exhaustion,
    rank_capacity_candidates,
    select_capacity_candidate,
    validate_model_device_placement,
)

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-([1-9][0-9]*)$")
_RESUME_BINDING_NAME = "resume_binding.json"
_NUMERICAL_FAILURE_NAME = "numerical_failure.json"
_DATALOADER_WORKER_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "measurement_kind",
        "config_sha256",
        "runner_sha256",
        "train_set_sha256",
        "capacity_plan_sha256",
        "selected_batch_size",
        "probe_partition",
        "probe_input_mode",
        "candidate_num_workers",
        "warmup_batches",
        "measurement_batches",
        "repeats",
        "logical_cpu_count_at_measurement",
        "system_ram_total_gib_at_measurement",
        "measurements",
        "selected_num_workers",
        "selection_rule",
        "test_accessed",
        "final_holdout_feedback_allowed",
        "claim_boundary",
    }
)
_DATALOADER_WORKER_MEASUREMENT_FIELDS = frozenset(
    {
        "num_workers",
        "status",
        "repeat_samples_per_second",
        "repeat_elapsed_seconds",
        "repeat_sample_counts",
        "median_samples_per_second",
        "error_type",
        "error_message",
    }
)


@dataclass(frozen=True)
class EncoderCapacityMeasurement(CapacityMeasurement):
    """Capacity result with enough timing detail to audit the ranking decision."""

    effective_batch_size: int = 0
    gradient_accumulation_steps: int = 0
    warmup_optimizer_steps: int = 0
    measured_optimizer_steps: int = 0
    step_seconds: tuple[float, ...] = ()
    step_samples_per_second: tuple[float, ...] = ()
    minimum_samples_per_second: float = 0.0
    p25_samples_per_second: float = 0.0
    p75_samples_per_second: float = 0.0
    maximum_samples_per_second: float = 0.0
    total_measured_seconds: float = 0.0
    timing_method: str = "synchronized_effective_batch_wall_clock_v2"
    throughput_statistic: str = "median"
    probe_input_mode: str = "text_role"
    rejection_reason: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None


class _EncodedDataset:
    def __init__(
        self,
        rows: list[Any],
        tokenizer: Any,
        input_mode: str,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.rows[index]
        encoded = self.tokenizer(
            detector_text(record, self.input_mode),
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        return {
            **encoded,
            "labels": 1 if record.label.value == "injection" else 0,
        }


def _experiment_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("install the locked experiment dependency group") from exc
    return (
        torch,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        (
            Trainer,
            TrainingArguments,
            EarlyStoppingCallback,
            DataCollatorWithPadding,
            TrainerCallback,
        ),
    )


def _current_device(torch: Any) -> Any:
    interface = getattr(torch, "accelerator", None)
    if interface is not None:
        device = interface.current_accelerator(check_available=True)
        if device is not None:
            return device
    return torch.device("cpu")


def _synchronize_accelerator(torch: Any) -> None:
    """Synchronize when an accelerator exists so wall-clock measurements are honest."""

    interface = getattr(torch, "accelerator", None)
    synchronize = getattr(interface, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _require_finite_tensor(
    torch: Any,
    value: Any,
    *,
    run_id: str,
    phase: str,
    tensor_name: str,
) -> Any:
    tensor = torch.as_tensor(value)
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return tensor
    non_finite_count = int((~finite).sum().item())
    raise FloatingPointError(
        "non-finite tensor detected "
        f"run_id={run_id} phase={phase} tensor={tensor_name} "
        f"count={non_finite_count} total={tensor.numel()}"
    )


def _seed_model_initialization(torch: Any, seed: int) -> None:
    """Bind newly initialized task heads to the preregistered run seed."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "manual_seed_all", None)):
        cuda.manual_seed_all(seed)


def _load_trainable_encoder_model(
    model_class: Any,
    torch: Any,
    config: dict[str, object],
) -> Any:
    """Load FP32 master parameters for FP32 or explicitly exploratory BF16 AMP."""

    mixed_precision = str(config["mixed_precision"])
    if mixed_precision not in {"fp32", "bf16"}:
        raise ValueError("trainable encoder precision must be fp32 or bf16 AMP")
    model = model_class.from_pretrained(
        str(config["backbone"]),
        revision=str(config["model_revision"]),
        trust_remote_code=False,
        num_labels=2,
        dtype=torch.float32,
    )
    unexpected = [
        f"{name}:{parameter.dtype}"
        for name, parameter in model.named_parameters()
        if bool(parameter.is_floating_point()) and parameter.dtype != torch.float32
    ]
    if unexpected:
        raise RuntimeError(
            "trainable encoder contains non-fp32 parameters: " + ", ".join(unexpected[:5])
        )
    return model


def _finite_class1_probabilities(
    torch: Any,
    logits: Any,
    *,
    run_id: str,
    phase: str,
) -> Any:
    tensor = torch.as_tensor(logits)
    if tensor.ndim != 2 or int(tensor.shape[1]) != 2:
        raise ValueError(
            "binary classifier logits must have shape [N, 2]: "
            f"run_id={run_id} phase={phase} shape={tuple(tensor.shape)}"
        )
    tensor = tensor.to(dtype=torch.float32)
    _require_finite_tensor(
        torch,
        tensor,
        run_id=run_id,
        phase=phase,
        tensor_name="logits",
    )
    probabilities = torch.softmax(tensor, dim=-1)[:, 1]
    _require_finite_tensor(
        torch,
        probabilities,
        run_id=run_id,
        phase=phase,
        tensor_name="class1_probabilities",
    )
    return probabilities


def _finite_trainer_class(trainer_class: Any, torch: Any, run_id: str) -> Any:
    class FiniteTrainer(trainer_class):
        def compute_loss(self, model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
            result = super().compute_loss(model, inputs, *args, **kwargs)
            loss = result[0] if isinstance(result, tuple) else result
            _require_finite_tensor(
                torch,
                loss,
                run_id=run_id,
                phase=f"training_step_{int(self.state.global_step)}",
                tensor_name="loss",
            )
            return result

    return FiniteTrainer


def _finite_optimizer_callback_class(callback_class: Any, torch: Any, run_id: str) -> Any:
    class FiniteOptimizerCallback(callback_class):
        def on_pre_optimizer_step(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            model = kwargs.get("model")
            if model is None:
                raise RuntimeError("finite optimizer callback did not receive the model")
            _require_finite_gradients(
                torch,
                model,
                run_id=run_id,
                phase=f"optimizer_step_{int(state.global_step) + 1}_backward",
            )
            return control

        def on_optimizer_step(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            model = kwargs.get("model")
            optimizer = kwargs.get("optimizer")
            if model is None or optimizer is None:
                raise RuntimeError("finite optimizer callback did not receive model and optimizer")
            phase = f"optimizer_step_{int(state.global_step) + 1}"
            _require_finite_model_state(torch, model, run_id=run_id, phase=phase)
            _require_finite_optimizer_state(
                torch,
                model,
                optimizer,
                run_id=run_id,
                phase=phase,
            )
            return control

    return FiniteOptimizerCallback


def _assert_finite_structure(torch: Any, value: Any, *, location: str) -> None:
    if torch.is_tensor(value):
        if bool(value.is_floating_point()) or bool(value.is_complex()):
            finite = torch.isfinite(value)
            if not bool(finite.all().item()):
                raise FloatingPointError(f"checkpoint contains non-finite tensor: {location}")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError(f"checkpoint contains non-finite scalar: {location}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_structure(torch, item, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_structure(torch, item, location=f"{location}[{index}]")


def _require_finite_nested_state(
    torch: Any,
    value: Any,
    *,
    run_id: str,
    phase: str,
    tensor_name: str,
) -> None:
    if torch.is_tensor(value):
        if bool(value.is_floating_point()) or bool(value.is_complex()):
            tensor = value.coalesce().values() if bool(value.is_sparse) else value
            _require_finite_tensor(
                torch,
                tensor,
                run_id=run_id,
                phase=phase,
                tensor_name=tensor_name,
            )
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError(
                "non-finite scalar detected "
                f"run_id={run_id} phase={phase} tensor={tensor_name}"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_nested_state(
                torch,
                item,
                run_id=run_id,
                phase=phase,
                tensor_name=f"{tensor_name}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_nested_state(
                torch,
                item,
                run_id=run_id,
                phase=phase,
                tensor_name=f"{tensor_name}[{index}]",
            )


def _require_finite_model_state(
    torch: Any,
    model: Any,
    *,
    run_id: str,
    phase: str,
) -> None:
    for name, parameter in model.named_parameters():
        if bool(parameter.is_floating_point()) or bool(parameter.is_complex()):
            _require_finite_tensor(
                torch,
                parameter.detach(),
                run_id=run_id,
                phase=phase,
                tensor_name=f"parameter:{name}",
            )


def _require_finite_gradients(
    torch: Any,
    model: Any,
    *,
    run_id: str,
    phase: str,
) -> None:
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        tensor = gradient.coalesce().values() if bool(gradient.is_sparse) else gradient
        _require_finite_tensor(
            torch,
            tensor,
            run_id=run_id,
            phase=phase,
            tensor_name=f"gradient:{name}",
        )


def _require_finite_optimizer_state(
    torch: Any,
    model: Any,
    optimizer: Any,
    *,
    run_id: str,
    phase: str,
) -> None:
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    for index, (parameter, state) in enumerate(optimizer.state.items()):
        name = parameter_names.get(id(parameter), f"index_{index}")
        _require_finite_nested_state(
            torch,
            state,
            run_id=run_id,
            phase=phase,
            tensor_name=f"optimizer_state:{name}",
        )


def _run_numerics_steps(
    torch: Any,
    model: Any,
    optimizer: Any,
    optimizer_batches: list[list[dict[str, Any]]],
    *,
    run_id: str,
    max_grad_norm: float,
    mixed_precision: str = "fp32",
) -> dict[str, object]:
    if not optimizer_batches or any(not batches for batches in optimizer_batches):
        raise ValueError("numerics canary requires non-empty optimizer steps")
    if mixed_precision not in {"fp32", "bf16"}:
        raise ValueError("numerics canary precision must be fp32 or bf16")
    _require_finite_model_state(torch, model, run_id=run_id, phase="initialization")
    model.train()
    micro_batch_count = 0
    for optimizer_step, micro_batches in enumerate(optimizer_batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        for micro_batch, inputs in enumerate(micro_batches, start=1):
            tensor_inputs = [value for value in inputs.values() if torch.is_tensor(value)]
            if not tensor_inputs:
                raise ValueError("numerics canary batch contains no tensor inputs")
            precision_context = (
                torch.autocast(
                    device_type=str(tensor_inputs[0].device.type),
                    dtype=torch.bfloat16,
                )
                if mixed_precision == "bf16"
                else nullcontext()
            )
            with precision_context:
                outputs = model(**inputs)
            logits = getattr(outputs, "logits", None)
            loss = getattr(outputs, "loss", None)
            if logits is None or loss is None:
                raise RuntimeError("numerics canary model output must expose logits and loss")
            phase = f"optimizer_step_{optimizer_step}_micro_batch_{micro_batch}"
            _require_finite_tensor(
                torch,
                logits,
                run_id=run_id,
                phase=phase,
                tensor_name="logits",
            )
            _require_finite_tensor(
                torch,
                loss,
                run_id=run_id,
                phase=phase,
                tensor_name="loss",
            )
            (loss / len(micro_batches)).backward()
            micro_batch_count += 1
        backward_phase = f"optimizer_step_{optimizer_step}_backward"
        _require_finite_gradients(
            torch,
            model,
            run_id=run_id,
            phase=backward_phase,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
            error_if_nonfinite=False,
        )
        _require_finite_tensor(
            torch,
            grad_norm,
            run_id=run_id,
            phase=backward_phase,
            tensor_name="gradient_norm",
        )
        optimizer.step()
        step_phase = f"optimizer_step_{optimizer_step}"
        _require_finite_model_state(torch, model, run_id=run_id, phase=step_phase)
        _require_finite_optimizer_state(
            torch,
            model,
            optimizer,
            run_id=run_id,
            phase=step_phase,
        )
    return {
        "status": "PASS",
        "optimizer_steps": len(optimizer_batches),
        "micro_batches": micro_batch_count,
    }


def _load_checkpoint_payload(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"checkpoint payload is unreadable: {path.name}") from exc


def _validate_resume_checkpoint(torch: Any, checkpoint: Path) -> None:
    match = _CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(f"invalid checkpoint directory name: {checkpoint.name}")
    expected_step = int(match.group(1))
    trainer_state_path = checkpoint / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise FileNotFoundError(f"checkpoint missing trainer_state.json: {checkpoint.name}")
    try:
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"checkpoint trainer state is unreadable: {checkpoint.name}") from exc
    if int(trainer_state.get("global_step", -1)) != expected_step:
        raise ValueError(f"checkpoint global step mismatch: {checkpoint.name}")
    _assert_finite_structure(torch, trainer_state, location=f"{checkpoint.name}.trainer_state")

    model_files = [
        path
        for path in (
            checkpoint / "model.safetensors",
            checkpoint / "pytorch_model.bin",
        )
        if path.is_file()
    ]
    if len(model_files) != 1:
        raise FileNotFoundError(
            f"checkpoint must contain exactly one supported model payload: {checkpoint.name}"
        )
    model_path = model_files[0]
    if model_path.suffix == ".safetensors":
        try:
            from safetensors import safe_open

            with safe_open(model_path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if not keys:
                    raise ValueError("empty safetensors payload")
                for key in keys:
                    _assert_finite_structure(
                        torch,
                        handle.get_tensor(key),
                        location=f"{checkpoint.name}.{model_path.name}.{key}",
                    )
        except FloatingPointError:
            raise
        except Exception as exc:
            raise RuntimeError(f"checkpoint model is unreadable: {checkpoint.name}") from exc
    else:
        _assert_finite_structure(
            torch,
            _load_checkpoint_payload(torch, model_path),
            location=f"{checkpoint.name}.{model_path.name}",
        )

    for name in ("optimizer.pt", "scheduler.pt"):
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint missing {name}: {checkpoint.name}")
        _assert_finite_structure(
            torch,
            _load_checkpoint_payload(torch, path),
            location=f"{checkpoint.name}.{name}",
        )
    rng_path = checkpoint / "rng_state.pth"
    if not rng_path.is_file() or rng_path.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint missing nonempty rng_state.pth: {checkpoint.name}")


def _prepare_resume_checkpoint(
    run_dir: Path,
    binding_metadata: dict[str, object],
    torch: Any,
) -> Path | None:
    binding_path = run_dir / _RESUME_BINDING_NAME
    expected_binding = {
        "schema_version": "1.0.0",
        "status": "BOUND",
        "metadata": binding_metadata,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_candidates = (
        [path for path in checkpoint_root.iterdir() if path.is_dir()]
        if checkpoint_root.is_dir()
        else []
    )
    unexpected = sorted(
        path.name
        for path in checkpoint_candidates
        if _CHECKPOINT_PATTERN.fullmatch(path.name) is None
    )
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint directories for {run_dir.name}: {unexpected}")
    checkpoints = sorted(
        checkpoint_candidates,
        key=lambda path: int(_CHECKPOINT_PATTERN.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
    )

    existing_entries = [
        path for path in run_dir.iterdir() if path.name not in {_RESUME_BINDING_NAME}
    ]
    if not binding_path.is_file():
        if existing_entries:
            raise RuntimeError(
                f"unbound existing run state cannot be resumed: {run_dir.name}; "
                "start from the new durable lineage or preserve and clear this stale run directory"
            )
        write_json(binding_path, expected_binding)
    else:
        try:
            observed_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"resume binding is unreadable: {run_dir.name}") from exc
        if observed_binding != expected_binding:
            raise RuntimeError(f"resume binding mismatch: {run_dir.name}")

    failure_path = run_dir / _NUMERICAL_FAILURE_NAME
    if failure_path.is_file():
        raise RuntimeError(
            f"numerically failed run cannot be resumed automatically: {run_dir.name}"
        )
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    _validate_resume_checkpoint(torch, latest)
    return latest


def _record_numerical_failure(run_dir: Path, run_id: str, exc: FloatingPointError) -> None:
    write_json(
        run_dir / _NUMERICAL_FAILURE_NAME,
        {
            "schema_version": "1.0.0",
            "status": "FAIL",
            "run_id": run_id,
            "recorded_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "resume_allowed": False,
            "claim_boundary": (
                "This record proves a non-finite numerical value was detected. It does not "
                "identify the first upstream operation that created the value."
            ),
        },
    )


def _write_predictions(
    path: Path,
    records: list[Any],
    scores: list[float],
    split: str,
    latency_ms: float,
    *,
    input_mode: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record, score in zip(records, scores, strict=True):
            row = prediction_row(
                record,
                float(score),
                split=split,
                latency_ms=latency_ms,
                input_mode=input_mode,
            )
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_track_predictions(
    run_dir: Path,
    records: list[Any],
    scores: list[float],
    split: str,
    latency_ms: float,
    *,
    input_mode: str | None = None,
) -> dict[str, Path]:
    if len(records) != len(scores):
        raise ValueError("record and score counts must match before track partitioning")
    scored = {record.episode_id: score for record, score in zip(records, scores, strict=True)}
    partitions = partition_by_track(records)
    paths = {
        CORE_TRACK: run_dir / f"core_{split}_predictions.jsonl",
        PROVENANCE_TRACK: run_dir / f"provenance_{split}_predictions.jsonl",
    }
    for track, members in partitions.items():
        _write_predictions(
            paths[track],
            members,
            [float(scored[member.episode_id]) for member in members],
            split,
            latency_ms,
            input_mode=input_mode,
        )
    return paths


def _load_public_detector(
    config: dict[str, object],
) -> tuple[Any, Any, Any, Any]:
    torch, model_class, tokenizer_class, _ = _experiment_imports()
    revision = str(config["model_revision"])
    tokenizer = tokenizer_class.from_pretrained(
        str(config["backbone"]),
        revision=str(config["tokenizer_revision"]),
        trust_remote_code=False,
    )
    model = model_class.from_pretrained(
        str(config["backbone"]),
        revision=revision,
        trust_remote_code=False,
    )
    device = _current_device(torch)
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _synchronize_accelerator(torch: Any) -> None:
    interface = getattr(torch, "accelerator", None)
    if interface is not None and interface.is_available():
        interface.synchronize()


def _score_public_detector_batch(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    texts: list[str],
    config: dict[str, object],
    *,
    phase: str,
) -> list[float]:
    tokens = tokenizer(
        texts,
        max_length=int(config["max_length"]),
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    if int(config["injection_label_id"]) != 1:
        raise ValueError("public detector injection label must remain class 1")
    with torch.inference_mode():
        probabilities = _finite_class1_probabilities(
            torch,
            model(**tokens).logits,
            run_id="public-detector",
            phase=phase,
        )
    return [float(value) for value in probabilities.detach().cpu().tolist()]


def _measure_public_detector_capacity(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    development_texts: list[str],
    config: dict[str, object],
) -> dict[str, object]:
    if not development_texts:
        raise ValueError("public detector capacity scout requires development records")
    interface = torch.accelerator
    _, total_bytes = interface.memory.get_memory_info()
    total_gib = total_bytes / (1024**3)
    measurements: list[CapacityMeasurement] = []
    repeat_rates: dict[str, list[float]] = {}
    warmup_batches = int(config["capacity_warmup_batches"])
    measurement_batches = int(config["capacity_measurement_batches"])
    repeats = int(config["capacity_repeats"])
    for batch_size in [int(value) for value in config["batch_candidates"]]:
        candidate_id = f"batch-{batch_size}"
        completed = True
        throughput = 0.0
        peak_gib = total_gib
        rates: list[float] = []
        probe_texts = [
            development_texts[index % len(development_texts)] for index in range(batch_size)
        ]
        try:
            interface.memory.empty_cache()
            for _ in range(warmup_batches):
                _score_public_detector_batch(
                    torch,
                    tokenizer,
                    model,
                    device,
                    probe_texts,
                    config,
                    phase="development_capacity_warmup",
                )
                interface.synchronize()
            interface.memory.reset_peak_memory_stats()
            for _ in range(repeats):
                interface.synchronize()
                started = time.perf_counter()
                for _ in range(measurement_batches):
                    _score_public_detector_batch(
                        torch,
                        tokenizer,
                        model,
                        device,
                        probe_texts,
                        config,
                        phase="development_capacity_measurement",
                    )
                interface.synchronize()
                elapsed = max(time.perf_counter() - started, 1e-9)
                rates.append(batch_size * measurement_batches / elapsed)
            throughput = float(statistics.median(rates))
            peak_gib = interface.memory.max_memory_reserved() / (1024**3)
        except RuntimeError as exc:
            if not is_capacity_exhaustion(torch, exc):
                raise
            completed = False
            rates = []
            interface.memory.empty_cache()
        repeat_rates[candidate_id] = rates
        measurements.append(
            CapacityMeasurement(
                candidate_id=candidate_id,
                batch_size=batch_size,
                samples_per_second=throughput,
                peak_reserved_gib=peak_gib,
                total_memory_gib=total_gib,
                completed=completed,
            )
        )
    result = select_capacity_candidate(
        measurements,
        maximum_utilization=float(config["target_memory_utilization"]),
    )
    result.update(
        {
            "schema_version": "1.0.0",
            "selection_partition": "development_only",
            "test_accessed": False,
            "final_holdout_feedback_allowed": False,
            "measurement_contract": {
                "timing": "synchronized_wall_clock",
                "warmup_batches": warmup_batches,
                "measurement_batches_per_repeat": measurement_batches,
                "repeats": repeats,
                "aggregation": "median_samples_per_second",
            },
            "repeat_samples_per_second": repeat_rates,
        }
    )
    return result


def _run_public_detector_loaded(
    config: dict[str, object],
    dataset_path: Path,
    output_path: Path,
    *,
    split: str,
    batch_size: int,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
) -> dict[str, object]:
    records = load_executable_episodes(dataset_path)
    texts = [detector_text(record, str(config["input_mode"])) for record in records]
    scores: list[float] = []
    _synchronize_accelerator(torch)
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        scores.extend(
            _score_public_detector_batch(
                torch,
                tokenizer,
                model,
                device,
                texts[start : start + batch_size],
                config,
                phase=f"{split}_inference",
            )
        )
    _synchronize_accelerator(torch)
    per_item_ms = (time.perf_counter() - started) * 1000 / len(records) if records else 0.0
    _write_predictions(
        output_path,
        records,
        scores,
        split,
        per_item_ms,
        input_mode=str(config["input_mode"]),
    )
    result = {
        "status": "PASS",
        "model_revision": str(config["model_revision"]),
        "dataset_sha256": sha256_file(dataset_path),
        "predictions_sha256": sha256_file(output_path),
        "count": len(records),
        "device_type": str(device.type),
        "batch_size": batch_size,
        "timing": "synchronized_wall_clock",
    }
    write_json(output_path.with_suffix(".manifest.json"), result)
    return result


def run_public_detector(
    config_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    split: str,
) -> dict[str, object]:
    protocol = validate_public_detector_protocol(config_path)
    if protocol["status"] != "PASS":
        raise ValueError(protocol["errors"])
    config = load_yaml(config_path)
    torch, tokenizer, model, device = _load_public_detector(config)
    return _run_public_detector_loaded(
        config,
        dataset_path,
        output_path,
        split=split,
        batch_size=int(config["batch_candidates"][0]),
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def run_public_detector_benchmark(
    config_path: Path,
    split_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    _require_confirmatory_authorization()
    protocol = validate_public_detector_protocol(config_path)
    if protocol["status"] != "PASS":
        raise ValueError(protocol["errors"])
    config = load_yaml(config_path)
    runtime = check_runtime_profile_path(Path(str(config["runtime_profile"])), split_dir)
    if runtime["status"] != "PASS" or runtime["hardware_observed"] is not True:
        raise RuntimeError(runtime["errors"])
    records, source_hashes = load_benchmark_partitions(
        split_dir,
        Path(str(config["contrast_dataset"])),
    )
    input_dir = output_root / "bound_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_paths: dict[str, Path] = {}
    for split in ("dev", "test"):
        path = input_dir / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for episode in records[split]:
                handle.write(canonical_json(episode.model_dump(mode="json")) + "\n")
        temporary.replace(path)
        input_paths[split] = path
    torch, tokenizer, model, device = _load_public_detector(config)
    placement = validate_model_device_placement(model, model_label="public_detector")
    development_texts = [
        detector_text(record, str(config["input_mode"])) for record in records["dev"]
    ]
    capacity = _measure_public_detector_capacity(
        torch,
        tokenizer,
        model,
        device,
        development_texts,
        config,
    )
    if capacity["status"] != "PASS":
        raise RuntimeError(capacity["errors"])
    capacity.update(
        {
            "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "development_input_sha256": sha256_file(input_paths["dev"]),
            "model_placement": placement,
        }
    )
    capacity_path = output_root / "capacity_plan.json"
    write_json(capacity_path, capacity)
    selected_batch_size = int(capacity["selected"]["batch_size"])
    dev_predictions = output_root / "dev_predictions.jsonl"
    test_predictions = output_root / "test_predictions.jsonl"
    dev = _run_public_detector_loaded(
        config,
        input_paths["dev"],
        dev_predictions,
        split="dev",
        batch_size=selected_batch_size,
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    test = _run_public_detector_loaded(
        config,
        input_paths["test"],
        test_predictions,
        split="test",
        batch_size=selected_batch_size,
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    thresholds_path = output_root / "thresholds.json"
    evaluation_path = output_root / "evaluation.json"
    thresholds = calibrate_thresholds(dev_predictions, thresholds_path)
    evaluation = evaluate_predictions(test_predictions, thresholds_path, evaluation_path)
    result = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "config_sha256": sha256_file(config_path),
        "source_hashes": source_hashes,
        "bound_input_hashes": {
            split: sha256_file(path) for split, path in input_paths.items()
        },
        "capacity_plan_sha256": sha256_file(capacity_path),
        "selected_batch_size": selected_batch_size,
        "shared_model_load_count": 1,
        "model_placement": placement,
        "development": dev,
        "test": test,
        "thresholds": thresholds,
        "evaluation": evaluation,
        "research_claim_eligible": evaluation["research_claim_eligible"],
        "claim_boundary": (
            "This pinned public detector is an external baseline evaluated on both frozen "
            "benchmark tracks. It is not evidence of model training or system containment."
        ),
    }
    write_json(output_root / "benchmark_manifest.json", result)
    return result


def run_encoder_matrix(
    config_path: Path,
    split_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    """Compatibility wrapper for the accelerator matrix followed by CPU analysis."""

    accelerator = run_encoder_accelerator_matrix(config_path, split_dir, output_root)
    analysis = run_encoder_test_analysis_matrix(config_path, split_dir, output_root)
    return {
        "status": "PASS",
        "development": accelerator["development"],
        "selection": accelerator["selection"],
        "prediction": accelerator["prediction"],
        "analysis": analysis,
        "required_run_count": 9,
    }


def run_encoder_accelerator_matrix(
    config_path: Path,
    split_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    """Run development, lock selection, and finish all accelerator-only test prediction."""

    _require_confirmatory_authorization()
    development = run_encoder_development_matrix(config_path, split_dir, output_root)
    selection = select_encoder_run(config_path, output_root)
    prediction = run_encoder_test_prediction_matrix(config_path, split_dir, output_root)
    return {
        "status": "PASS",
        "phase": "accelerator_encoder_matrix",
        "development": development,
        "selection": selection,
        "prediction": prediction,
        "required_run_count": 9,
        "analysis_complete": False,
        "research_claim_eligible": False,
    }


def run_encoder_development_matrix(
    config_path: Path,
    split_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    _require_confirmatory_authorization()
    protocol = validate_encoder_protocol(config_path)
    if protocol["status"] != "PASS":
        raise ValueError(protocol["errors"])
    return _run_encoder_development_matrix(
        config_path,
        split_dir,
        output_root,
        protocol=protocol,
        execution_scope="confirmatory_fp32",
    )


def _run_encoder_development_matrix(
    config_path: Path,
    split_dir: Path,
    output_root: Path,
    *,
    protocol: dict[str, object],
    execution_scope: str,
    partition_names: tuple[str, ...] = ("train", "dev", "test"),
) -> dict[str, object]:
    config = load_yaml(config_path)
    runtime = check_runtime_profile_path(Path(str(config["runtime_profile"])), split_dir)
    if runtime["status"] != "PASS" or runtime["hardware_observed"] is not True:
        raise RuntimeError(runtime["errors"])
    records, source_hashes = _load_training_records(
        config,
        split_dir,
        partition_names=partition_names,
    )
    torch, model_class, tokenizer_class, trainer_types = _experiment_imports()
    (
        trainer_class,
        arguments_class,
        early_stopping_class,
        collator_class,
        trainer_callback_class,
    ) = trainer_types
    tokenizer = tokenizer_class.from_pretrained(
        str(config["backbone"]),
        revision=str(config["tokenizer_revision"]),
        trust_remote_code=False,
    )
    collator = collator_class(tokenizer=tokenizer, pad_to_multiple_of=8)
    capacity_path = output_root / "capacity_plan.json"
    capacity = _load_or_measure_capacity(
        capacity_path,
        config_path,
        config,
        records["train"],
        torch,
        model_class,
        tokenizer,
        collator,
    )
    selected = capacity.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("capacity plan has no selected candidate")
    batch_size = int(selected["batch_size"])
    checkpointing = str(selected["candidate_id"]).endswith("checkpoint-on")
    effective_batch = int(config["effective_train_batch_size"])
    if effective_batch % batch_size != 0:
        raise ValueError("selected batch must divide the locked effective batch")
    accumulation = effective_batch // batch_size
    dataloader_worker_plan_path = output_root / "dataloader_worker_plan.json"
    dataloader_worker_plan = _load_or_measure_dataloader_workers(
        dataloader_worker_plan_path,
        config_path,
        config,
        records["train"],
        capacity_path,
        batch_size,
        torch,
        tokenizer,
        collator,
    )
    dataloader_num_workers = int(dataloader_worker_plan["selected_num_workers"])

    ledger = StageLedger(output_root / "stage_ledger")
    runner_sha256 = sha256_file(Path(__file__))
    completed: list[str] = []
    for entry in protocol["run_matrix"]:
        run_id = str(entry["run_id"])
        stage_id = f"development-{run_id}"
        stage_metadata = {
            "seed": int(entry["seed"]),
            "input_mode": str(entry["input_mode"]),
            "config_sha256": sha256_file(config_path),
            "source_hashes": source_hashes,
            "capacity_plan_sha256": sha256_file(capacity_path),
            "dataloader_worker_plan_sha256": sha256_file(dataloader_worker_plan_path),
            "dataloader_num_workers": dataloader_num_workers,
            "runner_sha256": runner_sha256,
            "mixed_precision": str(config["mixed_precision"]),
            "parameter_storage_dtype": "fp32",
            "execution_scope": execution_scope,
            "numerics_policy": str(config["numerics_policy"]),
            "gradient_checkpointing_use_reentrant": bool(
                config["gradient_checkpointing_use_reentrant"]
            ),
            "numerics_canary_optimizer_steps": int(
                config["numerics_canary_optimizer_steps"]
            ),
            "test_accessed": False,
        }
        if ledger.verified_complete(stage_id, stage_metadata):
            completed.append(run_id)
            continue
        mode = str(entry["input_mode"])
        seed = int(entry["seed"])
        run_dir = output_root / run_id
        resume_checkpoint = _prepare_resume_checkpoint(run_dir, stage_metadata, torch)
        print(
            f"[encoder-run] START run_id={run_id} "
            f"resume={resume_checkpoint if resume_checkpoint is not None else 'fresh'}",
            flush=True,
        )
        _seed_model_initialization(torch, seed)
        model = _load_trainable_encoder_model(model_class, torch, config)

        def compute_metrics(
            prediction: Any,
            *,
            _run_id: str = run_id,
        ) -> dict[str, float]:
            labels = prediction.label_ids
            probabilities = (
                _finite_class1_probabilities(
                    torch,
                    prediction.predictions,
                    run_id=_run_id,
                    phase="development_eval",
                )
                .detach()
                .cpu()
                .numpy()
            )
            return {"auprc": float(average_precision_score(labels, probabilities))}

        arguments = arguments_class(
            output_dir=str(run_dir / "checkpoints"),
            seed=seed,
            data_seed=seed,
            learning_rate=float(config["learning_rate"]),
            num_train_epochs=float(config["epochs"]),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            gradient_checkpointing=checkpointing,
            gradient_checkpointing_kwargs=(
                {
                    "use_reentrant": bool(
                        config["gradient_checkpointing_use_reentrant"]
                    )
                }
                if checkpointing
                else None
            ),
            bf16=str(config["mixed_precision"]) == "bf16",
            fp16=False,
            optim=str(config["optimizer"]),
            max_grad_norm=float(config["max_grad_norm"]),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_auprc",
            greater_is_better=True,
            report_to=[],
            save_total_limit=2,
            logging_nan_inf_filter=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_persistent_workers=bool(config["dataloader_persistent_workers"]),
            dataloader_prefetch_factor=int(config["dataloader_prefetch_factor"]),
        )
        finite_trainer_class = _finite_trainer_class(trainer_class, torch, run_id)
        finite_callback_class = _finite_optimizer_callback_class(
            trainer_callback_class,
            torch,
            run_id,
        )
        trainer = finite_trainer_class(
            model=model,
            args=arguments,
            train_dataset=_EncodedDataset(
                records["train"], tokenizer, mode, int(config["max_length"])
            ),
            eval_dataset=_EncodedDataset(
                records["dev"], tokenizer, mode, int(config["max_length"])
            ),
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[
                finite_callback_class(),
                early_stopping_class(
                    early_stopping_patience=int(config["early_stopping_patience"]),
                    early_stopping_threshold=float(config["early_stopping_threshold"]),
                )
            ],
        )
        _synchronize_accelerator(torch)
        training_started = time.perf_counter()
        try:
            trainer.train(
                resume_from_checkpoint=(
                    str(resume_checkpoint) if resume_checkpoint is not None else None
                )
            )
        except FloatingPointError as exc:
            _record_numerical_failure(run_dir, run_id, exc)
            raise
        _synchronize_accelerator(torch)
        training_wall_seconds = max(time.perf_counter() - training_started, 0.0)
        completed_epochs = float(trainer.state.epoch or 0.0)
        maximum_epochs = float(config["epochs"])
        training_stop_reason = (
            "development_early_stopping"
            if completed_epochs < maximum_epochs
            else "maximum_epochs_safety_ceiling"
        )
        final_dir = run_dir / "model"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        started = time.perf_counter()
        try:
            prediction = trainer.predict(
                _EncodedDataset(records["dev"], tokenizer, mode, int(config["max_length"]))
            )
            probabilities = (
                _finite_class1_probabilities(
                    torch,
                    prediction.predictions,
                    run_id=run_id,
                    phase="development_prediction",
                )
                .detach()
                .cpu()
                .tolist()
            )
        except FloatingPointError as exc:
            _record_numerical_failure(run_dir, run_id, exc)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        dev_predictions = run_dir / "dev_predictions.jsonl"
        _write_predictions(
            dev_predictions,
            records["dev"],
            probabilities,
            "dev",
            elapsed_ms / len(records["dev"]),
            input_mode=mode,
        )
        dev_track_paths = _write_track_predictions(
            run_dir,
            records["dev"],
            probabilities,
            "dev",
            elapsed_ms / len(records["dev"]),
            input_mode=mode,
        )
        dev_auprc = float(
            average_precision_score(
                [1 if record.label.value == "injection" else 0 for record in records["dev"]],
                probabilities,
            )
        )
        development_metrics = run_dir / "development_metrics.json"
        write_json(
            development_metrics,
            {
                "schema_version": "1.0.0",
                "status": "PASS",
                "run_id": run_id,
                "seed": seed,
                "input_mode": mode,
                "dev_auprc": dev_auprc,
                "training_wall_seconds": training_wall_seconds,
                "training_example_count": len(records["train"]),
                "training_global_steps": int(trainer.state.global_step),
                "dev_predictions_sha256": sha256_file(dev_predictions),
                "core_dev_predictions_sha256": sha256_file(dev_track_paths[CORE_TRACK]),
                "provenance_dev_predictions_sha256": sha256_file(
                    dev_track_paths[PROVENANCE_TRACK]
                ),
                "test_accessed": False,
            },
        )
        training_decision_path = run_dir / "training_decision.json"
        write_json(
            training_decision_path,
            {
                "schema_version": "1.0.0",
                "status": "PASS",
                "run_id": run_id,
                "decision_owner": "training_pipeline",
                "decision_split": "dev",
                "decision_metric": str(config["early_stopping_metric"]),
                "early_stopping_patience": int(config["early_stopping_patience"]),
                "early_stopping_threshold": float(config["early_stopping_threshold"]),
                "maximum_epochs_safety_ceiling": maximum_epochs,
                "completed_epochs": completed_epochs,
                "global_step": int(trainer.state.global_step),
                "best_development_metric": (
                    float(trainer.state.best_metric)
                    if trainer.state.best_metric is not None
                    else None
                ),
                "stop_reason": training_stop_reason,
                "resumed_from_checkpoint": (
                    str(resume_checkpoint) if resume_checkpoint is not None else None
                ),
                "final_holdout_feedback_allowed": False,
                "test_accessed": False,
                "dataloader_worker_plan_sha256": sha256_file(
                    dataloader_worker_plan_path
                ),
            },
        )
        manifest_path = run_dir / "train_manifest.json"
        write_json(
            manifest_path,
            {
                "schema_version": "2.0.0",
                "status": "PASS",
                "model_type": "transformer_sequence_classifier",
                "run_id": run_id,
                "seed": seed,
                "input_mode": mode,
                "model_revision": config["model_revision"],
                "source_hashes": source_hashes,
                "capacity_plan_sha256": sha256_file(capacity_path),
                "dataloader_worker_plan_sha256": sha256_file(
                    dataloader_worker_plan_path
                ),
                "dataloader_num_workers": dataloader_num_workers,
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation,
                "gradient_checkpointing": checkpointing,
                "mixed_precision": str(config["mixed_precision"]),
                "parameter_storage_dtype": "fp32",
                "execution_scope": execution_scope,
                "training_wall_seconds": training_wall_seconds,
                "optimizer": str(config["optimizer"]),
                "max_grad_norm": float(config["max_grad_norm"]),
                "numerics_policy": str(config["numerics_policy"]),
                "training_decision_sha256": sha256_file(training_decision_path),
                "test_accessed": False,
            },
        )
        model_artifacts = sorted(path for path in final_dir.rglob("*") if path.is_file())
        if not any(
            path.suffix == ".safetensors" or path.name.startswith("pytorch_model")
            for path in model_artifacts
        ):
            raise FileNotFoundError(f"saved model weights missing for {run_id}")
        model_binding_path = run_dir / "model_binding.json"
        write_json(
            model_binding_path,
            {
                "schema_version": "1.0.0",
                "status": "PASS",
                "run_id": run_id,
                "model_artifact_version": directory_fingerprint(final_dir),
                "artifact_count": len(model_artifacts),
            },
        )
        ledger.complete(
            stage_id,
            [
                manifest_path,
                development_metrics,
                dev_predictions,
                *dev_track_paths.values(),
                training_decision_path,
                model_binding_path,
                run_dir / _RESUME_BINDING_NAME,
                *model_artifacts,
            ],
            stage_metadata,
        )
        completed.append(run_id)
        # Trainer owns the wrapped accelerator model, optimizer, callbacks, and
        # dataloaders. Release it before the next model is constructed; otherwise
        # the nine-run matrix transiently holds two training stacks on the accelerator.
        del trainer, model
        torch.accelerator.memory.empty_cache()
    return {
        "status": "PASS",
        "completed_runs": completed,
        "required_run_count": len(protocol["run_matrix"]),
        "test_accessed": False,
        "execution_scope": execution_scope,
        "loaded_partitions": list(partition_names),
        "runtime_check": runtime,
        "capacity_plan": capacity,
        "dataloader_worker_plan": dataloader_worker_plan,
    }


def select_encoder_run(
    config_path: Path,
    output_root: Path,
) -> dict[str, object]:
    protocol = validate_encoder_protocol(config_path)
    if protocol["status"] != "PASS":
        raise ValueError(protocol["errors"])
    config = load_yaml(config_path)
    selection_path = output_root / "model_selection.json"
    candidates: list[dict[str, object]] = []
    for entry in protocol["run_matrix"]:
        run_id = str(entry["run_id"])
        path = output_root / run_id / "development_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"development metrics missing: {run_id}")
        test_predictions_exist = (output_root / run_id / "test_predictions.jsonl").exists()
        if test_predictions_exist and not selection_path.exists():
            raise ValueError("test predictions exist before model selection was locked")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        model_binding_path = output_root / run_id / "model_binding.json"
        if not model_binding_path.is_file():
            raise FileNotFoundError(f"model binding missing: {run_id}")
        model_binding = json.loads(model_binding_path.read_text(encoding="utf-8"))
        if model_binding.get("status") != "PASS":
            raise ValueError(f"model binding invalid: {run_id}")
        observed_model_version = directory_fingerprint(output_root / run_id / "model")
        if model_binding.get("model_artifact_version") != observed_model_version:
            raise ValueError(f"model artifact binding mismatch: {run_id}")
        if metrics.get("test_accessed") is not False:
            raise ValueError(f"development artifact accessed test: {run_id}")
        candidates.append(
            {
                "run_id": run_id,
                "seed": int(entry["seed"]),
                "input_mode": str(entry["input_mode"]),
                "dev_auprc": float(metrics["dev_auprc"]),
                "development_metrics_sha256": sha256_file(path),
                "model_artifact_version": str(model_binding["model_artifact_version"]),
                "model_binding_sha256": sha256_file(model_binding_path),
            }
        )
    system_mode = str(config["system_input_mode"])
    eligible = [item for item in candidates if item["input_mode"] == system_mode]
    selected = max(eligible, key=lambda item: (float(item["dev_auprc"]), -int(item["seed"])))
    result = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "selection_scope": "preregistered_system_input_mode_seed_only",
        "selection_metric": "dev_auprc",
        "system_input_mode": system_mode,
        "selected": selected,
        "all_development_candidates": candidates,
        "config_sha256": sha256_file(config_path),
        "test_accessed": False,
        "claim_boundary": (
            "The selected system seed is locked from development evidence only. All input modes "
            "remain preregistered ablations and are evaluated after this artifact exists."
        ),
    }
    write_json(selection_path, result)
    return result


def _prepare_encoder_test_context(
    config_path: Path,
    split_dir: Path,
    output_root: Path,
    *,
    require_accelerator: bool,
) -> dict[str, Any]:
    protocol = validate_encoder_protocol(config_path)
    if protocol["status"] != "PASS":
        raise ValueError(protocol["errors"])
    config = load_yaml(config_path)
    if require_accelerator:
        runtime = check_runtime_profile_path(Path(str(config["runtime_profile"])), split_dir)
        if runtime["status"] != "PASS" or runtime["hardware_observed"] is not True:
            raise RuntimeError(runtime["errors"])
    selection_path = output_root / "model_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("model selection must be locked before test evaluation")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "PASS" or selection.get("test_accessed") is not False:
        raise ValueError("model selection is not a development-only PASS artifact")
    if selection.get("config_sha256") != sha256_file(config_path):
        raise ValueError("model selection config binding mismatch")
    records, source_hashes = _load_training_records(config, split_dir)
    capacity_path = output_root / "capacity_plan.json"
    if not capacity_path.is_file():
        raise FileNotFoundError("capacity plan must be locked before test evaluation")
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    if capacity.get("config_sha256") != sha256_file(config_path):
        raise ValueError("capacity plan config binding mismatch")
    eval_batch_size = _selected_capacity_batch(
        capacity,
        effective_batch_size=int(config["effective_train_batch_size"]),
    )
    dataloader_worker_plan_path = output_root / "dataloader_worker_plan.json"
    dataloader_worker_plan = _load_locked_dataloader_worker_plan(
        dataloader_worker_plan_path,
        config_path,
        config,
        records["train"],
        capacity_path,
        eval_batch_size,
    )
    ledger = StageLedger(output_root / "stage_ledger")
    runner_sha256 = sha256_file(Path(__file__))
    selection_candidates = {
        str(item["run_id"]): item for item in selection["all_development_candidates"]
    }
    return {
        "protocol": protocol,
        "config": config,
        "selection_path": selection_path,
        "selection": selection,
        "selection_candidates": selection_candidates,
        "records": records,
        "source_hashes": source_hashes,
        "capacity_path": capacity_path,
        "dataloader_worker_plan_path": dataloader_worker_plan_path,
        "dataloader_num_workers": int(dataloader_worker_plan["selected_num_workers"]),
        "eval_batch_size": eval_batch_size,
        "ledger": ledger,
        "runner_sha256": runner_sha256,
    }


def _encoder_test_run_context(
    context: dict[str, Any],
    entry: dict[str, object],
    output_root: Path,
    *,
    require_model_artifact: bool = True,
) -> tuple[str, Path, Path, dict[str, object]]:
    run_id = str(entry["run_id"])
    run_dir = output_root / run_id
    final_dir = run_dir / "model"
    selection_candidates = context["selection_candidates"]
    if run_id not in selection_candidates:
        raise ValueError(f"model selection candidate missing: {run_id}")
    expected_model_version = str(selection_candidates[run_id]["model_artifact_version"])
    if require_model_artifact:
        observed_model_version = directory_fingerprint(final_dir)
        if observed_model_version != expected_model_version:
            raise ValueError(f"test model artifact binding mismatch: {run_id}")
    stage_metadata = {
        "model_selection_sha256": sha256_file(context["selection_path"]),
        "model_artifact_version": expected_model_version,
        "source_hashes": context["source_hashes"],
        "runner_sha256": context["runner_sha256"],
        "capacity_plan_sha256": sha256_file(context["capacity_path"]),
        "dataloader_worker_plan_sha256": sha256_file(
            context["dataloader_worker_plan_path"]
        ),
        "dataloader_num_workers": context["dataloader_num_workers"],
        "eval_batch_size": context["eval_batch_size"],
    }
    return run_id, run_dir, final_dir, stage_metadata


def _encoder_test_prediction_paths(run_dir: Path) -> tuple[Path, dict[str, Path], Path, Path]:
    return (
        run_dir / "test_predictions.jsonl",
        {
            CORE_TRACK: run_dir / "core_test_predictions.jsonl",
            PROVENANCE_TRACK: run_dir / "provenance_test_predictions.jsonl",
        },
        run_dir / "thresholds.json",
        run_dir / "test_prediction_manifest.json",
    )


def run_encoder_test_prediction_matrix(
    config_path: Path,
    split_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    """Run and durably bind only the accelerator-dependent final predictions."""

    _require_confirmatory_authorization()
    context = _prepare_encoder_test_context(
        config_path,
        split_dir,
        output_root,
        require_accelerator=True,
    )
    config = context["config"]
    records = context["records"]
    torch, model_class, tokenizer_class, trainer_types = _experiment_imports()
    trainer_class, arguments_class, _, collator_class, _ = trainer_types
    ledger = context["ledger"]
    completed: list[str] = []
    for entry in context["protocol"]["run_matrix"]:
        run_id, run_dir, final_dir, stage_metadata = _encoder_test_run_context(
            context, entry, output_root
        )
        stage_id = f"test-prediction-{run_id}"
        if ledger.verified_complete(stage_id, stage_metadata):
            completed.append(run_id)
            continue
        mode = str(entry["input_mode"])
        trainer: Any | None = None
        model: Any | None = None
        try:
            tokenizer = tokenizer_class.from_pretrained(final_dir, trust_remote_code=False)
            model = model_class.from_pretrained(final_dir, trust_remote_code=False)
            collator = collator_class(tokenizer=tokenizer, pad_to_multiple_of=8)
            arguments = arguments_class(
                output_dir=str(run_dir / "test_runtime"),
                per_device_eval_batch_size=context["eval_batch_size"],
                bf16=str(config["mixed_precision"]) == "bf16",
                fp16=False,
                report_to=[],
                dataloader_pin_memory=True,
                dataloader_num_workers=context["dataloader_num_workers"],
                dataloader_persistent_workers=bool(
                    config["dataloader_persistent_workers"]
                ),
                dataloader_prefetch_factor=int(config["dataloader_prefetch_factor"]),
            )
            trainer = trainer_class(model=model, args=arguments, data_collator=collator)
            started = time.perf_counter()
            prediction = trainer.predict(
                _EncodedDataset(records["test"], tokenizer, mode, int(config["max_length"]))
            )
            probabilities = (
                _finite_class1_probabilities(
                    torch,
                    prediction.predictions,
                    run_id=run_id,
                    phase="test_prediction",
                )
                .detach()
                .cpu()
                .tolist()
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
        except FloatingPointError as exc:
            _record_numerical_failure(run_dir, run_id, exc)
            raise
        finally:
            if trainer is not None:
                del trainer
            if model is not None:
                del model
            torch.accelerator.memory.empty_cache()
        test_predictions, _, thresholds_path, prediction_manifest_path = (
            _encoder_test_prediction_paths(run_dir)
        )
        _write_predictions(
            test_predictions,
            records["test"],
            probabilities,
            "test",
            elapsed_ms / len(records["test"]),
            input_mode=mode,
        )
        test_track_paths = _write_track_predictions(
            run_dir,
            records["test"],
            probabilities,
            "test",
            elapsed_ms / len(records["test"]),
            input_mode=mode,
        )
        calibrate_thresholds(run_dir / "dev_predictions.jsonl", thresholds_path)
        write_json(
            prediction_manifest_path,
            {
                "schema_version": "1.0.0",
                "status": "PASS",
                "phase": "accelerator_test_prediction",
                "run_id": run_id,
                "model_selection_sha256": sha256_file(context["selection_path"]),
                "model_artifact_version": stage_metadata["model_artifact_version"],
                "source_hashes": context["source_hashes"],
                "test_predictions_sha256": sha256_file(test_predictions),
                "core_test_predictions_sha256": sha256_file(test_track_paths[CORE_TRACK]),
                "provenance_test_predictions_sha256": sha256_file(
                    test_track_paths[PROVENANCE_TRACK]
                ),
                "thresholds_sha256": sha256_file(thresholds_path),
                "dataloader_worker_plan_sha256": sha256_file(
                    context["dataloader_worker_plan_path"]
                ),
                "dataloader_num_workers": context["dataloader_num_workers"],
                "test_accessed_after_selection": True,
                "analysis_complete": False,
                "research_claim_eligible": False,
            },
        )
        ledger.complete(
            stage_id,
            [
                test_predictions,
                *test_track_paths.values(),
                thresholds_path,
                prediction_manifest_path,
            ],
            stage_metadata,
        )
        completed.append(run_id)
    return {
        "status": "PASS",
        "phase": "accelerator_test_prediction",
        "completed_runs": completed,
        "required_run_count": 9,
        "model_selection_sha256": sha256_file(context["selection_path"]),
        "test_accessed_after_selection": True,
        "analysis_complete": False,
        "research_claim_eligible": False,
    }


def run_encoder_test_analysis_matrix(
    config_path: Path,
    split_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    """Run CPU-eligible uncertainty analysis from hash-bound prediction receipts."""

    _require_confirmatory_authorization()
    context = _prepare_encoder_test_context(
        config_path,
        split_dir,
        output_root,
        require_accelerator=False,
    )
    ledger = context["ledger"]
    completed: list[str] = []
    for entry in context["protocol"]["run_matrix"]:
        run_id, run_dir, _, stage_metadata = _encoder_test_run_context(
            context,
            entry,
            output_root,
            require_model_artifact=False,
        )
        prediction_stage_id = f"test-prediction-{run_id}"
        if not ledger.verified_complete(prediction_stage_id, stage_metadata):
            raise FileNotFoundError(f"verified test prediction receipt missing: {run_id}")
        test_predictions, test_track_paths, thresholds_path, prediction_manifest_path = (
            _encoder_test_prediction_paths(run_dir)
        )
        prediction_manifest = json.loads(
            prediction_manifest_path.read_text(encoding="utf-8")
        )
        if (
            prediction_manifest.get("status") != "PASS"
            or prediction_manifest.get("analysis_complete") is not False
            or prediction_manifest.get("test_predictions_sha256")
            != sha256_file(test_predictions)
            or prediction_manifest.get("thresholds_sha256") != sha256_file(thresholds_path)
        ):
            raise ValueError(f"test prediction manifest invalid: {run_id}")
        analysis_metadata = {
            **stage_metadata,
            "test_prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        }
        analysis_stage_id = f"test-analysis-{run_id}"
        outer_stage_id = f"test-{run_id}"
        analysis_complete = ledger.verified_complete(analysis_stage_id, analysis_metadata)
        outer_complete = ledger.verified_complete(outer_stage_id, stage_metadata)
        if analysis_complete and outer_complete:
            completed.append(run_id)
            continue
        evaluation_path = run_dir / "evaluation.json"
        evaluation = evaluate_predictions(test_predictions, thresholds_path, evaluation_path)
        test_manifest = run_dir / "test_manifest.json"
        write_json(
            test_manifest,
            {
                "schema_version": "2.0.0",
                "status": "PASS",
                "phase": "cpu_statistical_analysis",
                "run_id": run_id,
                "model_selection_sha256": sha256_file(context["selection_path"]),
                "source_hashes": context["source_hashes"],
                "test_prediction_manifest_sha256": sha256_file(prediction_manifest_path),
                "test_predictions_sha256": sha256_file(test_predictions),
                "core_test_predictions_sha256": sha256_file(test_track_paths[CORE_TRACK]),
                "provenance_test_predictions_sha256": sha256_file(
                    test_track_paths[PROVENANCE_TRACK]
                ),
                "thresholds_sha256": sha256_file(thresholds_path),
                "evaluation_sha256": sha256_file(evaluation_path),
                "research_claim_eligible": bool(evaluation["research_claim_eligible"]),
                "test_accessed_after_selection": True,
            },
        )
        ledger.complete(
            analysis_stage_id,
            [evaluation_path, test_manifest],
            analysis_metadata,
        )
        ledger.complete(
            outer_stage_id,
            [
                test_predictions,
                *test_track_paths.values(),
                thresholds_path,
                prediction_manifest_path,
                evaluation_path,
                test_manifest,
            ],
            stage_metadata,
        )
        completed.append(run_id)
    return {
        "status": "PASS",
        "phase": "cpu_statistical_analysis",
        "completed_runs": completed,
        "required_run_count": 9,
        "model_selection_sha256": sha256_file(context["selection_path"]),
        "test_accessed_after_selection": True,
        "analysis_complete": True,
    }


def run_encoder_test_matrix(
    config_path: Path,
    split_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    """Compatibility wrapper: run accelerator prediction, then CPU-eligible analysis."""

    prediction = run_encoder_test_prediction_matrix(config_path, split_dir, output_root)
    analysis = run_encoder_test_analysis_matrix(config_path, split_dir, output_root)
    return {
        "status": "PASS",
        "prediction": prediction,
        "analysis": analysis,
        "completed_runs": analysis["completed_runs"],
        "required_run_count": 9,
        "model_selection_sha256": analysis["model_selection_sha256"],
        "test_accessed_after_selection": True,
    }


def _load_training_records(
    config: dict[str, object],
    split_dir: Path,
    *,
    partition_names: tuple[str, ...] = ("train", "dev", "test"),
) -> tuple[dict[str, list[Any]], dict[str, str]]:
    return load_benchmark_partitions(
        split_dir,
        Path(str(config["contrast_dataset"])),
        splits=partition_names,
    )


def _capacity_measurement_payload(candidate: CapacityMeasurement) -> dict[str, object]:
    return {
        **asdict(candidate),
        "utilization": candidate.utilization,
    }


def _move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if callable(getattr(value, "to", None)) else value
        for key, value in batch.items()
    }


def _build_canary_optimizer_batches(
    config: dict[str, object],
    records: list[Any],
    tokenizer: Any,
    collator: Any,
    torch: Any,
    device: Any,
    *,
    input_mode: str,
    seed: int,
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    effective_batch = int(config["effective_train_batch_size"])
    if effective_batch % batch_size != 0:
        raise ValueError("canary batch must divide the locked effective batch")
    accumulation = effective_batch // batch_size
    optimizer_steps = int(config["numerics_canary_optimizer_steps"])
    required_records = optimizer_steps * effective_batch
    if len(records) < required_records:
        raise ValueError("training set is too small for the locked numerics canary")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(records), generator=generator).tolist()
    dataset = _EncodedDataset(records, tokenizer, input_mode, int(config["max_length"]))
    result: list[list[dict[str, Any]]] = []
    cursor = 0
    for _ in range(optimizer_steps):
        micro_batches: list[dict[str, Any]] = []
        for _ in range(accumulation):
            indices = order[cursor : cursor + batch_size]
            cursor += batch_size
            batch = _move_batch_to_device(
                dict(collator([dataset[index] for index in indices])),
                device,
            )
            labels = batch.get("labels")
            attention_mask = batch.get("attention_mask")
            if labels is None or attention_mask is None:
                raise RuntimeError("numerics canary batch lacks labels or attention mask")
            label_values = {int(value) for value in labels.detach().cpu().tolist()}
            if not label_values.issubset({0, 1}):
                raise ValueError("numerics canary observed a non-binary label")
            if not bool((attention_mask.sum(dim=1) > 0).all().item()):
                raise ValueError("numerics canary observed an empty encoded sequence")
            micro_batches.append(batch)
        result.append(micro_batches)
    return result


def _run_capacity_numerics_canary_once(
    config: dict[str, object],
    records: list[Any],
    torch: Any,
    model_class: Any,
    tokenizer: Any,
    collator: Any,
    device: Any,
    candidate: CapacityMeasurement,
    *,
    input_mode: str,
    seed: int,
) -> dict[str, object]:
    """Run one canary in an isolated frame so its accelerator tensors can die."""

    checkpointing = candidate.candidate_id.endswith("checkpoint-on")
    run_id = f"capacity-canary-{candidate.candidate_id}-{input_mode}-s{seed}"
    _seed_model_initialization(torch, seed)
    model = _load_trainable_encoder_model(model_class, torch, config).to(device)
    if checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(config["gradient_checkpointing_use_reentrant"])
            }
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=0.0,
    )
    optimizer_batches = _build_canary_optimizer_batches(
        config,
        records,
        tokenizer,
        collator,
        torch,
        device,
        input_mode=input_mode,
        seed=seed,
        batch_size=candidate.batch_size,
    )
    result = _run_numerics_steps(
        torch,
        model,
        optimizer,
        optimizer_batches,
        run_id=run_id,
        max_grad_norm=float(config["max_grad_norm"]),
        mixed_precision=str(config["mixed_precision"]),
    )
    return {
        "run_id": run_id,
        "input_mode": input_mode,
        "seed": seed,
        **result,
    }


def _run_capacity_numerics_canary(
    config: dict[str, object],
    records: list[Any],
    torch: Any,
    model_class: Any,
    tokenizer: Any,
    collator: Any,
    device: Any,
    candidate: CapacityMeasurement,
) -> dict[str, object]:
    run_results: list[dict[str, object]] = []
    for input_mode in [str(value) for value in config["input_modes"]]:
        for seed in [int(value) for value in config["seeds"]]:
            try:
                run_results.append(
                    _run_capacity_numerics_canary_once(
                        config,
                        records,
                        torch,
                        model_class,
                        tokenizer,
                        collator,
                        device,
                        candidate,
                        input_mode=input_mode,
                        seed=seed,
                    )
                )
            finally:
                # The helper frame has returned on success, so no prior model,
                # optimizer, loss, or encoded batch remains live when the cache
                # is released before the next matrix member is loaded.
                torch.accelerator.memory.empty_cache()
    return {
        "status": "PASS",
        "candidate_id": candidate.candidate_id,
        "optimizer_steps_per_run": int(config["numerics_canary_optimizer_steps"]),
        "run_count": len(run_results),
        "runs": run_results,
    }


def _capacity_precision_context(config: dict[str, object], torch: Any, device: Any) -> Any:
    if str(config["mixed_precision"]) == "bf16":
        return torch.autocast(device_type=str(device.type), dtype=torch.bfloat16)
    return nullcontext()


def _run_capacity_benchmark_optimizer_step(
    config: dict[str, object],
    torch: Any,
    model: Any,
    optimizer: Any,
    tokens: dict[str, Any],
    labels: Any,
    *,
    accumulation_steps: int,
    run_id: str,
    phase: str,
) -> None:
    """Run one complete effective-batch update using the real numerical guards."""

    optimizer.zero_grad(set_to_none=True)
    for micro_batch in range(1, accumulation_steps + 1):
        with _capacity_precision_context(config, torch, labels.device):
            loss = model(**tokens, labels=labels).loss
        _require_finite_tensor(
            torch,
            loss,
            run_id=run_id,
            phase=f"{phase}_micro_batch_{micro_batch}",
            tensor_name="loss",
        )
        (loss / accumulation_steps).backward()
    _require_finite_gradients(
        torch,
        model,
        run_id=run_id,
        phase=f"{phase}_backward",
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(config["max_grad_norm"]),
        error_if_nonfinite=False,
    )
    _require_finite_tensor(
        torch,
        grad_norm,
        run_id=run_id,
        phase=f"{phase}_backward",
        tensor_name="gradient_norm",
    )
    optimizer.step()
    _require_finite_model_state(torch, model, run_id=run_id, phase=phase)
    _require_finite_optimizer_state(
        torch,
        model,
        optimizer,
        run_id=run_id,
        phase=phase,
    )


def _measure_encoder_capacity_candidate(
    config: dict[str, object],
    records: list[Any],
    torch: Any,
    model_class: Any,
    tokenizer: Any,
    device: Any,
    interface: Any,
    *,
    batch_size: int,
    checkpointing: bool,
    total_gib: float,
) -> CapacityMeasurement:
    """Measure warmed, complete effective-batch updates in an isolated candidate frame."""

    candidate_id = (
        f"batch-{batch_size}-checkpoint-{'on' if checkpointing else 'off'}"
    )
    effective_batch = int(config["effective_train_batch_size"])
    accumulation_steps = effective_batch // batch_size
    warmup_steps = int(config["capacity_warmup_optimizer_steps"])
    measurement_steps = int(config["capacity_measurement_optimizer_steps"])
    probe_input_mode = str(config["capacity_probe_input_mode"])
    throughput = 0.0
    peak_gib = total_gib
    step_seconds: list[float] = []
    step_throughputs: list[float] = []
    rejection_reason: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    try:
        _seed_model_initialization(torch, int(config["seeds"][0]))
        model = _load_trainable_encoder_model(model_class, torch, config).to(device)
        model.train()
        if checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": bool(
                        config["gradient_checkpointing_use_reentrant"]
                    )
                }
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0,
            weight_decay=0.0,
        )
        batch = [records[index % len(records)] for index in range(batch_size)]
        tokens = tokenizer(
            [detector_text(record, probe_input_mode) for record in batch],
            max_length=int(config["max_length"]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        labels = torch.tensor(
            [1 if record.label.value == "injection" else 0 for record in batch],
            device=device,
        )
        for step in range(1, warmup_steps + 1):
            _run_capacity_benchmark_optimizer_step(
                config,
                torch,
                model,
                optimizer,
                tokens,
                labels,
                accumulation_steps=accumulation_steps,
                run_id=candidate_id,
                phase=f"warmup_optimizer_step_{step}",
            )
        interface.synchronize()
        reset_peak = getattr(interface.memory, "reset_peak_memory_stats", None)
        if callable(reset_peak):
            reset_peak()
        for step in range(1, measurement_steps + 1):
            interface.synchronize()
            started = time.perf_counter()
            _run_capacity_benchmark_optimizer_step(
                config,
                torch,
                model,
                optimizer,
                tokens,
                labels,
                accumulation_steps=accumulation_steps,
                run_id=candidate_id,
                phase=f"measured_optimizer_step_{step}",
            )
            interface.synchronize()
            elapsed = max(time.perf_counter() - started, 1e-9)
            step_seconds.append(elapsed)
            step_throughputs.append(effective_batch / elapsed)
        throughput = float(np.median(step_throughputs))
        peak_gib = interface.memory.max_memory_reserved() / (1024**3)
    except FloatingPointError as exc:
        rejection_reason = "non_finite_capacity_measurement"
        failure_type = type(exc).__name__
        failure_message = str(exc)
    except RuntimeError as exc:
        if not is_capacity_exhaustion(torch, exc):
            raise
        rejection_reason = "capacity_exhaustion"
        failure_type = type(exc).__name__
        failure_message = str(exc)
        max_reserved = getattr(interface.memory, "max_memory_reserved", None)
        if callable(max_reserved):
            peak_gib = max_reserved() / (1024**3)
    completed = rejection_reason is None and bool(step_throughputs)
    return EncoderCapacityMeasurement(
        candidate_id=candidate_id,
        batch_size=batch_size,
        samples_per_second=throughput,
        peak_reserved_gib=peak_gib,
        total_memory_gib=total_gib,
        completed=completed,
        effective_batch_size=effective_batch,
        gradient_accumulation_steps=accumulation_steps,
        warmup_optimizer_steps=warmup_steps,
        measured_optimizer_steps=len(step_throughputs),
        step_seconds=tuple(step_seconds),
        step_samples_per_second=tuple(step_throughputs),
        minimum_samples_per_second=min(step_throughputs, default=0.0),
        p25_samples_per_second=(
            float(np.percentile(step_throughputs, 25)) if step_throughputs else 0.0
        ),
        p75_samples_per_second=(
            float(np.percentile(step_throughputs, 75)) if step_throughputs else 0.0
        ),
        maximum_samples_per_second=max(step_throughputs, default=0.0),
        total_measured_seconds=sum(step_seconds),
        probe_input_mode=probe_input_mode,
        rejection_reason=rejection_reason,
        failure_type=failure_type,
        failure_message=failure_message,
    )


def _invalid_encoder_capacity_candidate(
    *,
    batch_size: int,
    checkpointing: bool,
    effective_batch: int,
    total_gib: float,
    config: dict[str, object],
) -> EncoderCapacityMeasurement:
    return EncoderCapacityMeasurement(
        candidate_id=(
            f"batch-{batch_size}-checkpoint-{'on' if checkpointing else 'off'}"
        ),
        batch_size=batch_size,
        samples_per_second=0.0,
        peak_reserved_gib=0.0,
        total_memory_gib=total_gib,
        completed=False,
        effective_batch_size=effective_batch,
        gradient_accumulation_steps=0,
        warmup_optimizer_steps=int(config["capacity_warmup_optimizer_steps"]),
        measured_optimizer_steps=0,
        probe_input_mode=str(config["capacity_probe_input_mode"]),
        rejection_reason="batch_size_must_divide_effective_batch",
    )


def _load_or_measure_dataloader_workers(
    output_path: Path,
    config_path: Path,
    config: dict[str, object],
    train_records: list[Any],
    capacity_path: Path,
    selected_batch_size: int,
    torch: Any,
    tokenizer: Any,
    collator: Any,
) -> dict[str, object]:
    """Measure bounded train-only input throughput and lock one worker count."""

    if output_path.is_file():
        return _load_locked_dataloader_worker_plan(
            output_path,
            config_path,
            config,
            train_records,
            capacity_path,
            selected_batch_size,
        )

    logical_cpu_count = int(os.cpu_count() or 1)
    candidates = [int(value) for value in config["dataloader_worker_candidates"]]
    measurements: list[dict[str, object]] = []
    for num_workers in candidates:
        if num_workers > logical_cpu_count:
            measurements.append(
                _failed_dataloader_worker_measurement(
                    num_workers,
                    status="SKIPPED",
                    error_type="logical_cpu_limit",
                    error_message=(
                        f"candidate requires {num_workers} logical CPUs but only "
                        f"{logical_cpu_count} were observed"
                    ),
                )
            )
            continue
        try:
            measurement = _measure_dataloader_worker_candidate(
                config,
                train_records,
                selected_batch_size,
                num_workers,
                torch,
                tokenizer,
                collator,
            )
        except Exception as exc:
            measurement = _failed_dataloader_worker_measurement(
                num_workers,
                status="FAIL",
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )
        measurements.append(measurement)

    try:
        selected_num_workers = _select_dataloader_worker_count(measurements)
        status = "PASS"
    except ValueError:
        selected_num_workers = None
        status = "FAIL"
    result: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": status,
        "measurement_kind": "development_train_dataloader_worker_scout",
        **_dataloader_worker_plan_bindings(
            config_path,
            train_records,
            capacity_path,
            selected_batch_size,
        ),
        "probe_partition": "train",
        "probe_input_mode": str(config["capacity_probe_input_mode"]),
        "candidate_num_workers": candidates,
        "warmup_batches": int(config["dataloader_worker_warmup_batches"]),
        "measurement_batches": int(config["dataloader_worker_measurement_batches"]),
        "repeats": int(config["dataloader_worker_repeats"]),
        "logical_cpu_count_at_measurement": logical_cpu_count,
        "system_ram_total_gib_at_measurement": psutil.virtual_memory().total / (1024**3),
        "measurements": measurements,
        "selected_num_workers": selected_num_workers,
        "selection_rule": (
            "maximum median train-only dataloader samples per second after bounded warm-up; "
            "deterministic ties prefer fewer workers"
        ),
        "test_accessed": False,
        "final_holdout_feedback_allowed": False,
        "claim_boundary": (
            "This plan tunes only input-pipeline concurrency from the training partition. It "
            "does not change model, optimizer, precision, effective batch size, early stopping, "
            "thresholds, seeds, input modes, or final-holdout governance."
        ),
    }
    write_json(output_path, result)
    if status != "PASS":
        raise RuntimeError("no dataloader worker candidate completed the train-only scout")
    return _load_locked_dataloader_worker_plan(
        output_path,
        config_path,
        config,
        train_records,
        capacity_path,
        selected_batch_size,
    )


def _load_locked_dataloader_worker_plan(
    output_path: Path,
    config_path: Path,
    config: dict[str, object],
    train_records: list[Any],
    capacity_path: Path,
    selected_batch_size: int,
) -> dict[str, object]:
    if not output_path.is_file():
        raise FileNotFoundError("dataloader worker plan must be locked before this stage")
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dataloader worker plan must be an object")
    return _validate_dataloader_worker_plan(
        raw,
        config_path=config_path,
        config=config,
        train_records=train_records,
        capacity_path=capacity_path,
        selected_batch_size=selected_batch_size,
    )


def _validate_dataloader_worker_plan(
    plan: dict[str, object],
    *,
    config_path: Path,
    config: dict[str, object],
    train_records: list[Any],
    capacity_path: Path,
    selected_batch_size: int,
) -> dict[str, object]:
    missing = sorted(_DATALOADER_WORKER_PLAN_FIELDS.difference(plan))
    unknown = sorted(set(plan).difference(_DATALOADER_WORKER_PLAN_FIELDS))
    if missing or unknown:
        raise ValueError(
            f"dataloader worker plan field mismatch: missing={missing}, unknown={unknown}"
        )
    if (
        plan["schema_version"] != "1.0.0"
        or plan["status"] != "PASS"
        or plan["measurement_kind"] != "development_train_dataloader_worker_scout"
    ):
        raise ValueError("dataloader worker plan is not a supported PASS artifact")
    expected_bindings = _dataloader_worker_plan_bindings(
        config_path,
        train_records,
        capacity_path,
        selected_batch_size,
    )
    for field, expected in expected_bindings.items():
        if plan[field] != expected:
            raise ValueError(f"dataloader worker plan {field} binding mismatch")
    expected_contract = {
        "probe_partition": "train",
        "probe_input_mode": str(config["capacity_probe_input_mode"]),
        "candidate_num_workers": [
            int(value) for value in config["dataloader_worker_candidates"]
        ],
        "warmup_batches": int(config["dataloader_worker_warmup_batches"]),
        "measurement_batches": int(config["dataloader_worker_measurement_batches"]),
        "repeats": int(config["dataloader_worker_repeats"]),
        "test_accessed": False,
        "final_holdout_feedback_allowed": False,
    }
    for field, expected in expected_contract.items():
        if plan[field] != expected:
            raise ValueError(f"dataloader worker plan {field} contract mismatch")
    measurements = plan["measurements"]
    if not isinstance(measurements, list):
        raise ValueError("dataloader worker measurements must be a list")
    observed_candidates: list[int] = []
    for measurement in measurements:
        if not isinstance(measurement, dict):
            raise ValueError("dataloader worker measurement must be an object")
        missing_measurement = sorted(
            _DATALOADER_WORKER_MEASUREMENT_FIELDS.difference(measurement)
        )
        unknown_measurement = sorted(
            set(measurement).difference(_DATALOADER_WORKER_MEASUREMENT_FIELDS)
        )
        if missing_measurement or unknown_measurement:
            raise ValueError("dataloader worker measurement field mismatch")
        num_workers = int(measurement["num_workers"])
        observed_candidates.append(num_workers)
        if measurement["status"] not in {"PASS", "FAIL", "SKIPPED"}:
            raise ValueError("dataloader worker measurement status invalid")
        if measurement["status"] == "PASS":
            rates = measurement["repeat_samples_per_second"]
            elapsed = measurement["repeat_elapsed_seconds"]
            counts = measurement["repeat_sample_counts"]
            if not all(isinstance(values, list) for values in (rates, elapsed, counts)):
                raise ValueError("dataloader worker repeat evidence invalid")
            if not (
                len(rates)
                == len(elapsed)
                == len(counts)
                == int(config["dataloader_worker_repeats"])
            ):
                raise ValueError("dataloader worker repeat count mismatch")
            median_rate = float(measurement["median_samples_per_second"])
            if (
                not math.isfinite(median_rate)
                or median_rate <= 0
                or median_rate != float(np.median([float(value) for value in rates]))
            ):
                raise ValueError("dataloader worker median throughput mismatch")
    if observed_candidates != expected_contract["candidate_num_workers"]:
        raise ValueError("dataloader worker candidate order mismatch")
    selected = _select_dataloader_worker_count(measurements)
    if plan["selected_num_workers"] != selected:
        raise ValueError("dataloader worker selection mismatch")
    if selected > int(os.cpu_count() or 1):
        raise RuntimeError("locked dataloader worker count exceeds current logical CPUs")
    if (
        not isinstance(plan["logical_cpu_count_at_measurement"], int)
        or int(plan["logical_cpu_count_at_measurement"]) < selected
    ):
        raise ValueError("dataloader worker CPU observation invalid")
    total_ram = float(plan["system_ram_total_gib_at_measurement"])
    if not math.isfinite(total_ram) or total_ram <= 0:
        raise ValueError("dataloader worker RAM observation invalid")
    return plan


def _dataloader_worker_plan_bindings(
    config_path: Path,
    train_records: list[Any],
    capacity_path: Path,
    selected_batch_size: int,
) -> dict[str, object]:
    return {
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "train_set_sha256": _training_record_set_sha256(train_records),
        "capacity_plan_sha256": sha256_file(capacity_path),
        "selected_batch_size": int(selected_batch_size),
    }


def _measure_dataloader_worker_candidate(
    config: dict[str, object],
    train_records: list[Any],
    selected_batch_size: int,
    num_workers: int,
    torch: Any,
    tokenizer: Any,
    collator: Any,
) -> dict[str, object]:
    dataset = _EncodedDataset(
        train_records,
        tokenizer,
        str(config["capacity_probe_input_mode"]),
        int(config["max_length"]),
    )
    warmup_batches = int(config["dataloader_worker_warmup_batches"])
    measurement_batches = int(config["dataloader_worker_measurement_batches"])
    repeats = int(config["dataloader_worker_repeats"])
    repeat_rates: list[float] = []
    repeat_elapsed: list[float] = []
    repeat_counts: list[int] = []
    for _ in range(repeats):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=selected_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            timeout=120,
        )
        iterator = iter(loader)
        try:
            for _ in range(warmup_batches):
                next(iterator)
            started = time.perf_counter()
            sample_count = 0
            for _ in range(measurement_batches):
                batch = next(iterator)
                labels = batch.get("labels") if isinstance(batch, Mapping) else None
                if labels is None:
                    raise ValueError("dataloader scout batch has no labels")
                sample_count += len(labels)
            elapsed_seconds = time.perf_counter() - started
            if elapsed_seconds <= 0 or sample_count <= 0:
                raise RuntimeError("dataloader scout produced an invalid timing sample")
            repeat_elapsed.append(elapsed_seconds)
            repeat_counts.append(sample_count)
            repeat_rates.append(sample_count / elapsed_seconds)
        finally:
            del iterator, loader
            gc.collect()
    return {
        "num_workers": num_workers,
        "status": "PASS",
        "repeat_samples_per_second": repeat_rates,
        "repeat_elapsed_seconds": repeat_elapsed,
        "repeat_sample_counts": repeat_counts,
        "median_samples_per_second": float(np.median(repeat_rates)),
        "error_type": None,
        "error_message": None,
    }


def _failed_dataloader_worker_measurement(
    num_workers: int,
    *,
    status: str,
    error_type: str,
    error_message: str,
) -> dict[str, object]:
    return {
        "num_workers": num_workers,
        "status": status,
        "repeat_samples_per_second": [],
        "repeat_elapsed_seconds": [],
        "repeat_sample_counts": [],
        "median_samples_per_second": None,
        "error_type": error_type,
        "error_message": error_message,
    }


def _select_dataloader_worker_count(measurements: list[dict[str, object]]) -> int:
    candidates: list[tuple[float, int]] = []
    seen: set[int] = set()
    for measurement in measurements:
        workers = int(measurement.get("num_workers", 0))
        if workers <= 0 or workers in seen:
            raise ValueError("dataloader worker candidates must be unique positive integers")
        seen.add(workers)
        if measurement.get("status") != "PASS":
            continue
        throughput = float(measurement.get("median_samples_per_second", 0))
        if not math.isfinite(throughput) or throughput <= 0:
            raise ValueError("dataloader worker throughput must be finite and positive")
        candidates.append((throughput, workers))
    if not candidates:
        raise ValueError("no passing dataloader worker candidate")
    return sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]


def _training_record_set_sha256(records: list[Any]) -> str:
    return hashlib.sha256(
        canonical_json([record.content_sha256 for record in records]).encode("utf-8")
    ).hexdigest().upper()


def _load_or_measure_capacity(
    output_path: Path,
    config_path: Path,
    config: dict[str, object],
    records: list[Any],
    torch: Any,
    model_class: Any,
    tokenizer: Any,
    collator: Any,
) -> dict[str, object]:
    config_sha256 = sha256_file(config_path)
    runner_sha256 = sha256_file(Path(__file__))
    train_set_sha256 = _training_record_set_sha256(records)
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_selected = existing.get("selected")
        existing_canary = existing.get("numerics_canary")
        if (
            existing.get("config_sha256") == config_sha256
            and existing.get("runner_sha256") == runner_sha256
            and existing.get("train_set_sha256") == train_set_sha256
            and existing.get("status") == "PASS"
            and isinstance(existing_selected, dict)
            and isinstance(existing_canary, dict)
            and existing_canary.get("status") == "PASS"
            and existing_canary.get("candidate_id")
            == existing_selected.get("candidate_id")
        ):
            return existing
        raise ValueError("capacity plan exists with stale source bindings")

    interface = torch.accelerator
    device = interface.current_accelerator(check_available=True)
    if device is None:
        raise RuntimeError("no observed compute device")
    _, total_bytes = interface.memory.get_memory_info()
    total_gib = total_bytes / (1024**3)
    candidates: list[CapacityMeasurement] = []
    effective_batch = int(config["effective_train_batch_size"])
    for batch_size in [int(value) for value in config["batch_candidates"]]:
        for checkpointing in [bool(value) for value in config["gradient_checkpointing_options"]]:
            if effective_batch % batch_size != 0:
                candidates.append(
                    _invalid_encoder_capacity_candidate(
                        batch_size=batch_size,
                        checkpointing=checkpointing,
                        effective_batch=effective_batch,
                        total_gib=total_gib,
                        config=config,
                    )
                )
                continue
            try:
                interface.memory.empty_cache()
                interface.memory.reset_peak_memory_stats()
                measurement = _measure_encoder_capacity_candidate(
                    config,
                    records,
                    torch,
                    model_class,
                    tokenizer,
                    device,
                    interface,
                    batch_size=batch_size,
                    checkpointing=checkpointing,
                    total_gib=total_gib,
                )
            finally:
                # The candidate helper has returned and released every tensor
                # in its frame before this cache operation runs.
                interface.memory.empty_cache()
            candidates.append(measurement)
    result = select_capacity_candidate(
        candidates,
        maximum_utilization=float(config["target_memory_utilization"]),
    )
    result["selection_rule"] = (
        "maximum median synchronized throughput over complete effective-batch optimizer steps "
        "after per-candidate warm-up, within the reserved-memory boundary; deterministic ties "
        "prefer lower utilization and then smaller batches"
    )
    canary_attempts: list[dict[str, object]] = []
    selected_candidate: CapacityMeasurement | None = None
    if result["status"] == "PASS":
        ranked = rank_capacity_candidates(
            candidates,
            maximum_utilization=float(config["target_memory_utilization"]),
        )
        for candidate in ranked:
            candidate_failed = False
            try:
                canary = _run_capacity_numerics_canary(
                    config,
                    records,
                    torch,
                    model_class,
                    tokenizer,
                    collator,
                    device,
                    candidate,
                )
            except FloatingPointError as exc:
                canary_attempts.append(
                    {
                        "status": "FAIL",
                        "candidate_id": candidate.candidate_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                candidate_failed = True
            except RuntimeError as exc:
                if not is_capacity_exhaustion(torch, exc):
                    raise
                canary_attempts.append(
                    {
                        "status": "FAIL",
                        "candidate_id": candidate.candidate_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                candidate_failed = True
            if candidate_failed:
                # The handled exception and its traceback are out of scope here,
                # so failed-canary tensors are no longer live before fallback.
                interface.memory.empty_cache()
                continue
            canary_attempts.append(canary)
            selected_candidate = candidate
            break
        if selected_candidate is None:
            result["status"] = "FAIL"
            result["errors"] = ["no_numerically_stable_capacity_candidate"]
            result["selected"] = None
            result["numerics_canary"] = {
                "status": "FAIL",
                "attempts": canary_attempts,
            }
        else:
            result["selected"] = _capacity_measurement_payload(selected_candidate)
            result["selection_rule"] = (
                f"{result['selection_rule']}; selected candidate must also pass the locked "
                "two-step finite-state canary for every input-mode and seed run"
            )
            result["numerics_canary"] = {
                "status": "PASS",
                "candidate_id": selected_candidate.candidate_id,
                "attempts": canary_attempts,
            }
    result.update(
        {
            "schema_version": "3.0.0",
            "config_sha256": config_sha256,
            "runner_sha256": runner_sha256,
            "train_set_sha256": train_set_sha256,
            "mixed_precision": str(config["mixed_precision"]),
            "capacity_benchmark_contract": {
                "revision": "synchronized-effective-batch-v2",
                "candidate_order": "batch_then_checkpointing_interleaved",
                "probe_input_mode": str(config["capacity_probe_input_mode"]),
                "warmup_optimizer_steps": int(config["capacity_warmup_optimizer_steps"]),
                "measurement_optimizer_steps": int(
                    config["capacity_measurement_optimizer_steps"]
                ),
                "throughput_statistic": "median",
            },
            "gradient_checkpointing_use_reentrant": bool(
                config["gradient_checkpointing_use_reentrant"]
            ),
        }
    )
    write_json(output_path, result)
    return result


def _selected_capacity_batch(
    capacity: dict[str, object],
    *,
    effective_batch_size: int,
) -> int:
    if capacity.get("status") != "PASS":
        raise ValueError("capacity plan is not PASS")
    selected = capacity.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("capacity plan has no selected candidate")
    batch_size = int(selected.get("batch_size", 0))
    if batch_size <= 0:
        raise ValueError("capacity plan selected an invalid batch size")
    if effective_batch_size <= 0 or effective_batch_size % batch_size != 0:
        raise ValueError("capacity batch must divide the locked effective batch")
    return batch_size


def _require_confirmatory_authorization() -> None:
    if os.environ.get("VIPIBENCH_CONFIRMATORY_RUN_APPROVED") != "YES":
        raise PermissionError(
            "VIPIBENCH_CONFIRMATORY_RUN_APPROVED=YES is required for confirmatory execution"
        )
