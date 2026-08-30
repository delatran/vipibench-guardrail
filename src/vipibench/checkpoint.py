from __future__ import annotations

import json
from pathlib import Path

from vipibench.dataio import sha256_file, write_json


class StageLedger:
    """Hash-verified stage completion markers for disconnect-safe orchestration."""

    def __init__(self, root: Path, *, artifact_root: Path | None = None) -> None:
        self.root = root
        self.artifact_root = (artifact_root or root.parent).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def marker_path(self, stage_id: str) -> Path:
        if not stage_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in stage_id
        ):
            raise ValueError(
                "stage_id must use lowercase ASCII letters, digits, hyphen, or underscore"
            )
        return self.root / f"{stage_id}.complete.json"

    def complete(self, stage_id: str, outputs: list[Path], metadata: dict[str, object]) -> Path:
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"cannot complete stage with missing outputs: {missing}")
        marker = self.marker_path(stage_id)
        payload = {
            "schema_version": "2.0.0",
            "stage_id": stage_id,
            "status": "PASS",
            "artifact_root": ".",
            "outputs": {
                self._portable_output_path(path): sha256_file(path) for path in outputs
            },
            "metadata": metadata,
        }
        if marker.exists():
            current = json.loads(marker.read_text(encoding="utf-8"))
            if current == payload:
                return marker
        write_json(marker, payload)
        return marker

    def verified_complete(
        self,
        stage_id: str,
        expected_metadata: dict[str, object] | None = None,
    ) -> bool:
        marker = self.marker_path(stage_id)
        if not marker.is_file():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if payload.get("status") != "PASS":
            return False
        if expected_metadata is not None and payload.get("metadata") != expected_metadata:
            return False
        outputs = payload.get("outputs")
        if not isinstance(outputs, dict):
            return False
        return all(
            self._resolve_output_path(path).is_file()
            and sha256_file(self._resolve_output_path(path)) == digest
            for path, digest in outputs.items()
        )

    def _portable_output_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.artifact_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"stage output must stay within artifact_root: {path}") from exc

    def _resolve_output_path(self, value: object) -> Path:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            return self.artifact_root / "__invalid_stage_output__"
        resolved = (self.artifact_root / relative).resolve()
        try:
            resolved.relative_to(self.artifact_root)
        except ValueError:
            return self.artifact_root / "__invalid_stage_output__"
        return resolved
