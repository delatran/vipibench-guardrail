from __future__ import annotations

from pathlib import Path

from vipibench.dataio import write_json
from vipibench.schema import DatasetRecord


def export_dataset_schema(output_path: Path) -> dict[str, object]:
    schema = DatasetRecord.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://example.invalid/vipibench/dataset-record.schema.json"
    schema["title"] = "ViPIBench Dataset Record"
    write_json(output_path, schema)
    return {"status": "PASS", "output": str(output_path)}
