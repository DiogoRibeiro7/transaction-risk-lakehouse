"""Saved scoring artifact helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataexcept import DataLoadingError, FileReadError, FileWriteError, wrapping
from pyspark.ml import PipelineModel

ARTIFACT_METADATA_FILENAME = "artifact_metadata.json"
BASE_MODEL_DIRNAME = "base_model"
CALIBRATOR_DIRNAME = "calibrator"


@dataclass(frozen=True)
class ScoringArtifact:
    """A deployable fraud-scoring artifact."""

    model: PipelineModel
    calibrator: PipelineModel | None = None


def load_scoring_artifact(path: str | Path) -> ScoringArtifact:
    """Load either a legacy Spark pipeline or a bundled scoring artifact."""
    artifact_path = Path(path)
    metadata_path = artifact_path / ARTIFACT_METADATA_FILENAME
    with wrapping(OSError, FileReadError, path=str(metadata_path)):
        has_metadata = metadata_path.exists()
    if not has_metadata:
        return ScoringArtifact(model=PipelineModel.load(str(artifact_path)))

    with wrapping((OSError, UnicodeError), FileReadError, path=str(metadata_path)):
        serialized = metadata_path.read_text(encoding="utf-8")
    with wrapping(json.JSONDecodeError, DataLoadingError, source=str(metadata_path)):
        metadata = json.loads(serialized)
    model = PipelineModel.load(str(artifact_path / BASE_MODEL_DIRNAME))
    calibrator = None
    if metadata.get("has_calibrator", False):
        calibrator = PipelineModel.load(str(artifact_path / CALIBRATOR_DIRNAME))
    return ScoringArtifact(model=model, calibrator=calibrator)


def save_scoring_artifact(
    model: PipelineModel,
    output_path: str | Path,
    calibrator: PipelineModel | None = None,
) -> None:
    """Save a base model plus an optional calibrator into one artifact directory."""
    artifact_path = Path(output_path)
    with wrapping(OSError, FileWriteError, path=str(artifact_path)):
        artifact_path.mkdir(parents=True, exist_ok=True)

    model.write().overwrite().save(str(artifact_path / BASE_MODEL_DIRNAME))
    if calibrator is not None:
        calibrator.write().overwrite().save(str(artifact_path / CALIBRATOR_DIRNAME))

    metadata = {
        "artifact_version": 1,
        "has_calibrator": calibrator is not None,
    }
    metadata_path = artifact_path / ARTIFACT_METADATA_FILENAME
    with wrapping((OSError, UnicodeError), FileWriteError, path=str(metadata_path)):
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
