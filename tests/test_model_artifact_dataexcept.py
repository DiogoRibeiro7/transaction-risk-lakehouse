"""Operational errors for the scoring artifact's JSON metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
from dataexcept import DataLoadingError, FileReadError, FileWriteError

from transaction_risk.models import artifact


class _FakeWriter:
    def overwrite(self) -> _FakeWriter:
        return self

    def save(self, _path: str) -> None:
        return None


class _FakeModel:
    def write(self) -> _FakeWriter:
        return _FakeWriter()


def test_artifact_metadata_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel()
    monkeypatch.setattr(artifact.PipelineModel, "load", lambda _path: model)

    artifact.save_scoring_artifact(model, tmp_path)
    loaded = artifact.load_scoring_artifact(tmp_path)

    assert loaded.model is model
    assert loaded.calibrator is None


def test_artifact_metadata_read_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata_path = tmp_path / artifact.ARTIFACT_METADATA_FILENAME
    metadata_path.touch()
    original = PermissionError("read denied")

    def fail_read(_path: Path, *, encoding: str) -> str:
        raise original

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(FileReadError) as caught:
        artifact.load_scoring_artifact(tmp_path)

    assert caught.value.path == str(metadata_path)
    assert caught.value.original is original
    assert caught.value.__cause__ is original


def test_artifact_metadata_decode_error(tmp_path: Path) -> None:
    metadata_path = tmp_path / artifact.ARTIFACT_METADATA_FILENAME
    metadata_path.write_text("{broken json}", encoding="utf-8")

    with pytest.raises(DataLoadingError) as caught:
        artifact.load_scoring_artifact(tmp_path)

    assert caught.value.source == str(metadata_path)
    assert isinstance(caught.value.original, ValueError)
    assert caught.value.__cause__ is caught.value.original


def test_artifact_metadata_write_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata_path = tmp_path / artifact.ARTIFACT_METADATA_FILENAME
    original = PermissionError("write denied")

    def fail_write(_path: Path, _data: str, *, encoding: str) -> None:
        raise original

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(FileWriteError) as caught:
        artifact.save_scoring_artifact(_FakeModel(), tmp_path)

    assert caught.value.path == str(metadata_path)
    assert caught.value.original is original
    assert caught.value.__cause__ is original
