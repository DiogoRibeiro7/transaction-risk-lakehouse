from __future__ import annotations

import importlib.util
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path("scripts") / "download_kaggle_ieee_cis.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("download_kaggle_ieee_cis", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

build_kaggle_download_command = SCRIPT_MODULE.build_kaggle_download_command
download_ieee_cis = SCRIPT_MODULE.download_ieee_cis
extract_selected_files = SCRIPT_MODULE.extract_selected_files
resolve_download_archive = SCRIPT_MODULE.resolve_download_archive


def test_build_kaggle_download_command() -> None:
    command = build_kaggle_download_command(
        competition="ieee-fraud-detection",
        download_dir=Path("data/raw"),
        force=True,
    )

    assert command == [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "ieee-fraud-detection",
        "-p",
        "data\\raw",
        "--force",
    ]


def test_resolve_download_archive_prefers_expected_name(tmp_path: Path) -> None:
    archive_path = tmp_path / "ieee-fraud-detection.zip"
    archive_path.write_bytes(b"zip")

    assert resolve_download_archive(tmp_path, "ieee-fraud-detection") == archive_path


def test_extract_selected_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "ieee-fraud-detection.zip"
    output_dir = tmp_path / "raw"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train_transaction.csv", "TransactionID\n1\n")
        archive.writestr("nested/train_identity.csv", "TransactionID\n1\n")

    extracted_paths = extract_selected_files(archive_path, output_dir)

    assert [path.name for path in extracted_paths] == ["train_transaction.csv", "train_identity.csv"]
    assert (output_dir / "train_transaction.csv").read_text(encoding="utf-8") == "TransactionID\n1\n"
    assert (output_dir / "train_identity.csv").read_text(encoding="utf-8") == "TransactionID\n1\n"


def test_download_ieee_cis_uses_existing_files_without_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    (output_dir / "train_transaction.csv").write_text("ok", encoding="utf-8")
    (output_dir / "train_identity.csv").write_text("ok", encoding="utf-8")

    def fail_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(SCRIPT_MODULE.subprocess, "run", fail_run)
    extracted_paths = download_ieee_cis(output_dir)

    assert [path.name for path in extracted_paths] == ["train_transaction.csv", "train_identity.csv"]


def test_download_ieee_cis_raises_clear_error_for_kaggle_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_called_process_error(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0],
            stderr="403 - Forbidden",
        )

    monkeypatch.setattr(SCRIPT_MODULE.subprocess, "run", raise_called_process_error)

    with pytest.raises(RuntimeError, match="accepted the competition rules"):
        download_ieee_cis(tmp_path / "raw", force_download=True)
