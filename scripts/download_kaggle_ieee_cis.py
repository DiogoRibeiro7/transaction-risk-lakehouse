"""Download and extract IEEE-CIS competition files with the Kaggle CLI."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

DEFAULT_COMPETITION = "ieee-fraud-detection"
DEFAULT_FILES = ("train_transaction.csv", "train_identity.csv")


def build_kaggle_download_command(
    competition: str,
    download_dir: Path,
    force: bool = False,
) -> list[str]:
    """Build the official Kaggle CLI command for competition downloads."""
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(download_dir),
    ]
    if force:
        command.append("--force")
    return command


def resolve_download_archive(download_dir: Path, competition: str) -> Path:
    """Resolve the competition ZIP archive path after a Kaggle CLI download."""
    expected = download_dir / f"{competition}.zip"
    if expected.exists():
        return expected

    zip_files = sorted(download_dir.glob("*.zip"))
    if len(zip_files) == 1:
        return zip_files[0]
    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP archive found in {download_dir}. Kaggle may not have downloaded the competition files."
        )
    raise FileNotFoundError(
        f"Multiple ZIP archives found in {download_dir}. Clean the directory or pass a dedicated download path."
    )


def extract_selected_files(
    archive_path: Path,
    output_dir: Path,
    selected_files: tuple[str, ...] = DEFAULT_FILES,
) -> list[Path]:
    """Extract a selected set of files from a Kaggle competition archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive_members = {Path(member).name: member for member in archive.namelist()}
        missing_files = [file_name for file_name in selected_files if file_name not in archive_members]
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(f"Archive {archive_path} is missing required files: {missing}.")

        extracted_paths: list[Path] = []
        for file_name in selected_files:
            member_name = archive_members[file_name]
            target_path = output_dir / file_name
            with archive.open(member_name) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_paths.append(target_path)
    return extracted_paths


def download_ieee_cis(
    output_dir: Path,
    competition: str = DEFAULT_COMPETITION,
    force_download: bool = False,
    force_extract: bool = False,
    keep_archive: bool = False,
) -> list[Path]:
    """Download and extract the IEEE-CIS train transaction and identity files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = [output_dir / file_name for file_name in DEFAULT_FILES]
    if all(path.exists() for path in existing_paths) and not (force_download or force_extract):
        logging.info("IEEE-CIS files already exist in %s", output_dir)
        return existing_paths

    command = build_kaggle_download_command(
        competition=competition,
        download_dir=output_dir,
        force=force_download,
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI is not installed or not on PATH. Install it with `pip install kaggle`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            "Kaggle download failed. Ensure you authenticated with the Kaggle CLI and accepted the "
            f"competition rules for `{competition}` on kaggle.com. CLI output: {stderr}"
        ) from exc

    archive_path = resolve_download_archive(output_dir, competition)
    extracted_paths = extract_selected_files(archive_path=archive_path, output_dir=output_dir)
    if not keep_archive:
        archive_path.unlink(missing_ok=True)
    return extracted_paths


def main() -> None:
    """Run the Kaggle IEEE-CIS downloader."""
    parser = argparse.ArgumentParser(
        description="Download IEEE-CIS transaction and identity CSV files with the Kaggle CLI"
    )
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    extracted_paths = download_ieee_cis(
        output_dir=Path(args.output_dir),
        competition=args.competition,
        force_download=args.force_download,
        force_extract=args.force_extract,
        keep_archive=args.keep_archive,
    )
    for path in extracted_paths:
        logging.info("Ready: %s", path)


if __name__ == "__main__":
    main()
