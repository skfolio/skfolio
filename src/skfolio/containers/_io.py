"""Directory preparation and manifests for saved containers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

import json
import shutil
from pathlib import Path

_METADATA_FILENAME = "_metadata.json"


def _validate_save_directory(path: Path, *, overwrite: bool, container: str) -> None:
    """Check a destination before validating payloads or modifying existing files."""
    if path.exists():
        if path.is_file():
            raise ValueError(f"Path '{path}' is a file, not a directory.")
        if not (path / _METADATA_FILENAME).exists():
            raise ValueError(
                f"Directory '{path}' exists but is not a saved {container} "
                f"(missing {_METADATA_FILENAME})."
            )
        _read_manifest(path, container=container)
        if not overwrite:
            raise FileExistsError(
                f"Directory '{path}' already contains a saved {container}. "
                "Use overwrite=True to replace it."
            )


def _prepare_save_directory(path: Path) -> None:
    """Replace a validated save destination with an empty directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_manifest(
    path: Path, *, container: str, version: int, **entries: object
) -> None:
    """Write the container identity, format version and payload metadata."""
    metadata = {"container": container, "version": version, **entries}
    with open(path / _METADATA_FILENAME, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def _read_manifest(
    path: Path, *, container: str, max_version: int | None = None
) -> dict:
    """Read a matching container manifest, accepting legacy AssetPanel files."""
    metadata_path = path / _METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Not a saved {container}: '{path}' (missing {_METADATA_FILENAME})."
        )
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    saved_container = metadata.get("container", "AssetPanel")
    if saved_container != container:
        raise ValueError(
            f"Expected a saved {container}, found {saved_container} at '{path}'."
        )
    if max_version is not None and metadata["version"] > max_version:
        raise ValueError(
            f"Unsupported {container} format version {metadata['version']}."
        )
    return metadata
