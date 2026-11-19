"""Directory preparation and manifests for saved containers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

import json
import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd

from skfolio.typing import AnyArray

_METADATA_FILENAME = "_metadata.json"


def _encode_observations(*, observations: AnyArray) -> tuple[AnyArray, str | None]:
    """Convert normalized observations to values and an optional timezone for saving.

    Dates with a timezone are stored as UTC values with a separate timezone string.
    Named zones retain their public identifier. Fixed offsets use `+HH:MM` or `-HH:MM`.
    Unsupported timezones raise `ValueError` before the save directory is modified.
    """
    if observations.dtype != object:
        return observations, None

    dates = pd.DatetimeIndex(observations)
    if dates.tz is None:
        return dates.to_numpy(), None

    timezone = getattr(dates.tz, "key", None) or getattr(dates.tz, "zone", None)
    if timezone is None:
        offset = dates.tz.utcoffset(None)
        if offset is None or offset % timedelta(minutes=1):
            raise ValueError(
                "Saving observations requires a named timezone or a fixed UTC offset "
                "in whole minutes."
            )
        minutes = offset // timedelta(minutes=1)
        hours, minutes_remainder = divmod(abs(minutes), 60)
        timezone = (
            "UTC"
            if minutes == 0
            else f"{'+' if minutes > 0 else '-'}{hours:02}:{minutes_remainder:02}"
        )

    # Loading must recognize the saved timezone identifier.
    try:
        pd.DatetimeIndex([], tz=timezone)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Cannot save observations with timezone {timezone!r}."
        ) from exc
    return dates.tz_convert("UTC").tz_localize(None).to_numpy(), timezone


def _decode_observations(*, observations: AnyArray, timezone: str | None) -> AnyArray:
    """Restore saved observations, converting UTC values to their saved timezone."""
    if timezone is None:
        return observations

    if not isinstance(timezone, str) or observations.dtype.kind != "M":
        raise ValueError(
            "A saved observations timezone requires a string and datetime values."
        )

    try:
        return (
            pd.DatetimeIndex(observations)
            .tz_localize("UTC")
            .tz_convert(timezone)
            .to_numpy()
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid observations timezone {timezone!r}.") from exc


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
