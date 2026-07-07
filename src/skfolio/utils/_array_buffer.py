"""Amortized O(1) append buffer for numpy arrays along axis 0."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.typing import AnyArray


class _ArrayBuffer:
    """Pre-allocated buffer that grows geometrically for amortized O(1) appends.

    Avoids the O(N^2) cost of repeated `np.concatenate` when accumulating
    arrays one slice (or small batch) at a time along axis 0.

    The buffer starts empty unless initial values are provided; shape and dtype
    are inferred from the first :meth:`append` call.
    """

    __slots__ = ("_buffer", "_size")

    def __init__(self, values: AnyArray | None = None):
        self._buffer: AnyArray | None = None
        self._size: int = 0
        if values is not None:
            self.append(values)

    @property
    def array(self) -> AnyArray | None:
        """Read-only view of the valid region, or `None` if empty."""
        if self._buffer is None:
            return None
        view = self._buffer[: self._size]
        view.flags.writeable = False
        return view

    def append(self, values: AnyArray) -> None:
        """Append *values* along axis 0.

        First call allocates an exact-fit buffer (optimal for batch mode).
        Subsequent calls use geometric (2x) growth for amortized O(1) cost.
        """
        values = np.asarray(values)
        if values.shape[0] == 0:
            return
        n_new = values.shape[0]
        if self._buffer is None:
            self._buffer = values.copy()
            self._size = n_new
            return
        if values.dtype != self._buffer.dtype:
            raise ValueError(
                f"dtype mismatch: buffer has {self._buffer.dtype}, got {values.dtype}"
            )
        if values.shape[1:] != self._buffer.shape[1:]:
            raise ValueError(
                f"Shape mismatch along trailing dimensions: "
                f"buffer has {self._buffer.shape[1:]}, got {values.shape[1:]}"
            )
        required_capacity = self._size + n_new
        if required_capacity > self._buffer.shape[0]:
            new_capacity = max(required_capacity, self._buffer.shape[0] * 2)
            new_buffer = np.empty(
                (new_capacity, *self._buffer.shape[1:]), dtype=self._buffer.dtype
            )
            new_buffer[: self._size] = self._buffer[: self._size]
            self._buffer = new_buffer
        self._buffer[self._size : self._size + n_new] = values
        self._size += n_new

    def truncate_to_last(self, n_rows: int) -> None:
        """Keep only the last *n_rows* rows (for rolling-window / max_history).

        *n_rows* must be positive. If the buffer already has fewer than
        *n_rows* rows the call is a no-op.

        Truncation allocates a fresh internal buffer instead of shifting rows in
        place: views returned by :attr:`array` may still be referenced by earlier
        snapshots.
        """
        if n_rows <= 0:
            raise ValueError(f"`n_rows` must be positive, got {n_rows}")
        if self._size > n_rows:
            new_buffer = np.empty_like(self._buffer)
            new_buffer[:n_rows] = self._buffer[self._size - n_rows : self._size]
            self._buffer = new_buffer
            self._size = n_rows

    def clear(self) -> None:
        """Discard all data and release the underlying memory."""
        self._buffer = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        if self._buffer is None:
            return "_ArrayBuffer(empty)"
        shape = (self._size, *self._buffer.shape[1:])
        return (
            f"_ArrayBuffer(len={self._size}, shape={shape}, dtype={self._buffer.dtype})"
        )


def _update_buffer(buffer: AnyArray, values: AnyArray, lag: int) -> None:
    """Update the lag buffer in-place with the last `lag` rows of `values`.

    All operations copy data into the pre-allocated `buffer`.  No view into `values` is
    retained, so the caller's input array can be freed.

    Parameters
    ----------
    buffer : ndarray of shape (lag, n_assets)
        Pre-allocated buffer to update.

    values : ndarray of shape (n_observations, n_assets)
        New observations (may be a view into a larger array).

    lag : int
        Buffer size (number of lagged rows to retain).
    """
    n_observations = values.shape[0]
    if n_observations == 0:
        return
    if n_observations >= lag:
        np.copyto(buffer, values[-lag:])
    else:
        buffer[:-n_observations] = buffer[n_observations:]
        buffer[-n_observations:] = values
