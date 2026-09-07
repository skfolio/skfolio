"""Test Dataset module."""

from __future__ import annotations

import gzip
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pandas as pd
import pytest

from skfolio.datasets import (
    _base,
    load_factors_dataset,
    load_ftse100_dataset,
    load_nasdaq_dataset,
    load_sp500_dataset,
    load_sp500_implied_vol_dataset,
    load_sp500_index,
)
from skfolio.datasets._base import (
    clear_data_home,
    download_dataset,
    get_data_home,
)


# These tests create and delete a data home. They must never point at the real
# `~/skfolio_data`: the suite runs under `-n=4 --dist=worksteal`, so wiping the shared
# cache races with every other test that reads a dataset from it, which surfaces as
# `FileNotFoundError` on one worker and forces the rest to re-download.
@pytest.fixture
def isolated_data_home(tmp_path, monkeypatch):
    """Point `SKFOLIO_DATA` at a per-test directory."""
    data_home = tmp_path / "skfolio_data"
    monkeypatch.setenv("SKFOLIO_DATA", str(data_home))
    return data_home


class TestGetDataHome:
    #  Returns the default path to skfolio data directory if no argument is passed
    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("SKFOLIO_DATA", raising=False)
        assert get_data_home() == os.path.expanduser(os.path.join("~", "skfolio_data"))

    #  Creates the skfolio data directory if it does not exist
    def test_create_directory(self, isolated_data_home):
        get_data_home()
        assert os.path.exists(isolated_data_home)


class TestClearDataHome:
    #  Deletes all content of data home cache when given a valid path.
    def test_delete_content_valid_path(self, tmp_path):
        # Set up
        data_home = tmp_path / "valid" / "path"
        os.makedirs(data_home)
        with open(os.path.join(data_home, "file1.txt"), "w") as f:
            f.write("test")

        # Execute
        clear_data_home(data_home)

        # Assert
        assert not os.path.exists(data_home)

    #  Deletes all content of default data home cache when no path is given.
    def test_delete_content_default_path(self, isolated_data_home):
        # Set up
        os.makedirs(isolated_data_home, exist_ok=True)
        with open(os.path.join(isolated_data_home, "file1.txt"), "w") as f:
            f.write("test")

        # Execute
        clear_data_home()

        # Assert
        assert not os.path.exists(isolated_data_home)

    #  Does not raise an error when given a non-existent path.
    def test_no_error_nonexistent_path(self, tmp_path):
        # Set up
        data_home = tmp_path / "nonexistent" / "path"

        # Execute and assert
        try:
            clear_data_home(data_home)
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")


class TestLoadSp500Dataset:
    #  Loads the S&P 500 dataset successfully
    def test_load_sp500_dataset_success(self):
        # Call the load_sp500_dataset function
        df = load_sp500_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_sp500_dataset_shape(self):
        # Call the load_sp500_dataset function
        df = load_sp500_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (8313, 20)

    #  DataFrame has the correct column names
    def test_load_sp500_dataset_columns(self):
        # Call the load_sp500_dataset function
        df = load_sp500_dataset()

        # Define the expected column names
        expected_columns = [
            "AAPL",
            "AMD",
            "BAC",
            "BBY",
            "CVX",
            "GE",
            "HD",
            "JNJ",
            "JPM",
            "KO",
            "LLY",
            "MRK",
            "MSFT",
            "PEP",
            "PFE",
            "PG",
            "RRC",
            "UNH",
            "WMT",
            "XOM",
        ]

        # Check if the column names of the DataFrame are correct
        assert list(df.columns) == expected_columns


class TestLoadSp500Index:
    def test_load_sp500_index_success(self):
        df = load_sp500_index()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_sp500_index_shape(self):
        df = load_sp500_index()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (8313, 1)

    #  DataFrame has the correct column names
    def test_load_sp500_dataset_columns(self):
        df = load_sp500_index()

        # Define the expected column names
        expected_columns = ["SP500"]

        # Check if the column names of the DataFrame are correct
        assert list(df.columns) == expected_columns


class TestLoadFactorsDataset:
    def test_load_factors_dataset_success(self):
        df = load_factors_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_factors_dataset_shape(self):
        df = load_factors_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (2264, 5)

    def test_load_factors_dataset_columns(self):
        df = load_factors_dataset()

        # Define the expected column names
        expected_columns = ["MTUM", "QUAL", "SIZE", "USMV", "VLUE"]

        # Check if the column names of the DataFrame are correct
        assert list(df.columns) == expected_columns


class TestLoadFtse100Dataset:
    def test_load_ftse100_dataset_success(self):
        df = load_ftse100_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_ftse100_dataset_shape(self):
        df = load_ftse100_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (5960, 64)


class TestNasdaqDataset:
    def test_load_nasdaq_dataset_success(self):
        df = load_nasdaq_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_nasdaq_dataset_shape(self):
        df = load_nasdaq_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (1362, 1455)


class TestSp500ImpliedVolDataset:
    def test_load_sp500_implied_vol_dataset_success(self):
        df = load_sp500_implied_vol_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_sp500_implied_vol_dataset_shape(self):
        df = load_sp500_implied_vol_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (3270, 20)


def _write_valid_gz(path: str) -> None:
    """Write a gzip-compressed CSV that `load_gzip_compressed_csv_data` can read."""
    csv = ",A,B\n2020-01-01,1.0,2.0\n2020-01-02,1.5,2.5\n"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(csv)


class TestDownloadDatasetCache:
    #  A failed download leaves neither a destination file nor temporary residue,
    #  so the next call retries instead of reading a truncated cache entry.
    @pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
    def test_failure_leaves_nothing_behind(self, tmp_path, monkeypatch, error):
        dest = tmp_path / "dataset.csv.gz"

        def interrupted(url, filename):
            # Write a partial file first, exactly as a cut-off download would.
            with open(filename, "wb") as f:
                f.write(b"\x1f\x8b\x08 truncated")
            raise error("connection lost")

        monkeypatch.setattr(_base.ur, "urlretrieve", interrupted)

        with pytest.raises(error):
            download_dataset("dataset", data_home=tmp_path)

        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    #  The destination does not exist while the download is in flight, so a
    #  concurrent reader can never observe a partially written file.
    def test_download_is_atomic_and_cached(self, tmp_path, monkeypatch):
        dest = tmp_path / "dataset.csv.gz"

        def downloading(url, filename):
            with open(filename, "wb") as f:
                f.write(b"\x1f\x8b\x08 truncated")
            assert not dest.exists()
            _write_valid_gz(filename)

        monkeypatch.setattr(_base.ur, "urlretrieve", downloading)
        df = download_dataset("dataset", data_home=tmp_path)

        assert list(df.columns) == ["A", "B"]
        assert len(df) == 2
        assert list(tmp_path.iterdir()) == [dest]
        monkeypatch.setattr(
            _base.ur,
            "urlretrieve",
            lambda url, filename: pytest.fail("should have used the cache"),
        )
        pd.testing.assert_frame_equal(
            download_dataset("dataset", data_home=tmp_path), df
        )

    def test_concurrent_downloads_with_open_cache(self, tmp_path, monkeypatch):
        dest = tmp_path / "dataset.csv.gz"
        downloaded = Barrier(2)
        publishing = Lock()
        rename = os.rename

        def downloading(url, filename):
            _write_valid_gz(filename)
            # Both callers must see an empty cache before either can publish.
            downloaded.wait(timeout=10)

        def rename_with_reader(src, dst):
            with publishing:
                if dest.exists():
                    # Keep the winner's file open while the other worker publishes.
                    with dest.open("rb"):
                        return rename(src, dst)
                return rename(src, dst)

        monkeypatch.setattr(_base.ur, "urlretrieve", downloading)
        monkeypatch.setattr(_base.os, "rename", rename_with_reader)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(download_dataset, "dataset", data_home=tmp_path)
                for _ in range(2)
            ]
            frames = [future.result(timeout=15) for future in futures]

        assert len(frames[0]) == 2
        pd.testing.assert_frame_equal(*frames)
        pd.testing.assert_frame_equal(
            frames[0], _base.load_gzip_compressed_csv_data(str(dest))
        )
        assert list(tmp_path.iterdir()) == [dest]

    def test_publish_error_propagates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _base.ur, "urlretrieve", lambda url, filename: _write_valid_gz(filename)
        )

        def denied(src, dst):
            raise PermissionError("cannot publish cache")

        monkeypatch.setattr(_base.os, "rename", denied)
        with pytest.raises(PermissionError, match="cannot publish cache"):
            download_dataset("dataset", data_home=tmp_path)
        assert list(tmp_path.iterdir()) == []

    #  A missing dataset still reports the original error when downloading is disabled.
    def test_missing_dataset_when_download_disabled(self, tmp_path):
        with pytest.raises(OSError, match="Data not found"):
            download_dataset(
                "some_dataset", data_home=str(tmp_path), download_if_missing=False
            )
