"""Datasets module."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import gzip
import os
import shutil
import urllib.request as ur
from importlib import resources
from pathlib import Path

import joblib
import pandas as pd

DATA_MODULE = "skfolio.datasets.data"


def get_data_home(data_home: str | Path | None = None) -> str:
    """Return the path of the skfolio data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default, the data directory is set to a folder named 'skfolio_data' in the
    user home folder.

    Alternatively, it can be set by the 'SKFOLIO_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str, optional
        The path to skfolio data directory. If `None`, the default path
        is `~/skfolio_data`.

    Returns
    -------
    data_home: str or path-like, optional
        The path to skfolio data directory.
    """
    if data_home is None:
        data_home = os.environ.get("SKFOLIO_DATA", os.path.join("~", "skfolio_data"))
    data_home = os.path.expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_home(data_home: str | Path | None = None) -> None:
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str or path-like, optional
        The path to scikit-learn data directory. If `None`, the default path
        is `~/skfolio_data`.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def load_gzip_compressed_csv_data(
    data_filename: str,
    data_module: str = DATA_MODULE,
    encoding="utf-8",
    datetime_index: bool = True,
) -> pd.DataFrame:
    """Loads gzip-compressed csv files with `importlib.resources`.

    1) Open resource file with `importlib.resources.open_binary`
    2) Decompress csv file with `gzip.open`
    3) Load decompressed data with `pd.read_csv`

    Parameters
    ----------
    data_filename : str
        Name of gzip-compressed csv file  (`'*.csv.gz'`) to be loaded from
        `data_module/data_file_name`. For example `'SPX500.csv.gz'`.

    data_module : str or module, default='skfolio.datasets.data'
        Module where data lives. The default is `'skfolio.datasets.data'`.

    encoding : str, default="utf-8"
        Name of the encoding that the gzip-decompressed file will be
        decoded with. The default is 'utf-8'.

    datetime_index: bool, default=True
        If this is set to True, the DataFrame index is converted to datetime with
        format="%Y-%m-%d".
        The default is `True`.

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        DataFrame with each row representing one observation and each column
        representing the asset price of a given observation.
    """
    path = resources.files(data_module).joinpath(data_filename)
    with path.open("rb") as compressed_file:
        compressed_file = gzip.open(compressed_file, mode="rt", encoding=encoding)
        df = pd.read_csv(compressed_file, sep=",", index_col=0)
        if datetime_index:
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return df


def download_dataset(
    data_filename: str,
    data_home: str | Path | None = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Download and save locally a dataset from the remote GitHub dataset folder.

    Parameters
    ----------
    data_filename : str
        Name of gzip-compressed csv file  (`'*.csv.gz'`) to be loaded from a remote
        GitHub dataset folder.

    data_home : str or path-like, optional
        Specify another download and cache folder for the datasets. By default,
        all skfolio data is stored in `~/skfolio_data` sub-folders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.
        The default is `True`.

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        DataFrame with each row representing one observation and each column
        representing the asset price of a given observation.
    """
    url = (
        f"https://github.com/skfolio/skfolio-datasets/raw/main/"
        f"datasets/{data_filename}.csv.gz"
    )

    data_home = get_data_home(data_home=data_home)
    filepath = os.path.join(data_home, f"{data_filename}.pkz")

    if os.path.exists(filepath):
        return joblib.load(filepath)

    if not download_if_missing:
        raise OSError("Data not found and `download_if_missing` is False")

    archive_path = os.path.join(data_home, os.path.basename(url))
    ur.urlretrieve(url, archive_path)
    df = load_gzip_compressed_csv_data(archive_path)
    joblib.dump(df, filepath, compress=6)
    os.remove(archive_path)
    return df


def load_sp500_dataset() -> pd.DataFrame:
    """Load the prices of 20 assets from the S&P 500 Index composition.

    This dataset is composed of the daily prices of 20 assets from the S&P 500
    composition starting from 1990-01-02 up to 2022-12-28.

    The data comes from the Yahoo public API.
    The price is the adjusted close which is the closing price after adjustments for
    all applicable splits and dividend distributions.
    The adjustment uses appropriate split and dividend multipliers, adhering to
    the Center for Research in Security Prices (CRSP) standards.

    ==============   ==================
    Observations     8313
    Assets           20
    ==============   ==================

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Prices DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> prices = load_sp500_dataset()
    >>> prices.head()
                    AAPL     AMD       BAC  ...       UNH       WMT      XOM
    1990-01-02  0.332589  4.1250  11.65625  ...  0.382813  5.890625  12.5000
    1990-01-03  0.334821  4.0000  11.75000  ...  0.375000  5.890625  12.3750
    1990-01-04  0.335938  3.9375  11.50000  ...  0.371094  5.859375  12.2500
    1990-01-05  0.337054  3.8125  11.25000  ...  0.355469  5.796875  12.1875
    1990-01-08  0.339286  3.8125  11.31250  ...  0.347656  5.875000  12.3750
    """
    data_filename = "sp500_dataset.csv.gz"
    df = load_gzip_compressed_csv_data(data_filename)
    return df


def load_sp500_index() -> pd.DataFrame:
    """Load the prices of the S&P 500 Index.

    This dataset is composed of the daily prices of the S&P 500 Index starting from
    1990-01-02 up to 2022-12-28.

    The data comes from the Yahoo public API.
    The price is the adjusted close which is the closing price after adjustments for
    all applicable splits and dividend distributions.
    The adjustment uses appropriate split and dividend multipliers, adhering to
    the Center for Research in Security Prices (CRSP) standards.

    ==============   ==================
    Observations     8313
    Assets           1
    ==============   ==================

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Prices DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_index
    >>> prices = load_sp500_index()
    >>> prices.head()
                 SP500
    Date
    1990-01-02  359.69
    1990-01-03  358.76
    1990-01-04  355.67
    1990-01-05  352.20
    1990-01-08  353.79
    """
    data_filename = "sp500_index.csv.gz"
    df = load_gzip_compressed_csv_data(data_filename)
    return df


def load_factors_dataset() -> pd.DataFrame:
    """Load the prices of 5 factor ETFs.

    This dataset is composed of the daily prices of 5 ETF representing common factors
    starting from 2014-01-02 up to 2022-12-28.

    The factors are:

        * "MTUM": Momentum
        * "QUAL": Quanlity
        * "SIZE": Size
        * "VLUE": Value
        * "USMV": low volatility

    The data comes from the Yahoo public API.
    The price is the adjusted close which is the closing price after adjustments for
    all applicable splits and dividend distributions.
    The adjustment uses appropriate split and dividend multipliers, adhering to
    the Center for Research in Security Prices (CRSP) standards.

    ==============   ==================
    Observations     2264
    Assets           5
    ==============   ==================

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Prices DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_factors_dataset
    >>> prices = load_factors_dataset()
    >>> prices.head()
                  MTUM    QUAL    SIZE    USMV    VLUE
    Date
    2014-01-02  52.704  48.351  48.986  29.338  47.054
    2014-01-03  52.792  48.256  48.722  29.330  46.999
    2014-01-06  52.677  48.067  48.722  29.263  46.991
    2014-01-07  53.112  48.455  48.731  29.430  47.253
    2014-01-08  53.502  48.437  48.731  29.422  47.253
    """
    data_filename = "factors_dataset.csv.gz"
    df = load_gzip_compressed_csv_data(data_filename)
    return df


def load_ftse100_dataset(data_home=None, download_if_missing=True) -> pd.DataFrame:
    """Load the prices of 64 assets from the FTSE 100 Index composition.

    This dataset is composed of the daily prices of 64 assets from the FTSE 100 Index
    starting from 2000-01-04 up to 2023-05-31.

    The data comes from the Yahoo public API.
    The price is the adjusted close which is the closing price after adjustments for
    all applicable splits and dividend distributions.
    The adjustment uses appropriate split and dividend multipliers, adhering to
    the Center for Research in Security Prices (CRSP) standards.
    The data contains NaN.

    ==============   ==================
    Observations     5960
    Assets           64
    ==============   ==================

    Parameters
    ----------
    data_home : str, optional
        Specify another download and cache folder for the datasets.
        By default, all skfolio data is stored in `~/skfolio_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Prices DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_ftse100_dataset
    >>> prices = load_ftse100_dataset()
    >>> prices.head()
                  AAL.L    ABF.L   AHT.L  ANTO.L  ...   VOD.L   WEIR.L    WPP.L    WTB.L
    Date                                          ...
    2000-01-04  535.354  205.926  97.590  40.313  ...  72.562  115.240  512.249  382.907
    2000-01-05  540.039  209.185  96.729  40.313  ...  69.042  118.483  462.080  381.972
    2000-01-06  553.289  229.048  95.581  40.452  ...  66.950  124.220  458.119  386.337
    2000-01-07  572.829  222.220  95.581  40.452  ...  70.716  121.725  475.283  405.046
    2000-01-10  578.852  224.548  92.711  40.685  ...  74.285  121.476  498.254  392.885
    """
    data_filename = "ftse100_dataset"
    df = download_dataset(
        data_filename, data_home=data_home, download_if_missing=download_if_missing
    )
    return df


def load_nasdaq_dataset(data_home=None, download_if_missing=True) -> pd.DataFrame:
    """Load the prices of 1455 assets from the NASDAQ Composite Index.

    This dataset is composed of the daily prices of 1455 assets from the NASDAQ
    Composite starting from 2018-01-02 up to 2023-05-31.

    The data comes from the Yahoo public API.
    The price is the adjusted close which is the closing price after adjustments for
    all applicable splits and dividend distributions.
    The adjustment uses appropriate split and dividend multipliers, adhering to
    the Center for Research in Security Prices (CRSP) standards.

    ==============   ==================
    Observations     1362
    Assets           1455
    ==============   ==================

    Parameters
    ----------
    data_home : str, optional
        Specify another download and cache folder for the datasets.
        By default, all skfolio data is stored in `~/skfolio_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    df : DataFrame of shape (n_observations, n_assets)
        Prices DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_nasdaq_dataset
    >>> prices = load_nasdaq_dataset()
    >>> prices.head()
                   AAL   AAOI    AAON    AAPL  ...  ZVRA   ZYME    ZYNE   ZYXI
    Date                                       ...
    2018-01-02  51.648  37.91  35.621  41.310  ...  66.4  7.933  12.995  2.922
    2018-01-03  51.014  37.89  36.247  41.303  ...  72.8  7.965  13.460  2.913
    2018-01-04  51.336  38.38  36.103  41.495  ...  78.4  8.430  12.700  2.869
    2018-01-05  51.316  38.89  36.681  41.967  ...  77.6  8.400  12.495  2.780
    2018-01-08  50.809  38.37  36.103  41.811  ...  82.4  8.310  12.550  2.825
    """
    data_filename = "nasdaq_dataset"
    df = download_dataset(
        data_filename, data_home=data_home, download_if_missing=download_if_missing
    )
    return df
