"""Datasets module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import gzip
import os
import shutil
import sys
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
    """Load gzip-compressed csv files with `importlib.resources`.

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
    **csv_reader_kwargs,
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
    # Use a CORS proxy when triggering requests from the browser
    url_prefix = "https://corsproxy.io/?" if sys.platform == "emscripten" else ""
    url = url_prefix + (
        f"https://github.com/BrandtWP/skfolio-datasets/raw/main/"
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
    df = load_gzip_compressed_csv_data(archive_path, **csv_reader_kwargs)
    joblib.dump(df, filepath, compress=6)
    os.remove(archive_path)
    return df


def load_sp500_dataset() -> pd.DataFrame:
    """Load the prices of 20 assets from the S&P 500 Index.

    This dataset contains daily adjusted closing prices for 20 selected constituents of
    the S&P 500 Index, covering the period from 1990-01-02 to 2022-12-28.

    .. caution::
        This dataset is provided solely for testing and example purposes. It is a stale
        dataset and does not reflect current or accurate market prices. It is not
        intended for investment, trading, or commercial use and should not be relied
        upon as authoritative market data.

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
    1990-01-02  0.264  4.125  4.599  0.144  ...  3.322  0.310  3.653  4.068
    1990-01-03  0.266  4.000  4.636  0.161  ...  3.322  0.304  3.653  4.027
    1990-01-04  0.267  3.938  4.537  0.159  ...  3.322  0.301  3.634  3.987
    1990-01-05  0.268  3.812  4.438  0.159  ...  3.322  0.288  3.595  3.966
    1990-01-08  0.269  3.812  4.463  0.147  ...  3.322  0.282  3.644  4.027
    """
    data_filename = "sp500_dataset.csv.gz"
    df = load_gzip_compressed_csv_data(data_filename)
    return df


def load_sp500_index() -> pd.DataFrame:
    """Load the prices of the S&P 500 Index.

    This dataset contains daily adjusted closing prices of the S&P 500 Index, covering
    the period from 1990-01-02 to 2022-12-28.

    .. caution::
        This dataset is provided solely for testing and example purposes. It is a stale
        dataset and does not reflect current or accurate market prices. It is not
        intended for investment, trading, or commercial use and should not be relied
        upon as authoritative market data.

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

    This dataset contains daily adjusted closing prices of 5 ETF representing common
    factors, covering the period from 2014-01-02 up to 2022-12-28.

    The factors are:

        * "MTUM": Momentum
        * "QUAL": Quality
        * "SIZE": Size
        * "VLUE": Value
        * "USMV": low volatility


    .. caution::
        This dataset is provided solely for testing and example purposes. It is a stale
        dataset and does not reflect current or accurate market prices. It is not
        intended for investment, trading, or commercial use and should not be relied
        upon as authoritative market data.

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

    This dataset contains daily adjusted closing prices of 64 assets from the FTSE 100
    Index, covering the period from 2000-01-04 up to 2023-05-31.
    The data contains NaN.

    .. caution::
        This dataset is provided solely for testing and example purposes. It is a stale
        dataset and does not reflect current or accurate market prices. It is not
        intended for investment, trading, or commercial use and should not be relied
        upon as authoritative market data.

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

    This dataset contains daily adjusted closing prices of 1455 assets from the NASDAQ
    Composite, covering the period from 2018-01-02 up to 2023-05-31.

    .. caution::
        This dataset is provided solely for testing and example purposes. It is a stale
        dataset and does not reflect current or accurate market prices. It is not
        intended for investment, trading, or commercial use and should not be relied
        upon as authoritative market data.

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


def load_sp500_implied_vol_dataset(
    data_home=None, download_if_missing=True
) -> pd.DataFrame:
    """Load the 3 months ATM implied volatility of the 20 assets from the
    SP500 dataset.

    This dataset is composed of the 3 months ATM implied volatility of 20 assets
    from the S&P 500 composition starting from 2010-01-04 up to 2022-12-28.

    .. caution::
        This dataset is provided solely for testing and example purposes. It is a stale
        dataset and does not reflect current or accurate market prices. It is not
        intended for investment, trading, or commercial use and should not be relied
        upon as authoritative market data.

    ==============   ==================
    Observations     3270
    Assets           20
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
        Implied volatility DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_implied_vol_dataset
    >>> implied_vol = load_sp500_implied_vol_dataset()
    >>> implied_vol.head()
                    AAPL       AMD       BAC  ...       UNH       WMT       XOM
    Date                                      ...
    2010-01-04  0.364353  0.572056  0.382926  ...  0.362751  0.171737  0.201485
    2010-01-05  0.371865  0.568791  0.374699  ...  0.368504  0.174764  0.203852
    2010-01-06  0.356746  0.558054  0.349220  ...  0.368514  0.171892  0.197475
    2010-01-07  0.361084  0.560475  0.354942  ...  0.355792  0.169083  0.200046
    2010-01-08  0.348085  0.543932  0.360345  ...  0.351130  0.170897  0.204832
    """
    data_filename = "sp500_implied_vol_dataset"
    df = download_dataset(
        data_filename, data_home=data_home, download_if_missing=download_if_missing
    )
    return df


def load_usd_rates_dataset(data_home=None, download_if_missing=True) -> pd.DataFrame:
    """Load the 1, 2, 3, 5, 7, 10, and 30 year daily end-of-day USD swap rates
    from 2000-07-03 to 2016-10-28.

    The data comes from the Investing.com public api.

    ==============   ==================
    Observations     1769
    Assets           7
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
        Implied volatility DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_usd_rates_dataset
    >>> usd_rates = load_usd_rates_dataset()
    >>> usd_rates.head()
                1Y    2Y    3Y    5Y    7Y   10Y   30Y
    Date                               ...
    2020-01-01  1.763  1.6909  1.6823  ...  1.774  1.873  2.0705
    2020-01-02  1.751  1.6370  1.6320  ...  1.737  1.826  2.0150
    2020-01-03  1.751  1.6370  1.6320  ...  1.737  1.826  2.0150
    2020-01-06  1.713  1.6100  1.5832  ...  1.650  1.741  1.9200
    2020-01-07  1.711  1.6040  1.5916  ...  1.664  1.762  1.9465
    """
    data_filename = "usd_rates_dataset"
    df = download_dataset(
        data_filename, data_home=data_home, download_if_missing=download_if_missing
    )
    return df


def load_bond_dataset(data_home=None, download_if_missing=True) -> pd.DataFrame:
    """Load the clean prices of a selection of 10 US corporate bonds from 2020-01-01
    through 2025-10-20.

    The data comes from the Luxembourg Stock Exchange's public website.

    ==============   ==================
    Observations     1231
    Assets           10
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
        Implied volatility DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_usd_rates_dataset
    >>> bond_prices = load_bond_dataset()
    >>> bond_prices.head()
                US606822BR40  US172967EW71  ...  US904764AH00  US254687FX90
    date                                    ...
    2021-01-04       107.809       178.590  ...       146.763       109.400
    2021-01-05       107.737       178.445  ...       146.558       109.085
    2021-01-06       106.994       173.600  ...       145.054       108.570
    2021-01-07       106.584       173.190  ...       144.869       108.250
    2021-01-08       106.464       173.995  ...       144.545       108.025
    """
    data_filename = "bond_dataset"
    df = download_dataset(
        data_filename, data_home=data_home, download_if_missing=download_if_missing
    )
    return df


def load_bond_metadata_dataset(
    data_home=None, download_if_missing=True
) -> pd.DataFrame:
    """Load the bond issue parameters for the bonds included in the bond_dataset.

    The data comes from FINRA's public website.

    ==============   ==================
    Bonds            10
    Parameters       10
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
        Implied volatility DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_bond_metadata_dataset
    >>> bond_info = load_bond_metadata_dataset()
    >>> bond_info.head()
                                                issuer  ... callable
    isin                                                ...
    US606822BR40         MITSUBISHI UFJ FINL GROUP INC  ...    False
    US172967EW71                         CITIGROUP INC  ...    False
    US86562MBP41        SUMITOMO MITSUI FINL GROUP INC  ...    False
    US925524AH30                            VIACOM INC  ...    False
    US233835AQ08  DAIMLERCHRYSLER NORTH AMER HLDG CORP  ...    False
    """
    data_filename = "bond_metadata_dataset"
    df = download_dataset(
        data_filename,
        data_home=data_home,
        download_if_missing=download_if_missing,
        datetime_index=False,
    )
    return df


def load_eur_rates_dataset(data_home=None, download_if_missing=True) -> pd.DataFrame:
    """Loads EUR risk-free yields for various maturities from 2004-09-06 to 2025-11-17.
    The yields are calculated using a Svensson interpolation fitted to AAA-rated sovereign
    bonds from eurozone issuers.

    The data comes from ECB's public API. For full details on their methodolgy, visit
    https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html

    ==============   ==================
    Observations     5420
    Parameters       8
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
        Implied volatility DataFrame

    Examples
    --------
    >>> from skfolio.datasets import load_eur_rates_dataset
    >>> eur_rates = load_eur_rates_dataset()
    >>> eur_rates.head()
                    3M        1Y    ...     15Y       30Y
    Date                            ...
    2004-09-06  2.034172  2.298838  ...  4.576354  4.988680
    2004-09-07  2.040893  2.328891  ...  4.569196  4.975495
    2004-09-08  2.044384  2.346666  ...  4.581290  4.978894
    2004-09-09  2.037111  2.308988  ...  4.527995  4.946545
    2004-09-10  2.034645  2.271566  ...  4.493569  4.918530
    """
    data_filename = "eur_risk_free_yields_dataset"
    df = download_dataset(
        data_filename, data_home=data_home, download_if_missing=download_if_missing
    )
    return df
