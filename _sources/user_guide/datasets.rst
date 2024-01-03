.. _datasets:

.. currentmodule:: skfolio.datasets

********
Datasets
********

`skfolio` comes with three natives datasets available via:

    * :func:`load_sp500_dataset`
    * :func:`load_sp500_index`
    * :func:`load_factors_dataset`

Larger datasets are downloaded from the GitHub repo and cached locally to a data
directory. They are available via:

    * :func:`load_ftse100_dataset`
    * :func:`load_nasdaq_dataset`

By default the data directory is set to a folder named "skfolio_data" in the user home
folder. Alternatively, it can be set by the `SKFOLIO_DATA` environment variable.
If the folder does not already exist, it is automatically created.


**Example:**

Loading the SPX 500 dataset, which is composed of the daily prices of 20 assets from the
S&P 500 composition starting from 1990-01-02 up to 2022-12-28:

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset

    prices = load_sp500_dataset()
    print(prices.head())


The data comes from the Yahoo public API.
The price is the adjusted close which is the closing price after adjustments for
all applicable splits and dividend distributions.
The adjustment uses appropriate split and dividend multipliers, adhering to
the Center for Research in Security Prices (CRSP) standards.