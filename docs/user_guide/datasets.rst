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
    * :func:`load_sp500_implied_vol_dataset`

By default the data directory is set to a folder named "skfolio_data" in the user home
folder. Alternatively, it can be set by the `SKFOLIO_DATA` environment variable.
If the folder does not already exist, it is automatically created.


.. caution::
    This dataset is provided solely for testing and example purposes. It is a stale
    dataset and does not reflect current or accurate market prices. It is not
    intended for investment, trading, or commercial use and should not be relied
    upon as authoritative market data.

**Example:**

Loading the SPX 500 dataset, which  contains daily adjusted closing prices for 20
selected constituents of the S&P 500 Index, covering the period from 1990-01-02 to
2022-12-28:

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset

    prices = load_sp500_dataset()
    print(prices.head())

