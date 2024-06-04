import numpy as np
import sklearn.utils.validation as skv


def _check_implied_vol(implied_vol, X):
    """Validate sample weights.

    Note that passing sample_weight=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(...)

    Parameters
    ----------
    sample_weight : {ndarray, Number or None}, shape (n_samples,)
        Input sample weights.

    X : {ndarray, list, sparse matrix}
        Input data.

    only_non_negative : bool, default=False,
        Whether or not the weights are expected to be non-negative.

        .. versionadded:: 1.0

    dtype : dtype, default=None
        dtype of the validated `sample_weight`.
        If None, and the input `sample_weight` is an array, the dtype of the
        input is preserved; otherwise an array with the default numpy dtype
        is be allocated.  If `dtype` is not one of `float32`, `float64`,
        `None`, the output will be of dtype `float64`.

    copy : bool, default=False
        If True, a copy of sample_weight will be created.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,)
        Validated sample weight. It is guaranteed to be "C" contiguous.
    """

    n_observations, n_assets = X.shape

    if implied_vol is None:
        raise ValueError("`implied_vol` cannot be None")
    else:
        sample_weight = skv.check_array(
            implied_vol,
            accept_sparse=False,
            ensure_2d=False,
            dtype=[np.float64, np.float32],
            order="C",
            copy=False,
            input_name="implied_vol",
        )
        if implied_vol.ndim != 2:
            raise ValueError(
                "Sample weights must be 2D array of shape (n_observation, n_assets)"
            )

        if implied_vol.shape != (n_observations, n_assets):
            raise ValueError(
                f"implied_vol.shape == {(implied_vol.shape)}, "
                f"expected {(n_observations, n_assets)}"
            )

    skv.check_non_negative((n_observations, n_assets), "`implied_vol`")

    return sample_weight
