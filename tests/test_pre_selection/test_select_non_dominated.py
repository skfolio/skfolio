import numpy as np
from sklearn import set_config

from skfolio.pre_selection import SelectNonDominated


def test_select_non_dominated(X):
    set_config(transform_output="pandas")

    model = SelectNonDominated(min_n_assets=10)
    model.fit(X)
    assert np.all(
        model.feature_names_in_[model.to_keep_] == model.get_feature_names_out()
    )
    new_X = model.transform(X)
    assert new_X.shape[0] == X.shape[0]
    assert new_X.shape[1] < X.shape[0]
    assert new_X.shape[1] >= 10
    assert np.all(new_X.columns == model.feature_names_in_[model.to_keep_])
