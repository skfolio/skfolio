import sklearn.utils.validation as skv
from sklearn import config_context

from skfolio.pre_selection import DropCorrelated


def test_validate_data(X):
    with config_context(transform_output="pandas"):
        model = DropCorrelated()
        _ = skv.validate_data(model, X)

    model = DropCorrelated()
    _ = skv.validate_data(model, X)
