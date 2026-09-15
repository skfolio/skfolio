"""Tests for the base classes in `skfolio.base`."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from skfolio.base import BaseAssetPanelTransformer, BaseComposition


class _Composition(BaseComposition):
    """Minimal concrete composition exposing the `estimators` parameter."""

    def __init__(self, estimators=None, alpha=1.0):
        super().__init__()
        self.estimators = estimators
        self.alpha = alpha

    def get_params(self, deep=True):
        return self._get_params("estimators", deep=deep)

    def set_params(self, **params):
        self._set_params("estimators", **params)
        return self


def _make_composition() -> _Composition:
    return _Composition(
        estimators=[("a", StandardScaler()), ("b", StandardScaler(with_std=False))]
    )


class TestBaseAssetPanelTransformer:
    def test_stateless_subclass_gets_partial_fit_transform(self):
        class Stateless(BaseAssetPanelTransformer, stateless=True):
            def fit_transform(self, X, y=None, **fit_params):
                return np.asarray(X) * 2

        transformer = Stateless()
        assert transformer.stateless is True
        np.testing.assert_array_equal(
            transformer.partial_fit_transform(np.ones(3)), np.full(3, 2.0)
        )

    def test_stateless_subclass_defining_partial_fit_transform_raises(self):
        with pytest.raises(
            TypeError,
            match=(
                "Classes declared with stateless=True must not define "
                "partial_fit_transform"
            ),
        ):

            class Invalid(BaseAssetPanelTransformer, stateless=True):
                def fit_transform(self, X, y=None, **fit_params):
                    return X

                def partial_fit_transform(self, X, y=None, **fit_params):
                    return X


class TestBaseCompositionGetParams:
    def test_get_params_shallow_returns_constructor_arguments_only(self):
        composition = _make_composition()
        params = composition.get_params(deep=False)
        assert set(params) == {"estimators", "alpha"}

    def test_get_params_deep_includes_named_estimators_and_their_params(self):
        composition = _make_composition()
        params = composition.get_params(deep=True)
        assert params["a"] is composition.estimators[0][1]
        assert params["b"] is composition.estimators[1][1]
        assert params["a__with_std"] is True
        assert params["b__with_std"] is False

    def test_get_params_deep_ignores_non_iterable_estimators(self):
        # `dict.update(5)` raises TypeError, which is swallowed so that
        # `set_params` can still be used to fix an invalid `estimators` value.
        composition = _Composition(estimators=5)
        params = composition.get_params(deep=True)
        assert params == {"estimators": 5, "alpha": 1.0}

    def test_get_params_deep_ignores_malformed_estimators_list(self):
        # A list whose items are not `(name, estimator)` pairs makes
        # `dict.update` raise ValueError, which is swallowed as well.
        composition = _Composition(estimators=[("only_a_name",)])
        params = composition.get_params(deep=True)
        assert params == {"estimators": [("only_a_name",)], "alpha": 1.0}


class TestBaseCompositionSetParams:
    def test_set_params_replaces_the_whole_estimators_list(self):
        composition = _make_composition()
        new_estimators = [("c", StandardScaler())]
        composition.set_params(estimators=new_estimators)
        assert composition.estimators is new_estimators

    def test_replace_estimator_keeps_order_and_other_items(self):
        composition = _make_composition()
        old_b = composition.estimators[1][1]
        new_a = StandardScaler(with_mean=False)
        composition._replace_estimator("estimators", "a", new_a)
        assert composition.estimators == [("a", new_a), ("b", old_b)]

    def test_set_params_replaces_a_single_estimator_by_name(self):
        composition = _make_composition()
        old_b = composition.estimators[1][1]
        new_a = StandardScaler(with_mean=False)
        composition.set_params(a=new_a)
        assert composition.estimators[0] == ("a", new_a)
        assert composition.estimators[1] == ("b", old_b)

    def test_set_params_forwards_nested_and_own_parameters(self):
        composition = _make_composition()
        composition.set_params(a__with_std=False, alpha=2.0)
        assert composition.estimators[0][1].with_std is False
        assert composition.alpha == 2.0

    def test_set_params_unknown_parameter_raises(self):
        composition = _make_composition()
        with pytest.raises(ValueError, match="Invalid parameter 'unknown'"):
            composition.set_params(unknown=1)


class TestBaseCompositionValidateNames:
    def test_valid_names_pass(self):
        _make_composition()._validate_names(["a", "b"])

    def test_duplicate_names_raise(self):
        with pytest.raises(
            ValueError, match=r"Names provided are not unique: \['a', 'a'\]"
        ):
            _make_composition()._validate_names(["a", "a"])

    def test_names_conflicting_with_constructor_arguments_raise(self):
        with pytest.raises(
            ValueError,
            match=(
                r"Estimator names conflict with constructor arguments: "
                r"\['alpha', 'estimators'\]"
            ),
        ):
            _make_composition()._validate_names(["estimators", "alpha", "ok"])

    def test_names_containing_double_underscore_raise(self):
        with pytest.raises(
            ValueError, match=r"Estimator names must not contain __: got \['a__b'\]"
        ):
            _make_composition()._validate_names(["a__b", "c"])
