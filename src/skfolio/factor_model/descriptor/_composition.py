"""Descriptor composition mixin with scikit-learn metadata routing."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sklearn.utils as sku
import sklearn.utils.metadata_routing as skm

from skfolio.factor_model.descriptor import BaseDescriptor, Passthrough
from skfolio.utils.composition import BaseComposition

__all__ = ["DescriptorCompositionMixin"]


class DescriptorCompositionMixin(BaseComposition):
    """Mixin class for all descriptor composition estimators in skfolio.

    This mixin provides `get_params` / `set_params` / metadata routing
    for the `descriptors` parameter, following the scikit-learn named
    estimator convention (similar to `Pipeline` or `ColumnTransformer`).

    Descriptors can be specified as `(name, estimator)` tuples or as
    `(name, field_name_string)` tuples. Strings are automatically wrapped
    as :class:`~skfolio.factor_model.descriptor.Passthrough` instances.
    """

    descriptors: list[tuple[str, BaseDescriptor | str]]

    @property
    def named_descriptors(self):
        """Dictionary to access any fitted factors by name.

        Returns
        -------
        :class:`~sklearn.utils.Bunch`
        """
        return sku.Bunch(**dict(self.descriptors))

    def set_params(self, **params):
        """Set the parameters of an factor from the ensemble.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super()._set_params("descriptors", **params)
        return self

    def get_params(self, deep=True):
        """Get the parameters of an estimator from the ensemble.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        return super()._get_params("descriptors", deep=deep)

    def get_metadata_routing(self):
        """Return metadata routing for descriptor estimators."""
        router = skm.MetadataRouter(owner=self.__class__.__name__)
        names, descriptors = self._validate_descriptors()
        for name, descriptor in zip(names, descriptors, strict=True):
            router.add(
                **{name: descriptor},
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
        return router

    def _validate_descriptors(self) -> tuple[list[str], list[BaseDescriptor]]:
        """Validate the `descriptors` parameter.

        Strings in the descriptor list are automatically wrapped as
        :class:`Passthrough` instances (shorthand for field lookups).

        Returns
        -------
        names : list[str]
            The list of descriptor names.
        descriptors : list[BaseDescriptor]
            The list of descriptor estimators (strings resolved to Passthrough).
        """
        if self.descriptors is None or len(self.descriptors) == 0:
            raise ValueError(
                "Invalid 'descriptors' attribute, 'descriptors' should be a "
                "list of (name, descriptor) tuples. Strings are accepted as "
                "shorthand for Passthrough(field_name)."
            )
        names, descriptors = zip(*self.descriptors, strict=True)
        descriptors = [Passthrough(d) if isinstance(d, str) else d for d in descriptors]

        # defined by MetaEstimatorMixin
        self._validate_names(names)

        return list(names), descriptors
