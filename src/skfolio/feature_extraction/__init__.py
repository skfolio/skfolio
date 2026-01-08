"""Feature Extraction module."""

from skfolio.feature_extraction._base import BaseFeatureExtractor

# Lazy import pattern for optional gen_fex dependency
# Import PPCA and PKPCA when gen_fex is available
try:
    from skfolio.feature_extraction._pkpca import PKPCA
    from skfolio.feature_extraction._ppca import PPCA
except ImportError:

    class _LazyImportError:
        """Placeholder class that raises an error when instantiated."""

        def __init__(self, *args, **kwargs):
            msg = (
                "gen_fex is required for feature extraction. "
                "Install it with: pip install 'skfolio[feature_extraction]', "
                "or using uv: uv pip install 'skfolio[feature_extraction]'"
            )
            raise ImportError(msg)

        def __getattr__(self, name):
            msg = (
                "gen_fex is required for feature extraction. "
                "Install it with: pip install 'skfolio[feature_extraction]', "
                "or using uv: uv pip install 'skfolio[feature_extraction]'"
            )
            raise ImportError(msg)

    # Set the module and qualname to make them look like the real classes
    PPCA = type("PPCA", (_LazyImportError,), {"__module__": __name__})
    PKPCA = type("PKPCA", (_LazyImportError,), {"__module__": __name__})


__all__ = [
    "PKPCA",
    "PPCA",
    "BaseFeatureExtractor",
]
