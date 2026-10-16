"""Execute authored documentation in reading order, with a fresh scope per page."""

import re
import textwrap
import warnings
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import plotly.io as pio
import pytest
from sklearn import config_context
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from threadpoolctl import threadpool_limits

from skfolio import datasets
from skfolio.datasets import _base, make_synthetic_characteristics
from skfolio.distribution import VineCopula
from skfolio.model_selection import OnlineGridSearch

ROOT = Path(__file__).resolve().parents[1]
# Gallery pages and generated API pages already have separate execution coverage.
DOCUMENTS = [
    ROOT / "README.rst",
    *sorted((ROOT / "docs").glob("*.rst")),
    # The factor-model guide's long online examples exceed the smoke-test budget.
    *sorted(
        path
        for path in (ROOT / "docs/user_guide").rglob("*.rst")
        if path.name != "factor_models.rst"
    ),
]


def python_blocks(path):
    """Yield (section, source line, code) for indented RST Python blocks."""
    lines = path.read_text(encoding="utf-8").splitlines()
    section = path.stem
    i = 0
    while i < len(lines):
        line = lines[i]
        if i and re.fullmatch(r"[=~*^\-]+", line.strip()):
            section = lines[i - 1].strip()
        if not re.fullmatch(r"\s*\.\. (?:code-block|code):: python\s*", line):
            i += 1
            continue
        indent = len(line) - len(line.lstrip())
        i += 1
        while i < len(lines) and (
            not lines[i].strip() or lines[i].lstrip().startswith(":")
        ):
            i += 1
        start = i
        while i < len(lines) and (
            not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
        ):
            i += 1
        yield section, start + 1, textwrap.dedent("\n".join(lines[start:i]))


def test_python_blocks(tmp_path):
    path = tmp_path / "example.rst"
    path.write_text(
        "First\n=====\n\n.. code-block:: python\n    :linenos:\n\n"
        "    import math\n\nSecond\n------\n\n.. code-block:: console\n\n"
        "    ignored shell command\n\n.. code:: python\n\n"
        "    result = math.sqrt(4)\n",
        encoding="utf-8",
    )
    blocks = list(python_blocks(path))
    assert [(section, line) for section, line, _ in blocks] == [
        ("First", 7),
        ("Second", 18),
    ]
    namespace = {}
    for _, _, source in blocks:
        exec(source, namespace)
    assert namespace["result"] == 2


@pytest.fixture
def small_documentation_data(request, monkeypatch):
    """Keep examples unchanged while exercising them on smaller input datasets."""

    def recent_prices(loader, *args, **kwargs):
        data = loader(*args, **kwargs)
        # Its worst-case return constraint needs the longer estimation history.
        if request.node.callspec.params["path"].stem == "uncertainty_set":
            return data
        # Retain the 2020 shock for the entropy-pooling stress views.
        data = data.loc["2019":"2021"].iloc[:601]
        return data.iloc[1:] if "implied_vol" in str(args[0]) else data

    def small_panel(n_assets=500, n_observations=2520, **kwargs):
        return make_synthetic_characteristics(
            n_assets=min(n_assets, 200),
            n_observations=min(n_observations, 600),
            **kwargs,
        )

    sample = VineCopula.sample

    def small_sample(self, n_samples=1, conditioning=None):
        return sample(self, n_samples=min(n_samples, 2000), conditioning=conditioning)

    monkeypatch.setattr(
        _base,
        "load_gzip_compressed_csv_data",
        partial(recent_prices, _base.load_gzip_compressed_csv_data),
    )
    monkeypatch.setattr(datasets, "make_synthetic_characteristics", small_panel)
    monkeypatch.setattr(VineCopula, "sample", small_sample)


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            path,
            id=path.relative_to(ROOT).as_posix(),
            marks=pytest.mark.network if path.stem == "metadata_routing" else (),
        )
        for path in DOCUMENTS
        if list(python_blocks(path))
    ],
)
def test_documentation(path, request, monkeypatch, tmp_path, small_documentation_data):
    namespace = {"__name__": "__documentation__"}
    if path.stem == "metadata_routing":
        monkeypatch.setattr(
            _base,
            "download_dataset",
            partial(request.getfixturevalue("remote_dataset"), _base.download_dataset),
        )

    monkeypatch.chdir(tmp_path)
    # pytest already distributes documents across workers.
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    monkeypatch.setattr(pio, "show", lambda *args, **kwargs: None)
    random_state = np.random.get_state()
    np.random.seed(0)
    try:
        with (
            config_context(),
            np.printoptions(),
            threadpool_limits(limits=1),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error", FitFailedWarning)
            for section, line, source in python_blocks(path):
                # The migration guide deliberately demonstrates removed 0.x APIs.
                historical = path.stem == "migration" and any(
                    old_api in source
                    for old_api in (
                        "alpha=0.2",
                        "expend_train=True",
                        "prior_estimator=FactorModel()",
                    )
                )
                try:
                    code = compile("\n" * (line - 1) + source, str(path), "exec")
                    with pytest.raises(TypeError) if historical else nullcontext():
                        exec(code, namespace)
                    for value in namespace.values():
                        if isinstance(
                            value, (GridSearchCV, RandomizedSearchCV, OnlineGridSearch)
                        ) and hasattr(value, "cv_results_"):
                            for name, scores in value.cv_results_.items():
                                if name.startswith(("mean_test_", "mean_score")):
                                    assert np.isfinite(scores).all(), name
                except (Exception, pytest.fail.Exception) as exc:
                    raise AssertionError(
                        f"{path.relative_to(ROOT)}:{line} ({section})"
                    ) from exc
    finally:
        np.random.set_state(random_state)
