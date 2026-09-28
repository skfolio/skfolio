"""Keep the package import graph acyclic.

`skfolio.utils` is the most-imported subpackage in the project, and three of its
modules used to reach back up into `containers`, `linear_model` and `preprocessing`.
Those cycles resolved at import time only because each closed at *submodule*
granularity -- `utils.stats` itself does not import `preprocessing` -- which made them
latent rather than harmless: an import later added to `utils/__init__.py` could turn a
working import into an `ImportError` surfacing far from the change that caused it.

`validate_asset_panel` still needs `AssetPanel`, and imports it inside the function
body so the dependency is resolved at call time instead of import time. These tests
exist so that import stays where it is: moving it back to module scope reintroduces
the cycle, and `test_no_package_import_cycles` says so by name.

The check is static rather than a `sys.modules` assertion, because importing any
submodule first executes `skfolio/__init__.py`, which imports the whole package -- so
at runtime every subpackage is loaded no matter which one was asked for.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

import skfolio

PACKAGE = Path(skfolio.__file__).parent
ROOT = PACKAGE.name

# Everything layered above `utils`. None of it may be imported while `skfolio.utils`
# is being imported.
ESTIMATOR_LAYER = (
    "alpha",
    "base",
    "containers",
    "descriptor",
    "distribution",
    "factor_exposure",
    "linear_model",
    "measures",
    "model_selection",
    "moments",
    "optimization",
    "population",
    "portfolio",
    "pre_selection",
    "preprocessing",
    "prior",
)


class _ImportTimeVisitor(ast.NodeVisitor):
    """Collect the imports a module runs when it is imported.

    Function bodies and `if TYPE_CHECKING:` blocks are skipped: neither executes on
    import, so neither can close an import cycle. Class bodies are kept, because they
    do execute.
    """

    def __init__(self, module: str) -> None:
        self.module = module
        self.imports: list[tuple[str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_If(self, node: ast.If) -> None:
        if "TYPE_CHECKING" in ast.dump(node.test):
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append((alias.name, node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:  # relative import: resolve against the containing package
            parts = self.module.split(".")[: -node.level]
            target = ".".join(parts + ([node.module] if node.module else []))
        else:
            target = node.module or ""
        self.imports.append((target, node.lineno))


def _subpackage(module: str) -> str:
    """The top-level `skfolio` subpackage a module belongs to, if any."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == ROOT else ""


@pytest.fixture(scope="module")
def import_edges() -> dict[tuple[str, str], list[str]]:
    """Map each `(from, to)` subpackage edge to the source lines that create it."""
    if not PACKAGE.exists():  # pragma: no cover - defensive
        pytest.skip("skfolio sources are not available")

    edges: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in sorted(PACKAGE.rglob("*.py")):
        module = ".".join(path.relative_to(PACKAGE.parent).with_suffix("").parts)
        visitor = _ImportTimeVisitor(module)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))

        source = _subpackage(module)
        for target_module, lineno in visitor.imports:
            target = _subpackage(target_module)
            if target and target != source:
                location = f"{path.relative_to(PACKAGE.parent)}:{lineno}"
                edges[(source, target)].append(f"{location} imports {target_module}")
    return dict(edges)


def _strongly_connected_components(
    graph: dict[str, set[str]],
) -> list[set[str]]:
    """Tarjan's algorithm, iterative so a deep graph cannot blow the stack."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[set[str]] = []
    counter = 0

    for root in sorted(graph):
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_i = work[-1]
            if child_i == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)

            children = sorted(graph.get(node, ()))
            if child_i < len(children):
                work[-1] = (node, child_i + 1)
                child = children[child_i]
                if child not in index:
                    work.append((child, 0))
                elif child in on_stack:
                    low[node] = min(low[node], index[child])
                continue

            if low[node] == index[node]:
                component = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                components.append(component)
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])

    return components


def test_no_package_import_cycles(import_edges) -> None:
    """No two `skfolio` subpackages may import each other at import time."""
    graph: dict[str, set[str]] = defaultdict(set)
    for source, target in import_edges:
        graph[source].add(target)

    cyclic = [c for c in _strongly_connected_components(graph) if len(c) > 1]
    cyclic += [{n} for n in graph if n in graph[n]]

    if cyclic:
        report = ["skfolio subpackages import each other at import time."]
        for component in cyclic:
            report.append(f"\ncycle between: {', '.join(sorted(component))}")
            for (source, target), locations in sorted(import_edges.items()):
                if source in component and target in component:
                    report.append(f"  {source} -> {target}")
                    report.extend(f"      {loc}" for loc in locations)
        report.append(
            "\nMove the offending import into the function that needs it, or "
            "relocate the code to the layer that owns it."
        )
        pytest.fail("\n".join(report))


def test_utils_does_not_import_estimator_layer(import_edges) -> None:
    """`skfolio.utils` is imported by everything, so it may import nothing above it."""
    offenders = {
        (source, target): locations
        for (source, target), locations in import_edges.items()
        if source == "utils" and target in ESTIMATOR_LAYER
    }
    if offenders:
        detail = "\n".join(
            f"  utils -> {target}\n" + "\n".join(f"      {loc}" for loc in locations)
            for (_, target), locations in sorted(offenders.items())
        )
        pytest.fail(
            f"skfolio.utils imports estimator-layer packages at import time:\n{detail}"
            "\n\nEverything imports `utils`, so it must stay at the bottom of the "
            "layering. Import inside the function that needs it, or move the code to "
            "the layer that owns it."
        )
