# Quality charter

skfolio is not rhiza-managed. This file only records decisions the maintainers have
already taken, so that `/rhiza:quality` runs stop re-filing them.

## Accepted deviations

### E-rated complexity blocks stay as they are

The remaining E-rated blocks were reviewed for decomposition, and the maintainers chose
to keep the original code:

- `CharacteristicsFactorModel._fit` (`src/skfolio/prior/_characteristics_factor_model.py`)
- `cross_val_predict` (`src/skfolio/model_selection/_validation.py`)
- `rolling_realized_factor_attribution` (`src/skfolio/attribution/_realized.py`)
- `Population.plot_measures` (`src/skfolio/population/_population.py`)

The same applies to splitting `src/skfolio/prior/_model/_factor_model.py` into mixins.

Reason: the attempted refactors (#325, #326, #327) were closed without merging. They
moved the complexity around rather than removing it. #325 added large parameter lists
and extra navigation. #326 spread one tightly connected class across four mixins that
still share state and private helpers. #327 replaced plain local control flow with a
large mutable `_FitState` object. The maintainers found several of the original
implementations easier to understand and maintain. See the recap in
[#301](https://github.com/skfolio/skfolio/issues/301#issuecomment-5794436916) and the
closing of #346.

Cyclomatic complexity is still a useful signal for deciding what to review. It is not a
refactor mandate in itself. Before recommending a decomposition of any block,
including a new one:

- Name the concrete maintenance problem and say what becomes easier to understand or
  change.
- Say whether the change removes complexity or duplication, or only moves it.
- Weigh the new helper interfaces, parameter counts, shared mutable state,
  dependencies, navigation and existing package conventions.
- Allow the answer to be "keep the original" when the abstraction costs more than it
  gives.

A radon grade, or a higher per-file maintainability index, is not enough evidence on its
own.

### Static type checking is deferred, not missing

Type checking is agreed in principle and will become a required CI check. It is planned
for after Python 3.10 support is dropped at its end of life (October 2026), with Python
3.11 as the new minimum. The choice of checker (Pyright, Pyrefly, ty or mypy) is still
open.

Reason: tracked in [#300](https://github.com/skfolio/skfolio/issues/300). #348 was
closed as a duplicate of it. Until #300 is resolved, report type checking as not gated
and point to #300. Don't file a new issue for it.

## Filing

- Before filing, check closed issues and their discussion as well as open ones. A
  finding that was already discussed and closed should not be re-filed without new
  evidence that answers the reason it was closed.
