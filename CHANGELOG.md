# CHANGELOG

<!-- version list -->

## v1.0.0 (2026-08-23)

### Bug Fixes

- Return picklable routed params from _route_params
  ([`47c2f07`](https://github.com/skfolio/skfolio/commit/47c2f07be25d9738d0ed6d8eb10e6669ed46867e))

- **containers**: Support Python 3.11 in AssetPanelView default selector
  ([`df82c3c`](https://github.com/skfolio/skfolio/commit/df82c3c1de1ee856df4009f92231677f60d02694))

### Build System

- **release**: Enable 1.0.0 bump for breaking changes
  ([`6e11523`](https://github.com/skfolio/skfolio/commit/6e11523e8ed03a7e5e1307505388de0a707c6f82))

- **release**: Restore changelog generation on python-semantic-release v10
  ([`340e235`](https://github.com/skfolio/skfolio/commit/340e2355f1e2ae6af6832c59dc8a594189c4d7ea))

### Chores

- **release**: Drop semantic-release keys that match upstream defaults
  ([`fe5e6f2`](https://github.com/skfolio/skfolio/commit/fe5e6f2107c04667d0d032a2a71dd241bdf1ec1e))

### Continuous Integration

- Build docs in fast mode to keep the PR check under 5 minutes
  ([`cbbfbb9`](https://github.com/skfolio/skfolio/commit/cbbfbb985a819ef499e9a7e25c7c41430cee7e34))

### Documentation

- Add documentation on missing data (NaNs)
  ([`d8a53da`](https://github.com/skfolio/skfolio/commit/d8a53dad9c28e3ade933c337c61f4e449229bd7d))

- Add factor model tutorials
  ([`6045669`](https://github.com/skfolio/skfolio/commit/60456694b06e01e2b242b2175dd1f9d5f0fcb293))

- Add migration guide for v1.0.0
  ([`d34f639`](https://github.com/skfolio/skfolio/commit/d34f639edb4bf6cb2c0c0d57740c8120929f6fc4))

- Correct Core Web Vitals extension
  ([`d209789`](https://github.com/skfolio/skfolio/commit/d2097892cb9de3b4b0783c3b09c395f51b1c2223))

- Fix typos
  ([`2762fd5`](https://github.com/skfolio/skfolio/commit/2762fd5266117ccf9c014dbd3a3730b54766ab71))

- Improve documentation
  ([`cc86f21`](https://github.com/skfolio/skfolio/commit/cc86f2150190c385646a4390580c848401676c26))

- Improve documentation
  ([`399b732`](https://github.com/skfolio/skfolio/commit/399b7324fd7023bbfe83c4b8f8c077f6c8c84c58))

- Improve documentation website
  ([`7c2b05d`](https://github.com/skfolio/skfolio/commit/7c2b05d2805f06c5b80f548397d8e97daadfb70a))

- Improve doc website SEO
  ([`18623e7`](https://github.com/skfolio/skfolio/commit/18623e7a5b597ed713b2ac3a6a9eee5502fb1a9b))

- Improve factor model documentation
  ([`fa978e0`](https://github.com/skfolio/skfolio/commit/fa978e0703e7d795fd881de1b66c2058e31e1b12))

- **examples**: Reduce gallery runtime
  ([`e5be8e9`](https://github.com/skfolio/skfolio/commit/e5be8e911ae259504f6afbb20c84082e0a679247))

- **uncertainty-set**: Document generalized norm-ball uncertainty sets
  ([`d699665`](https://github.com/skfolio/skfolio/commit/d6996653250d0856f066fb0f9d8fd3af8d30a6dd))

### Features

- Add 49 Descriptors used in cross-sectional Characteristics Factor Model
  ([`17ed51c`](https://github.com/skfolio/skfolio/commit/17ed51ca29faf84a6824c91f871f78f0dec10f52))

- Add `AssetPanel` container for skfolio cross-sectional workflows
  ([`c5db13c`](https://github.com/skfolio/skfolio/commit/c5db13c09d5cf3ea1f5df55a908886ce9d7d7ba0))

- Add a new optional `factor_model` attribute to `ReturnDistribution`
  ([`77f14df`](https://github.com/skfolio/skfolio/commit/77f14df2800a0358a34b44ebc5643bf1c89dabc5))

- Add alpha estimator using a user-provided ML regressor
  ([`399d3b2`](https://github.com/skfolio/skfolio/commit/399d3b2fa15b760176c8097d1d0edf6f9e5abd0d))

- Add exponentially weighted least-squares Sharpe-optimal alpha estimator
  ([`4825714`](https://github.com/skfolio/skfolio/commit/4825714e6cf5fdd626b38ad16464b43e240a0d11))

- Add factor exposure constraints to convex portfolio optimizations
  ([`fd630a7`](https://github.com/skfolio/skfolio/commit/fd630a7e17adc3e255b4be7491f16bdf7d22a960))

- Add Factor Exposure Transformers
  ([`b255d9a`](https://github.com/skfolio/skfolio/commit/b255d9a2b6c79f94fbb5d9f248d0e30d52d13cb7))

- Add factor model ex-post and ex-ante attributions
  ([`621fff7`](https://github.com/skfolio/skfolio/commit/621fff7ccb1fbad8937487b812c2850f5e723fef))

- Add investable universe and NaNs handling in convex portfolio optimizations
  ([`d7ffea3`](https://github.com/skfolio/skfolio/commit/d7ffea3fc5c2da718a62f851ca2168dae385a18b))

- **prior**: `TimeSeriesFactorModel` now requires `factors` as a keyword argument
  ([`2d2de44`](https://github.com/skfolio/skfolio/commit/2d2de447c97377f87ab1c85340b04c72b1e9c933))

- **prior**: Add CharacteristicsFactorModel
  ([`c990a9d`](https://github.com/skfolio/skfolio/commit/c990a9d1558e983c4a26beb1511cc8a1f47e6680))

- **prior**: Handle NaN in EmpiricalPrior return scenarios
  ([`7b4a7ac`](https://github.com/skfolio/skfolio/commit/7b4a7acfe6a723da8a03b622d2fb3ca4ccf64c7f))

- **uncertainty-set**: Generalize uncertainty sets beyond ellipsoids and add orthogonal uncertainty
  sets estimators
  ([`1f223b7`](https://github.com/skfolio/skfolio/commit/1f223b7fc26c44dc75bfcce4665d13d7c79c1aa1))

### Refactoring

- Remove deprecated alpha parameter from EWMu and EWCovariance
  ([`7f67118`](https://github.com/skfolio/skfolio/commit/7f671181accb91be0ff226c6f0ca5a756f4beb46))

- Remove deprecated expend_train parameter from WalkForward and keep expand_train
  ([`5b2101f`](https://github.com/skfolio/skfolio/commit/5b2101f52ca39b60be35c2cccb60357ca406f3a2))

- Remove deprecated FactorModel alias in favor of TimeSeriesFactorModel
  ([`1ffc002`](https://github.com/skfolio/skfolio/commit/1ffc00252f481af92f7339dfc8d55880368c4408))

### Testing

- Reduce test suite runtime
  ([`4100b17`](https://github.com/skfolio/skfolio/commit/4100b176f45f51909b3c1dab4fb29aa2e7b99868))

### Breaking Changes

- The deprecated `alpha` parameter has been removed from `EWMu` and `EWCovariance`. Use
  `half_life` instead.

- `WalkForward` no longer accepts the deprecated `expend_train` parameter. Use
  `expand_train` instead.

- `FactorModel` has been renamed in favor of `TimeSeriesFactorModel` and no longer
  accepts `residual_variance`. The name `FactorModel` now refers to the fitted factor
  model container on `ReturnDistribution.factor_model`.

- The `cholesky` attribute has been removed from `ReturnDistribution` and replaced by
  the `covariance_sqrt` property.

- `UncertaintySet` is a general norm ball. The fields `k` and `sigma` are replaced by
  `radius`, `geometry` and `norm`.

- `TimeSeriesFactorModel` no longer accepts factor returns as the positional `y`
  argument. Pass them with `factors=...` instead.


## v0.20.2 (2026-08-13)

### Bug Fixes

- **ShrunkMu**: Use Hermitian eigenvalue solver to keep outputs real
  ([#256](https://github.com/skfolio/skfolio/pull/256),
  [`bc4db81`](https://github.com/skfolio/skfolio/commit/bc4db81b321fa9c4b7a0b2206780ea711ca90c60))

### Chores

- Ruff formatting
  ([`aebb0c0`](https://github.com/skfolio/skfolio/commit/aebb0c0bc3a87df075d203c0a8895ad3f1fb92df))

### Continuous Integration

- Bump actions/checkout from 6 to 7 (#245) [skip ci]
  ([`109ed13`](https://github.com/skfolio/skfolio/commit/109ed13fee0125ff9b001b8be33643b17e791578))

- Bump actions/dependency-review-action from 4 to 5 (#238) [skip ci]
  ([`e5fe750`](https://github.com/skfolio/skfolio/commit/e5fe75088aecae1c2833903be03423c622762ed5))

- Bump codecov/codecov-action from 6 to 7 (#243) [skip ci]
  ([`c06db84`](https://github.com/skfolio/skfolio/commit/c06db8406385cf4bce83577d0f0710ee5a27e67e))

### Documentation

- Add dedicated LLM docs via llms.txt and per-page markdown
  ([#239](https://github.com/skfolio/skfolio/pull/239),
  [`45fa9af`](https://github.com/skfolio/skfolio/commit/45fa9af5573c0c9c0d71ad969b80050c6c1deecd))

- Add examples in MeanRisk docstring ([#236](https://github.com/skfolio/skfolio/pull/236),
  [`c174583`](https://github.com/skfolio/skfolio/commit/c17458370cd21ac503c7e95d6b945108cf5f2b24))

- Add examples to convex optimization docstrings
  ([#237](https://github.com/skfolio/skfolio/pull/237),
  [`5c902a5`](https://github.com/skfolio/skfolio/commit/5c902a5c1d7c753d09fba7371c81517d9c9767ce))

- **optimization**: Improve add_constraints docstring (#249) [skip ci]
  ([`8fc9fcb`](https://github.com/skfolio/skfolio/commit/8fc9fcb4a54b50568d0a1ac63aa662e126babdc6))


## v0.20.1 (2026-04-21)

### Bug Fixes

- Improve typing
  ([`3d2f2bc`](https://github.com/skfolio/skfolio/commit/3d2f2bce3db3a3228d75e25e617485dba4352d35))


## v0.20.0 (2026-04-20)

### Continuous Integration

- Bump actions/configure-pages from 5 to 6 (#229) [skip ci]
  ([`9a976af`](https://github.com/skfolio/skfolio/commit/9a976af4df1784d7b43078c7c49ce93e725d4df4))

- Bump actions/upload-pages-artifact from 4 to 5 (#233) [skip ci]
  ([`dfe8568`](https://github.com/skfolio/skfolio/commit/dfe8568eb9348bb950c611d13be0ce91de32d757))

### Features

- Add cross-sectional transformers
  ([`ae50462`](https://github.com/skfolio/skfolio/commit/ae5046294f9e2758c67e730ee25dbd1f62b409e1))


## v0.19.0 (2026-04-16)

### Features

- Add cross-sectional regression estimators
  ([`a0fd1c2`](https://github.com/skfolio/skfolio/commit/a0fd1c225f320e3ace93543232d9973d4d6c3159))

### Testing

- Fix unit tests
  ([`6cb2344`](https://github.com/skfolio/skfolio/commit/6cb2344a1c91939111ecca4e0f0caaf97427178a))

- Fix unit tests on underdetermined systems
  ([`04d30c0`](https://github.com/skfolio/skfolio/commit/04d30c0572c44aacbd96f07b92f2709ca612c401))


## v0.18.0 (2026-04-14)

### Chores

- Documentation improved
  ([`fcfc919`](https://github.com/skfolio/skfolio/commit/fcfc9192d85562235a7ef85854d7dba8db69499f))

### Continuous Integration

- Remove maxfail=1
  ([`4f24aef`](https://github.com/skfolio/skfolio/commit/4f24aefb0fa4bbf84d13f6d30226fe7c8185d8a3))

### Features

- Add covariance forecast evaluation utilities
  ([`634c537`](https://github.com/skfolio/skfolio/commit/634c53761767e2b17cdb8e94644fb4435da878b1))

- Add online learning utilities (online_score, online_predict, OnlineGridSearch and
  OnlineRandomizedSearch)
  ([`ea25dea`](https://github.com/skfolio/skfolio/commit/ea25dea733462d4081ff4118c366173070a18d7e))

- Add partial_fit support for EmpiricalPrior
  ([`3bb8cc7`](https://github.com/skfolio/skfolio/commit/3bb8cc7570a63ea8a5b83c3e83cfab064845c5ba))

- Add partial_fit support for MeanRisk
  ([`c600e26`](https://github.com/skfolio/skfolio/commit/c600e267fb6ae02557e5fbc8a0fb7d80d882a432))

### Testing

- Unit tests
  ([`0097100`](https://github.com/skfolio/skfolio/commit/0097100436acf0283b51fefd0d5d6566bfedd6fe))

- Unit tests
  ([`cc94134`](https://github.com/skfolio/skfolio/commit/cc94134c30e6e22dde6ed8f9aae0a09ff2a09366))

- Unit tests
  ([`8a16067`](https://github.com/skfolio/skfolio/commit/8a16067e948f34290beb4cc5dae342f457bc1296))

- Unit tests
  ([`bbea014`](https://github.com/skfolio/skfolio/commit/bbea0143d5ac724598c215546529319d987641f3))


## v0.17.0 (2026-04-05)

### Chores

- Add .gitattributes
  ([`223a1b6`](https://github.com/skfolio/skfolio/commit/223a1b606d26520d6666b533b9bb084953e35186))

- Improve docstrings
  ([`66c2af8`](https://github.com/skfolio/skfolio/commit/66c2af8701e7f16eb48d424893c98fa19f1f1b00))

### Continuous Integration

- Bump actions/deploy-pages from 4 to 5 (#226) [skip ci]
  ([`1d09142`](https://github.com/skfolio/skfolio/commit/1d09142a0e302b3ba50f15f4ce4f6879053db349))

- Bump codecov/codecov-action from 5 to 6 (#225) [skip ci]
  ([`f0d51a4`](https://github.com/skfolio/skfolio/commit/f0d51a469be4338c48b51bb10f17df49997eced0))

### Features

- **covariance**: Add regime-adjusted EW covariance estimator (RegimeAdjustedEWCovariance)
  ([`00c3eba`](https://github.com/skfolio/skfolio/commit/00c3ebade2585b8868636dce9ef0e236ded5839f))

- **moments**: Add online (partial_fit) and NaN-aware support to EWMu and EWCovariance
  ([`1298177`](https://github.com/skfolio/skfolio/commit/1298177a4e125007b2022aa57c75811028290de0))

- **variance**: Add empirical, EW, and regime-adjusted estimators (EmpiricalVariance, EWVariance,
  RegimeAdjustedEWVariance)
  ([`abe3f14`](https://github.com/skfolio/skfolio/commit/abe3f1449d8dba52c6267ffe0a574f7d8e30fa82))

### Refactoring

- **prior**: Deprecate FactorModel in favor of TimeSeriesFactorModel
  ([`e98cff9`](https://github.com/skfolio/skfolio/commit/e98cff9715268a8c723b99cbdfc08c0eaf8aba3b))


## v0.16.1 (2026-03-24)

### Bug Fixes

- **BaseOptimization**: Remove abstractmethod on __init__
  ([`84c7945`](https://github.com/skfolio/skfolio/commit/84c7945f151bcebd3ca881dcb0d4d721b846ac7d))

### Continuous Integration

- Bump actions/create-github-app-token from 2 to 3 (#222) [skip ci]
  ([`4fee1bb`](https://github.com/skfolio/skfolio/commit/4fee1bb8762668e527d5d85bf3dba66fd34e095e))


## v0.16.0 (2026-03-22)

### Features

- **MultipleRandomizedCV**: Add get_n_splits in MultipleRandomizedCV
  ([`5638dd3`](https://github.com/skfolio/skfolio/commit/5638dd350366c9e07783f3e9538ecd819d1d2f8c))

- **Population**: Add returns_df in Population
  ([`f538a68`](https://github.com/skfolio/skfolio/commit/f538a68004f3a011b832efe76f87592bfdc22674))


## v0.15.7 (2026-03-14)

### Bug Fixes

- BenchmarkTracker with dict-based weight constraints
  ([`802825a`](https://github.com/skfolio/skfolio/commit/802825a1e07451b1efba8626eaf163a9f9621c83))


## v0.15.6 (2026-03-08)

### Bug Fixes

- Add get_n_splits to CombinatorialPurgedCV ([#218](https://github.com/skfolio/skfolio/pull/218),
  [`1e0ab64`](https://github.com/skfolio/skfolio/commit/1e0ab64619d7e506c550b48f931ec1c51f0ab815))

### Continuous Integration

- Bump actions/upload-artifact from 6 to 7 (#217) [skip ci]
  ([`80fd889`](https://github.com/skfolio/skfolio/commit/80fd889bc8b36d974be7bfe7d5821bea246d02c0))


## v0.15.5 (2026-02-10)

### Bug Fixes

- Add max combinations check to CombinatorialPurgedCV to fail fast on impractical configurations
  ([`db2d976`](https://github.com/skfolio/skfolio/commit/db2d97650d49a58eaa4641adf321158853c036ed))

- **datasets**: Simplify dataset caching to use csv.gz directly
  ([`bb9e3ac`](https://github.com/skfolio/skfolio/commit/bb9e3ac4f6f15300bb701a42d9c185a5b372118b))

- **pandas**: Ensure compatibility with pandas >= 3.0
  ([`3d0c25b`](https://github.com/skfolio/skfolio/commit/3d0c25b642e72f761e808439fd11afb1537ad2a6))

### Documentation

- **readme**: Add resources and links
  ([`895a1d5`](https://github.com/skfolio/skfolio/commit/895a1d5102351481a2c1741c4a6a1978be134c3e))


## v0.15.4 (2026-01-29)

### Bug Fixes

- **typing**: Update cvxpy trace type hint for v1.8+
  ([`7851702`](https://github.com/skfolio/skfolio/commit/785170286bcbc80871b3ea7ad9eea1bd35b47737))

- **typing**: Use cvxpy.Expression instead of trace/Trace type hint
  ([`c50e8de`](https://github.com/skfolio/skfolio/commit/c50e8debf0ac291a7f7445f110431b0e72df3087))


## v0.15.3 (2025-12-19)

### Bug Fixes

- Handles NaNs gracefully in Portfolio
  ([`dd1ec2e`](https://github.com/skfolio/skfolio/commit/dd1ec2e5be4254fc20730b0dc12dab5df7fc025d))

### Continuous Integration

- Bump actions/upload-artifact from 5 to 6 (#208) [skip ci]
  ([`5a9e932`](https://github.com/skfolio/skfolio/commit/5a9e932ca0c34323aa459b8d6aaf47c162a9a3da))

- Bump python-semantic-release/python-semantic-release (#209) [skip ci]
  ([`0fd76b1`](https://github.com/skfolio/skfolio/commit/0fd76b1c96461d1dc15dc60681ec4c6430823272))

### Documentation

- Fix typo in prices_to_returns
  ([`37561c4`](https://github.com/skfolio/skfolio/commit/37561c403a854081d5404fa1183a6532353aba1b))

- Improve documentation about weight drift in Portfolio
  ([`97a34d7`](https://github.com/skfolio/skfolio/commit/97a34d7d04d621154f490ea98fde5ea4b24fdb8f))


## v0.15.2 (2025-12-02)

### Bug Fixes

- EntropyPooling now uses prior values from prior_estimator
  ([#203](https://github.com/skfolio/skfolio/pull/203),
  [`09c9813`](https://github.com/skfolio/skfolio/commit/09c9813c503a18a11146e2f22099db250aa128c6))

### Testing

- Mean views with prior_estimator
  ([`0db273d`](https://github.com/skfolio/skfolio/commit/0db273da361b6996327f10b57ca7682edd135a31))


## v0.15.1 (2025-12-02)

### Bug Fixes

- Add `n_eff` in Empirical Uncertainty Set estimators
  ([#205](https://github.com/skfolio/skfolio/pull/205),
  [`8f5fea5`](https://github.com/skfolio/skfolio/commit/8f5fea5e127398f842d7d12a14c2efbebf8dc8b3))

### Chores

- Typo in pyproject [skip ci]
  ([`382293f`](https://github.com/skfolio/skfolio/commit/382293f5043876c1dc8c00af7ad58944e71d5b6a))

### Continuous Integration

- Bump actions/checkout from 5 to 6 (#201) [skip ci]
  ([`4bbba9c`](https://github.com/skfolio/skfolio/commit/4bbba9cb9e141988de1bc83acf967df33901bcc3))

- Bump python-semantic-release/python-semantic-release (#200) [skip ci]
  ([`7028c49`](https://github.com/skfolio/skfolio/commit/7028c492ac4818eb204d20186d06c395af0ad3f3))


## v0.15.0 (2025-11-20)

### Features

- Add tracking error optimization ([#198](https://github.com/skfolio/skfolio/pull/198),
  [`4a1c207`](https://github.com/skfolio/skfolio/commit/4a1c207fd5e040592be68303b5ff3aa55431b24b))


## v0.14.3 (2025-11-12)

### Bug Fixes

- Maximum diversification with negative expected returns
  ([#197](https://github.com/skfolio/skfolio/pull/197),
  [`c2cc1ad`](https://github.com/skfolio/skfolio/commit/c2cc1adb60c759678ed872c943badeddb1977e21))

### Continuous Integration

- Bump actions/checkout from 3 to 5 (#185) [skip ci]
  ([`ff1f912`](https://github.com/skfolio/skfolio/commit/ff1f9129214266e1bb53673f80cde970a05cc3dc))

- Bump actions/create-github-app-token from 1 to 2 (#192) [skip ci]
  ([`5d673e7`](https://github.com/skfolio/skfolio/commit/5d673e73b73c1da4db2c80a43819a20b0d82b899))

- Bump actions/setup-python from 5 to 6 (#189) [skip ci]
  ([`ed63413`](https://github.com/skfolio/skfolio/commit/ed63413b33fbe94527a4c12bd59c10484520a729))

- Bump actions/upload-artifact from 4 to 5 (#190) [skip ci]
  ([`bf0887c`](https://github.com/skfolio/skfolio/commit/bf0887c8a6822983e644279675f70699166d12d0))

- Bump astral-sh/setup-uv from 4 to 7 (#191) [skip ci]
  ([`699ed58`](https://github.com/skfolio/skfolio/commit/699ed58e7b1cf33e8878a8b5f838e6069f26363a))

- Bump codecov-action from 4 to 5 (#186) [skip ci]
  ([`86082c4`](https://github.com/skfolio/skfolio/commit/86082c4cda3f75b09bb27646bb40f7c54a117ac6))


## v0.14.2 (2025-10-21)

### Bug Fixes

- Adds support for one-factor models using EmpiricalCovariance
  ([#188](https://github.com/skfolio/skfolio/pull/188),
  [`fd38213`](https://github.com/skfolio/skfolio/commit/fd3821378d602b78d94f8d5afbc71f4713bd4fe6))


## v0.14.1 (2025-10-14)

### Bug Fixes

- Improved docstrings
  ([`b28846e`](https://github.com/skfolio/skfolio/commit/b28846eb6a6e572db69987f34fceef461baab9ff))


## v0.14.0 (2025-10-09)

### Chores

- Add unit tests for python 3.13
  ([`365e6bd`](https://github.com/skfolio/skfolio/commit/365e6bd8ced3faf5a9b834277795bf34cac1a1d7))

- Increase security checks in ci/cd
  ([`3f141cf`](https://github.com/skfolio/skfolio/commit/3f141cf81503fd886b2de377de7ccd8bac95072f))

- Ruff format
  ([`dc224c7`](https://github.com/skfolio/skfolio/commit/dc224c7ac8aa5b0d744a5d31593020f2d9bcf538))

### Continuous Integration

- Add minimum deps tests in CI workflow
  ([`02b4878`](https://github.com/skfolio/skfolio/commit/02b48787387f83945e8a004a9afe9b354b21e5db))

- Fix CI workflow
  ([`76fe043`](https://github.com/skfolio/skfolio/commit/76fe043149dff615ea9742d4ffe1bcc4b8902068))

- Fix CI workflow
  ([`f2ad1cc`](https://github.com/skfolio/skfolio/commit/f2ad1cc6a18cc535d4f74f19566326506eb59c21))

### Documentation

- Add Enterprise Support to documentation
  ([`344eba2`](https://github.com/skfolio/skfolio/commit/344eba2cfd352697cf40fa9aa6e3c802fe32cb68))

- Add Failure and Fallback tutorial
  ([`45a6354`](https://github.com/skfolio/skfolio/commit/45a6354df4164c9fa285832451b153c0d65597e8))

- Add Skfolio Labs references
  ([`fe314c8`](https://github.com/skfolio/skfolio/commit/fe314c8d39b2607861ff53b87b52292c5251095e))

- Improve code references
  ([`9eb7294`](https://github.com/skfolio/skfolio/commit/9eb7294c56f9bcf3f8e6634a7ca39487cdf0bf3f))

- Simplify pull_request_template.md
  ([`b892f6d`](https://github.com/skfolio/skfolio/commit/b892f6d12ad7b3e4abced12737c6be69f76e0e07))

### Features

- Add fallback mechanism to the optimization classes
  ([`e45450c`](https://github.com/skfolio/skfolio/commit/e45450cdeefa5621a257aebdcff509a76f137448))

- Add NaN support to risk measures
  ([`f0e8305`](https://github.com/skfolio/skfolio/commit/f0e830535777b763885decfafa852a7f4ffa1a63))

- Cross_val_predict now propagates weights between consecutive folds for sequential CV strategies
  (e.g., WalkForward or MultipleRandomizedCV)
  ([`8c0ee99`](https://github.com/skfolio/skfolio/commit/8c0ee99d4df0a3af08636ed1a7fe8191479ed4ed))

- **portfolio**: Add FailedPortfolio for failed optimizations
  ([`dd79cf7`](https://github.com/skfolio/skfolio/commit/dd79cf7308da116b9f5965fbea5453ee62a3716c))


## v0.13.0 (2025-09-08)

### Bug Fixes

- Rebase compounded cumulative returns to start at 1.0
  ([`1ec1fcd`](https://github.com/skfolio/skfolio/commit/1ec1fcd1c21393e4047e1306055de0f5199b8409))

- Simplified Risk Budgeting formulation
  ([`296bb24`](https://github.com/skfolio/skfolio/commit/296bb2400c9d8cdb36d0276d3361254ff98cb900))

### Documentation

- Add note in the Population summary method about interactive table solutions
  ([`8db914a`](https://github.com/skfolio/skfolio/commit/8db914a9ccf15d7c79ebf6d4165215c478cad4f7))

- Docstring of Risk Budgeting improved
  ([`16385fb`](https://github.com/skfolio/skfolio/commit/16385fbe069f7335dc023e736cef14784ec42f7b))

- Schur complementary doc and example improved
  ([`c9ac1c9`](https://github.com/skfolio/skfolio/commit/c9ac1c976bfc4798b6243dba4283c654901dfce5))

- **WalkForward**: Add guidance on purged_size
  ([`03f594b`](https://github.com/skfolio/skfolio/commit/03f594bdc45f6ddc1a5babc6a19a77925650d4f4))

### Features

- Add plot_drawdowns in Portfolio and Population
  ([`c8fea08`](https://github.com/skfolio/skfolio/commit/c8fea08d3122f93f70b1e41dc5affc466aeefa19))

- **Population**: Added measure box plot
  ([`b504854`](https://github.com/skfolio/skfolio/commit/b504854c9670a6f542f1741a5c5a74508ef5e285))

### Breaking Changes

- Users relying on compounded cumulative returns previously starting at 1000.0 will now see outputs
  starting at 1.0. To restore the old behavior, explicitly pass `base=1000.0` to
  get_cumulative_returns().


## v0.12.0 (2025-09-06)

### Documentation

- Add documentation website redirects
  ([`e32d2d2`](https://github.com/skfolio/skfolio/commit/e32d2d22c7e00b1f6ec1174523a104a64511409c))

- Conf.py improved
  ([`b3e9381`](https://github.com/skfolio/skfolio/commit/b3e9381e0bf0faecf27b5b4b45a0951ffe6bcf21))

- Documentation website improved
  ([`e41e2d9`](https://github.com/skfolio/skfolio/commit/e41e2d967fba6f6b4009753b77a2a5992b41628c))

- Improve documentation website
  ([`31ae042`](https://github.com/skfolio/skfolio/commit/31ae042a35e1215829e7f6cadf5839fc348888bd))

- Sitemap
  ([`64c27ec`](https://github.com/skfolio/skfolio/commit/64c27ecd42917f100e6267b912dcad7573e7ba37))

### Features

- Add Schur Complementary Allocation from Peter Cotton
  ([#165](https://github.com/skfolio/skfolio/pull/165),
  [`aad9131`](https://github.com/skfolio/skfolio/commit/aad9131cefa2df79fdc3e2096606924ca797b446))


## v0.11.0 (2025-07-26)

### Documentation

- Add DOI for citations [skip ci]
  ([`8341e50`](https://github.com/skfolio/skfolio/commit/8341e506c60c771a4188b5f414ba6c04d7ad3bbf))

### Features

- Add Multiple Randomized Cross-Validation from Palomar
  ([#162](https://github.com/skfolio/skfolio/pull/162),
  [`b7a10f1`](https://github.com/skfolio/skfolio/commit/b7a10f180e33775e43827cc27d34315b96908fb1))


## v0.10.2 (2025-07-19)

### Bug Fixes

- Fixed `mu` estimation in EmpiricalPrior when `is_log_normal` is True
  ([`6084170`](https://github.com/skfolio/skfolio/commit/60841707c3efdc0830745b876e67a6087eafe9bc))

### Chores

- Added dockerfile and instructions to run jupyterlab inside docker (#157) [skip ci]
  ([`271055f`](https://github.com/skfolio/skfolio/commit/271055f20b81b49fcdc4ae49a53557b1601fc1b9))

### Documentation

- **readme**: Update citations ([#159](https://github.com/skfolio/skfolio/pull/159),
  [`5fb1335`](https://github.com/skfolio/skfolio/commit/5fb1335d8a4d5ed469250847a79d15a8df04a826))


## v0.10.1 (2025-06-17)

### Bug Fixes

- **measures**: Variance with sample_weight
  ([`e7e60cb`](https://github.com/skfolio/skfolio/commit/e7e60cb1c6e1dcb6afe1317221ee31a4870c0d01))

### Chores

- Typo and formatting
  ([`645e6f7`](https://github.com/skfolio/skfolio/commit/645e6f72843560b847eec4810b7f0bd7efaf500f))

- Typo README
  ([`de7113b`](https://github.com/skfolio/skfolio/commit/de7113b433cc524d4f3be0b8dfb7ca2688b9c1db))

- Typo README [skip ci]
  ([`82f029e`](https://github.com/skfolio/skfolio/commit/82f029e64494228604795a1536bccc5e24d57737))

- Typo README [skip ci]
  ([`4a325f1`](https://github.com/skfolio/skfolio/commit/4a325f13f70b56a33d785f5c0aeec3ad25dd9451))

### Documentation

- Fixing examples
  ([`4cfd09d`](https://github.com/skfolio/skfolio/commit/4cfd09d480b09e171a790b5d5f1636de37ff3257))

### Testing

- Entropy_pooling
  ([`c7fadf3`](https://github.com/skfolio/skfolio/commit/c7fadf3227cc45ea867e7f45f7ef8d511a94b0bb))


## v0.10.0 (2025-06-09)

### Features

- Entropy and Opinion Pooling estimators added
  ([`8842395`](https://github.com/skfolio/skfolio/commit/8842395220d3ccd20b1fb12a1d9b1e7b3479ea5d))

### Breaking Changes

- Prior Estimators attribute `prior_model_` has been renamed to `return_distribution_` and
  `PriorModel` to `ReturnDistribution`


## v0.9.1 (2025-05-15)

### Bug Fixes

- Equation perf improvement
  ([`5ed31d2`](https://github.com/skfolio/skfolio/commit/5ed31d2becbef82a57551b74c9c304324ee17e6d))

- Tracking error with max Ratio
  ([`ac91339`](https://github.com/skfolio/skfolio/commit/ac913397deae794d87582b3ff20d2e1ec2ea85fc))

### Chores

- Better error message for max Ratio failure
  ([`98dc598`](https://github.com/skfolio/skfolio/commit/98dc5985cd92c00bf2b10987cd5bef7b2025f42b))

- SelectNonExpiring message typo
  ([`e8c6701`](https://github.com/skfolio/skfolio/commit/e8c670121d09f4ef91bd34a208698b6317be1461))

### Continuous Integration

- Fix pre-commit ruff-format (#136) [skip ci]
  ([`43c38c1`](https://github.com/skfolio/skfolio/commit/43c38c1a572ff60dcef3cdc95969cbfdecc22dc3))

### Documentation

- Fix duplicate in docstring (#138) [skip ci]
  ([`e238bb7`](https://github.com/skfolio/skfolio/commit/e238bb7008bf9ff551c04856b50c823f1a2a8106))


## v0.9.0 (2025-04-05)

### Features

- Add pre-selection transformer DropZeroVariance
  ([#132](https://github.com/skfolio/skfolio/pull/132),
  [`8cdf103`](https://github.com/skfolio/skfolio/commit/8cdf1030bc82c134ab5796218c6ed804e6f80c7a))


## v0.8.1 (2025-03-21)

### Bug Fixes

- **population**: Surface plot compatible with plotly v6.0
  ([`c053625`](https://github.com/skfolio/skfolio/commit/c053625d641eda1deee480aafd8d10cd0e6b3c56))


## v0.8.0 (2025-03-21)

### Build System

- Change dependencies from cvxpy to cvxpy-base + clarabel (#127) [skip ci]
  ([`cd5d209`](https://github.com/skfolio/skfolio/commit/cd5d209aa6c5af65607d0243a2fc89993c7f81c6))

### Chores

- Ruff pydocstyle (#120) [skip ci]
  ([`3e373de`](https://github.com/skfolio/skfolio/commit/3e373de41bf7e5d7ba9e36f901ac56fb33ebdb5e))

- Spdx licence identifier (#119) [skip ci]
  ([`36ebe3a`](https://github.com/skfolio/skfolio/commit/36ebe3a2abfe03da53e7a7461a413f0baf91f675))

- **readme**: Update features [skip ci]
  ([`09af264`](https://github.com/skfolio/skfolio/commit/09af26457c2a669eae5e19a88d2c8b58df795ccd))

### Features

- Bivariate Copula estimators ([#118](https://github.com/skfolio/skfolio/pull/118),
  [`432a6d5`](https://github.com/skfolio/skfolio/commit/432a6d5548d6d4b0802d82e0d725a1ee3f4b26c2))

- Stress Test and Factor Stress Test ([#118](https://github.com/skfolio/skfolio/pull/118),
  [`432a6d5`](https://github.com/skfolio/skfolio/commit/432a6d5548d6d4b0802d82e0d725a1ee3f4b26c2))

- Synthetic Data estimator ([#118](https://github.com/skfolio/skfolio/pull/118),
  [`432a6d5`](https://github.com/skfolio/skfolio/commit/432a6d5548d6d4b0802d82e0d725a1ee3f4b26c2))

- Synthetic Data with Vine Copula ([#118](https://github.com/skfolio/skfolio/pull/118),
  [`432a6d5`](https://github.com/skfolio/skfolio/commit/432a6d5548d6d4b0802d82e0d725a1ee3f4b26c2))

- Univariates estimators ([#118](https://github.com/skfolio/skfolio/pull/118),
  [`432a6d5`](https://github.com/skfolio/skfolio/commit/432a6d5548d6d4b0802d82e0d725a1ee3f4b26c2))

- Vine Copula estimator ([#118](https://github.com/skfolio/skfolio/pull/118),
  [`432a6d5`](https://github.com/skfolio/skfolio/commit/432a6d5548d6d4b0802d82e0d725a1ee3f4b26c2))

### Breaking Changes

- If you relied on solver "SCS" explicitly, you will need to install it independently or use the
  default "CLARABEL"


## v0.7.0 (2025-01-01)

### Continuous Integration

- Sphinx build with uv ([#110](https://github.com/skfolio/skfolio/pull/110),
  [`b77584e`](https://github.com/skfolio/skfolio/commit/b77584e31125bd57800a54dd70256a499dcf5dcc))

- **workflow**: Use uv ([#107](https://github.com/skfolio/skfolio/pull/107),
  [`3218cfa`](https://github.com/skfolio/skfolio/commit/3218cfa5abf7d6179b02620a0295847f3ce8a79f))

### Features

- Support upcoming scikit-learn 1.7 ([#112](https://github.com/skfolio/skfolio/pull/112),
  [`b15cdb4`](https://github.com/skfolio/skfolio/commit/b15cdb4db16fe7bb83c2bc32017b46e09ee724b0))


## v0.6.0 (2024-11-19)

### Features

- **mean-risk**: Cardinality and Threshold constraints
  ([#102](https://github.com/skfolio/skfolio/pull/102),
  [`a4d8db5`](https://github.com/skfolio/skfolio/commit/a4d8db536b2656806dbf526f9f501badda04bd76))


## v0.5.2 (2024-11-17)

### Bug Fixes

- **datasets**: Use CORS proxy to load remote datasets from browser (JupyterLite)
  ([#101](https://github.com/skfolio/skfolio/pull/101),
  [`c7dc97e`](https://github.com/skfolio/skfolio/commit/c7dc97e63d6c00a576481f841f8aea65cce5e614))

### Chores

- **docs**: Pin `kaleido` version to fix `sphinx-build` (#97) [skip-ci]
  ([`3a7184e`](https://github.com/skfolio/skfolio/commit/3a7184e8e87947235e0d7d9b0b624bc5dae364ab))

### Continuous Integration

- Integrate JupyterLite into documentation site ([#98](https://github.com/skfolio/skfolio/pull/98),
  [`c1d5bc0`](https://github.com/skfolio/skfolio/commit/c1d5bc013ed3a42da42d6e2e7b7ff20416edd1e3))

- Specify `environment` in docs flow ([#99](https://github.com/skfolio/skfolio/pull/99),
  [`1efe4c8`](https://github.com/skfolio/skfolio/commit/1efe4c82e0901a15395686b762bee62950ffb42c))

### Documentation

- Sync dep versions between readme and pyproject (#100) [skip ci]
  ([`dc5671f`](https://github.com/skfolio/skfolio/commit/dc5671f3e9da3595136d5e3f88028fa8e19b0d3a))


## v0.5.1 (2024-11-09)

### Bug Fixes

- **risk-budgeting**: Weight constraint validation fixed
  ([`21da12a`](https://github.com/skfolio/skfolio/commit/21da12a27b9578f9de5cb1b74c4fd009b40efffa))

### Chores

- **numpy**: Fix numpy DeprecationWarning
  ([`28a36ed`](https://github.com/skfolio/skfolio/commit/28a36edaec6cbca2ae15b67dcb0ad8dff7255bf1))


## v0.5.0 (2024-11-04)

### Features

- **pre-selection**: Handle Incomplete Datasets, Inception, Expiry, Default, Delistings
  ([#94](https://github.com/skfolio/skfolio/pull/94),
  [`2bebfd2`](https://github.com/skfolio/skfolio/commit/2bebfd2d65d13501ae01741b47f3aaa322e12981))


## v0.4.3 (2024-10-24)

### Bug Fixes

- **herc**: HERC weight constraint bug fixed using minimum relative weight deviation
  ([`5d24fc3`](https://github.com/skfolio/skfolio/commit/5d24fc386dca61df39f98487c51dc914ee933a1c))

### Chores

- **readme**: Typo [skip ci]
  ([`d53a15b`](https://github.com/skfolio/skfolio/commit/d53a15bfbcd9da7f435a96876a259a2cd812315f))


## v0.4.2 (2024-10-06)

### Bug Fixes

- **optimization**: `linear_constraints` now supports equality strings
  ([`7a738f3`](https://github.com/skfolio/skfolio/commit/7a738f32eb8282b53f408cee59438ececed362ea))


## v0.4.1 (2024-09-22)

### Bug Fixes

- **combinatorial**: Added default type for compatibility with numpy 2.0.0
  ([`ae45598`](https://github.com/skfolio/skfolio/commit/ae455985f95130a152890f5601a68257852c6ca8))

- **pyproject**: Removing "numpy<2.0.0" following last cvxpy release
  ([`ba08925`](https://github.com/skfolio/skfolio/commit/ba089257801b4a3508b4fc52ab7163f700d9e8d6))

### Chores

- **examples**: Example for custom pre-selection with Volumes
  ([`3448281`](https://github.com/skfolio/skfolio/commit/34482817e591b4aa9406dde6510c3188b72c899c))


## v0.4.0 (2024-09-15)

### Bug Fixes

- **docs**: Typo in datasets docstring [skip ci]
  ([`c796630`](https://github.com/skfolio/skfolio/commit/c79663009291fb58bfd3101c8f2952608ad1ffb5))

- **docs**: Typo in measure docstring [skip ci]
  ([`17fd458`](https://github.com/skfolio/skfolio/commit/17fd458314fd72111dfe3d20d4d0344c0dcc9f54))

- **population**: Names and tags arguments
  ([`a020c28`](https://github.com/skfolio/skfolio/commit/a020c281745570ee814093108f37b5c790932d8c))

- **portfolio**: `plot_contribution` changed to stacked bars
  ([`cd6e7c0`](https://github.com/skfolio/skfolio/commit/cd6e7c0fddf5df807b80a271a4bdb992fa97563b))

- **pyproject**: Keeping major to 0.x.x after a breaking change
  ([`9b29180`](https://github.com/skfolio/skfolio/commit/9b291803c136331ce90e03aafac30781f850880a))

- **test**: Portfolio contribution
  ([`56947eb`](https://github.com/skfolio/skfolio/commit/56947ebcbe06c2a3fc2ad0f03ef2382196945681))

- **workflow**: Python-semantic-release version updated
  ([`f397565`](https://github.com/skfolio/skfolio/commit/f397565e56295438828872ab58943ab56ee11269))

### Chores

- **docstring**: WalkForward example
  ([`0745b6a`](https://github.com/skfolio/skfolio/commit/0745b6aef6fd220281a1d17e2d4515740ad3fe1d))

### Features

- **population**: `contribution` and `plot_contribution` added
  ([`128f619`](https://github.com/skfolio/skfolio/commit/128f619960494f827fb2205e1a9d5cca383092d8))

- **population**: Rolling_measure and plot_rolling_measure added at the Population level
  ([`c75d34d`](https://github.com/skfolio/skfolio/commit/c75d34db68a2f3e849b0d9d2a3e97c40025cde98))

- **portfolio**: `contribution` and `plot_contribution` implemented for MultiPeriodPortfolio
  ([`0996fac`](https://github.com/skfolio/skfolio/commit/0996face5d75674ea2aed3fb5c274740b91b800e))

- **portfolio**: `weights_per_observation` added
  ([`d0d3b33`](https://github.com/skfolio/skfolio/commit/d0d3b33f5ca3e27e45ac525c788ead119c83e92c))

- **walk-forward**: Possibility to split based on datetime periods and offsets
  ([`af386be`](https://github.com/skfolio/skfolio/commit/af386be84c9bc510d0ea2d9dc6d4cb9196cef5a4))

- **walk-forward**: Possibility to split based on datetime periods and pandas offsets
  ([`f782275`](https://github.com/skfolio/skfolio/commit/f782275a9179d9c16e607033efe45090c37cbe07))


## v0.3.1 (2024-07-01)

### Bug Fixes

- **covariance**: ImpliedCovariance param renamed to prior_covariance_estimator
  ([`1677bb4`](https://github.com/skfolio/skfolio/commit/1677bb41b135c6a0143e670ea5ba65ef04c76db1))


## v0.3.0 (2024-06-30)

### Bug Fixes

- **covariance**: Argument `nearest` is now defaulted to True for improved usability and emits a
  warning when cov is not PD ([#64](https://github.com/skfolio/skfolio/pull/64),
  [`a682f47`](https://github.com/skfolio/skfolio/commit/a682f47afb6367958acfc42875d4a2cad661ad05))

- **doc**: Typo in example filename (#62) [skip ci]
  ([`139ba0e`](https://github.com/skfolio/skfolio/commit/139ba0ea7f3cb0ffd9972bae1a7af0007500484e))

- **population**: Ensure plot index is sorted ([#64](https://github.com/skfolio/skfolio/pull/64),
  [`a682f47`](https://github.com/skfolio/skfolio/commit/a682f47afb6367958acfc42875d4a2cad661ad05))

### Features

- **covariance**: ImpliedCovariance added ([#64](https://github.com/skfolio/skfolio/pull/64),
  [`a682f47`](https://github.com/skfolio/skfolio/commit/a682f47afb6367958acfc42875d4a2cad661ad05))

- **datasets**: SP500 3 months ATM implied volatility dataset added
  ([#64](https://github.com/skfolio/skfolio/pull/64),
  [`a682f47`](https://github.com/skfolio/skfolio/commit/a682f47afb6367958acfc42875d4a2cad661ad05))

- **metadata-routing**: Metadata routing ([#64](https://github.com/skfolio/skfolio/pull/64),
  [`a682f47`](https://github.com/skfolio/skfolio/commit/a682f47afb6367958acfc42875d4a2cad661ad05))


## v0.2.3 (2024-06-20)

### Bug Fixes

- **pyproject**: Impose numpy < 2.0.0 until CVXPY is fully compatible with the new release
  ([`e02c172`](https://github.com/skfolio/skfolio/commit/e02c172877a5ce3899bf526522601c1b1a9588dd))


## v0.2.2 (2024-06-04)

### Bug Fixes

- **cluster**: Compute_optimal_n_cluster now handles less then 8 assets
  ([`d9ed4a4`](https://github.com/skfolio/skfolio/commit/d9ed4a4a4cbb096a45d8f09089c9d72c481edd45))

- **plotly**: Remove fix of create_dendrogram following new plotly release
  ([`2e712c8`](https://github.com/skfolio/skfolio/commit/2e712c8647f943b33958ec2fb3accbdb97542102))

- **plotly**: Remove global pandas plot backend setting
  ([`4ff07c3`](https://github.com/skfolio/skfolio/commit/4ff07c37692c83388cd1a45175e661a51726ddbd))

- **pre-processing**: Drop_interceptions_nan param added to prices_to_returns
  ([#57](https://github.com/skfolio/skfolio/pull/57),
  [`ba74842`](https://github.com/skfolio/skfolio/commit/ba7484263ad315d38601250d2e8d1dae5269149c))

- **pre-processing**: Param drop_inceptions_nan
  ([`75d8013`](https://github.com/skfolio/skfolio/commit/75d8013952e86fb96666e1f6f7436edceb05cde6))


## v0.2.1 (2024-05-22)

### Bug Fixes

- **prior**: Factor model updated with y propagation
  ([`8cbc5cc`](https://github.com/skfolio/skfolio/commit/8cbc5ccee48e60750b8d529168e3cf5c46b40043))

- **prior**: In BlackLitterman, y propagation was missing for embedded models
  ([`cc4adc1`](https://github.com/skfolio/skfolio/commit/cc4adc14a9a76ee1e449839ef490674c19ba34a6))

- **test**: Nested Black & Litterman tests added
  ([`1188b2c`](https://github.com/skfolio/skfolio/commit/1188b2ca98b905931d7e70230afe9e5df6c64593))

### Chores

- **doc**: Example link typo in user guide
  ([`d722814`](https://github.com/skfolio/skfolio/commit/d7228142fc993a3821c8db479519222570016f98))

- **docs**: Black & Litterman Factor model updated with deeply nested models
  ([`a5b038b`](https://github.com/skfolio/skfolio/commit/a5b038b58241740a658bf6c2213c5c5c5beca87c))

- **readme**: BL views updated
  ([`b849570`](https://github.com/skfolio/skfolio/commit/b849570a84ad9b755289087dd6d4022a7e79661e))


## v0.2.0 (2024-05-19)

### Bug Fixes

- **optimization**: Risk_free_rate added to the default portfolio_params
  ([#50](https://github.com/skfolio/skfolio/pull/50),
  [`92666a8`](https://github.com/skfolio/skfolio/commit/92666a80cba07e74d68358563a8c4ce4bc86bd38))

- **population**: Relax Population items to all inheritance of BasePortfolio
  ([#50](https://github.com/skfolio/skfolio/pull/50),
  [`92666a8`](https://github.com/skfolio/skfolio/commit/92666a80cba07e74d68358563a8c4ce4bc86bd38))

### Chores

- **readme**: Fix badge links and underscores in README.rst file [skip ci]
  ([`e4c4b65`](https://github.com/skfolio/skfolio/commit/e4c4b65470f38f416a496e6a445de3ee5da4e886))

- **readme**: Fix badge links and underscores in README.rst file [skip ci]
  ([`1a17e2a`](https://github.com/skfolio/skfolio/commit/1a17e2a773ec2a9f9ff9ec1a4f6bf87012a3c710))

- **typing**: Replaced `any` by the `Any` type ([#47](https://github.com/skfolio/skfolio/pull/47),
  [`d82c144`](https://github.com/skfolio/skfolio/commit/d82c14490289335cb178d1bf6727d1441b18e136))

### Features

- **combinatorial**: Optimal_folds_number function for CombinatorialPurgedCV
  ([#53](https://github.com/skfolio/skfolio/pull/53),
  [`ae83967`](https://github.com/skfolio/skfolio/commit/ae83967580c2ba1b1e25ba767659a75c5a97971e))


## v0.1.3 (2024-03-13)

### Bug Fixes

- **datasets**: Move datasets to another repo (skfolio-datasets)
  ([`4b31e7d`](https://github.com/skfolio/skfolio/commit/4b31e7d5186052f63ab3f8df3bfcfdecde85edc6))

- **ruff**: File reformatting to pass ruff format
  ([`f7a34b5`](https://github.com/skfolio/skfolio/commit/f7a34b536fe2769f7281cd0c14d7899d45972b92))

### Chores

- **docs**: Bibtex entry corrected [skip ci]
  ([`1b0c881`](https://github.com/skfolio/skfolio/commit/1b0c8811fe1d7f2dc6aef31bf82e9694a59617b5))

- **docs**: Contributing.md ruff cmd [skip ci]
  ([`4982e27`](https://github.com/skfolio/skfolio/commit/4982e274c9bce4813e3ef1e2f10dd6a299876324))

- **formatter**: Dropped black formatter. ([#37](https://github.com/skfolio/skfolio/pull/37),
  [`3ee55f4`](https://github.com/skfolio/skfolio/commit/3ee55f4829c16402136844ecbe01ac26425ce3e5))

- **formatter**: Replaced black formatter by ruff formatter
  ([#37](https://github.com/skfolio/skfolio/pull/37),
  [`3ee55f4`](https://github.com/skfolio/skfolio/commit/3ee55f4829c16402136844ecbe01ac26425ce3e5))

- **ruff**: Fix line_length discrepancy. ([#37](https://github.com/skfolio/skfolio/pull/37),
  [`3ee55f4`](https://github.com/skfolio/skfolio/commit/3ee55f4829c16402136844ecbe01ac26425ce3e5))

- **ruff**: Ignored redudant rules. ([#37](https://github.com/skfolio/skfolio/pull/37),
  [`3ee55f4`](https://github.com/skfolio/skfolio/commit/3ee55f4829c16402136844ecbe01ac26425ce3e5))

- **ruff**: Removed duplicated rule. ([#37](https://github.com/skfolio/skfolio/pull/37),
  [`3ee55f4`](https://github.com/skfolio/skfolio/commit/3ee55f4829c16402136844ecbe01ac26425ce3e5))

- **ruff**: Updated ruff's pyproject conf
  ([`4e68a8b`](https://github.com/skfolio/skfolio/commit/4e68a8b21f631dd392a0231a2343ebbac0dc03ea))

### Code Style

- One shot ruff formatting. ([#37](https://github.com/skfolio/skfolio/pull/37),
  [`3ee55f4`](https://github.com/skfolio/skfolio/commit/3ee55f4829c16402136844ecbe01ac26425ce3e5))

### Continuous Integration

- **tests**: Added a format step to the tests job.
  ([#37](https://github.com/skfolio/skfolio/pull/37),
  [`3ee55f4`](https://github.com/skfolio/skfolio/commit/3ee55f4829c16402136844ecbe01ac26425ce3e5))

### Refactoring

- **src**: Fix UP032. ([#37](https://github.com/skfolio/skfolio/pull/37),
  [`3ee55f4`](https://github.com/skfolio/skfolio/commit/3ee55f4829c16402136844ecbe01ac26425ce3e5))


## v0.1.2 (2024-02-05)

### Bug Fixes

- **portfolio**: Add effective number of assets
  ([`f4dbc5f`](https://github.com/skfolio/skfolio/commit/f4dbc5f6c904a14d6244a7c389cadfc28ba344ea))

### Chores

- **docs**: CONTRIBUTING.md improved [skip ci]
  ([`8955fed`](https://github.com/skfolio/skfolio/commit/8955fedc2a293066922c5e40029080a59d852e81))

- **docs**: Pip install in contributing.md [skip ci]
  ([`51a91b6`](https://github.com/skfolio/skfolio/commit/51a91b6a70bf28f4207e0333e44ad48f36a822a0))

- **docs**: Removed all-contributors [skip ci]
  ([`3b4dcf1`](https://github.com/skfolio/skfolio/commit/3b4dcf10f54b0d6df49b85d5b3931457a724a395))

- **docs**: Typo in example [skip ci]
  ([`6e7cb36`](https://github.com/skfolio/skfolio/commit/6e7cb3613323eefef68865c62db3e5d19c70fc4e))


## v0.1.1 (2024-01-28)

### Bug Fixes

- **optimization**: Left_inequality and right_inequality converted to numpy array
  ([`18073ad`](https://github.com/skfolio/skfolio/commit/18073adeba18a455a4cdd163c84e9833432dc890))


## v0.1.0 (2024-01-25)

### Bug Fixes

- **portfolio**: Len method removed and number of assets fixed
  ([`16205d5`](https://github.com/skfolio/skfolio/commit/16205d5f96eeff841a517c0f1716ae1603090f5f))

### Chores

- **docs**: Readme badge and logo improved [skip ci]
  ([`9c5238d`](https://github.com/skfolio/skfolio/commit/9c5238da0d0895be979a746ec3ea571e84cda0dd))

### Features

- **measure**: Effective number of assets [skip ci]
  ([`62692ef`](https://github.com/skfolio/skfolio/commit/62692ef757d90f52b984e859ec69a38c0151d9d7))


## v0.0.11 (2024-01-22)

### Bug Fixes

- **docs**: Ignore FutureWarnings in sphinx gallery examples
  ([`46db794`](https://github.com/skfolio/skfolio/commit/46db794937b29bb2ae59cb0a0d26281fabcbe7b7))

- **measure**: Default trading days in a year amended from 255 to 252
  ([`796ed1e`](https://github.com/skfolio/skfolio/commit/796ed1edb2717cac2473bc9c6bb28a2aeff6234a))

- **plotly**: Dendrogram fixes added
  ([`374154e`](https://github.com/skfolio/skfolio/commit/374154e156b199c8328e2aa06ab5d9685a0c6472))

- **plotly**: Fix ruff in _dendrogram.py
  ([`0c2c59e`](https://github.com/skfolio/skfolio/commit/0c2c59eeb70ad1709f64579933959f3166a3de28))

- **returns**: Pandas concat replaced by join
  ([`35f87b0`](https://github.com/skfolio/skfolio/commit/35f87b05ad2479925fe2123d6fdd9c83666e47c0))

- **typos**: Fix typos
  ([`e1e202a`](https://github.com/skfolio/skfolio/commit/e1e202a7d16210dd0d8c333887e1ef3e8499fa6e))

### Chores

- **docs**: Add author
  ([`c2fdd82`](https://github.com/skfolio/skfolio/commit/c2fdd820f47ba62cf79a9581241ebb682180cc89))

- **docs**: Add derivative work credits
  ([`c2fdd82`](https://github.com/skfolio/skfolio/commit/c2fdd820f47ba62cf79a9581241ebb682180cc89))

- **docs**: Add more derivative work credits
  ([`c2fdd82`](https://github.com/skfolio/skfolio/commit/c2fdd820f47ba62cf79a9581241ebb682180cc89))

- **docs**: Codecov badge [skip ci]
  ([`b369e87`](https://github.com/skfolio/skfolio/commit/b369e8770265762abdfa7a5798ce4b2c40e308d8))

- **docs**: Fix typo [skip ci]
  ([`e1e202a`](https://github.com/skfolio/skfolio/commit/e1e202a7d16210dd0d8c333887e1ef3e8499fa6e))

- **readme**: Download badge [skip ci]
  ([`6dc7957`](https://github.com/skfolio/skfolio/commit/6dc79570a1583e6d148260a2efa7ae7ea95c4b1b))

### Documentation

- Add @rriski as a contributor
  ([`e1e202a`](https://github.com/skfolio/skfolio/commit/e1e202a7d16210dd0d8c333887e1ef3e8499fa6e))


## v0.0.10 (2024-01-17)

### Bug Fixes

- Detoning typo ([#10](https://github.com/skfolio/skfolio/pull/10),
  [`60cd559`](https://github.com/skfolio/skfolio/commit/60cd559594ef92ee053f6a432cbf280f60bddde2))

- **population**: Lint noqa A003 not used [skip ci]
  ([`651a729`](https://github.com/skfolio/skfolio/commit/651a72947dbd2d22cedd0199131e4bd88eda92f5))

### Chores

- **docs**: Contributing [skip ci]
  ([`a1e79d3`](https://github.com/skfolio/skfolio/commit/a1e79d3cf732467dcd078cd414b466b889a2b1c5))

- **readme**: Ruff and website badges [skip ci]
  ([`73e7c0e`](https://github.com/skfolio/skfolio/commit/73e7c0ec1365c85d25cb6977dda75cadb78d4fb9))


## v0.0.9 (2024-01-04)

### Bug Fixes

- **docs**: Jupyterlite replaced by binder
  ([`5b4cde3`](https://github.com/skfolio/skfolio/commit/5b4cde35c2c0e6b786582e2e944cda026ef497d4))

### Chores

- **docs**: Readme updated [skip ci]
  ([`6272852`](https://github.com/skfolio/skfolio/commit/6272852ac77ad7fdfd3069e80f0bd2cd2de79c6b))


## v0.0.8 (2024-01-03)

### Bug Fixes

- **docs**: Docs and examples updated
  ([`0f5575e`](https://github.com/skfolio/skfolio/commit/0f5575e34dd0118e7e5f87fdaf10842a1cb803c5))

- **tests**: Random weights removed [skip ci]
  ([`6fa39fd`](https://github.com/skfolio/skfolio/commit/6fa39fdaefa4e1bc00def519420565fe758137a4))

- **tests**: Test_portfolio_clear_cache fixed
  ([`f72562b`](https://github.com/skfolio/skfolio/commit/f72562b7cffb9bdc8f111e40815ff47b10a2e905))

### Chores

- **docs**: Code of conduct and security [skip ci]
  ([`06a5755`](https://github.com/skfolio/skfolio/commit/06a57558f43dbfcc69aa14d6643bf3137ed0a427))

- **docs**: Contributing updated with branch name convention [skip ci]
  ([`b678be7`](https://github.com/skfolio/skfolio/commit/b678be7101859d7a1d0983bc0eb7fbf40bb5bc7c))

- **docs**: Improved the docs [skip ci]
  ([`d05b661`](https://github.com/skfolio/skfolio/commit/d05b661d22890fedf557469a639105115caf1898))

- **docs**: Readme update [skip ci]
  ([`23f5d5c`](https://github.com/skfolio/skfolio/commit/23f5d5cbc00609854f87cda4166e6c6848fad675))

- **docs**: Readme update [skip ci]
  ([`de262c5`](https://github.com/skfolio/skfolio/commit/de262c562f651b49089cb576b5346a1f3a5ae13f))

- **docs**: Readme update [skip ci]
  ([`9623272`](https://github.com/skfolio/skfolio/commit/9623272632e339dd0c377720569bffe478601606))

- **docs**: Readme update [skip ci]
  ([`40d8bef`](https://github.com/skfolio/skfolio/commit/40d8bef7dd6d7e1e1a5a2e72fe12b72e3b20885c))

- **docs**: Typo in docs [skip ci]
  ([`e8aed66`](https://github.com/skfolio/skfolio/commit/e8aed66849d952df89d48b03cac992f351181083))


## v0.0.7 (2023-12-27)

### Bug Fixes

- **docs**: New domain and SEO
  ([`912b086`](https://github.com/skfolio/skfolio/commit/912b08624406a1e73141fabfae9dd2f62a74c1b3))

- **workflow**: [skip ci] workflow dependency success condition
  ([`c5a115e`](https://github.com/skfolio/skfolio/commit/c5a115e02e4fb3d7aa26243c49f5d104667ce194))

### Chores

- **docs**: Sitemap [skip ci]
  ([`011e043`](https://github.com/skfolio/skfolio/commit/011e043b56859034388706e6253ca4e224503f76))


## v0.0.6 (2023-12-21)

### Bug Fixes

- **docs**: Doc theme and sidebar
  ([`2402e3a`](https://github.com/skfolio/skfolio/commit/2402e3afc79931fceec05ab20a9eb0bac83f584e))


## v0.0.5 (2023-12-20)

### Bug Fixes

- **docs**: Docs seo
  ([`136c7d1`](https://github.com/skfolio/skfolio/commit/136c7d1aadad698aef0137a7379e188bfae26dab))


## v0.0.4 (2023-12-20)

### Bug Fixes

- **docs**: Readme
  ([`1f5e86f`](https://github.com/skfolio/skfolio/commit/1f5e86f4b7d84aff58891633d8d879810b9093f8))

- **workflow**: Test semantic release app
  ([`73612c2`](https://github.com/skfolio/skfolio/commit/73612c2c63226b6c47a6800e05f41d7b2e5ce139))

- **workflow**: Test semantic release app
  ([`4b5cf7f`](https://github.com/skfolio/skfolio/commit/4b5cf7fd66626a23148c6823bc840bfe376b1ebb))

- **workflow**: Test semantic release app
  ([`fb2fd87`](https://github.com/skfolio/skfolio/commit/fb2fd87ccdf44c8adaca47ed2774a1154ede9e68))

- **workflow**: Test semantic release app
  ([`cc5a2a9`](https://github.com/skfolio/skfolio/commit/cc5a2a9e56003756eb06218bb216addf01d01fc5))

- **workflow**: Test semantic release app
  ([`709283f`](https://github.com/skfolio/skfolio/commit/709283f7ae64f5fab12a881704d51cfa0d4bf7c2))


## v0.0.3 (2023-12-18)

### Bug Fixes

- **workflow**: Release to pypi
  ([`4e7f722`](https://github.com/skfolio/skfolio/commit/4e7f722f0736de0a9e747979a1a86975da90d943))

### Chores

- **docs**: Docs workflow run
  ([`49d6069`](https://github.com/skfolio/skfolio/commit/49d60690f418d357b482507af7df4304ecf49386))

- **workflow**: Template, contributors
  ([`c6fe6ed`](https://github.com/skfolio/skfolio/commit/c6fe6ed4ed2672cdcccfcab3f4f11f7ef37d5baf))


## v0.0.2 (2023-12-18)

### Bug Fixes

- **convex**: Solver changed from ECOS/ECS to CLARABEL
  ([`1f05907`](https://github.com/skfolio/skfolio/commit/1f05907a25ef1e880eecd75d921cf7e382d7f6f6))

- **dataset**: Download dataset from github
  ([`2f0ff50`](https://github.com/skfolio/skfolio/commit/2f0ff50f8715dd3598ca4ccb7138659957967aeb))

- **tests**: Linux dataset path PermissionError
  ([`82017fe`](https://github.com/skfolio/skfolio/commit/82017fef7d6e7b17f5f2316ea7567d6483441f41))

### Chores

- **docs**: Docs workflow
  ([`edfeb41`](https://github.com/skfolio/skfolio/commit/edfeb417adb2d09ec38ed123a4f2fc5eaf2cbfcc))

- **docs**: Docs workflow on push
  ([`6d9eb09`](https://github.com/skfolio/skfolio/commit/6d9eb0981a77e2863df477fc1011185a87655ad6))

- **workflow**: Coverage report
  ([`f52ee96`](https://github.com/skfolio/skfolio/commit/f52ee968a5c67bfbd366945fe2376b8e9c6a7c5b))


## v0.0.1 (2023-12-17)

- Initial Release
