# Third-party notices

Original work in this repository is (c) Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0 (see `LICENSE`). The components below keep their own licenses.

## Data

### California Housing

- Not stored in this repository. `python -m mlserving train` downloads it at run time through `sklearn.datasets.fetch_california_housing`, which fetches https://ndownloader.figshare.com/files/5976036 and verifies the archive against the SHA-256 value pinned inside scikit-learn (`aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`). The training script also records a SHA-256 of the loaded arrays in each model's `metadata.json` (`dataset.content_sha256`).
- Upstream source: StatLib, https://lib.stat.cmu.edu/datasets/houses.zip. Reference: Pace, R. Kelley and Barry, Ronald (1997). Sparse Spatial Autoregressions. *Statistics and Probability Letters*, 33, 291 to 297.
- Origin: 1990 U.S. Census, one row per block group, 20,640 rows, 8 numeric features, target = median house value in units of USD 100,000 (capped at 5.00001 in the source).
- **License status: none stated.** The scikit-learn dataset description (`sklearn/datasets/descr/california_housing.rst`, scikit-learn 1.9.1, read 2026-10-15) names StatLib as the source and Pace and Barry as the reference and gives no license terms for the data. The records are U.S. Census statistics, but the compilation carries no explicit terms that were found. For that reason the repository commits no rows of the dataset: the committed model, `metadata.json` (per-feature minimum, maximum, mean and standard deviation) and `reference.json` (a 1,001-point quantile sketch per feature, not observations) are aggregate statistics derived from it. The drift demo uses held-out rows only in memory and commits only aggregate counts. The owner should confirm this reading before the repository is made public.
- Changes made: none to the values; the training script splits it 80/20 with a seed.

## Software (installed from PyPI, not redistributed in this repository)

| Package | Version | License |
|---|---|---|
| fastapi | 0.143.2 | MIT |
| starlette | 1.8.0 | BSD-3-Clause |
| pydantic | 2.14.1 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
| numpy | 2.4.6 | BSD-3-Clause (bundled components: 0BSD, MIT, Zlib, CC0-1.0) |
| scipy | 1.17.1 | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| joblib | 1.6.0 | BSD-3-Clause |
| prometheus-client | 0.26.0 | Apache-2.0 and BSD-2-Clause |
| mlflow-skinny | 3.17.1 | Apache-2.0 |
| pytest, pytest-cov | 9.1.1, 7.1.0 | MIT |
| httpx | 0.28.1 | BSD-3-Clause |
| ruff | 0.17.1 | MIT |
| mypy | 2.4.0 | MIT |
| pip-audit | 2.10.1 | Apache-2.0 |
| pre-commit | 4.7.0 | MIT |
| aiohttp | 3.14.5 | Apache-2.0 and MIT |
| matplotlib | 3.11.2 | Matplotlib license (PSF-based) |

Licenses were read from each installed package's metadata on 2026-10-15.

## Container base image and CI tools (referenced, not redistributed)

- `python:3.11.17-slim-bookworm` (Docker Official Image; Python is under the PSF License, the Debian layer under the licenses of its packages), pinned by digest in the `Dockerfile`.
- GitHub Actions used in CI: `actions/checkout`, `actions/setup-python` (MIT) and `hadolint/hadolint-action` (MIT; the hadolint tool it runs is GPL-3.0 and is executed in CI only, never shipped).

No pretrained models and no third-party source code are included.
