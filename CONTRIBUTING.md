# Contributing to pyprf

## Development setup

Clone the repository and install `pyprf` in editable mode, with the test dependencies. This also compiles the cython code:

```bash
git clone https://github.com/ingo-m/pyprf.git
cd pyprf
pip install -e ".[test]"
```

After changing the cython code (`pyprf/analysis/*.pyx`), run `pip install -e ".[test]"` again to recompile it.

Run the tests (with coverage):

```bash
pytest --cov=pyprf
```

The tests of the analysis compare the results of the pRF analysis on example data with reference results (`pyprf/analysis/testing*/exmpl_data_results_*.nii.gz`). The tests of the stimulus presentation (`tests/`) only need `numpy` and `pytest`, not PsychoPy. Changes to the stimulus presentation GUI have to be tested manually in PsychoPy.

## Workflow

* Development happens on the `devel` branch. For contributions, create a new branch from `devel` (or a fork), and open a pull request into `devel`.
* The tests run automatically on GitHub Actions for every push and pull request (on Linux, macOS, and Windows, with all supported Python versions).
* `devel` is merged into `main` for a release (see below).

## Making a release

Releases are made automatically by GitHub Actions (`.github/workflows/publish.yml`) when `devel` is merged into `main`, if the version number is new:

1. On `devel`, update the version number in `pyproject.toml` (e.g. `version = "3.0.1"`).
2. Merge `devel` into `main`, and push `main`.

The workflow then:

* builds wheels for Linux, macOS, and Windows, and runs the tests on each wheel,
* publishes the wheels and the source distribution to [PyPI](https://pypi.org/project/pyprf/),
* creates a GitHub release with tag `v<version>`, with a zip file of the stimulus presentation folder attached,
* the GitHub release triggers [Zenodo](https://doi.org/10.5281/zenodo.835161) to archive the new version (with a new DOI).

If the version number in `pyproject.toml` already has a tag (i.e. has been released before), nothing is released. So changes that do not need a release (e.g. documentation) can be merged into `main` without changing the version number.

### One-time setup

The release workflow needs the following settings (only once):

* PyPI: The project uses [trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is needed. On pypi.org, in the settings of the `pyprf` project, under *Publishing*, a GitHub publisher is configured with owner `ingo-m`, repository `pyprf`, workflow `publish.yml`, and environment `pypi`.
* Codecov: The repository secret `CODECOV_TOKEN` (GitHub repository settings, *Secrets and variables*, *Actions*) contains the upload token from [codecov.io](https://codecov.io/gh/ingo-m/pyprf).
* Zenodo: The GitHub integration for this repository is enabled on zenodo.org.
