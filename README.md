# Cleanhatch

The Cleanroom Airlock for AI & Time-Series Data Pipelines.

Early development of tools for AI and temporal data pipelines.

## Status

Version **0.0.1a0** is a pre-alpha package scaffold. It provides an importable
version and a command-line version query only. Data validation, temporal
queries, causal checks, and pipeline integrations are **not implemented**.
Do not use this release to validate data or make production decisions.

The source repository is currently private. Publishing on PyPI makes the
wheel, source distribution, metadata, and this README publicly downloadable,
regardless of the repository's visibility.

## Installation and version query

After publication on PyPI, with Python 3.10 or later:

```console
python -m pip install cleanhatch==0.0.1a0
cleanhatch --version
python -m cleanhatch --version
```

```python
from cleanhatch import __version__
```

Unsupported commands, including `cleanhatch validate`, fail rather than
reporting a successful validation.

## Local development

Run the source tests without building release artifacts:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Local Git identity and the dedicated SSH key are configured per repository,
not globally. This repository starts with new history and does not import
unrelated private documentation. Never commit credentials or private keys.

## Publishing

Release distributions are built only on GitHub Actions, not on a developer
machine. The `Build and publish` workflow tests Python 3.10 through 3.14,
builds the wheel and source distribution, checks their metadata, and tests
installation from both artifacts outside the source checkout.

Before the first release:

1. Enable 2FA on the PyPI account and keep its account email reachable.
   A GitHub `users.noreply.github.com` address is not a support inbox.
2. At <https://pypi.org/manage/account/publishing/>, add a pending publisher:

   | Field | Value |
   | --- | --- |
   | PyPI project | `cleanhatch` |
   | GitHub owner | `cleanhatch` |
   | GitHub repository | `cleanhatch` |
   | Workflow filename | `publish.yml` |
   | GitHub environment | `pypi` |

3. Configure the GitHub Actions environment `pypi` to match the publisher.
4. Run `Build and publish` manually from `main`. Pushes and pull requests
   test and build only; they do not publish. No PyPI API token is needed.
5. Verify <https://pypi.org/project/cleanhatch/0.0.1a0/> and install the
   published release in a fresh environment.

Do not rerun a successful publication of the same version: PyPI does not
permit replacing uploaded files. For the next release, update the version
in `src/cleanhatch/__init__.py` and the corresponding tests and documentation.

A pending publisher does not reserve a name. PyPI's
[name retention policy](https://docs.pypi.org/project-management/name-retention/)
allows removal of name-squatting projects with no functionality. This
version-query scaffold does not guarantee policy compliance, name retention,
or permanent ownership. Develop real functionality rather than relying on
an empty release as a name reservation.

## License

Copyright 2026 Uncio Lintel. Licensed under Apache-2.0; see [LICENSE](LICENSE).
