Contribution guide
==================

Development workflow:

1. Create a branch from the current development branch.
2. Use the dev container or install the editable checkout locally:

   .. code-block:: bash

      python -m pip install -e ".[all,dev,docs,notebooks]"

3. Add tests for public APIs, importer output, storage contracts, or numerical
   conventions. Use synthetic inputs, not local scanner data or private files.
4. Update the affected docs page when units, shapes, model assumptions, dependencies,
   or validation status change.

Run these checks from the repository root before opening a pull request:

.. code-block:: bash

   python -m ruff check mrax tests
   python -m isort --check-only mrax tests
   python -m pyright
   python -m coverage run -m pytest
   python -m coverage report
   python -m sphinx -W --keep-going -E -b html docs docs/_build/html
   git diff --check

For documentation-only changes, run the Sphinx build and ``git diff --check``. For code
changes, run the test suite. If a check cannot run in your environment, mention it in
the pull request.

Docs changes:

- Add or update method citations in :doc:`references`.
- Use generated data in examples unless the data are included in the repository.

Dependency and release changes:

- Runtime dependencies live in ``pyproject.toml`` when required by ``pip install mrax``.
- Development, documentation, and notebook authoring tools belong in contributor extras.
- Release notes list user-facing changes.

Documentation build
-------------------

To build the docs locally::

   cd docs
   sphinx-build -E -b html . _build/html

Open ``_build/html/index.html`` in your browser to view the rendered pages.

Importer extension points are covered in :doc:`importer_development`.
