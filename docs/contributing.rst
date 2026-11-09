Contribution guide
==================

Development workflow:

1. Create a branch from the current development branch.
2. Use the dev container or install the editable checkout locally:

   .. code-block:: bash

      python -m pip install -e ".[all,dev,docs,notebooks]"

3. Add tests for public APIs, importer output, storage contracts, or numerical
   conventions. Use independently generated synthetic inputs.
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
changes, run the test suite. Report each check's outcome and any environment
requirements in the pull request.

Docs changes:

- Add or update method citations in :doc:`references`.
- Use generated data in examples unless the data are included in the repository.

Dependency and release changes:

- Shared runtime dependencies live in ``pyproject.toml``. The ``full`` extra adds
  specialized scientific tools; ``gui`` adds those tools and the web backend.
  Keep imports usable without unrelated extras and make missing-feature errors
  name the required extra.
- Development, documentation, and notebook authoring tools belong in contributor extras.
- Release notes list user-facing changes.
- Build and qualify each installation profile in an isolated environment. Keep
  desktop installer recipes, browser runtimes and installer outputs outside this
  repository.

GUI development
---------------

After installing the editable checkout, build the frontend before starting
its GUI:

.. code-block:: bash

   npm --prefix frontend ci
   npm --prefix frontend run build
   mrax-gui

Run browser tests with ``npm --prefix frontend test``.

Documentation build
-------------------

To build the docs locally::

   cd docs
   sphinx-build -E -b html . _build/html

Open ``_build/html/index.html`` in your browser to view the rendered pages.

Importer extension points are covered in :doc:`importer_development`.
