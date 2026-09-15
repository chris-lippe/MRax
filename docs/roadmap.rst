Roadmap
=======

This is a rough sketch of likely development directions. Priorities may change.

- **Backend and GUI overhaul:** use .mrax files consistently for data strorage, use internal SQLite database for efficient storage, 
  make GUI more responsive and user-friendly and browser-based using FastAPI and React. Enable automatic testing using Playwright.
- **Public Reference Data:** provide a public reference dataset for testing and benchmarking.
- **GPU support:** improve documentation, examples, and memory handling for running
  fitting and reconstruction workflows on accelerators.
- **Performance optimization:** add benchmarks, speed up computations.
- **More importers:** add support for more vendor formats and make custom importer
  development easier for lab-specific data.
- **Batch and GUI workflows:** improve workflow, study
  browsing, ROI management, and fit inspection.
- **Additional optimizer functionality:** Rician noise estimation, 
  maximum likelihood estimation with Rician noise, Bayesian fitting,
  multi-stage fitting, and more regularization options. 
- **Simulation and synthetic data:** add simulation capabilities for research purposes. 
- **Default model library:** extend the built-in model collection with more
  quantitative MRI models, default bounds, units, and initialization strategies.
- **Examples:** grow the notebook gallery with compact end-to-end workflows for
  import, fitting, reconstruction, ROI analysis, and batch processing.
