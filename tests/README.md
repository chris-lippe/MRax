# Regression tests

| Area | Contracts |
| --- | --- |
| Models and fitting | Known signals and parameter recovery, observables, units, weighting, masks, bounds, checkpoint resume and cancellation |
| Solvers and reconstruction | Independent solver references, Fourier/NUFFT adjoints, measured samples, reconstruction stages and quantitative import-to-fit agreement |
| Import and geometry | Bruker, DICOM, Pulseq/MRD and NIfTI dimensions, timing, orientation and multiple slice packages |
| Processing and analysis | Denoising, noise estimates, normalization, drift, ROIs, timecourses and retained fit results |
| Workspaces and files | Editable/reusable pipelines, bundles, image exports, source immutability, overwrite confirmation and atomic failure handling |
| Browser | Complete data, viewing, segmentation, processing, reconstruction, fitting, metadata, analysis and save/load workflows |

## Routine checks

```sh
python -m pytest -q
npm --prefix frontend run build
npm --prefix frontend test
```

Python tests use CPU cases by default. Browser tests use Chromium. To check one
area, pass its test file. `npm --prefix frontend run test:smoke` runs one complete
GUI workflow. Use `--durations=10` with pytest to review runtime when extending the
suite. Timing comparisons should use the same environment.

## Explicit qualification

```sh
npm --prefix frontend run test:cross-browser
python -m pytest -q -m gpu tests/test_optimizer_workflows.py
```
