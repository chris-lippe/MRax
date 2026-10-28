# Test suite

| Suite | Coverage |
| --- | --- |
| `test_core_workflows.py` | Models, observables, losses, variance and study fitting |
| `test_optimizer_workflows.py` | Optax integration, bounds, schedules, callbacks and checkpoints |
| `test_fista_workflows.py` | FISTA recurrence, line search, restart and precision |
| `test_split_bregman_workflows.py` | Split-Bregman updates, primal solves and stability |
| `test_proximal_workflows.py` | TV operators, joint TV/box prox and physical-coordinate fitting |
| `test_acquisition_workflows.py` | Encoding, reconstruction, geometry and model-based inversion |
| `test_nufft_workflows.py` | Direct-DFT comparisons, adjoints, derivatives and sampling |
| `test_import_workflows.py` | Synthetic Bruker and DICOM imports |
| `test_pulseq_workflows.py` | Synthetic Pulseq, MRD and NIfTI alignment and timing |
| `test_gui_workflows.py` | Sessions, imports, ROI editing, plots and state |
| `test_save_export_workflows.py` | Bundles, image exports, package metadata and path safety |
| `test_device_workflows.py` | Device selection, placement and CPU/GPU parity |
| `test_preprocessing_workflows.py` | Drift, baselines, DCE and group statistics |

Shared file builders live in `_synthetic.py`.

Run `python -m pytest -q`. For headless GUI checks, set
`QT_QPA_PLATFORM=offscreen`. Tests select available CPU/GPU devices explicitly;
GPU-specific cases skip when a GPU is unavailable.
