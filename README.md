# SFL-Lab

**Split Federated Learning Formula & Algorithm Validation Workbench**

Enter a formula or controlled algorithm, validate its placement in a U-shaped SFL pipeline, and compare a fixed reference against the candidate using real PyTorch training. The workbench records evidence, explores trade-offs, and exports scientific reports.

## Run

Requires Python 3.12, Node.js 24 and a CPU Linux environment. No GPU, account, API key or external dataset is required for the bundled Digits workflow.

```bash
bash scripts/setup.sh
bash scripts/start.sh
```

The second command starts the dashboard and API on loopback port 8000. `SFL_LAB_PORT` selects another port. API documentation is available at `/docs`. For separate frontend development, run `npm --prefix frontend run dev` alongside the Python service.

Choose **Tiny CNN** for fast functional checks and **ResNet18** for the initial research architecture. Start with a paired comparison, inspect validation and SFL compliance, then enable full-evaluation suites. Training uses a serial worker with progress, logs, cancellation, history and exports.

## Implemented workflow

- Real client-front → server-middle → client-back execution with detached boundaries and explicit backward gradients. ResNet18 defaults to a 2–4–2 split of eight residual blocks, with the stem and prediction head client-side.
- Safe formulas at H1–H14, mathematical keyboard, executable templates, controlled algorithm blocks and preflight/runtime validation.
- Fixed reference versus candidate, matched seeds/initialization/partitions, separate train/validation/test data, validation-selected checkpoints and one final test evaluation per run.
- IID, Dirichlet, label-skew and quantity-imbalance partitions; FedProx; client accuracy, loss, drift and fairness dispersion.
- Activation/gradient clipping and Gaussian noise, conservative Gaussian RDP composition, scoped budget monitoring and actual pixel reconstruction probes.
- Measured tensor traffic, modeled network/energy scenarios, sweeps, component ablations, optional client/server scalability runs and offline split profiling.
- Paired statistics, acceptance criteria, explained recommendations, Pareto comparisons, interactive charts and JSON/CSV/PDF/Markdown reports. Charts support image export.
- SQLite history and a React/TypeScript dashboard served by FastAPI.

Read [the scientific methodology](docs/METHODOLOGY.md) for exact assumptions and limits. The [original brief](docs/PROJECT_BRIEF.md) is retained for traceability. Original-paper numerical reproduction, online layer migration and physically distributed deployment require further work. No published SeaShips scores or full-system DP guarantees are claimed.

See [verified execution and release scope](docs/VALIDATION.md) for the measured learning result and the exact boundaries of this release.

## Verify

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/verify_environment.py
.venv/bin/python scripts/verify_browser.py  # requires running service and Chromium
npm --prefix frontend run build
```

The verifier executes a three-seed paired Tiny CNN experiment with every evaluation suite enabled, then a ResNet18 comparison. It checks model updates, disjoint data, paired initialization, message counts, compliance, saved results and all report formats. The bounded budgets establish functionality, not research-quality accuracy or convergence. Evidence is saved to `artifacts/verification.json` and complete experiment exports. Add `--base-url` with the running local API address to verify HTTP transport too.

The browser verifier uses installed Chromium (`--browser` selects another executable), exercises actual UI-launched training, safe formula and algorithm editing, all thirteen result tabs, evidence and image/PDF downloads, cancellation, themes and a 390px mobile viewport. It writes screenshots and `artifacts/browser/verification.json`. It uses a previously completed full evaluation when available. A fresh Linux machine also needs Chromium's system libraries; the prepared cloud environment already has them.

## Reproducible CLI

```bash
.venv/bin/python -m sfl_lab defaults > experiment.json
# Edit the hypothesis, candidate, dataset and training budget.
.venv/bin/python -m sfl_lab run --config experiment.json --action compare --output artifacts/result.json
```

Actions are `baseline`, `candidate`, `compare`, and `full`. A conclusion requires both variants. Evidence includes configuration, seeds, software versions, Git state, split/model hashes, measurements and limitations.

Tested configurations are provided in `examples/`. `quick-study.json` is a short paired CPU study. `resnet-learning.json` is a larger Digits learning check; run it with `--action baseline` to reproduce the reference learning workflow. `parameter-grid.json` enables a real two-dimensional clipping/noise sweep. Grid and seeded random search use only executed cells; validation selects parameters.

## Data and persistence

Digits runs offline. CIFAR-10 has an optional Torchvision download adapter. Local images, including a researcher-supplied SeaShips copy, use `dataset: "imagefolder"` and a `dataset_path` with one directory per class. Inputs currently use 32×32 RGB preprocessing. Verify the original paper's data and protocol before claiming reproduction.

`artifacts/experiments.sqlite3` stores dashboard history; set `SFL_LAB_DATA_DIR` to relocate it. Data, evidence outputs, environments and built assets are ignored by Git. Retain exported JSON and the exact source commit for research provenance. Persistent model checkpoint/resume is not implemented.

```text
sfl_lab/      models, datasets, engine, formulas, science, analysis, API
frontend/     React dashboard and ECharts visualizations
tests/        numerical, security, training and API checks
scripts/      installation, startup and environment validation
docs/         specification, method and internal module contracts
artifacts/    local database and generated evidence (ignored)
data/         optional datasets (ignored)
```

Cloud tasks already have isolated checkouts; use the existing checkout without creating a Git worktree. Restart the service after restoring an environment snapshot because live processes do not survive publication/restoration.
