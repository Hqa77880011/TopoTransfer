# TopoTransfer

Implementation of **Selective graph conformal diagnostics under unlabeled topology shift**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation

The generator writes finite source/target graph scenarios as JSONL, with explicit topology shifts and recorded labels for evaluation. These are generated graph workloads; `--count` and `--seed` fix regeneration.

```bash
python -m topotransfer prepare --config configs/reproduce.json --output data/reproduce.jsonl
```

## Execution and evaluation

`run` evaluates the main method; `suite` evaluates the methods in `METHODS` on the same scenarios.

```bash
python -m topotransfer run --data data/reproduce.jsonl --output runs/main
python -m topotransfer suite --data data/reproduce.jsonl --output runs/comparison
```

## Outputs

Each `runs/<run>/<method>/` contains `records.jsonl` and `metrics.json`. The suite also writes `summary.csv`. The statistics command summarizes the main run records; plot inputs are method run directories.

```bash
python -m topotransfer.statistics --records runs/main/topotransfer/records.jsonl --output results/intervals.json
python -m topotransfer plot --inputs runs/comparison/topotransfer --output figures
```

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
