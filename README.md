# TopoTransfer

Implementation of **Selective graph conformal diagnostics under unlabeled topology shift**.

Calibrates an isotonic excess-miscoverage envelope across labeled environments, adds an environment conformal residual and finite-node term, then applies component support and leave-one-out stability checks.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/TopoTransfer.git
cd TopoTransfer
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

A source graph, at least 100 labeled calibration environments and an unlabeled target graph share node feature/class dimensions. Target labels are evaluator-only. Every graph has x, adjacency and y. Alpha, delta, tolerance and stability limits are frozen public parameters.

```bash
python -m topotransfer prepare --count 1 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m topotransfer run --data data/workload.jsonl --output runs/main
python -m topotransfer suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. topotransfer, global_envelope, feature_only, topology_only and source_only.

## Evaluation and visualization

Acceptance, accepted-environment false-certificate rate, target miscoverage, set size, envelope failure, support/stability refusal and runtime.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m topotransfer plot --inputs runs/comparison/topotransfer --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

The shared frozen base predictor is a graph-aware ridge classifier using node and neighbor features. APS thresholds use only source holdout labels. The calibration unit is the whole environment. Signatures combine degree histograms, spectral/triangle summaries and representation moments; the target never changes the envelope.


## Uncertainty and paired comparisons

```bash
python -m topotransfer.statistics --records runs/main/topotransfer/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.

## Registered compact reproduction setting

Twenty independent finite graph-shift cases, each with 100 independent calibration environments, a compact two-class graph-smoothed predictor and a fixed prespecified shift family. Target labels are confined to scoring; unobservable concept shift is outside the certificate premise.

Run from the repository root after installation. The following commands use the registered setting; they are provided for execution and have not been run as part of the source review.

```bash
python -m topotransfer prepare --config configs/reproduce.json --output data/reproduce.jsonl
python -m topotransfer suite --data data/reproduce.jsonl --output runs/reproduce
```

This setting needs no external dataset or pretrained weights. Generated inputs and declared fault/shift laws define the finite task; reported quantities retain that provenance.
