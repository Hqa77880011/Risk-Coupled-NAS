# Risk-Coupled-NAS

Implementation of **Risk-coupled evolutionary architecture search across device families.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Input CSV columns are `architecture,device,objective,block,cost,cap,budget,charge,error,split,genotype`. Each block is an independently acquired run-block statistic; within-block invocations are averaged, not counted as independent samples. `split` is `search` or `evaluation`. Timeouts retain the cap. Development error is fixed before search, and genotype is a JSON numeric vector. Hold evaluation blocks out of every search decision. The optional training entry point supports actual CIFAR weight training, development scoring, native precision conversion and one hardware run block per invocation.

```bash
python -m risknas prepare --input data/tape.csv --config config.json --output runs/full
```

## Run

```bash
python -m risknas run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

### Optional architecture measurements

```bash
python -m pip install -e ".[training]"
python -m risknas.train train --architecture a0 --genotype "[4,32,3,2,32]" --output runs/a0
python -m risknas.train measure --architecture a0 --genotype "[4,32,3,2,32]" --block 0 --output runs/a0
```

Retain each `measurement.csv` block before acquiring the next; hold evaluation blocks out of search.

## Evaluation

```bash
python -m risknas evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `uniform-interval`, `graph-point`, `uniform-point`, `largest-width`, `random-active`, `no-parent-gate`.

The run command evaluates these controls together.

## Outputs

All main outputs are written below `runs/full/`.

`measurements.csv` records charged streams and nested intervals. `search.json` separates certified, unresolved, infeasible and dominated candidates, plus observation-model conflicts. `metrics.csv` reports certified-return rate, held-out violation, feasible HV, false pruning, unresolved relevant candidates and acquisition efficiency. Optional architecture training writes development/test accuracy separately from search evidence.

```bash
python -m risknas plot --config config.json --output runs/full
```
