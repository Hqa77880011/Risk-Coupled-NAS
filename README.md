# Risk-Coupled-NAS

Implementation of **Risk-coupled evolutionary architecture search across device families.**

## Install

Use Python 3.11 in an isolated environment. Run from the repository root:

```bash
python -m pip install -e .
```

## Data

Input CSV columns are `architecture,device,objective,block,cost,cap,budget,charge,error,split,genotype`. Each block is an independently acquired run-block statistic; within-block invocations are averaged, not counted as independent samples. `split` is `search` or `evaluation`. Timeouts retain the cap. Development error is fixed before search, and genotype is a JSON numeric vector. Hold evaluation blocks out of every search decision. The optional training entry point supports actual CIFAR weight training, development scoring, native precision conversion and one hardware run block per invocation.

## Workflow

```bash
python -m risknas prepare --input data/tape.csv
python -m risknas run
python -m risknas evaluate
python -m risknas plot
```

Main commands accept `--config config.json` and `--output runs/default`. Each output directory records `settings.json`. Keep independent seeds, datasets and conditions in separate directories.

## Configuration

`delta` is the simultaneous certification error allocation; normalized deployment budgets come from CSV `budget/cap`. `run_budget` is charged acquisition work. `initial_blocks,forced_period,feasibility_weight` define initial/forced/graph allocation. `population,generation_blocks,unresolved_quota,evolution` define catalog-backed evolutionary admission. Additional optimizer, batch-size and model-width defaults are in `config.json`.

## Outputs

`measurements.csv` records charged streams and nested intervals. `search.json` separates certified, unresolved, infeasible and dominated candidates, plus observation-model conflicts. `metrics.csv` reports certified-return rate, held-out violation, feasible HV, false pruning, unresolved relevant candidates and acquisition efficiency. Optional architecture training writes development/test accuracy separately from search evidence.

## Method and comparisons

The anytime Hoeffding allocation sums over architecture introduction, device, objective and sample count. Nested intersections retain resolved predicates; an empty intersection stops certification. Graph incidence prioritizes streams blocking feasibility or dominance, with forced round-robin coverage. Evolution selects certified/unresolved parents and introduces nearby unseen catalog genotypes. A catalog is the declared genotype/evaluation domain, so missing hardware streams do not become invented measurements.

Controls: `full`, `uniform-interval`, `graph-point`, `uniform-point`, `largest-width`, `random-active`, `no-parent-gate`. The run command evaluates the allocation/decision controls together.

```bash
python -m risknas.compare --inputs "runs/*/metrics.csv" --reference full --metrics false_pruning unresolved_fraction feasible_hypervolume --unit seed
```

Comparison averages rows within independent units before bootstrap intervals and paired Wilcoxon/Holm inference. For participant/client analyses choose those independent unit columns instead of counting recordings as replicates.

## Architecture training and measured blocks

```bash
python -m pip install -e ".[training]"
python -m risknas.train train --architecture a0 --genotype "[4,32,3,2,32]" --output runs/a0
python -m risknas.train measure --architecture a0 --genotype "[4,32,3,2,32]" --block 0 --output runs/a0
python -m risknas.train test --architecture a0 --genotype "[4,32,3,2,32]" --output runs/a0
```

Genotypes are depth, width, odd kernel, expansion and precision bits. Supported deployment is CPU INT8/FP32 and CUDA FP16/FP32. Training/development partitions come only from CIFAR training data; final test accuracy is separate. One measure invocation writes one block to `measurement.csv`; retain that block before acquiring another. Device labels in the search tape should identify the full hardware/runtime protocol, not simply a generic backend. Relabel held-out block exports as `evaluation` before concatenating the tape.

