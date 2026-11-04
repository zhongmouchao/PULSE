# PULSE

PULSE (Positive-Unlabeled learning with mutation-guided Likelihood and Surveillance
Emergence) scores protein variants by combining surveillance sequence data with a
positive-unlabeled observation model: each observed peptide is detected only if one
of its nucleotide encodings both *emerges* from a high-prevalence parent pool
(mutation-rate weighted) and is *observed*, and unobserved peptides enter the
likelihood through on-the-go importance sampling during SGD.

This repository contains the PULSE implementation (`pulse/`) and the baseline
methods it is compared against (`baselines/`), for the eleven SARS-CoV-2 RBD
benchmarks spanning five variants (Wuhan-Hu-1, Alpha, Beta, Eta, Omicron XBB.1.5).

## Layout

```
pulse/        PULSE method (support, sampler, models, SGD training, scoring)
baselines/    ConstPU, 2Step, OCSVM, iForest, ESM-1v (self-contained, own data)
tasks/        one task config per variant (5 training problems)
sweeps/       paper.yaml (main configuration), ablation_grid.yaml (sensitivity)
scripts/      train_all.sh, score_all.sh
tests/        unit and toy-end-to-end tests
data/         place the training and benchmark CSVs here (see data/DATA.md)
wildtypes/    wild-type RBD sequences (used by the ESM-1v baseline)
docs/         implementation writeup
```

## Install

```bash
pip install -r requirements.txt
```

`fair-esm` is only needed for the ESM-1v baseline.

## Data

The repository ships no sequence data. Download the five training files
(~120 MB each) from
[Google Drive](https://drive.google.com/drive/folders/151zszaakRMy9KWyMiDdJGDcsEc4WRl6X?usp=sharing)
and place them in `data/`, together with the eleven benchmark files — see
[data/DATA.md](data/DATA.md) for the exact file list and formats (also
`data/DOWNLOAD.txt`). The five training problems share data by variant: one
fit per variant serves all of that variant's assays.

## Train PULSE (one run per variant)

```bash
python pulse/train_sgd.py --task SARS_Wuhan_Hu_1_RBD --sweep sweeps/paper.yaml
# or all five variants:
bash scripts/train_all.sh
```

Each run appends a tagged row to `results/results.csv` and saves a checkpoint
under `models/`.

## Evaluate on every assay of a variant

```bash
python pulse/score_benchmark.py --scan Wuhan_Hu_1_RBD \
    --benchmark data/SARS_expression_Wuhan_Hu_1_RBD_EVE_Truncated.csv --save-results
# or all eleven benchmarks:
bash scripts/score_all.sh
```

Checkpoints embed their full configuration, so scoring needs only the checkpoint
and the benchmark CSV — no training data or support caches.

## Baselines

See [baselines/README.md](baselines/README.md). After placing the shared data
in `data/`, run `python baselines/prepare_data.py` once to build the baselines'
internal layout; all CHEM features and unlabeled sets are then generated with
the same three-descriptor representation PULSE uses and cached on first use.

## Tests

```bash
python -m pytest tests/
```

## Citation

TODO: add citation.
