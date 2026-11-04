# RBD baselines

Baseline methods for the eleven SARS-CoV-2 RBD benchmarks, compared against
PULSE (whose implementation lives in `../pulse/`).

The baselines are:

- **ConstPU**, constant-labeling PU. This is the Protein-PU run used in the table, renamed. It draws 128 unobserved peptides at each step from the unobserved 3-nucleotide-substitution neighborhood and trains a NAM on CHEM features.
- **2Step**, with a fixed unlabeled set drawn within 3 nucleotide substitutions of an observed peptide.
- **OCSVM** and **iForest**, fit on the observed CHEM features only.
- **ESM-1v**, scored from `esm1v_model.py`. The weights download on the first run (about 650M parameters).

## Data

No sequence data is shipped. Download the five training files and eleven
benchmark files into `../data/` first (see `../data/DATA.md`), then run once
from this directory:

```bash
python prepare_data.py
```

This converts the shared files into the internal layout under `data/<property>/`
(eight properties; Alpha, Beta, and Eta each have binding and expression test
files, so there are eleven benchmark tasks). The CHEM features — the same
hydrophobicity / molar-refractivity / volume descriptors PULSE uses — are
computed on first use and cached as `data/<property>/<property>_chem.npz`;
the 2Step unlabeled sets are likewise generated with fixed seeds and cached on
first use.

Run every command from this directory, so `data/` resolves.

## Install

```bash
pip install -r requirements.txt
```

ConstPU, 2Step, OCSVM, and iForest need the packages above except `fair-esm`. ESM-1v needs PyTorch and a network connection the first time, so the ESM-1v weights can download.

ConstPU and 2Step are much faster on a GPU. OCSVM and iForest run on CPU. Pass `--device cpu` to force the CPU.

## ConstPU

The reported model uses `--constpu_unlabeled uniform`, batch size 128, NAM, CHEM, and penalty 0. The neighborhood radius is 3 inside the trainer. Leave `--obs_dist_max` at its default of 2; that flag only labels the result file for ConstPU.

`q` was chosen from `{0.1, 0.2, ..., 0.9}` by the mean raw ROC AUC over 10 folds. The selected values are:

| Problem | q |
|---|---|
| Binding Wuhan, Expression Wuhan, Alpha, Eta | 0.8 |
| Binding Omicron XBB.1.5, Expression Omicron XBB.1.5, cell entry, Beta | 0.9 |

Alpha, Beta, and Eta use one `q` for both assays.

A cross-validation job for one `q` does not score the test set:

```bash
python main_run.py \
  --method ConstPU \
  --property Binding_Wuhan \
  --use_unlabel True \
  --unlabeled_type RAND \
  --classifier_model NAM \
  --representation CHEM \
  --penalty_coeff 0 \
  --constpu_unlabeled uniform \
  --constpu_unobs_batch 128 \
  --constpu_q 0.8 \
  --output_dir result_constpu
```

The test Spearman rho is a second job, at the selected `q`:

```bash
python main_run.py \
  --method ConstPU \
  --property Binding_Wuhan \
  --use_unlabel True \
  --unlabeled_type RAND \
  --classifier_model NAM \
  --representation CHEM \
  --penalty_coeff 0 \
  --constpu_unlabeled uniform \
  --constpu_unobs_batch 128 \
  --constpu_q 0.8 \
  --constpu_final_only \
  --output_dir result_constpu
```

For Alpha, Beta, or Eta, one final job writes both assays. Results go to `result_constpu/<property>/ConstPU_uniform/`. The file name contains `q_0.8_final` or `q_0.9_final`.

## 2Step

The table uses 1,000 unlabeled peptides, penalty 0, NAM, and CHEM. Each job draws 10 unlabeled sets and, for each set, repeats the spy step with 10 seeds. That is 100 fits. `--twostep_set_index 0` runs only set 0 (seeds 0 through 9) if you want a smaller check.

```bash
python main_run.py \
  --method 2Step \
  --property Binding_Wuhan \
  --use_unlabel True \
  --unlabeled_type RAND \
  --classifier_model NAM \
  --representation CHEM \
  --penalty_coeff 0 \
  --obs_dist_max 3 \
  --rand_unlabeled_size 1000 \
  --output_dir result_2step
```

The log reports the mean and standard deviation of Spearman rho. Replace `1000` with `100` or `10000` for the other unlabeled-set sizes; each size generates and caches its own feature file on first use.

## OCSVM and iForest

Both fit on the observed peptides only.

```bash
python main_run.py \
  --method OCSVM \
  --property Binding_Wuhan \
  --use_unlabel False \
  --representation CHEM \
  --output_dir result_ocsvm

python main_run.py \
  --method iForest \
  --property Binding_Wuhan \
  --use_unlabel False \
  --representation CHEM \
  --output_dir result_iforest
```

## ESM-1v

The score is the pseudo-log-likelihood difference from the wild type. Alpha, Beta, and Eta need one command per test file.

```bash
python esm1v_model.py \
  --property Binding_Wuhan \
  --test_csv data/Binding_Wuhan/Binding_Wuhan_test.csv \
  --wt_file data/Binding_Wuhan/WT.txt \
  --output_dir result_esm1v

python esm1v_model.py \
  --property Alpha \
  --test_csv data/Alpha/Alpha_test_binding.csv \
  --wt_file data/Alpha/WT.txt \
  --output_dir result_esm1v
```

## What the test score is

Where the test file has a `freq` column, the printed Spearman rho is the correlation between the model score and that frequency. That is the number in the comparison table.
