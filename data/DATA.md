# Data files expected in this folder

**The repository ships no sequence data.** Download the five training files
(~120 MB each) from Google Drive:

    https://drive.google.com/drive/folders/151zszaakRMy9KWyMiDdJGDcsEc4WRl6X?usp=sharing

(also in `DOWNLOAD.txt` next to this file) and place them in this folder,
together with the eleven benchmark files. Then:

- PULSE works directly on these files (`tasks/*.yaml` already point here).
- For the baselines, run `python prepare_data.py` from `baselines/` once; it
  converts these files into the baselines' internal layout.

## Training (5 files, one per variant)

| File | Variant |
|---|---|
| `SARS_Wuhan_Hu_1_RBD_training_counts.csv` | Wuhan-Hu-1 |
| `SARS_Alpha_RBD_training_counts.csv` | Alpha |
| `SARS_Beta_RBD_training_counts.csv` | Beta |
| `SARS_Eta_RBD_training_counts.csv` | Eta |
| `SARS_Omicron_XBB15_RBD_training_counts.csv` | Omicron XBB.1.5 |

Format: one row per unique nucleotide sequence. Columns: `Motif Nucleotide`,
`Motif Peptide`, then one integer count column per surveillance period
(`YYYY` or `YYYY-MM` headers). Rows are expected pre-cleaned (uniform length,
no zero-count rows); the loader drops and reports any remaining violators.

## Benchmarks (11 files, two versions each)

Each benchmark exists in two versions: the **EVE-truncated** subset
(`..._EVE_Truncated.csv`, listed below) used for every number in the paper,
and the **full, untruncated** DMS library under the same name without the
suffix (e.g. `SARS_binding_Alpha_RBD.csv`). The truncated files are the
default everywhere; to score or prepare the full libraries instead, run
`bash scripts/score_all.sh ""` and `python baselines/prepare_data.py
--suffix ""`.


| File | Variant | Assay |
|---|---|---|
| `SARS_binding_Wuhan_Hu_1_RBD_EVE_Truncated.csv` | Wuhan-Hu-1 | binding |
| `SARS_expression_Wuhan_Hu_1_RBD_EVE_Truncated.csv` | Wuhan-Hu-1 | expression |
| `SARS_binding_Alpha_RBD_EVE_Truncated.csv` | Alpha | binding |
| `SARS_expression_Alpha_RBD_EVE_Truncated.csv` | Alpha | expression |
| `SARS_binding_Beta_RBD_EVE_Truncated.csv` | Beta | binding |
| `SARS_expression_Beta_RBD_EVE_Truncated.csv` | Beta | expression |
| `SARS_binding_Eta_RBD_EVE_Truncated.csv` | Eta | binding |
| `SARS_expression_Eta_RBD_EVE_Truncated.csv` | Eta | expression |
| `SARS_binding_Omicron_XBB15_RBD_EVE_Truncated.csv` | Omicron XBB.1.5 | binding |
| `SARS_expression_Omicron_XBB15_RBD_EVE_Truncated.csv` | Omicron XBB.1.5 | expression |
| `SARS_cell_entry_Omicron_XBB15_RBD_EVE_Truncated.csv` | Omicron XBB.1.5 | cell entry |

Format: one row per DMS measurement. Columns: `mutation` (e.g. `N1A`),
`mutated_sequence` (full-length amino-acid sequence), `DMS_score`.

**Why truncated:** EVE cannot score every variant in the original DMS
libraries; for a fair comparison, all methods in the paper (including PULSE)
are evaluated on the same truncated subsets, restricted to the variants EVE
can score.

Wild-type sequences for the ESM-1v baseline are in `../wildtypes/`.
