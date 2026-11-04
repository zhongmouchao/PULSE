"""evaluate.py — re-evaluate saved benchmark predictions under different
functional thresholds, WITHOUT torch and WITHOUT retraining.

Every training run saves models/<run_tag>_predictions.csv with one row per
benchmark sequence: sequence, [mutation], bench_score, p_a. This tool
recomputes AUC/AP for any functional label threshold on the DMS score:
  median      top 50% by score (the pipeline default)
  pct:Q       top Q percent functional, e.g. pct:25
  NUMBER      absolute DMS score cutoff, e.g. 0.2
Spearman on the raw scores is threshold-free and printed once per run.
Activity benchmarks (0/1 labels) have no threshold to vary; AUC/AP are
printed directly and --threshold is ignored for them.

Usage:
    python pulse/evaluate.py --list
    python pulse/evaluate.py A0A2Z5U3Z0_9INFA_Wu_2014
    python pulse/evaluate.py 'FLU_cell_entry_H3N2_MC22[observation.obs_dist_max=2, pool.fraction=0.01]' \
        --threshold median pct:25 0.2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import PROJECT_ROOT, safe_tag, bounded_filename

PRED_SUFFIX = '_predictions.csv'


def metrics_at(y, s, spec):
    """-> (description, n, n_pos, AUC, AP) for one threshold spec."""
    if spec == 'median':
        thr = float(np.median(y))
        desc = f'median ({thr:.4g})'
    elif spec.startswith('pct:'):
        q = float(spec[4:])
        thr = float(np.percentile(y, 100.0 - q))
        desc = f'top {q:g}% (score >= {thr:.4g})'
    else:
        thr = float(spec)
        desc = f'score >= {thr:g}'
    labels = (y >= thr).astype(int)
    if labels.min() == labels.max():
        return desc, len(labels), int(labels.sum()), float('nan'), float('nan')
    from sklearn.metrics import roc_auc_score, average_precision_score
    return (desc, len(labels), int(labels.sum()),
            float(roc_auc_score(labels, s)),
            float(average_precision_score(labels, s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_tags', nargs='*', help='run tag(s) as shown in results.csv')
    ap.add_argument('--threshold', nargs='+', default=['median'],
                    help="'median', 'pct:Q', or absolute score cutoffs")
    ap.add_argument('--models-dir',
                    default=os.path.join(PROJECT_ROOT, 'models'))
    ap.add_argument('--list', action='store_true', help='list saved runs')
    args = ap.parse_args()

    if args.list:
        if os.path.isdir(args.models_dir):
            for f in sorted(os.listdir(args.models_dir)):
                if f.endswith(PRED_SUFFIX):
                    print(f[:-len(PRED_SUFFIX)])
        return
    if not args.run_tags:
        ap.error('give at least one run tag (or --list)')

    from scipy.stats import spearmanr
    for tag in args.run_tags:
        path = os.path.join(args.models_dir, bounded_filename(tag, PRED_SUFFIX))
        if not os.path.exists(path):
            # pre-fix fallback: files written before the 250-char budget was
            # introduced could carry the unbounded name if it still fit in 255
            legacy = os.path.join(args.models_dir, safe_tag(tag) + PRED_SUFFIX)
            if os.path.exists(legacy):
                path = legacy
        if not os.path.exists(path):
            print(f'[{tag}] no predictions file at {path}')
            continue
        df = pd.read_csv(path)
        y = pd.to_numeric(df['bench_score'], errors='coerce').to_numpy()
        s = df['p_a'].to_numpy()
        ok = ~np.isnan(y)
        y, s = y[ok], s[ok]
        print(f'[{tag}]  n={len(df)} ({ok.sum()} scored)')
        if len(set(np.unique(y)) - {0.0, 1.0}) == 0:
            from sklearn.metrics import roc_auc_score, average_precision_score
            print(f'  activity labels (0/1): AUC {roc_auc_score(y, s):.4f}  '
                  f'AP {average_precision_score(y, s):.4f}  '
                  f'(no threshold to vary)')
            continue
        if len(y) > 2:
            print(f'  Spearman (threshold-free): {spearmanr(s, y).statistic:.4f}')
        for spec in args.threshold:
            desc, n, npos, auc, ap = metrics_at(y, s, spec)
            print(f'  thr {desc:<30} n_pos {npos:>5}/{n}  '
                  f'AUC {auc:.4f}  AP {ap:.4f}')


if __name__ == '__main__':
    main()
