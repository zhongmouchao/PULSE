"""score_benchmark.py — score benchmark file(s) with saved checkpoint(s),
locally or on a compute node: no training data, no support caches needed.

train_sgd.py saves models/<run_tag>.pt containing the model state AND the full
run config. This script rebuilds the model and featurizer from that embedded
config, scores the benchmark's peptides with p_a, and reports the same metrics
as a training run (DMS: AUC/AP at a functional label threshold plus Spearman on
raw scores; activity: AUC/AP). Predictions CSVs are written per checkpoint.

Single checkpoint:
    python pulse/score_benchmark.py models/<tag>.pt --benchmark new_bench.csv
Scan every checkpoint whose name/tag contains a substring:
    python pulse/score_benchmark.py --scan Wuhan_Hu_1_RBD --benchmark new_bench.csv
Options:
    --type/--seq-col/--score-col/--mutation-col  benchmark format overrides
    --threshold 'median' | NUMBER                functional label threshold
    --save-results                               append/replace rows in results CSV
    --results-csv PATH                           (default: results/results.csv)
Rows are labelled <run_tag> ++bench=<benchmark_stem>, so cross-benchmark
scorings never collide with training rows or the skip logic, and re-running
replaces a row instead of duplicating it.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import (TaskConfig, _from_dict, _validate,
                                  bounded_filename, PROJECT_ROOT)
from pulse.featurizers import build_featurizer
from pulse.models import build_model
from pulse.train_sgd import evaluate_benchmark


def score_checkpoint(pt_path, args, bench_stem):
    """Rebuild model+config from one checkpoint and score the benchmark.
    Returns (row_dict, predictions_csv_path)."""
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    run_tag = ckpt.get('run_tag', os.path.splitext(os.path.basename(pt_path))[0])

    # rebuild the run's config from the checkpoint, then point it at the NEW
    # benchmark (absolute path passes through resolve_paths unchanged)
    d = dict(ckpt['config'])
    d['data'] = dict(d['data'])
    d['data']['benchmark_file'] = os.path.abspath(args.benchmark)
    bc = dict(d['data'].get('benchmark') or {})
    if args.type:
        bc['type'] = args.type
    if args.seq_col:
        bc['sequence_column'] = args.seq_col
    if args.score_col:
        bc['score_column'] = args.score_col
    if args.mutation_col:
        bc['mutation_column'] = args.mutation_col
    if args.threshold is not None:
        bc['functional_threshold'] = (args.threshold if args.threshold == 'median'
                                      else float(args.threshold))
    d['data']['benchmark'] = bc
    cfg = TaskConfig(**_from_dict(d))
    _validate(cfg)

    feat = build_featurizer(cfg)
    peps = pd.read_csv(args.benchmark)[cfg.data.benchmark.sequence_column].astype(str)
    model = build_model(cfg, feat.featurize(peps.tolist()).shape[1])
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    metrics, preds = evaluate_benchmark(model, cfg, feat)

    beta = float(torch.exp(ckpt['d_raw'])) if 'd_raw' in ckpt else float('nan')
    p_o = float(torch.sigmoid(ckpt['b_raw'])) if 'b_raw' in ckpt else float('nan')

    out = (os.path.join(os.path.dirname(pt_path),
                        bounded_filename(run_tag, f'__{bench_stem}_predictions.csv'))
           if args.out is None else args.out)
    preds.to_csv(out, index=False)

    tr, mo = cfg.training, cfg.model
    row = {'task': cfg.task,
           'run_tag': f'{run_tag} ++bench={bench_stem}',
           'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'pool_fraction': cfg.pool.fraction,
           'obs_dist_max': cfg.observation.obs_dist_max,
           'support_max_dist': cfg.observation.support_max_dist,
           'rates': cfg.mutation_model.rates
                    if isinstance(cfg.mutation_model.rates, str) else 'explicit',
           'mu_total': cfg.mutation_model.mu_total,
           'n_max': cfg.proposal.n_max,
           'sgd_batch_size': tr.sgd_batch_size, 'num_sgd_steps': tr.num_sgd_steps,
           'reg_lambda': tr.reg_lambda, 'model': mo.classifier,
           'hidden_dims': str(mo.hidden_dims), 'dropout': mo.dropout,
           'representation': cfg.representation.name, 'seed': tr.seed,
           'beta': beta, 'p_o': p_o, **metrics}

    print(f'[{run_tag}]')
    print(f'  benchmark: {args.benchmark} ({cfg.data.benchmark.type}, '
          f'threshold={cfg.data.benchmark.functional_threshold})')
    print(f'  beta {beta:.4e}  p_o {p_o:.4f}')
    print(f'  n={metrics["n_benchmark"]}  AUC {metrics["auc"]:.4f}  '
          f'AP {metrics["ap"]:.4f}  Spearman {metrics["spearman"]:.4f}')
    print(f'  predictions -> {out}', flush=True)
    return row, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', nargs='?', help='models/<run_tag>.pt from train_sgd.py')
    ap.add_argument('--scan', help='score every checkpoint in --models-dir whose '
                                   'filename contains this substring')
    ap.add_argument('--models-dir', default=os.path.join(PROJECT_ROOT, 'models'))
    ap.add_argument('--benchmark', required=True, help='new benchmark CSV')
    ap.add_argument('--type', choices=['dms', 'activity'],
                    help='benchmark type (default: the checkpoint config)')
    ap.add_argument('--seq-col', help='sequence column (default: checkpoint config)')
    ap.add_argument('--score-col', help='score column (default: checkpoint config)')
    ap.add_argument('--mutation-col', help='mutation column (dms only)')
    ap.add_argument('--threshold', default=None,
                    help="functional label threshold: 'median' (default) or a number")
    ap.add_argument('--out', help='predictions CSV output path (single-checkpoint mode '
                                  'only; default: <checkpoint>__<benchmark>_predictions.csv)')
    ap.add_argument('--save-results', action='store_true',
                    help='append/replace rows in the results CSV (run_tag gets a '
                         '" ++bench=<benchmark>" suffix; never collides with '
                         'training rows or the skip logic)')
    ap.add_argument('--results-csv', help='results file for --save-results '
                                          '(default: results/results.csv next to the project)')
    args = ap.parse_args()

    if bool(args.checkpoint) == bool(args.scan):
        ap.error('give exactly one of: a checkpoint path, or --scan SUBSTRING')
    if args.out and args.scan:
        ap.error('--out only applies to single-checkpoint mode')

    if args.scan:
        if not os.path.isdir(args.models_dir):
            ap.error(f'no models dir at {args.models_dir}')
        targets = sorted(os.path.join(args.models_dir, f)
                         for f in os.listdir(args.models_dir)
                         if f.endswith('.pt') and args.scan in f)
        if not targets:
            ap.error(f'no checkpoints matching {args.scan!r} in {args.models_dir}')
    else:
        targets = [args.checkpoint]

    bench_stem = os.path.splitext(os.path.basename(args.benchmark))[0]
    print(f'{len(targets)} checkpoint(s) to score on {args.benchmark}')

    rows, failures = [], []
    for pt in targets:
        try:
            row, _ = score_checkpoint(pt, args, bench_stem)
            rows.append(row)
        except Exception as e:
            failures.append(pt)
            print(f'  -> FAILED: {os.path.basename(pt)}: {type(e).__name__}: {e}',
                  flush=True)

    if args.save_results and rows:
        # default: project-root results/results.csv (the artifacts default)
        path = args.results_csv or os.path.join(PROJECT_ROOT, 'results', 'results.csv')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new = pd.DataFrame(rows)
        if os.path.exists(path):        # idempotent: replace any prior rows for
            old = pd.read_csv(path)     # these run_tags, keep everything else
            old = old[~old['run_tag'].isin(new['run_tag'])]
            out_df = pd.concat([old, new], ignore_index=True)
        else:
            out_df = new
        tmp_path = path + '.tmp'        # atomic write (same filesystem)
        out_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        print(f'results -> {path}  ({len(rows)} row(s), run_tag suffix '
              f'"++bench={bench_stem}")')

    print(f'done: {len(rows)} scored, {len(failures)} failed')
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
