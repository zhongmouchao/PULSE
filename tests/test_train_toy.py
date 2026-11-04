"""End-to-end toy test of the SGD trainer (pulse/train_sgd.py).

Same 12-nt synthetic dataset as the other toy tests, plus an 8-peptide DMS
benchmark. Checks: the positive-side extra encodings (existence, non-table,
exact support vs independent brute force), the exact-zero contribution of
zero-encoding draws, a full short LR training run (determinism, results row),
and a NAM registry smoke run.

Run:  python tests/test_train_toy.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import load_task
from pulse.support import ensure_observed_support
from pulse.sampler import Sampler, positive_extra_encodings
from pulse.train_sgd import run_training, FlatSupports

A = 'AAACCCGGGTTT'   # pool, K P G F        (also in batch)
B = 'AAACCCGGGTTA'   # pool, K P G L
ROWS = [(A, 'KPGF', 50, 0, 10, 0), (B, 'KPGL', 40, 0, 0, 0),
        ('AAACCCGGGTAA', 'KPG*', 0, 10, 0, 0), ('AAACCCGGGTTC', 'KPGF', 0, 0, 20, 0),
        ('AAACCCGGGTCC', 'KPGS', 0, 0, 5, 0), ('GGGCCCGGGTTT', 'GPGF', 0, 0, 0, 3),
        ('AAANNNGGGTTT', 'KXGF', 0, 0, 2, 0)]
TABLE_NUCS = {r[0] for r in ROWS}
BENCH = [('VPGF', 0.9), ('KPGV', 0.8), ('KPRV', 0.7), ('GPGS', 0.6),
         ('KPGN', 0.4), ('SPGF', 0.3), ('KGGF', 0.2), ('KPGQ', 0.1)]


def brute_support(y, pool_seqs, freqs, R, max_dist=3):
    tot = 0.0
    for c, f in zip(pool_seqs, freqs):
        diffs = [(a, b) for a, b in zip(c, y)
                 if a in 'ACGT' and b in 'ACGT' and a != b]
        if 1 <= len(diffs) <= max_dist:
            p = 1.0
            for a, b in diffs:
                p *= R[a + '>' + b]
            tot += p * f
    return tot


def build(tmp):
    train_csv = os.path.join(tmp, 'toy_training_counts.csv')
    pd.DataFrame(ROWS, columns=['nuc', 'pep', '2001', '2002', '2003', 'unknown']
                 ).to_csv(train_csv, index=False)
    bench_csv = os.path.join(tmp, 'toy.csv')
    pd.DataFrame(BENCH, columns=['mutation', 'DMS_score']).to_csv(bench_csv, index=False)
    # rename: benchmark needs a sequence column
    df = pd.read_csv(bench_csv)
    df = df.rename(columns={'mutation': 'mutated_sequence'})
    df.to_csv(bench_csv, index=False)
    yaml_path = os.path.join(tmp, 'TOY.yaml')
    open(yaml_path, 'w').write(f"""task: TOY
data:
  training_file: {train_csv!r}
  benchmark_file: {bench_csv!r}
  nuc_column: nuc
  pep_column: pep
pool:
  fraction: 0.01
observation:
  obs_dist_max: 2
  support_max_dist: 3
training:
  num_sgd_steps: 300
  validation_interval: 25
  fixed_validation_size: 64
  sgd_batch_size: 32
  early_stop_checks: 25
  scheduler_patience: 50
  print_interval: 100000
cache:
  cache_dir: {os.path.join(tmp, 'cache')!r}
artifacts:
  results_csv: {os.path.join(tmp, 'results.csv')!r}
  models_dir: {os.path.join(tmp, 'models')!r}
""")
    return load_task(yaml_path)


def main():
    tmp = tempfile.mkdtemp()
    cfg = build(tmp)
    res = ensure_observed_support(cfg, verbose=False)
    sampler = Sampler(cfg, res, verbose=False)
    rates = res['rates']
    pool_seqs = sorted([A, B])
    fq = {A: 5 / 9, B: 4 / 9}
    fq_l = [fq[s] for s in pool_seqs]

    # ---- 1. positive-side extra encodings ----
    extra = positive_extra_encodings(sampler, res, verbose=True, keep_seqs=True)
    obs_peps = res['obs']['obs_peps']
    assert obs_peps == ['KPG*', 'KPGF', 'KPGS', 'KXGF']
    per_pep = {}
    for i, p in enumerate(obs_peps):
        seqs = extra['extra_seqs'][extra['extra_off'][i]:extra['extra_off'][i + 1]]
        sups = extra['extra_support'][extra['extra_off'][i]:extra['extra_off'][i + 1]]
        per_pep[p] = dict(zip(seqs, sups))
        for ys, sup in zip(seqs, sups):
            assert ys not in TABLE_NUCS, ys            # table strings excluded
            want = brute_support(ys, pool_seqs, fq_l, rates)
            assert abs(sup - want) / max(want, 1e-300) < 1e-12, (ys, sup, want)
    # KPGF (a pool peptide): synonymous variants from A expected
    kpgf = per_pep['KPGF']
    assert 'AAGCCCGGGTTT' in kpgf           # K: AAA->AAG synonymous
    assert 'AAACCAGGGTTT' in kpgf           # P: CCC->CCA synonymous
    anchor = 1.8e-4 * 5 / 9 + 1.8e-9 * 4 / 9   # A d1 (ti) + B d2 (ti*tv)
    assert abs(kpgf['AAGCCCGGGTTT'] - anchor) / anchor < 1e-12
    # KPGS (not a pool peptide): codon-path encodings from KPGF/KPGL
    assert len(per_pep['KPGS']) > 0
    print(f'  extras per peptide: {{k: len(v) for k, v in per_pep.items()}} = '
          f'{ {k: len(v) for k, v in per_pep.items()} }')

    # slim path (production default): bit-identical outputs minus the strings
    slim = positive_extra_encodings(sampler, res, verbose=False, keep_seqs=False)
    assert 'extra_seqs' not in slim and slim['n_extra'] == len(extra['extra_seqs'])
    assert np.array_equal(slim['extra_off'], extra['extra_off'])
    assert np.array_equal(slim['extra_support'], extra['extra_support'])
    print('  slim POS_L1 payload: identical supports/offsets, no strings — OK')

    # ---- 2. zero-encoding draw contributes exactly 0 ----
    xi = sampler.pool.PARENT_PEPS.index('KPGF')
    xs, q, encs = sampler._finish(xi, np.array([0, 1]),
                                  np.array([4, 5], dtype=np.int8))   # F, G
    assert xs == 'FGGF' and encs == [] and q > 0
    SF = FlatSupports([[]])
    sum_log = torch.zeros(SF.n).index_add_(0, SF.seg, SF.S)
    assert float(-torch.expm1(sum_log).item()) == 0.0                 # s = 0
    assert float(torch.log1p(torch.tensor(-0.7 * 0.0))) == 0.0        # g = 0

    # ---- 3. full short LR training run + results row + determinism ----
    row1 = run_training(cfg, verbose=False)
    assert np.isfinite(row1['best_val_loss']) and 0 < row1['p_o'] < 1
    assert row1['beta'] > 0 and row1['steps_run'] <= 300
    assert 0.0 <= row1['auc'] <= 1.0 and 0.0 <= row1['ap'] <= 1.0
    assert -1.0 <= row1['spearman'] <= 1.0 and row1['n_benchmark'] == 8
    res_csv = cfg.resolve_paths()['artifacts']['results_csv']
    df1 = pd.read_csv(res_csv)
    assert len(df1) == 1 and df1.iloc[0]['task'] == 'TOY'
    row2 = run_training(cfg, verbose=False)
    assert abs(row1['best_val_loss'] - row2['best_val_loss']) < 1e-9
    assert abs(row1['beta'] - row2['beta']) < 1e-12
    print(f'  LR run: val {row1["best_val_loss"]:.4f} @ {row1["best_step"]}, '
          f'beta {row1["beta"]:.3e}, p_o {row1["p_o"]:.3f}, '
          f'AUC {row1["auc"]:.3f}, Spearman {row1["spearman"]:.3f} '
          f'(deterministic: OK)')

    # ---- 4. NAM registry smoke run ----
    cfg_nam = load_task(os.path.join(tmp, 'TOY.yaml'), overrides={
        'model': {'classifier': 'NAM', 'hidden_dims': [4]},   # per-site width
        'training': {'num_sgd_steps': 100}})
    row3 = run_training(cfg_nam, over_label='model.classifier=NAM', verbose=False)
    assert row3['model'] == 'NAM' and np.isfinite(row3['best_val_loss'])
    df2 = pd.read_csv(res_csv)
    assert len(df2) == 3 and set(df2['model']) == {'LR', 'NAM'}

    cache_files = sorted(os.listdir(os.path.join(tmp, 'cache')))
    assert any(f.startswith('POS_L1_') for f in cache_files)
    print(f'  cache files: {cache_files}')
    print('\nALL TRAIN TOY TESTS PASSED')


if __name__ == '__main__':
    main()
