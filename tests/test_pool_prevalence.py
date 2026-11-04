"""Toy verification of prevalence-mode pool selection (pool.mode='prevalence').

Same 12-nt (4-residue) toy dataset as test_observed_support_toy.py, but the
parents are the sequences whose FILE-WIDE count share exceeds
pool.prevalence_threshold, and the observed set is ALL rows (pool members
included — B, pool-only in the temporal test, is observed here). Support
values are checked against the same independent brute-force reference, and
the cache is shown to separate thresholds while temporal-mode keys stay
bit-compatible.

Run:  python tests/test_pool_prevalence.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import load_task
from pulse.support import (ensure_observed_support, split_pool_prevalence,
                              pool_key_parts)

# ---------------- toy data (same sequences as the temporal toy) ----------------
A = 'AAACCCGGGTTT'   # K P G F, total 60  (42.9% of grand 140)
B = 'AAACCCGGGTTA'   # K P G L, total 40  (28.6%)
C = 'AAACCCGGGTAA'   # K P G *, total 10  ( 7.1%)
D = 'AAACCCGGGTTC'   # K P G F, total 20  (14.3%)
E = 'AAACCCGGGTCC'   # K P G S, total  5  ( 3.6%)
F = 'GGGCCCGGGTTT'   # G P G F, total  3  ( 2.1%); d >= 3 from all parents
H = 'AAANNNGGGTTT'   # K X G F, total  2  ( 1.4%)

ROWS = [  # nuc, pep, 2001, 2002, 2003, unknown
    (A, 'KPGF', 50, 0, 10, 0),
    (B, 'KPGL', 40, 0, 0, 0),
    (C, 'KPG*', 0, 10, 0, 0),
    (D, 'KPGF', 0, 0, 20, 0),
    (E, 'KPGS', 0, 0, 5, 0),
    (F, 'GPGF', 0, 0, 0, 3),
    (H, 'KXGF', 0, 0, 2, 0),
]
COLS = ['nuc', 'pep', '2001', '2002', '2003', 'unknown']
GRAND = 140.0


def brute_support(y, pool_seqs, freqs, R, R_out, max_dist):
    """Independent reference (identical to the temporal toy's): sum over
    parents with 1 <= d <= max_dist of the per-site rate product over
    differing mutually valid sites, times frequency; plus the always-on
    self-survival term for exact pool members."""
    tot = 0.0
    for c, f in zip(pool_seqs, freqs):
        diffs = [(a, b) for a, b in zip(c, y)
                 if a in 'ACGT' and b in 'ACGT' and a != b]
        if 1 <= len(diffs) <= max_dist:
            p = 1.0
            for a, b in diffs:
                p *= R[a + '>' + b]
            tot += p * f
    if y in pool_seqs:
        f = freqs[pool_seqs.index(y)]
        ps = 1.0
        for b in y:
            if b in 'ACGT':
                ps *= (1.0 - R_out[b])
        tot += ps * f
    return tot


def write_task(tmp, name, pool_block):
    train_csv = os.path.join(tmp, 'toy_training_counts.csv')
    if not os.path.exists(train_csv):
        pd.DataFrame(ROWS, columns=COLS).to_csv(train_csv, index=False)
    bench_csv = os.path.join(tmp, 'toy.csv')
    if not os.path.exists(bench_csv):
        pd.DataFrame({'mutation': ['A1V'], 'mutated_sequence': ['VPGF'],
                      'DMS_score': [0.5]}).to_csv(bench_csv, index=False)
    yaml_path = os.path.join(tmp, f'{name}.yaml')
    open(yaml_path, 'w').write(f"""task: {name}
data:
  training_file: {train_csv!r}
  benchmark_file: {bench_csv!r}
  nuc_column: nuc
  pep_column: pep
{pool_block}observation:
  obs_dist_max: 2
  support_max_dist: 3
cache:
  cache_dir: {os.path.join(tmp, 'cache')!r}
""")
    return yaml_path


def prev_block(threshold):
    return f'pool:\n  mode: prevalence\n  prevalence_threshold: {threshold}\n'


def main():
    tmp = tempfile.mkdtemp()

    # ---- 0. split function unit checks ----
    fn = pd.DataFrame(ROWS, columns=COLS)
    pool_cols, batch_cols, grand = split_pool_prevalence(fn, 0.10)
    assert pool_cols == [] and batch_cols == ['2001', '2002', '2003', 'unknown']
    assert grand == GRAND
    assert dict(zip(fn['nuc'], fn['Pool0_Count'])) == \
        {A: 60, B: 40, C: 0, D: 20, E: 0, F: 0, H: 0}
    assert dict(zip(fn['nuc'], fn['Batch_Count'])) == \
        {A: 60, B: 40, C: 10, D: 20, E: 5, F: 3, H: 2}   # batch = ALL rows
    try:
        split_pool_prevalence(pd.DataFrame(ROWS, columns=COLS), 0.50)
        raise AssertionError('expected ValueError for empty pool')
    except ValueError as e:
        print(f'  empty-pool guard ok: {e}')

    # ---- 1. cache-key ingredients: temporal bit-compat, prevalence adds thr ----
    cfg_t = SimpleNamespace(pool=SimpleNamespace(mode='time_fraction'))
    parts_t = pool_key_parts(cfg_t, ['2001'])
    assert parts_t == {'pool_cols': ['2001']}, parts_t     # no new keys: old caches valid
    cfg_p = SimpleNamespace(pool=SimpleNamespace(mode='prevalence',
                                                 prevalence_threshold=0.10))
    parts_p = pool_key_parts(cfg_p, [])
    assert parts_p == {'pool_cols': [], 'pool_mode': 'prevalence',
                       'prev_thr': 0.1}, parts_p

    # ---- 2. end-to-end at threshold 0.10: parents {A, B, D}, observed = all ----
    cfg = load_task(write_task(tmp, 'TOY_PREV', prev_block(0.10)))
    res = ensure_observed_support(cfg, verbose=True)
    pool, obs = res['pool'], res['obs']
    assert res['pool_cols'] == []
    assert pool.seqs == sorted([A, B, D]) and abs(pool.total - 120.0) < 1e-9
    freqs = {s: float(pool.freq[i]) for i, s in enumerate(pool.seqs)}
    assert abs(freqs[A] - 60 / 120) < 1e-15 and abs(freqs[B] - 40 / 120) < 1e-15 \
        and abs(freqs[D] - 20 / 120) < 1e-15

    kept = set(obs['obs_seqs'])
    assert kept == {A, B, C, D, E, H}, kept     # B observed (pool member); F d=3 excluded
    assert obs['obs_excluded_distant'] == [F]

    # ---- 3. support values vs independent brute force ----
    rates = res['rates']
    R_out = {b: sum(v for k, v in rates.items() if k[0] == b) for b in 'ACGT'}
    ps = sorted([A, B, D])
    fq = [freqs[s] for s in ps]
    for s, got in zip(obs['obs_seqs'], obs['obs_support']):
        want = brute_support(s, ps, fq, rates, R_out, max_dist=3)
        rel = abs(got - want) / max(want, 1e-300)
        print(f'  {s}  got {got:.6e}  want {want:.6e}  rel {rel:.2e}')
        assert rel < 1e-12, s
    # hand anchor: H sees only B (d=1, A>T transversion) and D (d=1, C>T
    # transition); d=0 to A contributes nothing, and H has no self term.
    ti, tv = rates['T>C'], rates['A>T']
    h_want = tv * 40 / 120 + ti * 20 / 120
    got_h = obs['obs_support'][obs['obs_seqs'].index(H)]
    assert abs(got_h - h_want) / h_want < 1e-12

    # ---- 4. peptide grouping: KPGL (B) is now observed; KPGF groups A and D ----
    i_kpgf = obs['obs_peps'].index('KPGF')
    encs = obs['enc_flat'][obs['pep_off'][i_kpgf]:obs['pep_off'][i_kpgf + 1]]
    assert {obs['obs_seqs'][i] for i in encs} == {A, D}
    assert 'KPGL' in obs['obs_peps']

    # ---- 5. cache separation across thresholds (same cache dir) ----
    cache_dir = os.path.join(tmp, 'cache')
    n_l0 = lambda: len([f for f in os.listdir(cache_dir) if f.startswith('OBS_L0_')])
    assert n_l0() == 1
    cfg2 = load_task(write_task(tmp, 'TOY_PREV2', prev_block(0.20)))
    res2 = ensure_observed_support(cfg2, verbose=False)
    assert res2['pool'].seqs == sorted([A, B])          # D (14.3%) below 0.20
    assert n_l0() == 2, sorted(os.listdir(cache_dir))
    res1b = ensure_observed_support(cfg, verbose=False)  # first config unaffected
    assert np.array_equal(res1b['obs']['obs_support'], obs['obs_support'])
    assert n_l0() == 2
    print(f'  cache files: {sorted(os.listdir(cache_dir))}')

    # ---- 6. config validation ----
    bad1 = write_task(tmp, 'TOY_BAD1', 'pool:\n  mode: bogus\n')
    try:
        load_task(bad1)
        raise AssertionError('expected ValueError for bad pool.mode')
    except ValueError as e:
        print(f'  bad mode rejected: {e}')
    bad2 = write_task(tmp, 'TOY_BAD2', prev_block(1.5))
    try:
        load_task(bad2)
        raise AssertionError('expected ValueError for threshold out of range')
    except ValueError as e:
        print(f'  bad threshold rejected: {e}')

    print('\nALL PREVALENCE-MODE TESTS PASSED')


if __name__ == '__main__':
    main()
