"""Toy verification of the observed-support stage (pulse/support.py).

A 12-nt (4-residue) synthetic dataset with known distances, one synonymous
pair, one stop-codon batch sequence, one NNN-degenerate batch sequence, one
pool member reappearing in the batch (self-support), and one distance-3
sequence that must be excluded at obs_dist_max=2. Support values are checked
against an INDEPENDENT brute-force implementation below.

Run:  python tests/test_observed_support_toy.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import load_task
from pulse.support import ensure_observed_support

# ---------------- toy data ----------------
A = 'AAACCCGGGTTT'   # pool, K P G F        (also reappears in batch: self-support case)
B = 'AAACCCGGGTTA'   # pool, K P G L        (d=1 from A)
C = 'AAACCCGGGTAA'   # batch, K P G *       (d=2 from A, d=1 from B)
D = 'AAACCCGGGTTC'   # batch, K P G F       (synonymous with A: d=1 from A, d=1 from B)
E = 'AAACCCGGGTCC'   # batch, K P G S       (d=2 from A, d=2 from B)
F = 'GGGCCCGGGTTT'   # batch, G P G F       (d=3 from all pool -> excluded at obs_dist_max=2)
H = 'AAANNNGGGTTT'   # batch, K X G F       (degenerate: d=0 to A over valid sites)

ROWS = [  # nuc, pep, 2001, 2002, 2003, unknown
    (A, 'KPGF', 50, 0, 10, 0),   # pool member (2001) AND batch member (2003)
    (B, 'KPGL', 40, 0, 0, 0),    # pool only
    (C, 'KPG*', 0, 10, 0, 0),    # batch (2002 is a batch column: cut is at 2001)
    (D, 'KPGF', 0, 0, 20, 0),
    (E, 'KPGS', 0, 0, 5, 0),
    (F, 'GPGF', 0, 0, 0, 3),     # 'unknown' is always batch
    (H, 'KXGF', 0, 0, 2, 0),
]

def brute_support(y, pool_seqs, freqs, R, R_out, max_dist):
    """Independent reference: sum over parents with 1 <= d <= max_dist of the
    per-site rate product over differing mutually valid sites, times frequency;
    plus the (always-on) self-survival term for exact pool members."""
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


def main():
    tmp = tempfile.mkdtemp()
    train_csv = os.path.join(tmp, 'toy_training_counts.csv')
    pd.DataFrame(ROWS, columns=['nuc', 'pep', '2001', '2002', '2003', 'unknown']
                 ).to_csv(train_csv, index=False)
    bench_csv = os.path.join(tmp, 'toy.csv')
    pd.DataFrame({'mutation': ['A1V'], 'mutated_sequence': ['VPGF'],
                  'DMS_score': [0.5]}).to_csv(bench_csv, index=False)
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
cache:
  cache_dir: {os.path.join(tmp, 'cache')!r}
""")

    cfg = load_task(yaml_path)
    res = ensure_observed_support(cfg, verbose=True)
    pool, obs = res['pool'], res['obs']

    # ---- 1. split and pool ----
    assert res['pool_cols'] == ['2001'], res['pool_cols']
    assert pool.seqs == sorted([A, B]) and abs(pool.total - 90.0) < 1e-9
    freqs = {s: float(pool.freq[i]) for i, s in enumerate(pool.seqs)}
    assert abs(freqs[A] - 50 / 90) < 1e-15 and abs(freqs[B] - 40 / 90) < 1e-15

    # ---- 2. distance filter ----
    kept = set(obs['obs_seqs'])
    assert kept == {A, C, D, E, H}, kept            # F (d=3) excluded; G == A string
    assert obs['obs_excluded_distant'] == [F]

    # ---- 3. support values vs independent brute force ----
    rates = res['rates']
    R_out = {b: sum(v for k, v in rates.items() if k[0] == b) for b in 'ACGT'}
    pool_seqs = sorted([A, B])
    fq = [freqs[s] for s in pool_seqs]
    for s, got in zip(obs['obs_seqs'], obs['obs_support']):
        want = brute_support(s, pool_seqs, fq, rates, R_out, max_dist=3)
        rel = abs(got - want) / max(want, 1e-300)
        print(f'  {s}  got {got:.6e}  want {want:.6e}  rel {rel:.2e}')
        assert rel < 1e-12, s
    # hand anchors: D = ti*5/9 + tv*4/9; H = only B at d=1 over valid sites
    ti, tv = rates['T>C'], rates['A>T']
    assert abs(ti - 1.8e-4) < 1e-15 and abs(tv - 1e-5) < 1e-15
    d_want = ti * 5 / 9 + 1e-5 * 4 / 9
    h_want = tv * 4 / 9
    got_d = obs['obs_support'][obs['obs_seqs'].index(D)]
    got_h = obs['obs_support'][obs['obs_seqs'].index(H)]
    assert abs(got_d - d_want) / d_want < 1e-12 and abs(got_h - h_want) / h_want < 1e-12

    # ---- 4. peptide grouping: KPGF must group the A-string and D encodings ----
    i_kpgf = obs['obs_peps'].index('KPGF')
    encs = obs['enc_flat'][obs['pep_off'][i_kpgf]:obs['pep_off'][i_kpgf + 1]]
    assert {obs['obs_seqs'][i] for i in encs} == {A, D}
    assert obs['obs_peps'] == ['KPG*', 'KPGF', 'KPGS', 'KXGF'], obs['obs_peps']

    # ---- 5. peptide aggregation: sum over encodings, self terms included ----
    self_term = (1 - 2e-4) ** 12 * 5 / 9
    kpgf_want = self_term + tv * 4 / 9 + d_want   # S(A-string) + S(D)
    pep_sup = obs['obs_pep_support']
    got_kpgf = pep_sup[i_kpgf]
    print(f'  S(KPGF) got {got_kpgf:.6e}  want {kpgf_want:.6e}')
    assert abs(got_kpgf - kpgf_want) / kpgf_want < 1e-12
    for pi, p in enumerate(obs['obs_peps']):      # every peptide vs its encodings
        encs_pi = obs['enc_flat'][obs['pep_off'][pi]:obs['pep_off'][pi + 1]]
        want = obs['obs_support'][encs_pi].sum()
        assert abs(pep_sup[pi] - want) <= 1e-12 * max(want, 1e-300), p

    # ---- 6. support_max_dist=1: distance-2 parents drop out ----
    cfg_d1 = load_task(yaml_path, overrides={'observation': {'support_max_dist': 1}})
    obs_d1 = ensure_observed_support(cfg_d1, verbose=False)['obs']
    c_d1 = obs_d1['obs_support'][obs_d1['obs_seqs'].index(C)]
    e_d1 = obs_d1['obs_support'][obs_d1['obs_seqs'].index(E)]
    assert abs(c_d1 - tv * 4 / 9) / (tv * 4 / 9) < 1e-12    # A's d=2 term gone
    assert e_d1 == 0.0                                      # d=2 from all parents

    # ---- 7. cache round-trip: second call loads identical values ----
    res2 = ensure_observed_support(cfg, verbose=False)
    assert np.array_equal(res2['obs']['obs_support'], obs['obs_support'])
    cache_files = sorted(os.listdir(os.path.join(tmp, 'cache')))
    assert any(f.startswith('OBS_L0_') for f in cache_files)
    assert any(f.startswith('OBS_L2_') for f in cache_files)
    print(f'  cache files: {cache_files}')

    print('\nALL TOY TESTS PASSED')


if __name__ == '__main__':
    main()
