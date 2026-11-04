"""Toy verification of the degenerate-observed filter
(observation.max_degenerate_frac).

Same 12-nt (4-residue) temporal toy dataset as test_observed_support_toy.py.
H ('KXGF', 1 of 4 residues = 25% X) must be excluded when the filter is on at
0.01, while C ('KPG*' — a stop codon is NOT degenerate) must be kept. With the
filter off (None, the default) behavior and cache keys are bit-identical to
the historical pipeline.

Run:  python tests/test_degenerate_filter.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import load_task
from pulse.support import ensure_observed_support, obs_key2_parts

A = 'AAACCCGGGTTT'   # K P G F  pool + batch
B = 'AAACCCGGGTTA'   # K P G L  pool only
C = 'AAACCCGGGTAA'   # K P G *  batch (stop codon is NOT degenerate: kept)
D = 'AAACCCGGGTTC'   # K P G F  batch
E = 'AAACCCGGGTCC'   # K P G S  batch
F = 'GGGCCCGGGTTT'   # G P G F  batch, d=3 from all pool -> distance-excluded
H = 'AAANNNGGGTTT'   # K X G F  batch, 1/4 = 25% degenerate -> filtered at 0.01

ROWS = [
    (A, 'KPGF', 50, 0, 10, 0),
    (B, 'KPGL', 40, 0, 0, 0),
    (C, 'KPG*', 0, 10, 0, 0),
    (D, 'KPGF', 0, 0, 20, 0),
    (E, 'KPGS', 0, 0, 5, 0),
    (F, 'GPGF', 0, 0, 0, 3),
    (H, 'KXGF', 0, 0, 2, 0),
]
COLS = ['nuc', 'pep', '2001', '2002', '2003', 'unknown']


def write_task(tmp, name, mdf_line=''):
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
pool:
  fraction: 0.01
observation:
  obs_dist_max: 2
  support_max_dist: 3
{mdf_line}cache:
  cache_dir: {os.path.join(tmp, 'cache')!r}
""")
    return yaml_path


def main():
    tmp = tempfile.mkdtemp()

    # ---- 0. key ingredients: off (None) is bit-identical to historical ----
    cfg_off = SimpleNamespace(observation=SimpleNamespace(
        obs_dist_max=2, support_max_dist=3, max_degenerate_frac=None))
    parts_off = obs_key2_parts(cfg_off, 'K0', {'A>C': 1e-5})
    assert 'deg_frac' not in parts_off, parts_off
    cfg_on = SimpleNamespace(observation=SimpleNamespace(
        obs_dist_max=2, support_max_dist=3, max_degenerate_frac=0.01))
    parts_on = obs_key2_parts(cfg_on, 'K0', {'A>C': 1e-5})
    assert parts_on['deg_frac'] == 0.01 and \
        all(parts_on[k] == parts_off[k] for k in parts_off), parts_on
    print('  key parts: off is historical, on adds deg_frac')

    # ---- 1. filter OFF: H kept (historical behavior) ----
    cfg0 = load_task(write_task(tmp, 'TOY_OFF'))
    obs0 = ensure_observed_support(cfg0, verbose=False)['obs']
    assert set(obs0['obs_seqs']) == {A, C, D, E, H}
    assert obs0['obs_excluded_distant'] == [F]
    assert obs0['obs_excluded_degenerate'] == []
    h_off = obs0['obs_support'][obs0['obs_seqs'].index(H)]

    # ---- 2. filter ON at 0.01: H excluded, stop-codon C kept, rest unchanged ----
    cfg1 = load_task(write_task(tmp, 'TOY_ON',
                                '  max_degenerate_frac: 0.01\n'))
    obs1 = ensure_observed_support(cfg1, verbose=True)['obs']
    assert set(obs1['obs_seqs']) == {A, C, D, E}, set(obs1['obs_seqs'])
    assert obs1['obs_excluded_degenerate'] == [H], obs1['obs_excluded_degenerate']
    assert obs1['obs_excluded_distant'] == [F]      # distance filter unaffected
    assert H not in obs1['obs_seqs']
    # every remaining sequence keeps the exact support it had with the filter off
    for s in (A, C, D, E):
        v0 = obs0['obs_support'][obs0['obs_seqs'].index(s)]
        v1 = obs1['obs_support'][obs1['obs_seqs'].index(s)]
        assert abs(v0 - v1) <= 1e-12 * max(v0, 1e-300), s
    print(f'  H excluded (had support {h_off:.3e} when kept); C (stop codon) kept; '
          f'all other supports unchanged')

    # ---- 3. cache separation: on/off are different OBS_L2 files, off reloads ----
    cache_dir = os.path.join(tmp, 'cache')
    l2_files = lambda: sorted(f for f in os.listdir(cache_dir)
                              if f.startswith('OBS_L2_'))
    assert len(l2_files()) == 2, l2_files()
    obs0b = ensure_observed_support(cfg0, verbose=False)['obs']
    assert np.array_equal(obs0b['obs_support'], obs0['obs_support'])
    assert len(l2_files()) == 2
    print(f'  cache files: {l2_files()}')

    # ---- 4. validation ----
    for bad, val in [('TOY_BAD1', '1.5'), ('TOY_BAD2', '-0.1')]:
        p = write_task(tmp, bad, f'  max_degenerate_frac: {val}\n')
        try:
            load_task(p)
            raise AssertionError(f'expected ValueError for {val}')
        except ValueError as e:
            print(f'  rejected {val}: {e}')

    print('\nALL DEGENERATE-FILTER TESTS PASSED')


if __name__ == '__main__':
    main()
