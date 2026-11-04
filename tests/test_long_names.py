"""Verification of bounded_filename() and the evaluate.py legacy-name fallback.

Why: a 7-knob run tag (~224 chars after safe_tag) plus the
'__<benchmark>_predictions.csv' suffix exceeds the 255-byte filename limit and
crashes with OSError [Errno 36]. bounded_filename caps names at NAME_BUDGET
(250) with a 10-char md5 of the FULL tag when needed, and is bit-identical to
the old safe_tag+suffix scheme for every tag that already fit — so existing
model/prediction files keep their names.

Run:  python tests/test_long_names.py
"""
import os
import subprocess
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import NAME_BUDGET, bounded_filename, safe_tag

failures = []


def check(cond, msg):
    print(('  ok  ' if cond else '  FAIL') + ' ' + msg)
    if not cond:
        failures.append(msg)


# 1) bit-compat: any tag whose legacy name fits keeps the legacy name
tag = 'TASK_mutation_model.rates_ti90_tv5_observation.obs_dist_max_1_pool.fraction_0.01'
for suffix in ('.pt', '_predictions.csv', '__BENCH_predictions.csv'):
    check(bounded_filename(tag, suffix) == safe_tag(tag) + suffix,
          f'short tag + {suffix!r} keeps legacy name')

# the real failing tag from the Errno 36 report
real = ('SARS_binding_Wuhan_Hu_1_RBD_mutation_model.rates_sars_cov2_'
        'observation.max_degenerate_frac_0.01_observation.obs_dist_max_3_'
        'pool.mode_prevalence_pool.prevalence_threshold_0.001_'
        'proposal.n_max_3_training.early_stop_checks_50_')
name = bounded_filename(real, '__SARS_expression_Wuhan_Hu_1_RBD_predictions.csv')
check(len(name.encode()) <= NAME_BUDGET, f'real failing combo bounded to {len(name)} chars')
check(name.endswith('__SARS_expression_Wuhan_Hu_1_RBD_predictions.csv'),
      'suffix preserved on bounded name')

# 2) very long tag: bounded, deterministic, suffix intact
long_tag = 'x' * 300
a = bounded_filename(long_tag, '_predictions.csv')
b = bounded_filename(long_tag, '_predictions.csv')
check(len(a.encode()) <= NAME_BUDGET, f'300-char tag bounded to {len(a)} chars')
check(a == b, 'bounded name is deterministic')
check(a.endswith('_predictions.csv'), 'suffix preserved')
check(a != safe_tag(long_tag) + '_predictions.csv', 'hashed name differs from (illegal) legacy name')

# 3) two tags differing only AFTER the truncation point get distinct names
t1 = 'a' * 240 + '_FIRST'
t2 = 'a' * 240 + '_SECOND'
n1, n2 = bounded_filename(t1, '_predictions.csv'), bounded_filename(t2, '_predictions.csv')
check(n1 != n2, 'tags identical through char 240 still map to distinct names')
check(len(n1) <= NAME_BUDGET and len(n2) <= NAME_BUDGET, 'both within budget')

# 4) evaluate.py fallback: a legacy file in the 251..255-char band (legal on
# disk, but over the new budget -> hashed name would miss it) must still load.
legacy_tag = 't' * 238                      # legacy name: 238 + 16 = 254 chars
with tempfile.TemporaryDirectory() as tmp:
    legacy_name = safe_tag(legacy_tag) + '_predictions.csv'
    assert 250 < len(legacy_name) <= 255
    pd.DataFrame({'bench_score': [0.1, 0.2, 0.9, 0.8],
                  'p_a': [0.1, 0.3, 0.7, 0.9]}).to_csv(
        os.path.join(tmp, legacy_name), index=False)
    proc = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'pulse', 'evaluate.py'),
         legacy_tag, '--models-dir', tmp],
        capture_output=True, text=True)
    check(proc.returncode == 0, f'evaluate.py ran (stderr: {proc.stderr.strip()[:200]})')
    check('no predictions file' not in proc.stdout,
          'legacy-named predictions file found via fallback')
    check('Spearman' in proc.stdout, 'metrics printed for legacy file')

print()
if failures:
    print(f'FAILED ({len(failures)} checks)')
    sys.exit(1)
print('PASS')
