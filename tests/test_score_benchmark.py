"""Smoke test for pulse/score_benchmark.py (local scoring of NEW benchmark
files from a saved checkpoint, without training data or caches).

Builds a deterministic toy checkpoint (model state + embedded config), runs
the CLI on (1) a standard DMS benchmark and (2) an activity benchmark with
non-standard column names via CLI overrides, and checks the written p_a
scores equal an in-process forward pass.

Run:  python tests/test_score_benchmark.py
"""
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import load_task
from pulse.featurizers import build_featurizer
from pulse.models import build_model

ROWS = [  # tiny training file — only needed so the config validates
    ('AAACCCGGGTTT', 'KPGF', 50, 10),
    ('AAACCCGGGTTA', 'KPGL', 40, 0),
    ('AAACCCGGGTAA', 'KPG*', 0, 10),
]


def main():
    tmp = tempfile.mkdtemp()
    train_csv = os.path.join(tmp, 'toy_training_counts.csv')
    pd.DataFrame(ROWS, columns=['nuc', 'pep', '2001', '2002']).to_csv(train_csv, index=False)
    yaml_path = os.path.join(tmp, 'TOY.yaml')
    open(yaml_path, 'w').write(f"""task: TOY
data:
  training_file: {train_csv!r}
  benchmark_file: {train_csv!r}
  nuc_column: nuc
  pep_column: pep
""")
    cfg = load_task(yaml_path)
    feat = build_featurizer(cfg)

    torch.manual_seed(7)
    model = build_model(cfg, feat.featurize(['KPGF']).shape[1])
    ckpt_path = os.path.join(tmp, 'TOYCKPT.pt')
    torch.save({'run_tag': 'TOYCKPT', 'model_state': model.state_dict(),
                'd_raw': torch.tensor([0.3]), 'b_raw': torch.tensor([0.1]),
                'config': asdict(cfg)}, ckpt_path)

    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'pulse', 'score_benchmark.py')

    # ---- 1. DMS benchmark, standard columns ----
    dms_csv = os.path.join(tmp, 'new_dms.csv')
    dms = pd.DataFrame({'mutation': ['K1V', 'P2A', 'G3R', 'F4L'],
                        'mutated_sequence': ['VPGF', 'KAGF', 'KPRF', 'KPGL'],
                        'DMS_score': [0.9, 0.1, 0.4, 0.8]})
    dms.to_csv(dms_csv, index=False)
    r = subprocess.run([sys.executable, script, ckpt_path, '--benchmark', dms_csv],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    print(r.stdout)
    out1 = ckpt_path.replace('.pt', '__new_dms_predictions.csv')
    assert os.path.exists(out1), os.listdir(tmp)
    got = pd.read_csv(out1)
    peps = dms['mutated_sequence'].tolist()
    with torch.no_grad():
        want = model(torch.tensor(feat.featurize(peps))).squeeze(-1).numpy()
    assert np.allclose(got['p_a'].to_numpy(), want, atol=1e-6), 'p_a mismatch vs in-process'
    assert list(got['bench_score']) == [0.9, 0.1, 0.4, 0.8]
    print('PASS dms: predictions match in-process forward pass')

    # ---- 2. activity benchmark, non-standard columns + CLI overrides ----
    act_csv = os.path.join(tmp, 'new_act.csv')
    pd.DataFrame({'seq': ['KPGF', 'KPGL', 'KPRF'],
                  'label': [1, 0, 0]}).to_csv(act_csv, index=False)
    r = subprocess.run([sys.executable, script, ckpt_path, '--benchmark', act_csv,
                        '--type', 'activity', '--seq-col', 'seq', '--score-col', 'label'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    print(r.stdout)
    out2 = ckpt_path.replace('.pt', '__new_act_predictions.csv')
    got2 = pd.read_csv(out2)
    with torch.no_grad():
        want2 = model(torch.tensor(feat.featurize(['KPGF', 'KPGL', 'KPRF']))).squeeze(-1).numpy()
    assert np.allclose(got2['p_a'].to_numpy(), want2, atol=1e-6)
    print('PASS activity: CLI column overrides work, predictions match')

    # ---- 3. --save-results: appends a suffixed row, idempotent on re-run ----
    res_csv = os.path.join(tmp, 'results', 'results.csv')
    cmd = [sys.executable, script, ckpt_path, '--benchmark', dms_csv,
           '--save-results', '--results-csv', res_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(res_csv)
    assert len(df) == 1 and df['task'][0] == 'TOY'
    assert df['run_tag'][0] == 'TOYCKPT ++bench=new_dms', df['run_tag'][0]
    assert abs(df['auc'][0] - 1.0) < 1e-12 and abs(df['spearman'][0] - 1.0) < 1e-12
    assert df['model'][0] == cfg.model.classifier and df['n_benchmark'][0] == 4
    r = subprocess.run(cmd, capture_output=True, text=True)      # re-run replaces,
    assert r.returncode == 0, r.stderr                           # not duplicates
    df2 = pd.read_csv(res_csv)
    assert len(df2) == 1, f're-run duplicated the row: {len(df2)} rows'
    print('PASS --save-results: row appended, re-run replaces instead of duplicating')

    # ---- 4. --scan: every matching checkpoint scored, labelled by its tag ----
    mdir = os.path.join(tmp, 'models')
    os.makedirs(mdir)
    torch.manual_seed(11)
    mb = build_model(cfg, feat.featurize(['KPGF']).shape[1])
    torch.save({'run_tag': 'TOYSCAN_A', 'model_state': model.state_dict(),
                'd_raw': torch.tensor([0.3]), 'b_raw': torch.tensor([0.1]),
                'config': asdict(cfg)}, os.path.join(mdir, 'TOYSCAN_A.pt'))
    torch.save({'run_tag': 'TOYSCAN_B', 'model_state': mb.state_dict(),
                'd_raw': torch.tensor([-0.2]), 'b_raw': torch.tensor([0.0]),
                'config': asdict(cfg)}, os.path.join(mdir, 'TOYSCAN_B.pt'))
    torch.save({'run_tag': 'OTHER', 'model_state': mb.state_dict(),
                'config': asdict(cfg)}, os.path.join(mdir, 'OTHER.pt'))  # no match
    res_csv2 = os.path.join(tmp, 'results2', 'results.csv')
    r = subprocess.run([sys.executable, script, '--scan', 'TOYSCAN',
                        '--models-dir', mdir, '--benchmark', dms_csv,
                        '--save-results', '--results-csv', res_csv2],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    print(r.stdout)
    df3 = pd.read_csv(res_csv2)
    assert sorted(df3['run_tag']) == ['TOYSCAN_A ++bench=new_dms',
                                      'TOYSCAN_B ++bench=new_dms'], df3['run_tag'].tolist()
    for t in ('TOYSCAN_A', 'TOYSCAN_B'):
        assert os.path.exists(os.path.join(mdir, f'{t}__new_dms_predictions.csv')), t
    r = subprocess.run([sys.executable, script, '--scan', 'NOPE',
                        '--models-dir', mdir, '--benchmark', dms_csv],
                       capture_output=True, text=True)
    assert r.returncode != 0     # empty scan is a clean error, not a silent pass
    print('PASS --scan: matches scored and labelled, non-matches skipped, '
          'empty scan errors out')

    print('\nALL SCORE_BENCHMARK TESTS PASSED')


if __name__ == '__main__':
    main()
