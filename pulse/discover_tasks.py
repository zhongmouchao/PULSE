"""Discover new tasks by pairing files in the training and evaluation folders.

Folder names are auto-detected (first existing of):
  training:   training_data/, Training_Data/
  evaluation: evaluation_data/, Evaluation_Data/, Benchmarks/

A pair = a training counts CSV whose stem matches an evaluation CSV stem after
stripping a training suffix ('_training_counts', '_training', '_counts').
For every pair not yet referenced by an existing tasks/*.yaml, a minimal task
config is auto-created (columns and benchmark type are inferred from the CSV
headers; everything else takes the loader defaults).

Existing task configs are NEVER overwritten — customize them freely.

CLI:
    python pulse/discover_tasks.py             # scan and create new configs
    python pulse/discover_tasks.py --dry-run   # report only, write nothing
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import PROJECT_ROOT, TASKS_DIR, _read_yaml

TRAIN_DIR_CANDIDATES = ('data', 'training_data', 'Training_Data')
BENCH_DIR_CANDIDATES = ('data', 'evaluation_data', 'Evaluation_Data', 'Benchmarks')
TRAIN_SUFFIXES = ('_training_counts', '_training', '_counts')

DMS_COLS = ('mutation', 'mutated_sequence', 'DMS_score')
ACTIVITY_COLS = ('aa_sequence', 'activity')


def default_dirs():
    """(training_dir, evaluation_dir): first existing candidates, matched
    case-insensitively but returned with the TRUE on-disk casing (matters when
    configs move between macOS and the case-sensitive cluster); if neither
    exists yet, fall back to the preferred (lowercase) names."""
    entries = {}
    for e in os.listdir(PROJECT_ROOT):
        p = os.path.join(PROJECT_ROOT, e)
        if os.path.isdir(p):
            entries.setdefault(e.lower(), e)
    def pick(cands):
        for c in cands:
            if c.lower() in entries:
                return os.path.join(PROJECT_ROOT, entries[c.lower()])
        return os.path.join(PROJECT_ROOT, cands[0])
    return pick(TRAIN_DIR_CANDIDATES), pick(BENCH_DIR_CANDIDATES)


TRAIN_DIR, BENCH_DIR = default_dirs()


def _bench_stems_for(train_stem):
    yield train_stem
    for suf in TRAIN_SUFFIXES:
        if train_stem.endswith(suf):
            yield train_stem[:-len(suf)]


def find_pairs(train_dir=TRAIN_DIR, bench_dir=BENCH_DIR):
    trains = {os.path.splitext(f)[0]: f for f in os.listdir(train_dir)
              if f.endswith('.csv')} if os.path.isdir(train_dir) else {}
    benches = {os.path.splitext(f)[0]: f for f in os.listdir(bench_dir)
               if f.endswith('.csv')} if os.path.isdir(bench_dir) else {}
    print(f'  {os.path.relpath(train_dir)}: {len(trains)} CSVs; '
          f'{os.path.relpath(bench_dir)}: {len(benches)} CSVs')
    if not trains:
        print(f'  WARNING: no training CSVs found in {train_dir} '
              f'(missing or empty folder?)')
    if not benches:
        print(f'  WARNING: no benchmark CSVs found in {bench_dir} '
              f'(missing or empty folder?)')
    pairs, lonely_trains = [], []
    used_benches = set()
    for t_stem, t_file in sorted(trains.items()):
        hit = next((b for b in _bench_stems_for(t_stem) if b in benches), None)
        if hit is None:
            lonely_trains.append(t_file)
        else:
            pairs.append((t_file, benches[hit], hit))
            used_benches.add(hit)
    lonely_benches = [f for s, f in sorted(benches.items()) if s not in used_benches]
    return pairs, lonely_trains, lonely_benches


def configured_files(tasks_dir=TASKS_DIR):
    """(training_file, benchmark_file) pairs already referenced by task configs."""
    used = set()
    if os.path.isdir(tasks_dir):
        for f in os.listdir(tasks_dir):
            if not f.endswith('.yaml'):
                continue
            try:
                d = _read_yaml(os.path.join(tasks_dir, f)) or {}
            except Exception as e:
                print(f'  warning: could not parse {f}: {e}')
                continue
            data = d.get('data') or {}
            used.add((data.get('training_file', ''), data.get('benchmark_file', '')))
    return used


def _rel(dirpath, filename):
    """Path of filename relative to the project root (as recorded in task configs)."""
    return os.path.relpath(os.path.join(dirpath, filename), PROJECT_ROOT)


BENCH_METRICS = {'dms': 'AUC + AP + Spearman', 'activity': 'AUC + AP'}


def detect_bench_type(be_columns, bench_file=''):
    """'dms' (continuous scores -> AUC + AP + Spearman) or 'activity'
    (+/- labels -> AUC + AP), from the CSV header."""
    if all(c in be_columns for c in DMS_COLS):
        return 'dms'
    if all(c in be_columns for c in ACTIVITY_COLS):
        return 'activity'
    raise ValueError(f'{bench_file}: unrecognized benchmark columns {list(be_columns)}; '
                     f'expected {DMS_COLS} (dms) or {ACTIVITY_COLS} (activity)')


def infer_task_yaml(train_file, bench_file, task_name,
                    train_dir=TRAIN_DIR, bench_dir=BENCH_DIR):
    """Minimal task config text; columns/benchmark type inferred from headers.
    Returns (yaml_text, bench_type)."""
    tr = pd.read_csv(os.path.join(train_dir, train_file), nrows=2)
    be = pd.read_csv(os.path.join(bench_dir, bench_file), nrows=2)

    def _pick(cols, *keys, exclude=None):
        for c in cols:
            if exclude is not None and c == exclude:
                continue
            if any(k in c.lower() for k in keys):
                return c
        return None

    nuc_col = _pick(tr.columns, 'nucleotide', 'nuc_seq') or tr.columns[0]
    pep_col = _pick(tr.columns, 'peptide', 'pep', 'aa_seq', exclude=nuc_col) or tr.columns[1]

    bench_type = detect_bench_type(be.columns, bench_file)
    if bench_type == 'dms':
        bench_block = """    type: dms
    mutation_column: mutation
    sequence_column: mutated_sequence
    score_column: DMS_score
    functional_threshold: median"""
    else:
        bench_block = """    type: activity
    sequence_column: aa_sequence"""

    # SARS-CoV-2 tasks get the SARS-CoV-2 mutational spectrum; everything else
    # takes the default ti90_tv5. Edit the YAML to override per task.
    rates_block = """
mutation_model:
  mu_total: 0.0002
  rates: sars_cov2
""" if task_name.startswith('SARS') else ''

    # Production training settings baked into every auto-generated config, so a
    # discovered task is runnable as-is (overnight-safe); all other knobs take
    # the loader defaults. Edit the YAML only for special studies.
    prod_block = """
# ---- production defaults: runnable as-is, no tuning required ----
training:
  num_sgd_steps: 8000     # overnight-safe cap at batch 128; early stopping usually ends sooner
  early_stop_checks: 100  # stop after 100 checks x 10 steps = 1000 steps without improvement
  print_interval: 100
"""

    return f"""# Auto-generated by pulse/discover_tasks.py — edit freely, never overwritten.
task: {task_name}
description: >
  TODO: describe this task.

data:
  training_file: {_rel(train_dir, train_file)}
  benchmark_file: {_rel(bench_dir, bench_file)}
  nuc_column: {nuc_col!r}
  pep_column: {pep_col!r}
  benchmark:
{bench_block}
{rates_block}{prod_block}""", bench_type


def discover(write=True, train_dir=TRAIN_DIR, bench_dir=BENCH_DIR, tasks_dir=TASKS_DIR):
    print(f'scanning {os.path.relpath(train_dir)}/ and {os.path.relpath(bench_dir)}/')
    pairs, lonely_trains, lonely_benches = find_pairs(train_dir, bench_dir)
    used = configured_files(tasks_dir)
    used_train = {u[0] for u in used} | {u[1] for u in used}

    print(f'pairs found: {len(pairs)}')
    created, skipped = [], []
    for train_file, bench_file, bench_stem in pairs:
        rel_t = _rel(train_dir, train_file)
        rel_b = _rel(bench_dir, bench_file)
        bench_type = detect_bench_type(
            pd.read_csv(os.path.join(bench_dir, bench_file), nrows=2).columns, bench_file)
        tag = f'[{bench_type}] -> metrics: {BENCH_METRICS[bench_type]}'
        if rel_t in used_train or rel_b in used_train:
            skipped.append((train_file, bench_file))
            print(f'  already configured: {bench_stem}  {tag}')
            continue
        text, _ = infer_task_yaml(train_file, bench_file, bench_stem, train_dir, bench_dir)
        path = os.path.join(tasks_dir, bench_stem + '.yaml')
        if write:
            if os.path.exists(path):
                skipped.append((train_file, bench_file))
                print(f'  already configured: {bench_stem}  {tag}')
                continue
            os.makedirs(tasks_dir, exist_ok=True)
            with open(path, 'w') as f:
                f.write(text)
        created.append((bench_stem, path))
        print(f'  {"would create" if not write else "NEW task"}: {bench_stem}  {tag}')

    for f in lonely_trains:
        print(f'  training file with NO matching benchmark: {f}')
    for f in lonely_benches:
        print(f'  benchmark with NO matching training file: {f}')
    n_existing = len([f for f in os.listdir(tasks_dir) if f.endswith('.yaml')]) \
        if os.path.isdir(tasks_dir) else 0
    print(f'summary: {len(pairs)} pairs ({len(created)} new, {len(skipped)} already configured), '
          f'{n_existing} task configs total, '
          f'{len(lonely_trains)} unmatched training / {len(lonely_benches)} unmatched benchmark files')
    return created, skipped, lonely_trains, lonely_benches


if __name__ == '__main__':
    discover(write=('--dry-run' not in sys.argv))
