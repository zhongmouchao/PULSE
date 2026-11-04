"""generate_support.py — the one-time observed-support precompute stage.

Runs ensure_observed_support for one task (or all tasks) over a grid of
config combinations. Every combination is cached independently
(OBS_L0 neighbor pass / OBS_L2 observed support), so rerunning a grid only
computes what is new. Training (pulse/train_sgd.py) loads these caches.

Usage:
    python pulse/generate_support.py --task SARS_viral_growth
    python pulse/generate_support.py --task SARS_viral_growth --sweep sweeps/default_grid.yaml
    python pulse/generate_support.py --task X --set pool.fraction=0.1 --set observation.obs_dist_max=3
    python pulse/generate_support.py --all --sweep sweeps/default_grid.yaml
"""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from pulse.task_config import load_task, list_tasks
from pulse.support import ensure_observed_support


def parse_set(values):
    out = {}
    for kv in values or []:
        key, _, val = kv.partition('=')
        if not key or not _:
            raise ValueError(f'--set expects section.key=value, got {kv!r}')
        parts = key.split('.')
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


def deep_merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def grid_combos(grid):
    """Cartesian product of a {'section.key': [values]} dict -> list of override dicts."""
    keys = sorted(grid)
    combos = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        over = {}
        for k, v in zip(keys, vals):
            parts = k.split('.')
            d = over
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
        combos.append(over)
    return combos


def combo_label(over):
    parts = []
    def walk(d, pre=''):
        for k in sorted(d):
            if isinstance(d[k], dict):
                walk(d[k], pre + k + '.')
            else:
                parts.append(f'{pre}{k}={d[k]}')
    walk(over)
    return ', '.join(parts) if parts else '(defaults)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', help='task name (tasks/<name>.yaml)')
    ap.add_argument('--all', action='store_true', help='run every task in tasks/')
    ap.add_argument('--sweep', help='YAML file with a `grid:` dict of dotted-key lists')
    ap.add_argument('--set', action='append', default=[], help='section.key=value overrides')
    args = ap.parse_args()

    if not args.task and not args.all:
        ap.error('give --task NAME or --all')

    sweep_overrides = [parse_set(args.set)]
    if args.sweep:
        spec = yaml.safe_load(open(args.sweep)) or {}
        grid, defaults = spec.get('grid') or {}, spec.get('defaults') or {}
        both = sorted(set(grid) & set(defaults))
        if both:
            ap.error(f'sweep YAML: key(s) in both grid and defaults: {both}')
        # `defaults` are fixed values applied to every combo — folded in as
        # single-element grid dimensions, so they do NOT multiply runs but DO
        # appear in combo_label / run_tag (no result collisions across values).
        # `grid` values may also be scalars; they are auto-wrapped into lists.
        grid = {k: (v if isinstance(v, list) else [v])
                for k, v in {**defaults, **grid}.items()}
        sweep_overrides = []
        for combo in grid_combos(grid):
            merged = parse_set(args.set)
            deep_merge(merged, combo)
            sweep_overrides.append(merged)

    tasks = list_tasks() if args.all else [args.task]
    print(f'tasks: {tasks}; combinations per task: {len(sweep_overrides)}')

    for task in tasks:
        for over in sweep_overrides:
            label = combo_label(over)
            t0 = time.time()
            try:
                cfg = load_task(task, overrides=over)
                res = ensure_observed_support(cfg, verbose=False)
                obs = res['obs']
                n_zero = int((obs['obs_support'] == 0).sum())
                print(f'[{task}] {label}\n  -> pool {res["pool"].n} parents | '
                      f'{len(obs["obs_seqs"])} observed kept '
                      f'({len(obs["obs_excluded_distant"])} distance-excluded'
                      f', {len(obs.get("obs_excluded_degenerate", []))} degenerate-excluded), '
                      f'{len(obs["obs_peps"])} observed peptides, '
                      f'{n_zero} at zero support | {time.time() - t0:.0f}s')
            except Exception as e:
                print(f'[{task}] {label}\n  -> FAILED: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
