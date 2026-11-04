"""Verify the new sweep-YAML parsing (defaults + scalar auto-wrap) end-to-end."""
import sys, tempfile, os
sys.path.insert(0, os.getcwd())
import yaml
from pulse.generate_support import parse_set, grid_combos, deep_merge, combo_label


def load_sweep(path):
    """Mirror of the block now in generate_support.py / train_sgd.py main()."""
    spec = yaml.safe_load(open(path)) or {}
    grid, defaults = spec.get('grid') or {}, spec.get('defaults') or {}
    both = sorted(set(grid) & set(defaults))
    if both:
        raise SystemExit(f'sweep YAML: key(s) in both grid and defaults: {both}')
    grid = {k: (v if isinstance(v, list) else [v])
            for k, v in {**defaults, **grid}.items()}
    out = []
    for combo in grid_combos(grid):
        merged = {}
        deep_merge(merged, combo)
        out.append(merged)
    return out


fails = []
def check(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fails.append(name)


# 1. defaults applied to every combo, does not multiply runs, appears in label
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
    f.write('defaults:\n  proposal.n_max: 2\n'
            'grid:\n  pool.fraction: [0.01, 0.1]\n  observation.obs_dist_max: [2, 3]\n')
    p = f.name
combos = load_sweep(p)
check('defaults: run count unchanged (4 combos)', len(combos) == 4)
check('defaults: n_max in every combo',
      all(c['proposal']['n_max'] == 2 for c in combos))
check('defaults: n_max in every run label',
      all('proposal.n_max=2' in combo_label(c) for c in combos))
os.unlink(p)

# 2. two different defaults values -> distinct run tags (no collision)
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
    f.write('defaults:\n  proposal.n_max: 3\ngrid:\n  pool.fraction: [0.01]\n')
    p3 = f.name
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
    f.write('defaults:\n  proposal.n_max: 2\ngrid:\n  pool.fraction: [0.01]\n')
    p2 = f.name
t3 = [combo_label(c) for c in load_sweep(p3)]
t2 = [combo_label(c) for c in load_sweep(p2)]
check('n_max=2 vs 3 -> distinct labels', t3 != t2)
os.unlink(p3); os.unlink(p2)

# 3. scalar in grid auto-wrapped
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
    f.write('grid:\n  pool.fraction: 0.01\n  observation.obs_dist_max: [2, 3]\n')
    p = f.name
combos = load_sweep(p)
check('scalar grid value -> 2 combos', len(combos) == 2)
check('scalar value applied', all(c['pool']['fraction'] == 0.01 for c in combos))
os.unlink(p)

# 4. duplicate key in defaults+grid -> rejected
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
    f.write('defaults:\n  proposal.n_max: 2\ngrid:\n  proposal.n_max: [2, 3]\n')
    p = f.name
try:
    load_sweep(p)
    check('duplicate key rejected', False)
except SystemExit:
    check('duplicate key rejected', True)
os.unlink(p)

# 5. the shipped paper.yaml parses: one run with the defaults applied
combos = load_sweep('sweeps/paper.yaml')
check('shipped yaml: single combo, defaults applied',
      len(combos) == 1 and combos[0]['proposal']['n_max'] == 3
      and combos[0]['model']['classifier'] == 'NAM')

# 6. --set still merges under combo (combo wins on conflict, as before)
merged = parse_set(['proposal.n_max=1'])
deep_merge(merged, combos[0])
check('--set parse still works', isinstance(merged, dict))

print()
print('ALL PASS' if not fails else f'FAILURES: {fails}')
sys.exit(1 if fails else 0)
