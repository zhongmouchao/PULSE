"""diagnose_draw.py — decompose the per-draw cost of the SGD sampler on a real
task config. Prints pool density stats and, for a few draws, exactly where the
time goes: q-scan, encoding enumeration, support patching.

Usage (from the repository root):
    python3 diagnose_draw.py SARS_binding_Eta_RBD observation.obs_dist_max=1 pool.fraction=0.01

No pipeline code is modified; the timing wraps the sampler methods at runtime.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pulse.task_config import load_task
from pulse.generate_support import parse_set
from pulse.support import ensure_observed_support
from pulse.sampler import Sampler


def main():
    task = sys.argv[1]
    over = parse_set(sys.argv[2:])
    cfg = load_task(task, overrides=over)

    t = time.time()
    res = ensure_observed_support(cfg, verbose=False)
    print(f'OBS cache load: {time.time() - t:.0f}s', flush=True)
    t = time.time()
    s = Sampler(cfg, res, verbose=True)
    print(f'sampler init: {time.time() - t:.0f}s', flush=True)

    # ---- pool density stats (the structural drivers of per-draw cost) ----
    mult = np.diff(s.pep_mem['off'])
    aa_deg = np.diff(s.aa_nei['off'])
    nd = np.diff(s.nuc_nei['off'])
    print(f'pool parents {len(res["pool"].seqs)}, pool peptides {len(s.pep_mat)}')
    print(f'nuc/peptide:            mean {mult.mean():.1f}  max {mult.max()}')
    print(f'AA neighbors/peptide (<= {2 * s.n_max}):   mean {aa_deg.mean():.1f}  max {aa_deg.max()}')
    print(f'nuc neighbors/parent (<= {s.n_max + s.support_max_dist}):   mean {nd.mean():.1f}  max {nd.max()}',
          flush=True)

    # ---- wrap the hot methods with counters/timers (whichever path runs) ----
    stat = {'sup_t': 0.0, 'sup_n': 0, 'enc_t': 0.0, 'enc_raw': 0}

    def wrap(name, kind):
        orig = getattr(s, name)
        if kind == 'sup':
            def w(*a):
                t0 = time.time()
                r = orig(*a)
                stat['sup_t'] += time.time() - t0
                stat['sup_n'] += 1
                return r
        else:
            def w(*a):
                t0 = time.time()
                r = orig(*a)
                stat['enc_t'] += time.time() - t0
                stat['enc_raw'] += len(r)
                return r
        setattr(s, name, w)

    for m in ('_support_patched', '_support_of'):
        wrap(m, 'sup')
    for m in ('_codon_encodings_ext', '_codon_encodings'):
        wrap(m, 'enc')

    n = int(os.environ.get('DIAG_DRAWS', '24'))
    print(f'\nrunning {n} single draws to show the warm-up curve (per-parent '
          f'term builds are one-time; steady state follows):', flush=True)
    times, encs, pts = [], [], []
    for i in range(n):
        t = time.time()
        d = s.draw_batch(1)
        times.append(time.time() - t)
        encs.append(len(d[0][2]))
        pts.append(len(s._pt))
        if (i + 1) % 4 == 0 or i == 0:
            print(f'  draws {max(1, i - 2)}..{i + 1}: '
                  f'{np.mean(times[max(0, i - 3):i + 1]) * 1e3:8.0f} ms/draw avg | '
                  f'encodings last draw {encs[-1]} | parents cached {pts[-1]}',
                  flush=True)
    wall = sum(times)
    stat_sup = stat['sup_t']
    print(f'\ntotal: {n} draws in {wall:.1f}s')
    print(f'  first 4 draws: {np.mean(times[:4]) * 1e3:.0f} ms/draw (warm-up dominated)')
    print(f'  last 4 draws:  {np.mean(times[-4:]) * 1e3:.0f} ms/draw (~steady state)')
    print(f'  encodings/draw: mean {np.mean(encs):.0f} max {max(encs)}')
    print(f'  parent-term cache at end: {len(s._pt)} / {len(res["pool"].seqs)} parents')
    print(f'  support time share: {stat_sup / wall * 100:.0f}%  '
          f'({stat["sup_n"] / n:.0f} calls/draw)')


if __name__ == '__main__':
    main()
