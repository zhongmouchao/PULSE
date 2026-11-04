"""Toy verification of the SGD proposal (pulse/sampler.py).

Same 12-nt (4-residue) synthetic dataset as the observed-support toy. The
proposal's q values, exact Z, table-distance counts, codon-path encodings and
per-encoding supports are checked against an INDEPENDENT full enumeration of
the proposal's outcome space (59,356 paths), and sampling is checked for
rejection correctness and seed determinism.

Run:  python tests/test_sampler_toy.py
"""
import itertools
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse.task_config import load_task
from pulse.support import ensure_observed_support
from pulse.sampler import Sampler, pep_codes, pep_str

# ---------------- toy data (identical to test_observed_support_toy) ----------------
A = 'AAACCCGGGTTT'   # pool, K P G F        (also in batch)
B = 'AAACCCGGGTTA'   # pool, K P G L
C = 'AAACCCGGGTAA'   # batch, K P G *
D = 'AAACCCGGGTTC'   # batch, K P G F
E = 'AAACCCGGGTCC'   # batch, K P G S
F = 'GGGCCCGGGTTT'   # batch, G P G F
H = 'AAANNNGGGTTT'   # batch, K X G F
ROWS = [(A, 'KPGF', 50, 0, 10, 0), (B, 'KPGL', 40, 0, 0, 0),
        (C, 'KPG*', 0, 10, 0, 0), (D, 'KPGF', 0, 0, 20, 0),
        (E, 'KPGS', 0, 0, 5, 0), (F, 'GPGF', 0, 0, 0, 3),
        (H, 'KXGF', 0, 0, 2, 0)]
TABLE_PEPS = {'KPGF', 'KPGL', 'KPG*', 'KPGS', 'GPGF', 'KXGF'}
LAM = 12 * 2e-4                  # auto lambda for 12 nt, mu = 2e-4
ROOTS = {'KPGF': 5 / 9, 'KPGL': 4 / 9}
LETTERS = 'ACDEFGHIKLMNPQRSTVWY'


def brute_outcome_space():
    """Full enumeration of the pre-rejection proposal: q_raw[x'] summed over
    every (root, n, positions, residues) path."""
    pn = {n: LAM ** n / math.factorial(n) for n in (1, 2, 3)}
    norm = sum(pn.values())
    q_raw = {}
    for root, r in ROOTS.items():
        rc = pep_codes(root)
        for n in (1, 2, 3):
            p_n = pn[n] / norm
            n_paths = math.comb(4, n) * 19 ** n
            w = r * p_n / n_paths
            for poss in itertools.combinations(range(4), n):
                others = [[a for a in LETTERS if a != root[j]] for j in poss]
                for res in itertools.product(*others):
                    xc = rc.copy()
                    for j, a in zip(poss, res):
                        xc[j] = pep_codes(a)[0]
                    xs = pep_str(xc)
                    q_raw[xs] = q_raw.get(xs, 0.0) + w
    return q_raw


def brute_support(y, pool_seqs, freqs, R, max_dist=3):
    """Nucleotide support of y: sum over pool parents at 1 <= d <= max_dist of
    the rate product over differing mutually valid sites, times frequency."""
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
    res = ensure_observed_support(cfg, verbose=False)
    rates = res['rates']

    q_raw = brute_outcome_space()
    total = sum(q_raw.values())
    assert abs(total - 1.0) < 1e-12, f'brute outcome space not normalized: {total}'

    sampler = Sampler(cfg, res, verbose=True)

    # ---- 1. lambda, roots, table set ----
    assert abs(sampler.lam - LAM) < 1e-15
    assert sampler.pool.PARENT_PEPS == ['KPGF', 'KPGL']
    assert np.allclose(sampler.root_w, [5 / 9, 4 / 9])
    assert sampler.table_set == TABLE_PEPS
    assert sampler.z_mode == 'exact'

    # ---- 2. pool x table distance counts (brute per pool peptide) ----
    for pi, p in enumerate(sampler.pool.PARENT_PEPS):
        want = [0, 0, 0]
        for t in TABLE_PEPS:
            d = sum(1 for a, b in zip(p, t) if a != b)
            bad = any(a != b and (a not in LETTERS or b not in LETTERS)
                      for a, b in zip(p, t))
            if not bad and 1 <= d <= 3:
                want[d - 1] += 1
        assert (sampler.N_table[pi] == want).all(), (p, sampler.N_table[pi], want)
    print(f'  N_table: {sampler.N_table.tolist()}')

    # ---- 3. exact Z vs brute table mass ----
    z_want = 1.0 - sum(q_raw.get(t, 0.0) for t in TABLE_PEPS)
    print(f'  Z: got {sampler.Z:.10f}  want {z_want:.10f}')
    assert abs(sampler.Z - z_want) / z_want < 1e-12

    # ---- 4. deterministic _finish: q and encodings for KPGV ----
    def enc_strs(encs):
        """Fast path returns bytes keys (arr.tobytes()); decode to ACGT
        strings for comparison. Toy sequences are pure ACGT, so the digit
        map is exact. Legacy string keys pass through unchanged."""
        dig = 'ACGT'
        return [(''.join(dig[b] for b in k) if isinstance(k, bytes) else k, s)
                for k, s in encs]
    xi = sampler.pool.PARENT_PEPS.index('KPGF')
    xs, q, encs = sampler._finish(xi, np.array([3]), np.array([17], dtype=np.int8))
    encs = enc_strs(encs)
    assert xs == 'KPGV'
    q_want = q_raw['KPGV'] / z_want
    print(f'  q(KPGV): got {q:.6e}  want {q_want:.6e}')
    assert abs(q - q_want) / q_want < 1e-12
    enc_d = dict(encs)
    # A (TTT) and B (TTA) produce the SAME 4 unique strings (dedup!):
    want_encs = {'AAACCCGGGGTT', 'AAACCCGGGGTA', 'AAACCCGGGGTC', 'AAACCCGGGGTG'}
    assert set(enc_d) == want_encs, set(enc_d)
    pool_seqs = sorted([A, B])
    fq = {A: 5 / 9, B: 4 / 9}
    for ys, sup in encs:
        want = brute_support(ys, pool_seqs, [fq[s] for s in pool_seqs], rates)
        rel = abs(sup - want) / max(want, 1e-300)
        print(f'  enc {ys}  got {sup:.6e}  want {want:.6e}  rel {rel:.2e}')
        assert rel < 1e-12, ys
    # hand anchor: AAACCCGGGGTT = A at d=1 (T->G tv) + B at d=2 (tv*tv)
    anchor = 1e-5 * 5 / 9 + 1e-10 * 4 / 9
    assert abs(enc_d['AAACCCGGGGTT'] - anchor) / anchor < 1e-12

    # ---- 5. second _finish: KPGF root, positions 2,3 (G->R, F->V), m=2 ----
    xs2, q2, encs2 = sampler._finish(xi, np.array([2, 3]),
                                     np.array([14, 17], dtype=np.int8))  # R, V
    encs2 = enc_strs(encs2)
    assert xs2 == 'KPRV'
    assert abs(q2 - q_raw['KPRV'] / z_want) / (q_raw['KPRV'] / z_want) < 1e-12
    # expected encodings: parents A (GGG, TTT) and B (GGG, TTA), budget k1+k2<=3
    from pulse.sampler import REACH as R2
    GGG, TTT_, TTA = 2 * 16 + 2 * 4 + 2, 3 * 16 + 3 * 4 + 3, 3 * 16 + 3 * 4 + 0
    dig = 'ACGT'
    cs = lambda tc: dig[tc >> 4] + dig[(tc >> 2) & 3] + dig[tc & 3]
    want_set = set()
    for c4 in (TTT_, TTA):
        for tc3, k1 in R2[GGG][14]:          # codon3 -> R
            for tc4, k2 in R2[c4][17]:       # codon4 -> V
                if k1 + k2 <= 3:
                    want_set.add('AAACCC' + cs(tc3) + cs(tc4))
    assert {ys for ys, _ in encs2} == want_set, (len(encs2), len(want_set))
    for ys, sup in encs2:
        want = brute_support(ys, pool_seqs, [fq[s] for s in pool_seqs], rates)
        assert abs(sup - want) / max(want, 1e-300) < 1e-12, ys

    # ---- 6. sampled batch: rejection, q values, determinism ----
    batch = sampler.draw_batch(4000)
    assert all(xs not in TABLE_PEPS for xs, _, _ in batch)
    for xs, q, encs in batch[::37]:          # spot-check every 37th draw
        want = q_raw[xs] / z_want
        assert abs(q - want) / want < 1e-9, (xs, q, want)
        for ys, sup in enc_strs(encs):
            w2 = brute_support(ys, pool_seqs, [fq[s] for s in pool_seqs], rates)
            assert abs(sup - w2) / max(w2, 1e-300) < 1e-12, ys
    n_distinct = len({xs for xs, _, _ in batch})
    print(f'  4000 draws: {n_distinct} distinct peptides, none in table')

    sampler2 = Sampler(cfg, res, verbose=False)
    b1 = [xs for xs, _, _ in sampler2.draw_batch(50)]
    sampler3 = Sampler(cfg, res, verbose=False)
    b2 = [xs for xs, _, _ in sampler3.draw_batch(50)]
    assert b1 == b2, 'same seed must reproduce the same stream'

    # fixed validation batch via dedicated rng
    va = [xs for xs, _, _ in sampler.draw_batch(64, rng=np.random.default_rng(43))]
    vb = [xs for xs, _, _ in sampler.draw_batch(64, rng=np.random.default_rng(43))]
    assert va == vb

    # ---- 7. MC Z pilot within tolerance of exact ----
    mc = Sampler(cfg, res, verbose=False, force_mc=True)
    assert mc.z_mode.startswith('MC')
    se = math.sqrt(z_want * (1 - z_want) / 200000)
    print(f'  Z_MC: {mc.Z:.6f} vs exact {z_want:.6f} (se {se:.1e})')
    assert abs(mc.Z - z_want) < 5 * se + 1e-6

    print('\nALL SAMPLER TOY TESTS PASSED')


if __name__ == '__main__':
    main()
