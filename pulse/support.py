"""Support machinery for the PULSE SGD pipeline (on-the-go design).

This module holds the SHARED sequence/rate machinery plus the one-time
OBSERVED-support precompute. There is no candidate enumeration in this
pipeline: unobserved peptides are sampled during SGD (pulse/sampler.py,
built on these same primitives) and their per-encoding support is computed
on the fly with the same exact sum.

Observed support:
  1. keep batch sequences whose minimal distance to the pool is
     <= observation.obs_dist_max (distances over mutually valid sites);
  2. for each kept nucleotide y,
         S(y) = sum over pool parents C with 1 <= d(C, y) <= support_max_dist
                of P(C -> y) * freq_0(C),
     where P(C -> y) is the product of per-site mutation rates over the
     differing sites -- the minimal pathway. A batch sequence identical to a
     pool member always adds the self-survival term P(y->y) * freq_0(y)
     (P(y->y) = product over valid sites of 1 - outgoing rate; nucleotide
     level, exact string match only). Sequences left with zero support sit at
     the loss floor downstream.
  3. kept encodings are grouped by translated peptide: obs_pep_support is the
     SUM of encoding supports per peptide (each encoding carrying its own
     self term where applicable); the observed loss term combines encodings
     as s(x) = 1 - prod_{y in Yhat(x)} (1 - p_o * p_e(y)) downstream.

Cache tiers (pickles under the task's cache dir):
  OBS_L0  per (data file, pool split, drop_gap):  pool structures, table/pool
          neighbor CSR (cutoff = max(obs_dist_max, support_max_dist)), and the
          true minimal distance of every table row to the pool
  OBS_L2  per OBS_L0 + (rates, obs_dist_max, support_max_dist):
          per-nucleotide observed support + peptide grouping
"""
import hashlib
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from pulse.task_config import resolve_rates

# ==================== codon / residue machinery ====================
_AAS = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
_ORDER = 'TCAG'
B2D = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
AA64 = ['?'] * 64
for _i, _a in enumerate(_ORDER):
    for _j, _b in enumerate(_ORDER):
        for _k, _c in enumerate(_ORDER):
            AA64[B2D[_a] * 16 + B2D[_b] * 4 + B2D[_c]] = _AAS[_i * 16 + _j * 4 + _k]
AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
_AA2CODE = {a: i for i, a in enumerate(AA_LETTERS)}
AA_CODE64 = np.array([_AA2CODE.get(a, -1) for a in AA64], dtype=np.int8)   # -1 = stop

_ci = np.arange(64)
NEWC = np.zeros((64, 3, 4), dtype=np.uint8)   # NEWC[codon, position, new_base] = new codon
NEWC[:, 0, :] = (np.arange(4)[None, :] * 16 + ((_ci // 4) % 4)[:, None] * 4 + (_ci % 4)[:, None])
NEWC[:, 1, :] = ((_ci // 16)[:, None] * 16 + np.arange(4)[None, :] * 4 + (_ci % 4)[:, None])
NEWC[:, 2, :] = ((_ci // 16)[:, None] * 16 + ((_ci // 4) % 4)[:, None] * 4 + np.arange(4)[None, :])


def codon_aa(c):
    if c == '---':
        return '-'
    if any(b not in 'ACGT' for b in c):
        return 'X'
    return AA64[B2D[c[0]] * 16 + B2D[c[1]] * 4 + B2D[c[2]]]


def translate_nuc(s):
    return ''.join(codon_aa(s[i:i + 3]) for i in range(0, len(s), 3))


def aa_code(ch):
    if ch == '-':
        return 21
    return _AA2CODE.get(ch, 20)          # 'X', '*' and anything else -> 20


def nuc_arr(s):
    return np.array([B2D.get(c, 4) for c in s], dtype=np.uint8)


# ==================== rate model ====================
def rate_matrices(rates):
    """rates: dict 'C>T' -> per-site rate. Returns R (4x4), logR (4x4, diagonal 0),
    R_out (4,) total outgoing rate per base, r_max."""
    R = np.zeros((4, 4))
    for key, r in rates.items():
        a, b = key.split('>')
        R[B2D[a], B2D[b]] = r
    with np.errstate(divide='ignore'):
        logR = np.log(R)
    np.fill_diagonal(logR, 0.0)          # diagonal never used; avoid -inf in indexing
    return R, logR, R.sum(axis=1), float(R.max())


# ==================== data loading + pool/batch split ====================
def load_counts(cfgd):
    """cfgd = cfg.resolve_paths()['data'] merged with filter settings."""
    fn = pd.read_csv(cfgd['training_file'])
    nuc_col, pep_col = cfgd['nuc_column'], cfgd['pep_column']
    # Length filter: keep only rows whose nuc AND pep lengths match the FIRST
    # row's (matrix stacking downstream requires uniform length). Rare after
    # padding; the drop count is reported, never silent.
    ref_nuc, ref_pep = len(fn[nuc_col].iloc[0]), len(fn[pep_col].iloc[0])
    keep = (fn[nuc_col].str.len() == ref_nuc) & (fn[pep_col].str.len() == ref_pep)
    n_drop = int((~keep).sum())
    if n_drop:
        print(f'  length filter: dropping {n_drop} / {len(fn)} rows not matching '
              f'first-row lengths (nuc={ref_nuc}, pep={ref_pep})')
        fn = fn.loc[keep].reset_index(drop=True)
    # Zero-count filter: rows with no surveillance counts in any time column
    # (or 'unknown') contribute nothing to pool or batch; drop and report.
    count_cols = [c for c in fn.columns if _is_time_col(c) or c == 'before 2000'] + \
                 (['unknown'] if 'unknown' in fn.columns else [])
    if count_cols:
        row_tot = fn[count_cols].sum(axis=1)
        n_zero = int((row_tot <= 0).sum())
        if n_zero:
            print(f'  zero-count filter: dropping {n_zero} / {len(fn)} rows with '
                  f'no surveillance counts')
            fn = fn.loc[row_tot > 0].reset_index(drop=True)
    if cfgd.get('drop_gap_or_ambiguous'):
        clean = fn[nuc_col].str.fullmatch(r'[ACGT]+')
        n_drop = int((~clean).sum())
        if n_drop:
            print(f'  drop_gap_or_ambiguous: dropping {n_drop} / {len(fn)} rows')
        fn = fn.loc[clean].reset_index(drop=True)
    return fn, nuc_col, pep_col


def _is_time_col(c):
    """Recognized surveillance time columns: years ('2020'), zero-padded months
    ('2020-01'), days ('2020-01-15'). All sort chronologically as strings.
    'before 2000' is handled separately (it sorts before any year)."""
    parts = c.split('-')
    return (len(parts[0]) == 4 and len(parts) in (1, 2, 3)
            and all(p.isdigit() for p in parts))


def split_pool_batch(fn, fraction):
    """Pool = whole time columns up to the first column whose cumulative count
    reaches `fraction` of the grand total; 'unknown' is always batch."""
    time_cols = (['before 2000'] if 'before 2000' in fn.columns else []) + \
                sorted(c for c in fn.columns
                       if c != 'before 2000' and _is_time_col(c))
    if not time_cols:
        raise ValueError(
            f'no surveillance time columns recognized; columns found: '
            f'{list(fn.columns)}. Expected years (2020), months (2020-01), '
            f"days (2020-01-15), or 'before 2000'.")
    totals = fn[time_cols].sum()
    grand = float(totals.sum())
    if grand <= 0:
        raise ValueError(f'time columns {time_cols} contain zero total counts')
    cum = (totals.cumsum() / grand).to_numpy()
    cut = int(np.searchsorted(cum, fraction))
    pool_cols = time_cols[:cut + 1]
    batch_cols = time_cols[cut + 1:] + (['unknown'] if 'unknown' in fn.columns else [])
    fn['Pool0_Count'] = fn[pool_cols].sum(axis=1)
    fn['Batch_Count'] = fn[batch_cols].sum(axis=1)
    return pool_cols, batch_cols, grand


def split_pool_prevalence(fn, threshold):
    """Prevalence mode: pool parents are ROWS whose file-wide count share is
    strictly above `threshold` of the grand total over every count column
    (time, 'before 2000', 'unknown'). Observed (batch) = ALL rows, pool
    members included — the analog of a temporal pool member reappearing in
    batch columns. Produces the same Pool0_Count / Batch_Count structure as
    split_pool_batch, so nothing downstream changes."""
    count_cols = ((['before 2000'] if 'before 2000' in fn.columns else []) +
                  sorted(c for c in fn.columns if _is_time_col(c)) +
                  (['unknown'] if 'unknown' in fn.columns else []))
    if not count_cols:
        raise ValueError('no count columns (time / before 2000 / unknown) found')
    total = fn[count_cols].sum(axis=1)
    grand = float(total.sum())
    if grand <= 0:
        raise ValueError(f'count columns {count_cols} contain zero total counts')
    is_pool = (total / grand) > threshold
    if not is_pool.any():
        raise ValueError(f'pool.prevalence_threshold={threshold}: no sequence exceeds '
                         f'this share of the grand total (max = {(total / grand).max():.4g})')
    fn['Pool0_Count'] = total.where(is_pool, 0)
    fn['Batch_Count'] = total
    return [], count_cols, grand


def pool_key_parts(cfg, pool_cols):
    """Pool-selection ingredients for the OBS_L0 / SAMP_L1 cache keys. Temporal
    mode keeps the historical ingredient set (pool_cols only) so existing caches
    stay valid bit-for-bit; prevalence mode adds mode + threshold (pool_cols is
    empty there and cannot discriminate thresholds)."""
    parts = {'pool_cols': pool_cols}
    if cfg.pool.mode == 'prevalence':
        parts['pool_mode'] = 'prevalence'
        parts['prev_thr'] = round(float(cfg.pool.prevalence_threshold), 12)
    return parts


def obs_key2_parts(cfg, key0, rates):
    """Ingredients for the OBS_L2 cache key — and for the L2 key embedded in
    POS_L1 (sampler.py). Used by BOTH call sites so the two can never diverge.
    The historical set is returned unchanged while max_degenerate_frac is off
    (None), so existing OBS_L2 / POS_L1 caches stay valid bit-for-bit; enabling
    the degenerate filter adds deg_frac."""
    parts = {'L0': key0,
             'rates': {k: round(v, 20) for k, v in sorted(rates.items())},
             'obs_dist_max': cfg.observation.obs_dist_max,
             'support_max_dist': cfg.observation.support_max_dist}
    if cfg.observation.max_degenerate_frac is not None:
        parts['deg_frac'] = round(float(cfg.observation.max_degenerate_frac), 12)
    return parts


# ==================== pool structures ====================
class Pool:
    """Unique pool nucleotide sequences (sorted), frequencies normalized within
    the pool, uint8 base matrix (4 = gap/ambiguous), codon table, and the
    parent peptide grouping."""
    def __init__(self, fn, nuc_col, pep_col):
        counts = fn.loc[fn['Pool0_Count'] > 0].groupby(nuc_col)['Pool0_Count'].sum()
        self.seqs = sorted(counts.index)
        self.total = float(counts.sum())
        self.freq = (counts.loc[self.seqs] / self.total).to_numpy(dtype=np.float64)
        self.idx_of = {s: i for i, s in enumerate(self.seqs)}
        self.set = set(self.seqs)
        self.mat = np.stack([nuc_arr(s) for s in self.seqs])
        self.n = len(self.seqs)
        self.L = self.mat.shape[1]
        self.n_res = self.L // 3
        _pm3 = self.mat.reshape(self.n, self.n_res, 3).astype(np.int16)
        self.codon64 = np.where((_pm3 < 4).all(axis=2),
                                _pm3[..., 0] * 16 + _pm3[..., 1] * 4 + _pm3[..., 2],
                                255).astype(np.uint8)
        _pep_of_nuc = fn.drop_duplicates(nuc_col).set_index(nuc_col)[pep_col]
        pep_str = [_pep_of_nuc[s] for s in self.seqs]
        self.PARENT_PEPS = sorted(set(pep_str))
        _pid = {p: i for i, p in enumerate(self.PARENT_PEPS)}
        self.pep_id = np.array([_pid[p] for p in pep_str], dtype=np.int64)


# ==================== sparse Hamming neighbor pass ====================
def sparse_hamming(tgt_mat, par_mat, cutoff, chunk=512):
    """For each target row: CSR lists of parents within Hamming distance
    <= cutoff (degenerate sites ignored on both sides), plus the TRUE minimal
    distance per target (computed over all parents, not truncated)."""
    Pv = (par_mat < 4).astype(np.float32)
    Pb = [(par_mat == b).astype(np.float32) for b in range(4)]
    nT = len(tgt_mat)
    offs = np.zeros(nT + 1, dtype=np.int64)
    idx_parts, dist_parts = [], []
    min_k = np.full(nT, np.inf)
    for s0 in range(0, nT, chunk):
        T = tgt_mat[s0:s0 + chunk]
        valid = (T < 4).astype(np.float32) @ Pv.T
        matches = np.zeros_like(valid)
        for b in range(4):
            matches += (T == b).astype(np.float32) @ Pb[b].T
        k = valid - matches
        min_k[s0:s0 + len(T)] = k.min(axis=1)
        near = k <= cutoff
        cnt = near.sum(axis=1)
        offs[s0 + 1: s0 + 1 + len(T)] = cnt
        _r, _c = np.where(near)
        idx_parts.append(_c.astype(np.int32))
        dist_parts.append(k[_r, _c].astype(np.float32))
    offs = np.cumsum(offs)
    return {'off': offs,
            'idx': (np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int32)),
            'dist': (np.concatenate(dist_parts) if dist_parts else np.zeros(0, np.float32)),
            'min_k': min_k}


# ==================== exact support of one target ====================
def exact_support_sparse(t_arr, t_row, pool, nei, logR, max_dist=None):
    """Exact minimal-pathway support of one target nucleotide sequence:
    sum over its neighbor parents with 1 <= distance <= max_dist (all
    distances if None) of exp(sum of logR over differing, mutually valid
    sites) * parent frequency. Distance-0 entries (the sequence itself, or
    gap/ambiguity variants equal over mutually valid sites) never contribute;
    the self path is added separately via p_self_of."""
    o0, o1 = nei['off'][t_row], nei['off'][t_row + 1]
    k = nei['dist'][o0:o1].astype(np.float64)
    nei_idx = nei['idx'][o0:o1]
    if max_dist is not None:
        m = k <= max_dist
        k, nei_idx = k[m], nei_idx[m]
    if len(nei_idx) == 0:
        return 0.0
    NB = pool.mat[nei_idx]
    validNB = NB < 4
    validT = t_arr < 4
    lr = logR[np.clip(NB, 0, 3).astype(np.int64),
              np.clip(t_arr, 0, 3).astype(np.int64)[None, :]]
    logP = np.where(validNB & validT[None, :] & (NB != t_arr[None, :]),
                    lr, 0.0).sum(axis=1)
    return float((np.where(k > 0, np.exp(logP), 0.0) * pool.freq[nei_idx]).sum())


def p_self_of(seq_arr, R_out):
    """Zero-mutation survival probability: product over valid sites of
    (1 - total outgoing rate of the site's base)."""
    valid = seq_arr < 4
    if not valid.any():
        return 0.0
    return float(np.exp(np.log1p(-R_out[seq_arr[valid]]).sum()))


# ==================== cache helpers ====================
def cache_key(**parts):
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cache_load(cache_dir, name):
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f), path
    return None, path


def cache_save(cache_dir, name, obj):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)
    return path


# ==================== observed-support driver ====================
def ensure_observed_support(cfg, verbose=True):
    """Compute (or load from cache) the observed-side support for one task
    config. One-time per (data, pool split, rates, obs_dist_max,
    support_max_dist); every training rerun reuses it."""
    t0 = time.time()
    rp = cfg.resolve_paths()
    cache_dir = rp['cache']['cache_dir']
    use_cache = cfg.cache.use_support_cache
    log = print if verbose else lambda *a, **k: None

    # ---- data + pool/batch split ----
    cfgd = dict(rp['data'])
    cfgd['drop_gap_or_ambiguous'] = cfg.filter.drop_gap_or_ambiguous
    fn, nuc_col, pep_col = load_counts(cfgd)
    if cfg.pool.mode == 'prevalence':
        pool_cols, batch_cols, grand = split_pool_prevalence(
            fn, cfg.pool.prevalence_threshold)
    else:
        pool_cols, batch_cols, grand = split_pool_batch(fn, cfg.pool.fraction)
    key0 = cache_key(data=os.path.basename(rp['data']['training_file']),
                     n_rows=len(fn), **pool_key_parts(cfg, pool_cols),
                     drop_gap=cfg.filter.drop_gap_or_ambiguous)
    if cfg.pool.mode == 'prevalence':
        log(f'[{cfg.task}] pool prevalence>{cfg.pool.prevalence_threshold:g}: '
            f'{int((fn["Pool0_Count"] > 0).sum())} parent rows '
            f'({int(fn["Pool0_Count"].sum())}/{int(grand)} counts); batch = all rows')
    else:
        log(f'[{cfg.task}] pool fraction {cfg.pool.fraction}: pool columns '
            f'{pool_cols[0]}..{pool_cols[-1]} ({int(fn["Pool0_Count"].sum())}/{int(grand)} counts)')

    # ---- OBS_L0: pool + table/pool neighbor pass ----
    cutoff = max(cfg.observation.obs_dist_max, cfg.observation.support_max_dist)
    pool = Pool(fn, nuc_col, pep_col)
    table_seqs = fn[nuc_col].to_numpy()
    L0_name = f'OBS_L0_{key0}.pkl'
    L0, L0_path = (cache_load(cache_dir, L0_name) if use_cache else (None, None))
    if L0 is None:
        CH = 4096
        parts = []
        for s0 in range(0, len(fn), CH):
            tm = np.stack([nuc_arr(s) for s in table_seqs[s0:s0 + CH]])
            parts.append(sparse_hamming(tm, pool.mat, cutoff))
        offs = [0]
        for p in parts:
            offs.extend((p['off'][1:] + offs[-1]).tolist())
        nei = {'off': np.array(offs, dtype=np.int64),
               'idx': np.concatenate([p['idx'] for p in parts]),
               'dist': np.concatenate([p['dist'] for p in parts]),
               'min_k': np.concatenate([p['min_k'] for p in parts])}
        L0 = {'nei': nei, 'pool_cols': pool_cols, 'batch_cols': batch_cols}
        if use_cache:
            cache_save(cache_dir, L0_name, L0)
        log(f'  OBS_L0 computed: {pool.n} pool parents, table pass over '
            f'{len(fn)} rows ({time.time() - t0:.0f}s)')
    else:
        nei = L0['nei']
        log(f'  OBS_L0 loaded: {L0_path}')

    # ---- OBS_L2: observed support + peptide grouping ----
    rates, weights = resolve_rates(cfg.mutation_model)
    R, logR, R_out, r_max = rate_matrices(rates)
    key2 = cache_key(**obs_key2_parts(cfg, key0, rates))
    L2_name = f'OBS_L2_{key2}.pkl'
    L2, L2_path = (cache_load(cache_dir, L2_name) if use_cache else (None, None))
    if L2 is None:
        nuc_to_row = {s: i for i, s in enumerate(table_seqs)}
        batch_seqs = sorted(set(table_seqs[(fn['Batch_Count'] > 0).to_numpy()]))
        n_batch = len(batch_seqs)
        # optional degenerate filter (off when max_degenerate_frac is None):
        # exclude observed sequences whose X + '-' share of PEPTIDE length
        # strictly exceeds the threshold; disjoint from the distance exclusion
        mdf = cfg.observation.max_degenerate_frac
        degenerate = []
        if mdf is not None:
            pep_of_row = fn[pep_col].to_numpy()
            deg_set = set()
            for s in batch_seqs:
                p = pep_of_row[nuc_to_row[s]]
                if (p.count('X') + p.count('-')) / len(p) > mdf:
                    deg_set.add(s)
            degenerate = [s for s in batch_seqs if s in deg_set]
            batch_seqs = [s for s in batch_seqs if s not in deg_set]
        min_k = nei['min_k']
        kept, excluded = [], []
        for s in batch_seqs:
            (kept if min_k[nuc_to_row[s]] <= cfg.observation.obs_dist_max
             else excluded).append(s)
        sup = np.zeros(len(kept), dtype=np.float64)
        for i, s in enumerate(kept):
            r = nuc_to_row[s]
            sa = exact_support_sparse(nuc_arr(s), r, pool, nei, logR,
                                      max_dist=cfg.observation.support_max_dist)
            if s in pool.set:   # self-survival: always allowed (exact match)
                j = pool.idx_of[s]
                sa += p_self_of(pool.mat[j], R_out) * float(pool.freq[j])
            sup[i] = sa
        # group kept encodings by translated peptide (CSR over sorted peptides)
        peps = [translate_nuc(s) for s in kept]
        uniq_peps = sorted(set(peps))
        pid = {p: i for i, p in enumerate(uniq_peps)}
        pep_of_enc = np.array([pid[p] for p in peps], dtype=np.int64)
        enc_flat = np.argsort(pep_of_enc, kind='stable')
        pep_off = np.concatenate(
            [[0], np.cumsum(np.bincount(pep_of_enc, minlength=len(uniq_peps)))])
        # peptide-level support = SUM over its nucleotide encodings; each
        # encoding's support already bundles its own self-survival term (if it
        # is a pool member) with the mutation contributions of all parents
        pep_support = np.zeros(len(uniq_peps), dtype=np.float64)
        np.add.at(pep_support, pep_of_enc, sup)
        L2 = {'obs_seqs': kept, 'obs_support': sup, 'obs_peps': uniq_peps,
              'obs_pep_support': pep_support,
              'pep_of_enc': pep_of_enc, 'pep_off': pep_off, 'enc_flat': enc_flat,
              'obs_excluded_distant': excluded,
              'obs_excluded_degenerate': degenerate}
        if use_cache:
            cache_save(cache_dir, L2_name, L2)
        log(f'  OBS_L2 computed: {len(kept)} / {n_batch} observed batch sequences '
            f'within {cfg.observation.obs_dist_max} mutations'
            + ('' if mdf is None else f'; {len(degenerate)} degenerate-excluded')
            + f'; {len(uniq_peps)} observed peptides ({time.time() - t0:.0f}s)')
    else:
        log(f'  OBS_L2 loaded: {L2_path} '
            f'({len(L2["obs_seqs"])} observed sequences, {len(L2["obs_peps"])} peptides)')

    if verbose and len(L2['obs_support']):
        s = L2['obs_support']
        log(f'  observed support: min={s.min():.3e}, median={np.median(s):.3e}, '
            f'max={s.max():.3e}; zero-support (loss floor): {int((s == 0).sum())}')

    return {'cfg': cfg, 'fn': fn, 'nuc_col': nuc_col, 'pep_col': pep_col,
            'pool': pool, 'pool_cols': L0['pool_cols'], 'batch_cols': L0['batch_cols'],
            'nei': nei, 'table_seqs': table_seqs,
            'rates': rates, 'R': R, 'logR': logR, 'R_out': R_out,
            'obs': L2}
