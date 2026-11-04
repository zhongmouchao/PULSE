"""On-the-go proposal engine for the PULSE SGD pipeline.

Called once per SGD step to produce a batch of UNOBSERVED peptides. There is
no candidate enumeration and no EPS threshold anywhere: the unobserved space
is sampled, never materialized.

Per draw:
  1. root peptide x ~ r, the pool-frequency distribution over pool peptides
     (counts aggregated over synonymous pool nucleotides);
  2. mutation count n ~ Poisson(lam) truncated to {1..min(n_max, L_x)} and
     renormalized per root (L_x = # canonical residues of x; lam is
     data-derived, L_nuc * mu_total, or a config float);
  3. n distinct canonical positions uniformly without replacement, new
     residue uniform among the other 19 at each;
  4. reject & resample if x' is ANY table peptide (pool / batch / unknown);
  5. q(x') = q_raw(x') / Z with
         q_raw(x') = sum_m sum_{parents p: d_aa(p,x')=m} r(p) *
                     P_p(n=m) / (C(L_p, m) * 19^m),
     the self path from the actual root included. Z = 1 - q_raw(table) is
     computed EXACTLY from a one-time pool-peptide x table-peptide distance
     pass (MC pilot fallback for very large pools, force_mc for testing);
  6. encodings of x': for every pool parent nucleotide y whose peptide is an
     AA-neighbor of x' (1 <= d_aa <= n_max), the minimal-codon-path variants
     -- mutations confined to the codons of the differing residues, total nt
     changes <= n_max, any target codon of the new residue allowed. A parent
     whose actual codon cannot reach the target residue within budget
     contributes ZERO encodings (codon context). A draw with no reachable
     encoding is wasted (g = 0) but unbiased. Each unique encoding y' gets
     the exact multi-distance support: sum over ALL pool parents at
     nucleotide distance 1..support_max_dist of (minimal-pathway rate
     product) * freq_0 -- the same support as the observed side.

One-time structures (cached as SAMP_L1_<hash>.pkl, keyed on the OBS_L0 key +
n_max + support_max_dist): pool-peptide root weights, peptide code matrix,
canonical counts, pool-pool AA neighbor CSR (cutoff 2*n_max), pool-pool
nucleotide neighbor CSR (cutoff n_max + support_max_dist), table-peptide
hash set, pool-peptide x table-peptide distance counts (exact Z mode).

Interface:
    res = ensure_observed_support(cfg)
    sampler = Sampler(cfg, res)
    batch = sampler.draw_batch(128)          # list of (x'_str, q, [(key, support), ...])
    val   = sampler.draw_batch(500, rng=np.random.default_rng(cfg.training.val_seed))

    # key: arr.tobytes() of the encoding (dedup identity, with an ambiguity
    # signature per generating parent so entries match legacy string dedup
    # exactly). Downstream only the support values are consumed (batch_tensors).
"""
import itertools
import math
import os
import time

import numpy as np

from pulse.support import (
    AA_CODE64, cache_key, cache_load, cache_save, obs_key2_parts, pool_key_parts,
    sparse_hamming, translate_nuc,
)

# ==================== peptide codes ====================
# 0..19 canonical amino acids (AA_LETTERS order), 20 'X', 21 '-', 22 '*'.
_NC = 20
_AA_LETTERS = 'ACDEFGHIKLMNPQRSTVWY'
_CODE = {a: i for i, a in enumerate(_AA_LETTERS)}
_CODE.update({'X': 20, '-': 21, '*': 22})
_DECODE = [_AA_LETTERS[i] if i < 20 else {20: 'X', 21: '-', 22: '*'}[i]
           for i in range(23)]
_DIGITS = 'ACGT'           # nucleotide code -> base char (matches B2D / pool.mat)


def pep_codes(s):
    return np.array([_CODE.get(ch, 20) for ch in s], dtype=np.int8)


def pep_str(codes):
    return ''.join(_DECODE[int(c)] for c in codes)


# ==================== codon reachability ====================
# REACH[c][a]: list of (target_codon, k) with k = nucleotide Hamming(c, target)
# >= 1, over all codons tc of amino acid a (a in 0..19). Codon indices use the
# ACGT-digit encoding of support.py (c = b0*16 + b1*4 + b2).
def _build_reach():
    reach = [[[] for _ in range(20)] for _ in range(64)]
    for c in range(64):
        d0, d1, d2 = c >> 4, (c >> 2) & 3, c & 3
        for tc in range(64):
            a = int(AA_CODE64[tc])
            if a < 0:
                continue
            k = (d0 != (tc >> 4)) + (d1 != ((tc >> 2) & 3)) + (d2 != (tc & 3))
            if k >= 1:
                reach[c][a].append((tc, k))
    return reach


REACH = _build_reach()


# ==================== AA pair scan ====================
def _aa_pair_scan(P, T, m_max, want_csr=False, chunk=256):
    """Scan P x T peptide pairs. A pair is REACHABLE at distance m when the
    two rows differ at exactly m positions, every differing position being
    canonical (< _NC) on both sides. Returns:
      counts[i, m-1] = # T rows reachable from P[i] at distance m (1..m_max)
      csr (if want_csr): {'off','idx','m'} neighbor lists of kept pairs."""
    Tlt = (T < _NC).astype(np.float32)
    Tge = (T >= _NC).astype(np.float32)
    Teq = [(T == c).astype(np.float32) for c in range(23)]
    counts = np.zeros((len(P), m_max), dtype=np.int64)
    offs = np.zeros(len(P) + 1, dtype=np.int64)
    cols_, ms_ = [], []
    for s0 in range(0, len(P), chunk):
        Pc = P[s0:s0 + chunk]
        cnb = (Pc < _NC).astype(np.float32) @ Tlt.T          # canonical-both
        eq = np.zeros_like(cnb)
        for a in range(20):
            eq += (Pc == a).astype(np.float32) @ Teq[a].T
        eqn = np.zeros_like(cnb)
        for c in (20, 21, 22):
            eqn += (Pc == c).astype(np.float32) @ Teq[c].T
        cnp = (Pc >= _NC).sum(axis=1, keepdims=True).astype(np.float32)
        # bad positions: a difference involving a non-canonical residue
        bad = cnp - eqn + (Pc < _NC).astype(np.float32) @ Tge.T
        m = cnb - eq                                          # canonical diffs
        keep = (bad == 0) & (m >= 1) & (m <= m_max)
        mi = m.astype(np.int8)
        for mm in range(1, m_max + 1):
            counts[s0:s0 + len(Pc), mm - 1] = (keep & (mi == mm)).sum(axis=1)
        if want_csr:
            r_, c_ = np.where(keep)
            offs[s0 + 1: s0 + 1 + len(Pc)] = np.bincount(r_, minlength=len(Pc))
            cols_.append(c_.astype(np.int32))
            ms_.append(mi[r_, c_])
    csr = None
    if want_csr:
        csr = {'off': np.cumsum(offs),
               'idx': np.concatenate(cols_) if cols_ else np.zeros(0, np.int32),
               'm': np.concatenate(ms_) if ms_ else np.zeros(0, np.int8)}
    return counts, csr


# ==================== the sampler ====================
class Sampler:
    def __init__(self, cfg, res, verbose=True, force_mc=False):
        self.cfg, self.res, self.pool = cfg, res, res['pool']
        self.logR = res['logR']
        self.n_max = cfg.proposal.n_max
        self.support_max_dist = cfg.observation.support_max_dist
        log = print if verbose else lambda *a, **k: None
        t0 = time.time()

        # ---- lambda: data-derived or config float (needed before any build) ----
        mu = cfg.mutation_model.mu_total
        self.lam = (float(cfg.proposal.lam) if cfg.proposal.lam != 'auto'
                    else self.pool.L * mu)

        # ---- cache key: OBS_L0 ingredients + proposal distances + lam ----
        # (lam enters Z through P_p(n); without it a mu_total / proposal.lam
        # change would silently reuse a stale SAMP_L1 cache)
        rp = cfg.resolve_paths()
        cache_dir = rp['cache']['cache_dir']
        fn = res['fn']
        key0 = cache_key(data=os.path.basename(rp['data']['training_file']),
                         n_rows=len(fn), **pool_key_parts(cfg, res['pool_cols']),
                         drop_gap=cfg.filter.drop_gap_or_ambiguous)
        skey = cache_key(L0=key0, n_max=self.n_max,
                         support_max_dist=self.support_max_dist, lam=self.lam)
        self.key0, self.skey = key0, skey
        name = f'SAMP_L1_{skey}.pkl'
        use_cache = cfg.cache.use_support_cache and not force_mc
        S, path = (cache_load(cache_dir, name) if use_cache else (None, None))
        if S is None:
            S = self._build(res, force_mc, log)
            if use_cache:
                cache_save(cache_dir, name, S)
            log(f'  SAMP_L1 computed ({time.time() - t0:.0f}s)')
        else:
            log(f'  SAMP_L1 loaded: {path}')
        self.__dict__.update(S)

        # ---- per-parent support terms (lazy): y0 -> (u_idx, d0, logP0, freq_u).
        # d0/logP0 vs pool.mat[y0] are fixed for the whole run; an encoding's
        # support then only needs a <=9-site patch instead of a full (n_nei x L)
        # recomputation. Mathematically identical to _support_of (per-site
        # decomposition of the same common-valid Hamming / rate sums).
        self._pt = {}
        # per-parent non-ACGT positions as (pos, letter) pairs — the ambiguity
        # signatures used for exact string-level encoding dedup (see _finish)
        self._amb = [tuple((int(pos), self.pool.seqs[y][int(pos)])
                           for pos in np.where(self.pool.mat[y] >= 4)[0])
                     for y in range(len(self.pool.seqs))]

        # ---- proposal weights (cheap; rebuilt every init) ----
        n_max, lam = self.n_max, self.lam
        L_can = self.L_can
        self._cap = np.minimum(n_max, L_can).astype(np.int64)
        # Pn[p, m] = P(n = m) for root p (m = 1..n_max), renormalized per root
        Pn = np.zeros((len(L_can), n_max + 1))
        facts = np.array([lam ** m / math.factorial(m) for m in range(n_max + 1)])
        for cap in range(1, n_max + 1):
            rows = self._cap == cap
            Pn[rows, 1:cap + 1] = facts[1:cap + 1] / facts[1:cap + 1].sum()
        self.Pn = Pn
        # W[p, m] = r(p) * P_p(n=m) / (C(L_p, m) * 19^m): q_raw building block
        root_w = self.root_w / self.root_w.sum()
        self.root_w = root_w
        W = np.zeros_like(Pn)
        with np.errstate(divide='ignore', invalid='ignore'):
            for m in range(1, n_max + 1):
                comb = np.array([math.comb(int(L), m) if L >= m else 0
                                 for L in L_can], dtype=np.float64)
                denom = comb * (19.0 ** m)
                W[:, m] = np.where(denom > 0, root_w * Pn[:, m] / denom, 0.0)
        self.W = W
        # Z = 1 - q_raw(table): exact from counts, or MC
        if self.N_table is not None:
            self.Z = float(1.0 - (W[:, 1:] * self.N_table).sum())
            self.z_mode = 'exact'
        else:
            self.Z = float(self.Z_mc)
            self.z_mode = f'MC(K={self.Z_mc_K})'
        if not (0.0 < self.Z <= 1.0):
            raise RuntimeError(f'proposal rejection mass leaves Z={self.Z}; check inputs')
        if self.Z < 0.5:
            log(f'  WARNING: Z={self.Z:.3f} -- more than half of all draws are rejected')
        self.root_cum = np.cumsum(root_w)
        self._pn_cum = {cap: np.cumsum(Pn[np.argmax(self._cap == cap), 1:cap + 1])
                        if (self._cap == cap).any() else None
                        for cap in range(1, n_max + 1)}
        self._canon_pos = [np.where(row < _NC)[0] for row in self.pep_mat]
        self.rng = np.random.default_rng(cfg.training.seed)
        log(f'  proposal: lam={self.lam:.4g}, n_max={n_max}, Z={self.Z:.6f} ({self.z_mode}), '
            f'{len(self.table_set)} table peptides, {self.aa_nei["off"][-1]} AA neighbors')

    # ---------------- one-time build ----------------
    def _build(self, res, force_mc, log):
        pool, fn = self.pool, res['fn']
        n_max = self.n_max
        pep_mat = np.stack([pep_codes(p) for p in pool.PARENT_PEPS])
        L_aa = pep_mat.shape[1]
        L_can = (pep_mat < _NC).sum(axis=1).astype(np.int64)
        root_w = np.bincount(pool.pep_id, weights=pool.freq,
                             minlength=len(pool.PARENT_PEPS)).astype(np.float64)
        root_w[L_can == 0] = 0.0          # nothing mutable at a fully degenerate root
        if root_w.sum() <= 0:
            raise RuntimeError('no pool peptide with a canonical residue to mutate')
        # table peptides (rejection set + exact-Z pass)
        table_strs = sorted({translate_nuc(s) for s in set(res['table_seqs'])})
        if any(len(t) != L_aa for t in table_strs):
            raise ValueError('non-uniform peptide length among table sequences; '
                             'check padding of the training file')
        table_set = set(table_strs)
        # pool peptide -> member nucleotides CSR
        order = np.argsort(pool.pep_id, kind='stable')
        pep_mem = {'off': np.concatenate(
            [[0], np.cumsum(np.bincount(pool.pep_id, minlength=len(pool.PARENT_PEPS)))]),
            'idx': order.astype(np.int32)}
        # pool x pool AA neighbors (candidate parents of any x' drawn from a root)
        log(f'  SAMP_L1: {len(pep_mat)} pool peptides, {len(table_strs)} table peptides; '
            f'AA neighbor pass (cutoff {2 * n_max})...')
        _, aa_nei = _aa_pair_scan(pep_mat, pep_mat, 2 * n_max, want_csr=True)
        if aa_nei['off'][-1] > 5e7:
            raise RuntimeError(f'AA neighbor table too dense ({aa_nei["off"][-1]} pairs)')
        # pool x pool nucleotide neighbors (candidate support parents of encodings)
        nuc_nei = sparse_hamming(pool.mat, pool.mat, cutoff=n_max + self.support_max_dist)
        # exact Z counts, or MC fallback when the exact pass is too big.
        # _aa_pair_scan materializes ~25 float32 matrices of size
        # (L_aa x n_table) at once; with millions of table peptides that is
        # hundreds of GB (OOM-killed the full-spike prevalence jobs), so
        # huge tables take the MC pilot. The 16 GB budget keeps every
        # previously-feasible exact run exact (RBD: 25 x 201 x 250k x 4B
        # ~= 2 GB stays exact; 2M-peptide full-spike tables -> MC).
        exact_bytes = 25 * L_aa * max(len(table_strs), 1) * 4
        N_table, Z_mc, Z_mc_K = None, None, None
        if force_mc or len(pep_mat) * len(table_strs) > 3e8 or exact_bytes > 16e9:
            Z_mc_K = 200000
            rng = np.random.default_rng(1234567)
            root_w_tmp = root_w / root_w.sum()
            cap = np.minimum(n_max, L_can)
            facts = np.array([self.lam ** m / math.factorial(m)
                              for m in range(n_max + 1)])
            cum = np.cumsum(root_w_tmp)
            hits = 0
            for _ in range(Z_mc_K):
                xi = int(np.searchsorted(cum, rng.random()))
                c = int(cap[xi])
                w = facts[1:c + 1] / facts[1:c + 1].sum()
                n = 1 + int(np.searchsorted(np.cumsum(w), rng.random()))
                pc = np.where(pep_mat[xi] < _NC)[0]
                pos = pc[rng.choice(len(pc), size=n, replace=False)]
                old = pep_mat[xi, pos]
                v = rng.integers(0, 19, size=n)
                xc = pep_mat[xi].copy()
                xc[pos] = v + (v >= old)
                if pep_str(xc) in table_set:
                    hits += 1
            Z_mc = 1.0 - hits / Z_mc_K
            log(f'  Z by MC pilot: {Z_mc:.4f} (K={Z_mc_K})')
        else:
            T_mat = np.stack([pep_codes(t) for t in table_strs])
            N_table, _ = _aa_pair_scan(pep_mat, T_mat, n_max)
        return {'pep_mat': pep_mat, 'L_can': L_can, 'root_w': root_w,
                'table_set': table_set, 'pep_mem': pep_mem,
                'aa_nei': aa_nei, 'nuc_nei': nuc_nei,
                'N_table': N_table, 'Z_mc': Z_mc, 'Z_mc_K': Z_mc_K}

    # ---------------- per-draw pieces ----------------
    def _draw_pre_rejection(self, rng):
        xi = int(np.searchsorted(self.root_cum, rng.random()))
        cap = int(self._cap[xi])
        n = 1 + int(np.searchsorted(self._pn_cum[cap], rng.random()))
        pc = self._canon_pos[xi]
        pos = pc[rng.choice(len(pc), size=n, replace=False)]
        old = self.pep_mat[xi, pos]
        v = rng.integers(0, 19, size=n)
        return xi, pos, (v + (v >= old)).astype(np.int8)

    def _finish(self, xi, positions, newres):
        """Deterministic core: given root xi and the mutation, return
        (x'_str, q, [(y'_str, support), ...])."""
        n = len(positions)
        xc = self.pep_mat[xi].copy()
        xc[positions] = newres
        xs = pep_str(xc)
        # ---- q_raw over all candidate parents (root's AA neighbors + root itself)
        o0, o1 = self.aa_nei['off'][xi], self.aa_nei['off'][xi + 1]
        cand = self.aa_nei['idx'][o0:o1]
        Pn_ = self.pep_mat[cand]
        diff = Pn_ != xc[None, :]
        cb = (Pn_ < _NC) & (xc[None, :] < _NC)
        m_p = (diff & cb).sum(axis=1)
        bad = (diff & ~cb).any(axis=1)
        ok = (~bad) & (m_p >= 1) & (m_p <= self.n_max)
        kept = cand[ok]
        mk = m_p[ok].astype(np.int64)
        q = float(self.W[xi, n] + self.W[kept, mk].sum()) / self.Z
        # ---- minimal-codon-path encodings: the ROOT itself plus every kept
        # AA-neighbor parent (the root is not in its own neighbor CSR)
        dmask = diff[ok] & cb[ok]
        root_dmask = np.zeros(len(xc), dtype=bool)
        root_dmask[np.asarray(positions)] = True
        parent_iter = [(xi, root_dmask)] + [(p, dmask[row]) for row, p in enumerate(kept)]
        # Dedup on the encoding ARRAY (bytes key) + an ambiguity SIGNATURE —
        # exactly equivalent to legacy string dedup: the final string is the
        # encoding's ACGT digits plus the generating parent's non-ACGT letters
        # at NON-VARIANT codon positions (variant codons are always overwritten
        # with ACGT, so string differences there merge — e.g. synonymous TTT /
        # TTA parents both yield TTG; while mat-identical alias parents like
        # 'N' vs 'R', whose letters sit in invalid codons that can never be
        # variant codons, stay distinct). One entry per distinct final string,
        # each with the (generator-independent) support of the array.
        enc = {}
        for p, dm in parent_iter:
            js = np.where(dm)[0]
            jset = set(js.tolist())
            a, b = self.pep_mem['off'][p], self.pep_mem['off'][p + 1]
            for y in self.pep_mem['idx'][a:b]:
                amb = self._amb[y]
                sig = (tuple((pos, ch) for pos, ch in amb if pos // 3 not in jset)
                       if amb else None)
                for key, sites, newb in self._codon_encodings_ext(y, js, xc):
                    ent = enc.get(key)
                    if ent is None:
                        enc[key] = (y, sites, newb, {sig})
                    else:
                        ent[3].add(sig)
        out = []
        for key, (y0, sites, newb, sigs) in enc.items():
            sup = self._support_patched(y0, sites, newb)
            out.extend([(key, sup)] * len(sigs))
        return xs, q, out

    def _codon_encodings(self, y, js, tgt_codes):
        """All minimal-codon-path variants of pool nucleotide y toward target
        residues tgt_codes at differing-residue positions js, within the total
        nt-change budget n_max. Returns a list of (y'_str, y'_arr)."""
        codons = self.pool.codon64[y, js]
        if (codons == 255).any():
            return []
        option_lists = []
        for j, c, tgt in zip(js, codons, tgt_codes[js]):
            opts = [(j, tc, k) for (tc, k) in REACH[int(c)][int(tgt)]]
            if not opts:
                return []
            option_lists.append(opts)
        out = []
        for combo in itertools.product(*option_lists):
            if sum(k for _, _, k in combo) > self.n_max:
                continue
            arr = self.pool.mat[y].copy()
            chars = list(self.pool.seqs[y])
            for j, tc, k in combo:
                arr[3 * j] = tc >> 4
                arr[3 * j + 1] = (tc >> 2) & 3
                arr[3 * j + 2] = tc & 3
                chars[3 * j] = _DIGITS[tc >> 4]
                chars[3 * j + 1] = _DIGITS[(tc >> 2) & 3]
                chars[3 * j + 2] = _DIGITS[tc & 3]
            out.append((''.join(chars), arr))
        return out

    def _support_of(self, ya, y0):
        """Exact support of one encoding: sum over pool nucleotide parents at
        distance 1..support_max_dist of rate product * freq_0. Candidate
        parents = nucleotide neighbors of the generating parent y0 at cutoff
        n_max + support_max_dist (triangle inequality on the common-valid-site
        Hamming distance, which is a metric here because y' inherits exactly
        the valid-site pattern of y0)."""
        o0, o1 = self.nuc_nei['off'][y0], self.nuc_nei['off'][y0 + 1]
        u_idx = self.nuc_nei['idx'][o0:o1]
        if len(u_idx) == 0:
            return 0.0
        U = self.pool.mat[u_idx]
        valid = (U < 4) & (ya[None, :] < 4)
        diffm = valid & (U != ya[None, :])
        d = diffm.sum(axis=1)
        sel = (d >= 1) & (d <= self.support_max_dist)
        if not sel.any():
            return 0.0
        Us = U[sel].astype(np.int64)
        lr = self.logR[np.clip(Us, 0, 3),
                       np.clip(ya, 0, 3).astype(np.int64)[None, :]]
        logP = np.where(diffm[sel], lr, 0.0).sum(axis=1)
        return float((np.exp(logP) * self.pool.freq[u_idx[sel]]).sum())

    # ---------------- fast path: per-parent terms + <=9-site patching ----------------
    def _parent_terms(self, y0):
        """Lazy cache of (u_idx, d0, logP0, freq_u) for parent y0: neighbor ids
        (cutoff n_max + support_max_dist), common-valid distances and rate sums
        vs pool.mat[y0]. One (n_nei, L) pass per parent per run, then reused by
        every encoding generated from y0."""
        ent = self._pt.get(y0)
        if ent is None:
            o0, o1 = self.nuc_nei['off'][y0], self.nuc_nei['off'][y0 + 1]
            u_idx = self.nuc_nei['idx'][o0:o1]
            U = self.pool.mat[u_idx]
            y0a = self.pool.mat[y0]
            valid = (U < 4) & (y0a < 4)
            diffm = valid & (U != y0a[None, :])
            d0 = diffm.sum(axis=1).astype(np.int16)
            lr = self.logR[np.clip(U, 0, 3), np.clip(y0a, 0, 3)[None, :]]
            logP0 = np.where(diffm, lr, 0.0).sum(axis=1)
            ent = (u_idx, d0, logP0, self.pool.freq[u_idx])
            self._pt[y0] = ent
        return ent

    def _support_patched(self, y0, sites, newb):
        """Exact support of the encoding that differs from parent y0 at the
        given sites (new bases newb, all canonical). Identical value to
        _support_of(arr, y0): d and logP decompose per site, so only the
        variant sites are patched off the cached parent terms."""
        u_idx, d0, logP0, fr = self._parent_terms(y0)
        if len(u_idx) == 0:
            return 0.0
        Us = self.pool.mat[u_idx][:, sites]          # (n_nei, k), k <= 9
        y0a = self.pool.mat[y0]
        d = d0.astype(np.int32)
        for c in range(len(sites)):
            u_s, ob, nb = Us[:, c], y0a[sites[c]], int(newb[c])
            old_d = (u_s < 4) & (ob < 4) & (u_s != ob)
            new_d = (u_s < 4) & (u_s != nb)
            d += new_d.astype(np.int32) - old_d.astype(np.int32)
        sel = (d >= 1) & (d <= self.support_max_dist)
        if not sel.any():
            return 0.0
        lp = logP0[sel].copy()
        Us_sel, fr_sel = Us[sel], fr[sel]
        for c in range(len(sites)):
            u_s, ob, nb = Us_sel[:, c], y0a[sites[c]], int(newb[c])
            old_m = (u_s < 4) & (ob < 4) & (u_s != ob)
            new_m = (u_s < 4) & (u_s != nb)
            if old_m.any():
                lp[old_m] -= self.logR[np.clip(u_s[old_m], 0, 3),
                                       ob if ob < 4 else 0]
            if new_m.any():
                lp[new_m] += self.logR[u_s[new_m], nb]
        return float((np.exp(lp) * fr_sel).sum())

    def _codon_encodings_ext(self, y, js, tgt_codes):
        """Fast-path variant of _codon_encodings: same enumeration and budget
        rules, but returns (arr.tobytes(), sites, newb) per variant instead of
        building the 603-char string (the string is never used downstream)."""
        codons = self.pool.codon64[y, js]
        if (codons == 255).any():
            return []
        option_lists = []
        for j, c, tgt in zip(js, codons, tgt_codes[js]):
            opts = [(j, tc, k) for (tc, k) in REACH[int(c)][int(tgt)]]
            if not opts:
                return []
            option_lists.append(opts)
        out = []
        for combo in itertools.product(*option_lists):
            if sum(k for _, _, k in combo) > self.n_max:
                continue
            arr = self.pool.mat[y].copy()
            sites, newb = [], []
            for j, tc, k in combo:
                arr[3 * j] = tc >> 4
                arr[3 * j + 1] = (tc >> 2) & 3
                arr[3 * j + 2] = tc & 3
                sites += [3 * j, 3 * j + 1, 3 * j + 2]
                newb += [tc >> 4, (tc >> 2) & 3, tc & 3]
            out.append((arr.tobytes(), np.array(sites, dtype=np.int64),
                        np.array(newb, dtype=np.uint8)))
        return out

    # ---------------- public API ----------------
    def draw_batch(self, B, rng=None):
        """B i.i.d. accepted draws: list of (x'_str, q, [(y'_str, support), ...]).
        Pass a dedicated rng (e.g. seeded by val_seed) for a fixed validation
        batch; default is the training stream seeded by training.seed."""
        rng = rng or self.rng
        out = []
        while len(out) < B:
            xi, pos, new = self._draw_pre_rejection(rng)
            xc = self.pep_mat[xi].copy()
            xc[pos] = new
            if pep_str(xc) in self.table_set:
                continue                      # reject & resample
            out.append(self._finish(xi, pos, new))
        return out


# ==================== positive-side extra encodings ====================
def positive_extra_encodings(sampler, res, verbose=True, keep_seqs=False):
    """Unobserved nucleotide encodings of the OBSERVED peptides (preprint
    Eq. 4 parity: Yhat(x) spans observed batch encodings AND generated
    candidates encoding x). For each kept observed peptide x:
      enc2: minimal-codon-path encodings from pool parents at AA distance
            1..n_max (same machinery and budget as the sampler);
      enc3: if x is itself a pool peptide, single-position synonymous codon
            variants (k = 1..n_max) of its pool member nucleotides.
    Table nucleotide strings are excluded (the observed encodings of x already
    enter via the observed-side support). Every extra encoding carries the
    exact multi-distance support, same definition as everywhere else.
    Cached as POS_L1 keyed on the SAMP_L1 + OBS_L2 ingredients.

    Memory note: production callers only consume supports + offsets, so by
    default the encoding STRINGS are never accumulated (they dominate memory
    at full-spike scale: ~20M x 3.8 kb strings ~ 70 GB). Supports are
    accumulated per peptide as small float64 chunks and concatenated once.
    keep_seqs=True (tests only) also returns the strings and never touches
    the cache.

    Returns {'extra_support': float64 array, 'extra_off': CSR offsets over
    res['obs']['obs_peps'], 'n_extra': int} — plus 'extra_seqs' (flat list of
    encodings, aligned with extra_support) when keep_seqs=True. Old caches
    carrying 'extra_seqs' still load; the field is simply unused.
    """
    cfg = res['cfg']
    log = print if verbose else lambda *a, **k: None
    t0 = time.time()
    rp = cfg.resolve_paths()
    rates = res['rates']
    key2 = cache_key(**obs_key2_parts(cfg, sampler.key0, rates))
    name = f'POS_L1_{cache_key(SAMP=sampler.skey, L2=key2)}.pkl'
    if cfg.cache.use_support_cache and not keep_seqs:
        cached, path = cache_load(rp['cache']['cache_dir'], name)
        if cached is not None:
            n = cached.get('n_extra', len(cached.get('extra_seqs', ())))
            log(f'  POS_L1 loaded: {path} ({n} extra encodings)')
            return cached

    obs = res['obs']
    pool = res['pool']
    obs_peps = obs['obs_peps']
    obs_codes = np.stack([pep_codes(p) for p in obs_peps])
    table_nucs = set(res['table_seqs'])
    # pool parents at AA distance 1..n_max of each observed peptide
    log(f'  POS_L1: AA neighbor scan {len(obs_peps)} observed x '
        f'{len(sampler.pep_mat)} pool peptides (cutoff {sampler.n_max})...')
    t_scan = time.time()
    _, csr = _aa_pair_scan(obs_codes, sampler.pep_mat, sampler.n_max,
                           want_csr=True)
    log(f'  POS_L1: scan done ({time.time() - t_scan:.0f}s, '
        f'{int(csr["off"][-1])} neighbor pairs); building extras...')
    pool_pid = {p: i for i, p in enumerate(pool.PARENT_PEPS)}

    extra_seqs = [] if keep_seqs else None
    sup_chunks, counts = [], []
    n_extra = 0
    mat = sampler.pool.mat
    sup_fast = sampler._support_patched
    t_loop = time.time()
    for i, x in enumerate(obs_peps):
        enc = {}
        # enc2: codon-path encodings from AA-neighbor parents
        o0, o1 = csr['off'][i], csr['off'][i + 1]
        for p in csr['idx'][o0:o1]:
            P = sampler.pep_mat[p]
            diff = P != obs_codes[i]
            js = np.where(diff & (P < _NC) & (obs_codes[i] < _NC))[0]
            if len(js) == 0:
                continue
            a, b = sampler.pep_mem['off'][p], sampler.pep_mem['off'][p + 1]
            for y in sampler.pep_mem['idx'][a:b]:
                for ys, arr in sampler._codon_encodings(y, js, obs_codes[i]):
                    if ys not in table_nucs:
                        enc.setdefault(ys, (arr, y))
        # enc3: synonymous single-position variants when x is a pool peptide
        pi = pool_pid.get(x)
        if pi is not None:
            a, b = sampler.pep_mem['off'][pi], sampler.pep_mem['off'][pi + 1]
            for y in sampler.pep_mem['idx'][a:b]:
                for j in range(len(x)):
                    if obs_codes[i][j] >= _NC:
                        continue
                    for ys, arr in sampler._codon_encodings(
                            y, np.array([j]), obs_codes[i]):
                        if ys not in table_nucs:
                            enc.setdefault(ys, (arr, y))
        if enc:
            if keep_seqs:
                extra_seqs.extend(enc.keys())
            # fast path: identical value to _support_of(arr, y0) — per-parent
            # cached terms patched at the variant sites (exactness-tested in
            # tests/test_sampler_exact.py and tests/test_pos_fastpath.py).
            # sites is nonempty in practice (table-identical encodings are
            # excluded above); the fallback keeps the edge well-defined.
            vals = []
            for ys, (arr, y0) in enc.items():
                sites = np.nonzero(arr != mat[y0])[0]
                vals.append(sup_fast(y0, sites, arr[sites]) if sites.size
                            else sampler._support_of(arr, y0))
            sup_chunks.append(np.asarray(vals, dtype=np.float64))
        counts.append(len(enc))
        n_extra += len(enc)
        if (i + 1) % 20000 == 0:
            el = time.time() - t_loop
            eta = el / (i + 1) * (len(obs_peps) - i - 1)
            log(f'  POS_L1: {i + 1}/{len(obs_peps)} peptides, '
                f'{n_extra} extras, {el:.0f}s elapsed, '
                f'ETA ~{eta / 60:.0f} min', flush=True)

    extra_off = np.concatenate(
        ([0], np.cumsum(np.asarray(counts, dtype=np.int64)))).astype(np.int64)
    extra_support = (np.concatenate(sup_chunks) if sup_chunks
                     else np.zeros(0, dtype=np.float64))
    out = {'extra_support': extra_support, 'extra_off': extra_off,
           'n_extra': int(extra_support.shape[0])}
    if cfg.cache.use_support_cache and not keep_seqs:
        cache_save(rp['cache']['cache_dir'], name, out)
    if keep_seqs:
        out['extra_seqs'] = extra_seqs
    log(f'  POS_L1 computed: {n_extra} extra encodings over '
        f'{len(obs_peps)} observed peptides ({time.time() - t0:.0f}s)')
    return out
