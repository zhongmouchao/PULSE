# PULSE with SGD: Implementation Notes (pool + batch version)

This document describes how the PULSE pipeline (`pulse/`) implements
stochastic gradient descent for the PULSE peptide-observation model. It
follows the structure and notation of the reference writeup
`SGD_PULSE_reference.pdf` (the June one-batch version) and highlights where
the current pool + batch implementation differs.

The two structural changes relative to June are:

1. **Pool + batch surveillance design.** The observed nucleotide table is
   split by time into a **pool** (the first whole time columns holding
   ~`pool.fraction` of all counts, default 1%) and a **batch** (everything
   later, plus the `unknown` column). Emergence support always flows from
   pool parents; the batch holds the sequences whose emergence we explain.
2. **Multi-mutation proposal.** The unobserved space is sampled with up to
   `n_max` (default 3) simultaneous amino-acid mutations per draw, with the
   mutation count drawn from a truncated Poisson. The proposal probability
   `q` and its rejection normalization `Z` are computed exactly.

---

## 1. Likelihood

Let $\mathcal{D}^Y_{\text{pool}}$ be the set of pool nucleotide sequences
(parents), $\mathcal{D}^Y_{\text{obs}}$ the set of observed batch nucleotide
sequences that pass the distance filter (Section 3), and
$\mathcal{D}^X_{\text{obs}}$ the corresponding observed amino-acid
sequences. For an amino-acid sequence $x$, let $\hat{\mathcal{Y}}(x)$ be its
tractable encoding set (Section 3.3). The approximated log-likelihood is

$$
\hat{\ell}(\theta, \beta, p_o)
= \sum_{x \in \mathcal{D}^X_{\text{obs}}}
\log \Big( p_a(x; \theta) \, s(x; \beta, p_o) \Big)
+ \sum_{x' \in \mathcal{U}}
\log \Big( 1 - p_a(x'; \theta) \, s(x'; \beta, p_o) \Big),
$$

where the **detection probability** of any peptide $x$ is

$$
s(x; \beta, p_o)
= 1 - \prod_{y \in \hat{\mathcal{Y}}(x)}
\big(1 - p_o \, p_e(y; \beta)\big),
\qquad
p_e(y; \beta) = 1 - \exp\big(-\beta \, S(y)\big),
$$

$p_a(x;\theta)$ is the activity probability from the classifier (LR or NAM)
over peptide features, $p_o$ the observation probability, $S(y)$ the
emergence support of nucleotide $y$ from the pool (Section 3.2), and
$\mathcal{U}$ the unobserved peptide space — every peptide reachable from
the pool that is **not** any table peptide.

Correspondence with the June notation: $\beta$ plays the role of $\alpha T$
(the population scale $T$ is absorbed into $\beta$), and pool frequency
$\text{freq}_0(y)$ plays the role of $f(y)$.

The unobserved sum over $\mathcal{U}$ cannot be enumerated, so it is
estimated by importance sampling (Section 5). The observed sum is computed
exactly at every step.

## 2. Why the unobserved sum must be importance-sampled

The June writeup (its Section 3) showed that naively subsampling observed
sequences and mutating only those does **not** give an unbiased gradient,
because the same unobserved $x'$ would carry different encoding sets
$\hat{\mathcal{Y}}(x')$ in different minibatches. The fix is the same here:
for each **sampled** $x'$ we compute its contribution **deterministically** —
its encoding set, the support of each encoding, and its proposal probability
$q(x')$ — from one-time pool structures that never change during training.
The estimator (Section 5) is then unbiased for the finite sum over
$\mathcal{U}$.

## 3. Observed side (one-time precompute, cached)

All of Section 3 is computed once per (data, pool split, rate model,
`obs_dist_max`, `support_max_dist`) and cached to disk
(`OBS_L0`/`OBS_L2`/`POS_L1` files in `support_cache/`); training reruns load
it.

### 3.1 Pool / batch split

Time columns are recognized as `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`
(zero-padded, so string order = chronological order), plus a literal
`before 2000` column that sorts first. Walking columns in chronological
order, the pool takes whole columns up to and including the first column
whose cumulative count reaches `pool.fraction` of the grand total (whole
columns are kept even when they overshoot the fraction). Every later column
plus `unknown` is batch. Each row then gets `Pool0_Count` and
`Batch_Count`.

Row filters applied at load time, each reporting its drop count:

- **length filter**: rows whose nucleotide or peptide length differs from
  the first row's are dropped;
- **zero-count filter**: rows with no counts in any time column or
  `unknown` are dropped;
- **stop codons**: nucleotide sequences translating with an internal `*`
  are excluded from supports (they never become parents or encodings);
- optional `drop_gap_or_ambiguous` (off by default): rows whose nucleotide
  string is not pure `ACGT` are dropped. By default, `---`/`NNN` rows are
  kept: gap/ambiguous sites are simply excluded from distance and rate
  computations on both sides ("mutually valid sites"), and the encoded
  amino acids `X`/`-` carry a $(0,0,0)$ descriptor row.

### 3.2 Support of an observed batch sequence

A unique batch nucleotide sequence $y$ (Batch_Count > 0) is **kept** if its
minimal distance to the pool is $\le$ `obs_dist_max` (default 2; distance =
Hamming over mutually valid sites) and **excluded** otherwise — exclusions
are reported.

The support of a kept sequence aggregates every independent way the pool
can produce it:

$$
S(y) \;=\;
\underbrace{\sum_{y_p \in \mathcal{D}^Y_{\text{pool}},\;
1 \le d(y_p, y) \le d_S}
P_{\min}(y_p \!\to\! y)\; \text{freq}_0(y_p)}_{\text{mutation from other parents}}
\;+\;
\underbrace{\mathbf{1}[y \in \text{pool}]\;
P_{\text{self}}(y)\; \text{freq}_0(y)}_{\text{self-survival (exact match only)}}
$$

with $d_S$ = `support_max_dist` (default 3) and

$$
P_{\min}(y_p \!\to\! y) = \prod_{i:\, y_p[i] \ne y[i],\; \text{both valid}}
\mu\big(y_p[i] \!\to\! y[i]\big),
\qquad
P_{\text{self}}(y) = \prod_{i \text{ valid}}
\Big(1 - \sum_{b' \ne y[i]} \mu\big(y[i] \!\to\! b'\big)\Big).
$$

$P_{\min}$ is the minimal-pathway rate product: because per-site mutation
rates are tiny, paths with extra mutations are negligible, so the direct
product over the differing sites dominates and is used exactly. Note the
two distances are independent knobs: `obs_dist_max` decides **which** batch
sequences are observed; `support_max_dist` decides **which parents** may
contribute support (kept sequences farther than `support_max_dist` from all
parents get mutation support 0 — only self-survival if applicable). A
distance-0 "parent" that is merely a gap/ambiguity variant of $y$ (same
over mutually valid sites, different string) contributes nothing — matching
the rule that self-support is exact-match only.

Per-site rates $\mu(b \!\to\! b')$ come from the mutation model:

- default `ti90_tv5`, *per-base* normalization: transition $1.8\times10^{-4}$,
  transversion $1\times10^{-5}$, so ti + 2 tv $= 2\times10^{-4}$ at every site;
- opt-in `sars_cov2`, *spectrum* normalization: the 12-pathway weight table
  (C$\to$T 46.5, G$\to$T 18.2, A$\to$G 9.4, T$\to$C 9.4, G$\to$A 7.4, the
  remaining seven transversions 1.3) scaled so the genome-average
  per-nucleotide rate is $\mu_{\text{total}} = 2\times10^{-4}$
  (rate $= 4\mu_{\text{total}} w / \sum_{12} w$). Cross-pathway ratios then
  equal the weight ratios exactly; per-base totals differ (C sites mutate
  fastest).

### 3.3 Peptide aggregation and $\hat{\mathcal{Y}}(x)$

Observed batch nucleotides are translated and grouped; **peptide support is
the sum over its observed encodings** (each encoding's $S(y)$ already
bundles its own self-survival term).

For the detection product, $\hat{\mathcal{Y}}(x)$ of an observed peptide
spans not only its observed batch encodings but also **extra encodings**
(computed once, cached as `POS_L1`):

- **enc2**: minimal-codon-path encodings of $x$ from pool parents whose
  peptide is at amino-acid distance $1 \dots n_{\max}$ from $x$ (same
  machinery and nucleotide budget as the sampler, Section 4.3);
- **enc3**: when $x$ is itself a pool peptide, single-position synonymous
  codon variants ($k = 1 \dots n_{\max}$ changes in one codon) of its pool
  member nucleotides.

Table nucleotide strings are excluded from the extras (observed encodings
already enter with their own support). Every extra encoding carries the
exact support of Section 3.2.

## 4. The proposal distribution over unobserved peptides

There is no candidate enumeration and no support threshold anywhere on the
unobserved side: the space is sampled, never materialized.

### 4.1 One raw draw

1. **Root**: sample a pool peptide $x \sim r$, where $r$ is the
   pool-frequency distribution aggregated over synonymous pool nucleotides:
   $r(x) = \sum_{y_p \in \text{pool},\, \text{pep}(y_p)=x} \text{freq}_0(y_p)$.
   Roots with no canonical residue (all `X`/`-`/`*`) get weight 0.
2. **Mutation count**: $n \sim$ Poisson$(\lambda)$ truncated to
   $\{1, \dots, \min(n_{\max}, L_x)\}$ and renormalized per root, where
   $L_x$ is the number of canonical residues of $x$ and
   $\lambda = L_{\text{nuc}} \cdot \mu_{\text{total}}$ by default
   (`proposal.lam: auto`; any positive float overrides). This is the
   expected number of nucleotide mutations per sequence per replication
   cycle, so the draw mass is concentrated at $n = 1$ with a tail decaying
   at scale $\mu_{\text{total}}$.
3. **Mutation**: pick $n$ distinct canonical positions of $x$ uniformly
   without replacement; at each, the new residue is uniform among the other
   19.
4. **Rejection**: if $x'$ is **any table peptide** (a translation of any
   nucleotide row of the training file — pool, batch, or `unknown`), reject
   and resample.

### 4.2 Exact proposal probability

For a parent peptide $p$ at amino-acid distance $m \in \{1, \dots, n_{\max}\}$
from $x'$, the probability that one raw draw from $p$ produces $x'$ is

$$
q_{\text{mut}}(x' \mid p) \;=\; \frac{P_p(n = m)}
{\binom{L_p}{m}\, 19^{m}},
\qquad
P_p(n=m) = \frac{\lambda^m / m!}{\sum_{j=1}^{\min(n_{\max}, L_p)} \lambda^j / j!},
$$

and the raw proposal mass of $x'$ sums over **all** candidate parents —
the actual root plus every pool peptide at distance $1 \dots n_{\max}$:

$$
q_{\text{raw}}(x') \;=\; \sum_{m=1}^{n_{\max}}
\sum_{p:\, d_{aa}(p, x') = m} r(p)\, q_{\text{mut}}(x' \mid p).
$$

Omitting any parent would make the importance weight wrong and the
estimator biased — this is the same requirement as in the June writeup,
extended to $m > 1$. (Implementation: candidate parents come from a
one-time pool×pool amino-acid neighbor table with cutoff $2 n_{\max}$ —
a draw from root $x$ with $n \le n_{\max}$ mutations can only coincide with
a parent within $2 n_{\max}$ of $x$.)

After rejection, the normalized proposal is

$$
q(x') = \frac{q_{\text{raw}}(x')}{Z},
\qquad
Z = 1 - q_{\text{raw}}(\text{table})
= 1 - \sum_{p,\, m}
\frac{r(p)\, P_p(n=m)}{\binom{L_p}{m} 19^{m}} \, N_{\text{table}}(p, m),
$$

where $N_{\text{table}}(p, m)$ = number of table peptides at amino-acid
distance $m$ from pool peptide $p$. $Z$ is computed **exactly** from a
one-time pool-peptide × table-peptide distance pass. Only when that product
is too large ($> 3\times10^8$ pairs) does a Monte-Carlo pilot
($K = 200{,}000$ raw draws) estimate $Z$ instead; the mode is printed at
startup. A warning is printed if $Z < 0.5$ (more than half of all draws
rejected).

### 4.3 Encodings of a sampled peptide and their support

For an accepted draw $x'$ we deterministically build its encoding set
$\hat{\mathcal{Y}}(x')$:

- **Parents**: the actual root plus every pool peptide at amino-acid
  distance $1 \dots n_{\max}$ from $x'$.
- **Codon paths**: for parent peptide $p$, member nucleotide $y_p$, and
  differing-residue positions $J$: mutations are confined to the codons of
  $J$; at position $j$ any target codon of the new residue is allowed;
  the total number of nucleotide changes over all of $J$ must be
  $\le n_{\max}$. If $y_p$'s actual codon at $j$ cannot reach the target
  residue within budget, that parent contributes **zero** encodings —
  codon context is respected exactly.
- **Support**: each unique encoding $y'$ receives the exact multi-distance
  support of Section 3.2 — the sum over all pool parents at nucleotide
  distance $1 \dots$ `support_max_dist` of $P_{\min} \cdot \text{freq}_0$
  (candidate parents are found by a triangle-inequality neighbor search
  around the generating parent at cutoff $n_{\max} + d_S$). The same $y'$
  reached from two parents is merged once into the set; its support
  already aggregates all parents, so nothing is double-counted.

A draw with no reachable encoding has $s(x') = 0$ **exactly** and
contributes $g = \log(1 - 0) = 0$: a wasted draw, but the estimator stays
unbiased.

## 5. The SGD estimator

Writing the unobserved contribution of peptide $x'$ as

$$
g(x'; \theta, \beta, p_o) = \log\big(1 - p_a(x'; \theta)\, s(x'; \beta, p_o)\big),
$$

the unobserved part of the likelihood is the finite sum
$\hat{L}_{\text{unobs}} = \sum_{x' \in \mathcal{U}} g(x')$.
With $B$ accepted draws $x'_b \sim q$ per step,

$$
\widetilde{L}_{\text{unobs}} = \frac{1}{B} \sum_{b=1}^{B}
\frac{g(x'_b)}{q(x'_b)},
\qquad
\mathbb{E}_q\big[\widetilde{L}_{\text{unobs}}\big]
= \hat{L}_{\text{unobs}},
$$

because $q(x') > 0$ for every $x' \in \mathcal{U}$ (any reachable peptide
has positive raw mass and $Z > 0$) and $q(x') = 0$ on table peptides by
rejection. Since each $g(x'_b)$ is computed deterministically from the
one-time structures, the stochastic gradient is unbiased for the
approximated finite-sum objective.

The full per-step objective minimized by the optimizer is

$$
-\Bigg[
\underbrace{\sum_{x \in \mathcal{D}^X_{\text{obs}}}
\big(\log p_a(x) + \log s(x)\big)}_{\text{exact, every step}}
\;+\;
\underbrace{\frac{1}{B}\sum_{b=1}^{B}
\frac{\log\big(1 - p_a(x'_b)\, s(x'_b)\big)}{q(x'_b)}}_{\text{importance-sampled}}
\Bigg]
\;+\;
\lambda_{\text{reg}}\Big(\sum_{w \in \theta} w^2 + p_o^2\Big).
$$

One term per observed peptide, regardless of how many times it was seen in
surveillance (surveillance counts entered earlier, through the pool
frequencies and the pool/batch split — this is the September one-batch
convention, deliberately different from June).

## 6. Pseudocode

```
Algorithm: PULSE SGD (pool + batch, multi-mutation proposal)

Input:  training counts table, benchmark table, config
One-time (cached):
  1: filter rows (length, zero-count, stop codons); split pool / batch
  2: pool parents: unique pool nucleotides, freq_0, codon table,
     peptide grouping
  3: OBS_L0: table x pool nucleotide neighbor pass
     (cutoff max(obs_dist_max, support_max_dist)); true minimal distance
     per table row
  4: OBS_L2: keep batch sequences with min distance <= obs_dist_max;
     support S(y) per kept sequence (mutation sum + self-survival);
     group by peptide; peptide support = sum over encodings
  5: SAMP_L1: root weights r(x); pool-pool amino-acid neighbor CSR
     (cutoff 2*n_max); pool-pool nucleotide neighbor CSR
     (cutoff n_max + support_max_dist); table-peptide set;
     N_table counts -> exact Z (or MC pilot)
  6: POS_L1: extra encodings (enc2, enc3) of observed peptides
Training:
  7: draw the FIXED validation batch once (val_seed, 500 draws)
  8: for step t = 1, 2, ... do
  9:    batch <- B accepted draws (default B = 128):
 10:       repeat
 11:          x  ~ r ;  n ~ truncPoisson(lambda | x)
 12:          n distinct canonical positions; new residue ~ uniform(19)
 13:       until x' not a table peptide
 14:       q(x')   = [ W(x, n) + sum over AA-neighbor parents W(p, m) ] / Z
 15:       Yhat(x') = minimal-codon-path encodings (budget n_max);
 16:                  exact support per encoding
 17:    loss = -( positive_term + mean_b[ log(1 - p_a s)/q ] ) + reg
 18:    AdamW step on (theta, d_raw, b_raw); grad-norm clip 1.0
 19:    every 10 steps: validate on the fixed batch; keep best checkpoint;
 20:       stop after early_stop_checks (default 100) checks without
 21:       min_delta (default 1e-4) improvement
 22: restore best checkpoint; report beta = exp(d_raw), p_o = sigmoid(b_raw)
 23: score benchmark with p_a; AUC / AP / Spearman; append results row;
     save model .pt + predictions .csv under models/<run_tag>
```

## 7. Parameterization and numerical guards

- $\beta = \exp(d_{raw})$ — free (no bound), initialized at
  $d_{raw} = -\log(\text{median of positive supports})$ so that
  $\beta S$ starts in a gradient-sensitive zone.
- $p_o = \text{sigmoid}(b_{raw})$, initialized at 0.5.
- Probability clamps: $p_e, p_o, p_a \in [\texttt{p\_floor}, 1]$ with
  `p_floor` $= 10^{-8}$ by default; $\log(1 - p_o p_e)$ is clamped at
  `p_floor`. These clamps are applied **uniformly on both sides of the
  likelihood** — observed peptides and sampled unobserved draws flow
  through the same detection computation, so every encoding's $p_e$
  (observed or generated) is floored, as is every $p_a$. An observed
  peptide with zero support therefore still gets $p_e = $ `p_floor`
  instead of an exact 0 — the numerical floor is the only bound in the
  system; $\beta$ itself is unbounded.
- Zero-encoding rows are handled structurally, not by clamps: a peptide
  (observed or drawn) with an empty encoding set has
  $\sum \log(\cdot) = 0 \Rightarrow s = 0$ exactly, because the padded
  support slots are masked out before the sum.
- L2 penalty on network weights and $p_o$ (`reg_lambda`, default 0);
  $\beta$ is never penalized.
- Checkpoint selection: the full objective (exact positive term +
  importance-sampled unlabeled term) is evaluated every
  `validation_interval` (10) steps on all observed peptides plus the fixed
  validation draws; the best checkpoint is restored at the end. AdamW
  (lr 0.01, weight decay 0.01) with ReduceLROnPlateau (patience 200 steps,
  factor 0.5).

## 8. Key defaults

| setting | default | meaning |
|---|---|---|
| `pool.fraction` | 0.01 | pool = first whole time columns holding ≥ 1% of counts |
| `observation.obs_dist_max` | 2 | batch sequences farther from pool are excluded |
| `observation.support_max_dist` | 3 | parents beyond this distance contribute no support |
| `proposal.n_max` | 3 | max mutations per draw / encoding nt-change budget |
| `proposal.lam` | auto | $\lambda = L_{\text{nuc}} \cdot \mu_{\text{total}}$ |
| `mutation_model` | ti90_tv5, $\mu_{\text{total}}=2\times10^{-4}$ | sars_cov2 spectrum opt-in |
| `training.sgd_batch_size` | 128 | accepted draws per step |
| `training.fixed_validation_size` | 500 | fixed validation draws |
| `training.early_stop_checks` | 100 | × 10 steps = 1000-step patience |
| `training.p_floor` | 1e-8 | clamp floor for $p_e, p_o, p_a$ |

## 9. Differences from the June reference at a glance

| | June (one batch) | current (pool + batch) |
|---|---|---|
| parents of support | all observed sequences | pool sequences only |
| proposal space | 1 AA mutation | 1..3 AA mutations, truncated Poisson count |
| $q_{\text{raw}}$ | $\sum_{p:\,d_{aa}=1} r(p)/19L$ | $\sum_{m=1}^{3}\sum_{p:\,d_{aa}=m} r(p)P_p(n=m)/(\binom{L_p}{m}19^m)$ |
| $Z$ | exact, 1-mutation counts | exact, 1..3-mutation counts (MC fallback) |
| encodings of $x'$ | 1-nt codon changes, one differing residue | minimal codon paths over ≤3 residues, ≤3 nt changes total, codon-context aware |
| support per encoding | transition/transversion rate × observed count | exact minimal-pathway rate product over pool parents at distance 1..3, × freq_0 |
| self path | — | explicit self-survival for exact pool members (always on) |
| observed-side $\hat{\mathcal{Y}}(x)$ | observed encodings only | + extra generated encodings (POS_L1) |
| surveillance counts in loss | per-count weighting | one term per peptide |
