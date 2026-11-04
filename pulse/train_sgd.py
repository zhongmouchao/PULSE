"""train_sgd.py — SGD training of the PULSE peptide-observation model.

One run = one task config (+ optional overrides / grid combo). All heavy
precomputes come from cache (OBS_L0/L2 via generate_support.py; SAMP_L1 and
POS_L1 are computed on first use and cached), so training reruns — other
batch sizes, models, representations, seeds — are cheap. Each run appends
one tagged row to artifacts.results_csv.

Loss (peptide-observation model; the unlabeled term is importance-sampled
on the go rather than summed over an enumerated candidate universe):
  observed term (EXACT, all observed peptides x, every step):
      log p_a(x) + log(1 - prod_{y in Yhat(x)} (1 - p_o * p_e(y)))
      Yhat(x) = observed batch encodings + POS_L1 extra encodings
  unlabeled (B fresh proposal draws per step):
      U_hat = mean_b [ log(1 - p_a(x'_b) * s(x'_b)) / q(x'_b) ]
      s(x') = 1 - prod_{y' in enc(x')} (1 - p_o * p_e(y'))
      (a draw with no encodings has s = 0 EXACTLY -> contributes 0)
  p_e(y) = 1 - exp(-beta * S(y)); beta = exp(d_raw) free and unpenalized;
  p_o = sigmoid(b_raw); loss = -(observed + U_hat) + reg.

Checkpoint selection: every validation_interval steps, the objective is
evaluated on all observed peptides + a FIXED set of proposal draws
(val_seed); best checkpoint restored at the end; early stopping after
early_stop_checks checks without min_delta improvement.

Usage:
    python pulse/train_sgd.py --task SARS_viral_growth
    python pulse/train_sgd.py --task X --set training.sgd_batch_size=64
    python pulse/train_sgd.py --task X --sweep sweeps/ablation_batch.yaml
    python pulse/train_sgd.py --all --sweep sweeps/default_grid.yaml
"""
import argparse
import os
import random
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pulse.task_config import load_task, list_tasks, bounded_filename
from pulse.support import ensure_observed_support
from pulse.sampler import Sampler, positive_extra_encodings
from pulse.featurizers import build_featurizer, featurize_chunked
from pulse.models import build_model
from pulse.generate_support import parse_set, grid_combos, deep_merge, combo_label


# ==================== flat (CSR) support container ====================
class FlatSupports:
    """Flat CSR-style encoding supports — replaces padded (S, mask) matrices.

    S: (E,) float32 tensor of per-encoding supports; seg: (E,) int64 tensor
    mapping each encoding to its row (peptide or draw); n: number of rows.
    Per-row sums are computed with index_add_, which is mathematically
    identical to the masked dense row-sum (padded slots contributed exactly
    0) but needs O(E) memory instead of O(n x max_encoding_count).
    """
    def __init__(self, lists):
        n = len(lists)
        lengths = np.fromiter((len(v) for v in lists), dtype=np.int64, count=n)
        self.n = n
        if lengths.sum() > 0:
            self.S = torch.tensor(np.concatenate(lists), dtype=torch.float32)
            self.seg = torch.tensor(np.repeat(np.arange(n), lengths),
                                    dtype=torch.int64)
        else:
            self.S = torch.zeros(0, dtype=torch.float32)
            self.seg = torch.zeros(0, dtype=torch.int64)

    @classmethod
    def from_flat(cls, S_flat, seg, n):
        """Build directly from a concatenated support array + segment ids
        (segments need not be contiguous)."""
        obj = cls.__new__(cls)
        obj.n = n
        obj.S = torch.tensor(np.ascontiguousarray(S_flat), dtype=torch.float32)
        obj.seg = torch.tensor(np.ascontiguousarray(seg), dtype=torch.int64)
        return obj


def batch_tensors(draws, feat):
    """Sampler draws -> (X, FlatSupports, q) tensors."""
    peps = [d[0] for d in draws]
    X = torch.tensor(feat.featurize(peps))
    S = FlatSupports([[s for _, s in encs] for _, _, encs in draws])
    q = torch.tensor([d[1] for d in draws], dtype=torch.float32)
    return X, S, q


# ==================== benchmark evaluation ====================
def evaluate_benchmark(model, cfg, feat):
    """Score the benchmark peptides with p_a. DMS benchmarks: AUC/AP with the
    functional label = score >= threshold (median by default) plus Spearman
    against the raw score. Activity benchmarks (binary labels): AUC/AP.
    Returns (metrics dict, per-sequence predictions DataFrame)."""
    rp = cfg.resolve_paths()
    df = pd.read_csv(rp['data']['benchmark_file'])
    bc = cfg.data.benchmark
    peps = df[bc.sequence_column].astype(str).tolist()
    y = pd.to_numeric(df[bc.score_column], errors='coerce').to_numpy()
    ok = ~np.isnan(y)
    with torch.no_grad():
        scores = model(torch.tensor(feat.featurize(peps))).squeeze(-1).numpy()
    peps_ok = list(np.array(peps, dtype=object)[ok])
    muts_ok = (df[bc.mutation_column].astype(str).to_numpy()[ok]
               if bc.type == 'dms' and bc.mutation_column in df.columns else None)
    scores, y = scores[ok], y[ok]
    out = {'n_benchmark': int(ok.sum()),
           'auc': float('nan'), 'ap': float('nan'), 'spearman': float('nan')}
    if bc.type == 'dms':
        from scipy.stats import spearmanr
        if len(y) > 2:
            out['spearman'] = float(spearmanr(scores, y).statistic)
        thr = (float(np.median(y)) if bc.functional_threshold == 'median'
               else float(bc.functional_threshold))
        labels = (y >= thr).astype(int)
    else:  # activity: score column already holds 0/1 labels
        labels = y.astype(int)
    if len(labels) and labels.min() != labels.max():
        from sklearn.metrics import roc_auc_score, average_precision_score
        out['auc'] = float(roc_auc_score(labels, scores))
        out['ap'] = float(average_precision_score(labels, scores))
    preds = pd.DataFrame({'sequence': peps_ok, 'bench_score': y, 'p_a': scores})
    if muts_ok is not None:
        preds.insert(1, 'mutation', muts_ok)
    return out, preds


# ==================== one training run ====================
def run_training(cfg, over_label='', verbose=True):
    t0 = time.time()
    tr = cfg.training
    torch.manual_seed(tr.seed)
    np.random.seed(tr.seed)
    random.seed(tr.seed)

    res = ensure_observed_support(cfg, verbose=False)
    sampler = Sampler(cfg, res, verbose=False)
    feat = build_featurizer(cfg)
    obs = res['obs']
    extra = positive_extra_encodings(sampler, res, verbose=verbose)
    obs_peps = obs['obs_peps']

    # positive side: per observed peptide, observed + extra encodings.
    # Flat CSR layout — exactly what the OBS_L2 / POS_L1 caches already
    # store: one concatenated (E,) support array + per-encoding segment ids.
    # O(E) memory; no padded (n_peptides x max_encodings) matrix is built.
    # Chunked featurization: peak = final float32 matrix + one chunk (the
    # all-at-once numpy build + torch copy would double the peak — fatal at
    # full-spike scale, e.g. 1M x 3807 float32 = 15 GB x 2).
    X_POS = torch.from_numpy(featurize_chunked(feat, obs_peps))
    n_pos = len(obs_peps)
    obs_cnt = np.diff(obs['pep_off'])        # obs encodings per peptide
    ext_cnt = np.diff(extra['extra_off'])    # POS_L1 extras per peptide
    S_pos_np = np.concatenate([obs['obs_support'][obs['enc_flat']],
                               extra['extra_support']])
    seg_pos_np = np.concatenate([np.repeat(np.arange(n_pos), obs_cnt),
                                 np.repeat(np.arange(n_pos), ext_cnt)])
    POS_SUP = FlatSupports.from_flat(S_pos_np, seg_pos_np, n_pos)

    pos_sup = S_pos_np[S_pos_np > 0]
    d_init = -float(np.log(np.median(pos_sup))) if len(pos_sup) else 0.0

    model = build_model(cfg, X_POS.shape[1])
    d_raw = torch.tensor([d_init], requires_grad=True)   # beta = exp(d_raw), free
    b_raw = torch.zeros(1, requires_grad=True)           # p_o = sigmoid(b_raw)
    opt = torch.optim.AdamW(list(model.parameters()) + [d_raw, b_raw],
                            lr=tr.lr, weight_decay=tr.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=max(1, tr.scheduler_patience // tr.validation_interval),
        factor=tr.scheduler_factor)
    pf = tr.p_floor

    print(f'  setup done ({time.time() - t0:.0f}s): {n_pos} observed peptides, '
          f'{POS_SUP.S.shape[0]} encodings; drawing validation batch...', flush=True)
    t_vd = time.time()
    val = batch_tensors(sampler.draw_batch(
        tr.fixed_validation_size, rng=np.random.default_rng(tr.val_seed)), feat)
    t_val_draw = time.time() - t_vd
    print(f'  validation batch: {tr.fixed_validation_size} draws in '
          f'{t_val_draw:.1f}s ({t_val_draw / tr.fixed_validation_size * 1e3:.1f} ms/draw)',
          flush=True)

    def detection_terms(SF):
        """SF: FlatSupports. Per-row detection probability
        s = 1 - prod(1 - p_o * p_e) via segment sums; a row with no
        encodings gets sum_log = 0 -> s = 0 EXACTLY (contributes g = 0)."""
        beta = torch.exp(d_raw)
        p_o = torch.clamp(torch.sigmoid(b_raw), pf, 1 - pf)
        p_e = torch.clamp(-torch.expm1(-beta * SF.S), pf, 1.0)
        log1m = torch.log(torch.clamp(1 - p_o * p_e, min=pf))
        sum_log = torch.zeros(SF.n, dtype=torch.float32).index_add_(
            0, SF.seg, log1m)
        return p_o, -torch.expm1(sum_log)

    def positive_term():
        _, p_pos = detection_terms(POS_SUP)
        pa = torch.clamp(model(X_POS).squeeze(-1), pf, 1 - pf)
        return torch.sum(torch.log(pa) + torch.log(torch.clamp(p_pos, min=pf)))

    def unlabeled_term(B):
        _, p_un = detection_terms(B[1])
        pa = torch.clamp(model(B[0]).squeeze(-1), pf, 1 - pf)
        # = log(1 - p_a * s), exact at s = 0; floored like every other log in
        # this file. In float32 the pa clamp max (1 - 1e-8) ROUNDS TO 1.0 and
        # p_un can be exactly 1.0 (draws with ~200 encodings saturate), so
        # log1p(-pa*p_un) could be log(0) = -inf -> inf grads -> grad clipping
        # turns 0*inf into NaN and poisons every parameter in one step.
        f = torch.log(torch.clamp(1 - pa * p_un, min=pf))
        return (f / B[2]).mean()             # unbiased estimate of the full sum

    def full_loss(B):
        p_o = torch.sigmoid(b_raw)
        reg = tr.reg_lambda * (sum(torch.sum(p * p) for p in model.parameters())
                               + p_o ** 2)
        return -(positive_term() + unlabeled_term(B)) + reg

    best = {'val': float('inf'), 'step': 0, 'state': None, 'd': None, 'b': None}
    checks, hist, steps_run = 0, [], 0
    t_draw, t_step, t_val = 0.0, 0.0, 0.0     # per-phase wall-clock accumulators
    for step in range(1, tr.num_sgd_steps + 1):
        steps_run = step
        opt.zero_grad()
        t_d = time.time()
        B = batch_tensors(sampler.draw_batch(tr.sgd_batch_size), feat)
        t_draw += time.time() - t_d
        t_s = time.time()
        loss = full_loss(B)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr.max_grad_norm)
        torch.nn.utils.clip_grad_norm_([d_raw, b_raw], tr.max_grad_norm)
        opt.step()
        t_step += time.time() - t_s
        hist.append(loss.item())
        if step % tr.validation_interval == 0 or step == tr.num_sgd_steps:
            t_v = time.time()
            with torch.no_grad():
                vl = full_loss(val).item()
            t_val += time.time() - t_v
            sched.step(vl)
            if vl < best['val'] - tr.min_delta:
                best.update(val=vl, step=step,
                            state={k: v.clone() for k, v in model.state_dict().items()},
                            d=d_raw.detach().clone(), b=b_raw.detach().clone())
                checks = 0
            else:
                checks += 1
                if checks >= tr.early_stop_checks:
                    break
        if verbose and step % tr.print_interval == 0:
            print(f'  step {step}/{tr.num_sgd_steps}  '
                  f'train {np.mean(hist[-10:]):.4f}  val {best["val"]:.4f} '
                  f'@{best["step"]}  beta {torch.exp(d_raw).item():.4e}  '
                  f'p_o {torch.sigmoid(b_raw).item():.4f}  '
                  f'({time.time() - t0:.0f}s | draw {t_draw / step:.2f}s/step, '
                  f'fwd+bwd {t_step / step:.3f}s/step, val {t_val:.0f}s total)',
                  flush=True)

    model.load_state_dict(best['state'])
    beta, p_o = float(torch.exp(best['d'])), float(torch.sigmoid(best['b']))
    metrics, preds = evaluate_benchmark(model, cfg, feat)

    rates = cfg.mutation_model.rates
    run_tag = compute_run_tag(cfg, over_label)
    row = {'task': cfg.task, 'run_tag': run_tag,
           'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'pool_fraction': cfg.pool.fraction,
           'obs_dist_max': cfg.observation.obs_dist_max,
           'support_max_dist': cfg.observation.support_max_dist,
           'rates': rates if isinstance(rates, str) else 'explicit',
           'mu_total': cfg.mutation_model.mu_total,
           'lam': sampler.lam, 'n_max': cfg.proposal.n_max,
           'sgd_batch_size': tr.sgd_batch_size, 'num_sgd_steps': tr.num_sgd_steps,
           'reg_lambda': tr.reg_lambda, 'model': cfg.model.classifier,
           'hidden_dims': str(cfg.model.hidden_dims), 'dropout': cfg.model.dropout,
           'representation': cfg.representation.name, 'seed': tr.seed,
           'best_val_loss': best['val'], 'best_step': best['step'],
           'steps_run': steps_run, 'beta': beta, 'p_o': p_o,
           'n_obs_peptides': len(obs_peps),
           'wall_time_s': round(time.time() - t0, 1), **metrics}
    path = cfg.resolve_paths()['artifacts']['results_csv']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path),
                               index=False)
    # ---- persist model checkpoint + benchmark predictions for downstream
    # re-evaluation (e.g. different functional thresholds via pulse/evaluate.py)
    models_dir = cfg.resolve_paths()['artifacts']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    pt_path = os.path.join(models_dir, bounded_filename(run_tag, '.pt'))
    torch.save({'run_tag': run_tag, 'model_state': best['state'],
                'd_raw': best['d'], 'b_raw': best['b'],
                'config': asdict(cfg)}, pt_path)
    pred_path = os.path.join(models_dir, bounded_filename(run_tag, '_predictions.csv'))
    preds.to_csv(pred_path, index=False)
    print(f'[{cfg.task}] {run_tag}\n'
          f'  best val {best["val"]:.4f} @ step {best["step"]}  '
          f'beta {beta:.4e}  p_o {p_o:.4f}  '
          f'AUC {metrics["auc"]:.4f}  AP {metrics["ap"]:.4f}  '
          f'Spearman {metrics["spearman"]:.4f}  ({row["wall_time_s"]}s)\n'
          f'  results -> {path}\n'
          f'  model -> {pt_path}\n'
          f'  predictions -> {pred_path}')
    return row


def compute_run_tag(cfg, over_label):
    """Deterministic run identity: same task + same overrides -> same tag."""
    return cfg.artifacts.run_tag or (
        cfg.task if over_label in ('', '(defaults)') else f'{cfg.task}[{over_label}]')


def existing_run_tags(path):
    """run_tags already present in the results CSV (empty set if none/readable)."""
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path, usecols=['run_tag'])['run_tag'])
    except Exception:
        return set()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', help='task name (tasks/<name>.yaml)')
    ap.add_argument('--all', action='store_true', help='run every task in tasks/')
    ap.add_argument('--sweep', help='YAML file with a `grid:` dict of dotted-key lists')
    ap.add_argument('--set', action='append', default=[], help='section.key=value overrides')
    ap.add_argument('--force', action='store_true',
                    help='rerun even if the results CSV already has this run_tag')
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
    print(f'tasks: {tasks}; runs per task: {len(sweep_overrides)}')
    for task in tasks:
        for over in sweep_overrides:
            label = combo_label(over)
            try:
                cfg = load_task(task, overrides=over)
                results_path = cfg.resolve_paths()['artifacts']['results_csv']
                run_tag = compute_run_tag(cfg, label)
                if not args.force and run_tag in existing_run_tags(results_path):
                    print(f'[{task}] {label}\n  -> SKIP: {results_path} already has '
                          f'run_tag {run_tag!r} (use --force to rerun)')
                    continue
                run_training(cfg, over_label=label)
            except Exception as e:
                print(f'[{task}] {label}\n  -> FAILED: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
