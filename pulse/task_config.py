"""Task config layer: one YAML per task in tasks/, strict load + validate.

Design (on-the-go SGD): unobserved peptides are SAMPLED during training
(proposal over pool roots with a truncated-Poisson mutation count), never
enumerated. Hence there is no candidate/EPS configuration. The only heavy
precompute is the observed-sequence support (pulse/support.py).

Usage:
    from pulse.task_config import load_task
    cfg = load_task('SARS_viral_growth')                          # by name
    cfg = load_task('SARS_viral_growth', overrides={'observation': {'obs_dist_max': 3}})

CLI:
    python pulse/task_config.py SARS_viral_growth              # print summary
"""
import hashlib
import os
import re
from dataclasses import dataclass, field, fields, asdict

try:
    import yaml
except ImportError:
    yaml = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(PROJECT_ROOT, 'tasks')


def safe_tag(run_tag):
    """Filesystem-safe version of a run tag (model/prediction filenames)."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(run_tag))


NAME_BUDGET = 250      # ext4/APFS filename component limit is 255 bytes

def bounded_filename(run_tag, suffix):
    """safe_tag(run_tag) + suffix when that fits the filename budget; otherwise
    a recognizable prefix + 10-char hash of the FULL tag, so many-knob sweep
    tags cannot crash file writes (Errno 36). Deterministic and collision-safe;
    bit-identical to the historical scheme for every tag that fits, so all
    existing model/prediction files keep their names."""
    base = safe_tag(run_tag)
    if len(base) + len(suffix) <= NAME_BUDGET:
        return base + suffix
    h = hashlib.md5(str(run_tag).encode()).hexdigest()[:10]
    keep = NAME_BUDGET - len(suffix) - 11          # '_' separator + hash
    return f'{base[:keep]}_{h}{suffix}'


@dataclass
class BenchmarkConfig:
    type: str = 'dms'                      # dms | activity
    mutation_column: str = 'mutation'
    sequence_column: str = 'mutated_sequence'
    score_column: str = 'DMS_score'
    functional_threshold: object = 'median'  # 'median' | float


@dataclass
class DataConfig:
    training_file: str = ''
    benchmark_file: str = ''
    nuc_column: str = 'Motif Nucleotide'
    pep_column: str = 'Motif Peptide'
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


@dataclass
class PoolConfig:
    mode: str = 'time_fraction'        # 'time_fraction' | 'prevalence'
    fraction: float = 0.01             # time_fraction: earliest time columns up to
                                       # this cumulative share of total counts
    prevalence_threshold: float = 0.01 # prevalence: parents are rows whose file-wide
                                       # count share is strictly above this


@dataclass
class MutationConfig:
    mu_total: float = 0.0002       # mutation rate per nucleotide per cycle
    rates: object = 'ti90_tv5'     # preset name, or explicit {'C>T': w, ...} 12-pathway weights
    normalization: object = None   # 'per_base' | 'spectrum'; None = preset-implied
                                   # (per_base for ti90_tv5 / explicit maps, spectrum for sars_cov2)


# Relative weight per ordered pathway (parent base > target base). How weights
# become per-site rates depends on the normalization:
#   per_base: rate(b->b') = mu_total * w / W_out(b), W_out(b) = sum of b's 3
#     outgoing weights. Every site's outgoing pathways sum to exactly mu_total
#     (e.g. ti90_tv5: ti = 1.8e-4, tv = 1e-5, ti + 2*tv = 2e-4).
#   spectrum: rate = 4 * mu_total * w / sum(all 12 weights). All cross-pathway
#     ratios equal the weight ratios exactly; the genome-average per-nucleotide
#     rate is mu_total, but per-base totals differ (e.g. sars_cov2: C sites
#     mutate at 3.93e-4, A/T sites at 9.6e-5).
RATE_PRESETS = {
    # all viruses except SARS-CoV-2: transitions 90, transversions 5
    'ti90_tv5': {'C>T': 90.0, 'A>G': 90.0, 'G>A': 90.0, 'T>C': 90.0,
                 'G>T': 5.0, 'C>G': 5.0, 'A>C': 5.0, 'A>T': 5.0,
                 'C>A': 5.0, 'G>C': 5.0, 'T>A': 5.0, 'T>G': 5.0},
    # SARS-CoV-2 mutational spectrum (opt-in bias; spectrum normalization)
    'sars_cov2': {'C>T': 46.5, 'A>G': 9.4, 'G>A': 7.4, 'T>C': 9.4,
                  'G>T': 18.2, 'C>G': 1.3, 'A>C': 1.3, 'A>T': 1.3,
                  'C>A': 1.3, 'G>C': 1.3, 'T>A': 1.3, 'T>G': 1.3},
}
PRESET_NORMALIZATION = {'ti90_tv5': 'per_base', 'sars_cov2': 'spectrum'}
_ALL_PATHWAYS = list(RATE_PRESETS['ti90_tv5'])


def resolve_rates(mut):
    """-> (rates dict ('C>T' -> per-site rate), weight dict). Raises on bad input."""
    w = RATE_PRESETS.get(mut.rates) if isinstance(mut.rates, str) else None
    implied = PRESET_NORMALIZATION.get(mut.rates) if isinstance(mut.rates, str) else None
    if w is None:
        if not isinstance(mut.rates, dict):
            raise ValueError(f'mutation_model.rates must be one of {sorted(RATE_PRESETS)} '
                             f'or an explicit pathway->weight map, got {mut.rates!r}')
        keys = set(mut.rates)
        if keys != set(_ALL_PATHWAYS):
            raise ValueError(f'mutation_model.rates map must have exactly the 12 pathways '
                             f'{_ALL_PATHWAYS}; missing {sorted(set(_ALL_PATHWAYS) - keys)}, '
                             f'extra {sorted(keys - set(_ALL_PATHWAYS))}')
        w = {k: float(v) for k, v in mut.rates.items()}
    if any(v <= 0 for v in w.values()):
        raise ValueError('mutation_model.rates weights must be > 0')
    if mut.mu_total <= 0:
        raise ValueError('mutation_model.mu_total must be > 0')
    norm = mut.normalization or implied or 'per_base'
    if norm == 'per_base':
        out = {}
        for b in ('A', 'C', 'G', 'T'):
            outgoing = [(k, v) for k, v in w.items() if k[0] == b]
            w_out = sum(v for _, v in outgoing)
            for k, v in outgoing:
                out[k] = mut.mu_total * v / w_out
        return out, w
    if norm == 'spectrum':
        W = sum(w.values())
        return {k: 4 * mut.mu_total * v / W for k, v in w.items()}, w
    raise ValueError(f"mutation_model.normalization must be 'per_base' or 'spectrum', "
                     f'got {norm!r}')


@dataclass
class ObservationConfig:
    """Support of OBSERVED batch sequences (one-time precompute, cached):
    keep batch sequences with minimal pool distance <= obs_dist_max, then
    S(y) = sum over pool parents at distance 1..support_max_dist of
    (minimal-pathway rate product) * parent frequency. Self-survival of pool
    members (nucleotide level, exact string match) is always included."""
    obs_dist_max: int = 2            # inclusion filter on observed batch sequences
    support_max_dist: int = 3        # parents beyond this distance contribute nothing
    max_degenerate_frac: object = None  # None = off; otherwise exclude observed
                                        # sequences whose X+'-' share of peptide
                                        # length strictly exceeds this fraction


@dataclass
class ProposalConfig:
    """On-the-go proposal over unobserved peptides (SGD sampler):
    root peptide x ~ r (pool frequency); mutation count n ~ P(n), a Poisson
    truncated to {1..n_max} and renormalized; n distinct canonical positions
    and new residues uniform; reject if x' is any table peptide."""
    n_max: int = 3                   # largest mutation count a draw can produce
    count_dist: str = 'poisson'      # only 'poisson' for now
    lam: object = 'auto'             # 'auto' (data-derived from pool + rates) or float > 0
    root: str = 'pool_frequency'     # root distribution over pool peptides


@dataclass
class FilterConfig:
    drop_gap_or_ambiguous: bool = False


@dataclass
class ModelConfig:
    """Activity model p_a(x) — registry in pulse/models.py. 'LR' (linear +
    sigmoid) or 'NAM' (neural additive model: each position gets its OWN
    one-hidden-layer net of width hidden_dims[0] over its chemical3 triplet;
    peptide logit = sum over sites scaled by 1/sqrt(n_sites), then sigmoid).
    A new architecture = a new registered class there plus any new
    hyperparameter fields here."""
    classifier: str = 'LR'
    hidden_dims: list = field(default_factory=lambda: [4])   # NAM only (one entry)
    dropout: float = 0.0   # reserved for future architectures; ignored by LR/NAM


@dataclass
class RepresentationConfig:
    """Peptide representation — registry in pulse/featurizers.py.
    'chemical3': per-residue (hydrophobicity, mol, volume) triplets flattened
    (non-canonical residues -> (0,0,0)); requires equal-length peptides.
    Future: 'esm2' etc."""
    name: str = 'chemical3'


@dataclass
class TrainingConfig:
    reg_lambda: float = 0.0          # L2 on network weights + p_o (beta unpenalized)
    sgd_batch_size: int = 128        # unlabeled draws per SGD step (sweepable)
    num_sgd_steps: int = 30000
    validation_interval: int = 10    # validation check every N steps
    fixed_validation_size: int = 500 # fixed draws for checkpoint selection
    early_stop_checks: int = 100     # stop after this many checks without min_delta improvement;
                                     # patience in steps = checks x validation_interval (100x10 = 1000)
    min_delta: float = 1e-4
    lr: float = 0.01                 # AdamW
    weight_decay: float = 0.01       # AdamW
    max_grad_norm: float = 1.0
    scheduler_patience: int = 200    # ReduceLROnPlateau patience, in STEPS
    scheduler_factor: float = 0.5
    print_interval: int = 500
    p_floor: float = 1e-8            # numerical clamp floor for p_e / p_o / p_a
    seed: int = 42
    val_seed: int = 43


@dataclass
class CacheConfig:
    use_support_cache: bool = True
    cache_dir: str = 'support_cache'


@dataclass
class ArtifactsConfig:
    run_tag: object = None                 # None -> auto-generated
    results_csv: str = 'results/results.csv'
    models_dir: str = 'models'             # per-run .pt checkpoint + benchmark predictions


@dataclass
class TaskConfig:
    task: str = ''
    description: str = ''
    data: DataConfig = field(default_factory=DataConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    mutation_model: MutationConfig = field(default_factory=MutationConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)

    def resolve_paths(self, base_dir=None):
        """Absolute data/cache/results paths (relative ones anchor at the project root)."""
        base = base_dir or PROJECT_ROOT
        def r(p):
            return p if (p is None or os.path.isabs(p)) else os.path.normpath(os.path.join(base, p))
        d = asdict(self)
        d['data']['training_file'] = r(self.data.training_file)
        d['data']['benchmark_file'] = r(self.data.benchmark_file)
        d['cache']['cache_dir'] = r(self.cache.cache_dir)
        d['artifacts']['results_csv'] = r(self.artifacts.results_csv)
        d['artifacts']['models_dir'] = r(self.artifacts.models_dir)
        return d

    def summary(self):
        d = asdict(self)
        lines = [f'task: {self.task}', f'  data: {self.data.training_file} -> {self.data.benchmark_file}']
        for sec in ('pool', 'mutation_model', 'observation', 'proposal', 'filter',
                    'model', 'representation', 'training', 'cache', 'artifacts'):
            lines.append(f'  {sec}: {d[sec]}')
        return '\n'.join(lines)


def _read_yaml(path):
    text = open(path).read()
    if yaml is not None:
        return yaml.safe_load(text)
    raise ImportError('pyyaml is required to read task configs')


def _deep_merge(base, over, path=''):
    for k, v in over.items():
        if not isinstance(base, dict) or k not in base:
            raise ValueError(f'unknown config key: {path + k!r}')
        if isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v, path + k + '.')
        else:
            base[k] = v


_SECTIONS = {
    'data': (DataConfig, {'benchmark': BenchmarkConfig}),
    'pool': (PoolConfig, {}),
    'mutation_model': (MutationConfig, {}),
    'observation': (ObservationConfig, {}),
    'proposal': (ProposalConfig, {}),
    'filter': (FilterConfig, {}),
    'model': (ModelConfig, {}),
    'representation': (RepresentationConfig, {}),
    'training': (TrainingConfig, {}),
    'cache': (CacheConfig, {}),
    'artifacts': (ArtifactsConfig, {}),
}


def _from_dict(d):
    for k in d:
        if k not in ('task', 'description') and k not in _SECTIONS:
            raise ValueError(f'unknown config section: {k!r}')
    kw = {'task': d.get('task', ''), 'description': d.get('description', '')}
    for sec, (cls, subs) in _SECTIONS.items():
        sd = dict(d.get(sec) or {})
        for sub_name, sub_cls in subs.items():
            sub_val = sd.pop(sub_name, None)
            if sub_val is not None:
                bad = set(sub_val) - {f.name for f in fields(sub_cls)}
                if bad:
                    raise ValueError(f'unknown config key: {sec}.{sub_name}.{sorted(bad)[0]!r}')
                sd[sub_name] = sub_cls(**sub_val)
        bad = set(sd) - {f.name for f in fields(cls)}
        if bad:
            raise ValueError(f'unknown config key: {sec}.{sorted(bad)[0]!r}')
        kw[sec] = cls(**sd)
    return kw


def _validate(cfg):
    if not cfg.task:
        raise ValueError('config must set `task`')
    if not cfg.data.training_file:
        raise ValueError(f'{cfg.task}: data.training_file is required')
    if not cfg.data.benchmark_file:
        raise ValueError(f'{cfg.task}: data.benchmark_file is required')
    if cfg.pool.mode not in ('time_fraction', 'prevalence'):
        raise ValueError(f"{cfg.task}: pool.mode must be 'time_fraction' or 'prevalence'")
    if cfg.pool.mode == 'time_fraction' and not (0.0 < cfg.pool.fraction < 1.0):
        raise ValueError(f'{cfg.task}: pool.fraction must be in (0, 1)')
    if cfg.pool.mode == 'prevalence' and not (0.0 < cfg.pool.prevalence_threshold < 1.0):
        raise ValueError(f'{cfg.task}: pool.prevalence_threshold must be in (0, 1)')
    resolve_rates(cfg.mutation_model)   # raises on bad preset/map/mu_total
    obs = cfg.observation
    if not isinstance(obs.obs_dist_max, int) or obs.obs_dist_max < 1:
        raise ValueError(f'{cfg.task}: observation.obs_dist_max must be an int >= 1')
    if not isinstance(obs.support_max_dist, int) or obs.support_max_dist < 1:
        raise ValueError(f'{cfg.task}: observation.support_max_dist must be an int >= 1')
    mdf = obs.max_degenerate_frac
    if mdf is not None and not (isinstance(mdf, (int, float)) and 0.0 <= mdf < 1.0):
        raise ValueError(f'{cfg.task}: observation.max_degenerate_frac must be '
                         f'None or a number in [0, 1)')
    prop = cfg.proposal
    if not isinstance(prop.n_max, int) or prop.n_max < 1:
        raise ValueError(f'{cfg.task}: proposal.n_max must be an int >= 1')
    if prop.count_dist != 'poisson':
        raise ValueError(f"{cfg.task}: proposal.count_dist must be 'poisson'")
    if not (prop.lam == 'auto' or (isinstance(prop.lam, (int, float)) and prop.lam > 0)):
        raise ValueError(f"{cfg.task}: proposal.lam must be 'auto' or a number > 0")
    if prop.root != 'pool_frequency':
        raise ValueError(f"{cfg.task}: proposal.root must be 'pool_frequency'")
    if cfg.data.benchmark.type not in ('dms', 'activity'):
        raise ValueError(f"{cfg.task}: benchmark.type must be 'dms' or 'activity'")
    th = cfg.data.benchmark.functional_threshold
    if not (th == 'median' or isinstance(th, (int, float))):
        raise ValueError(f"{cfg.task}: benchmark.functional_threshold must be 'median' or a number")
    if cfg.training.num_sgd_steps <= 0 or cfg.training.sgd_batch_size <= 0:
        raise ValueError(f'{cfg.task}: training steps/batch size must be > 0')
    hd = cfg.model.hidden_dims
    if not cfg.model.classifier:
        raise ValueError(f'{cfg.task}: model.classifier must be a registry name')
    if not (isinstance(hd, list) and all(isinstance(h, int) and h > 0 for h in hd)):
        raise ValueError(f'{cfg.task}: model.hidden_dims must be a list of positive ints')
    if cfg.model.classifier == 'NAM' and len(hd) != 1:
        raise ValueError(f'{cfg.task}: NAM gives each site its own '
                         f'one-hidden-layer net; model.hidden_dims must have '
                         f'one entry (per-site width), e.g. [4]')
    if not (0.0 <= cfg.model.dropout < 1.0):
        raise ValueError(f'{cfg.task}: model.dropout must be in [0, 1)')
    if not cfg.representation.name:
        raise ValueError(f'{cfg.task}: representation.name must be a registry name')
    tr = cfg.training
    if tr.lr <= 0 or tr.weight_decay < 0 or tr.max_grad_norm <= 0:
        raise ValueError(f'{cfg.task}: bad optimizer settings (lr/weight_decay/max_grad_norm)')
    if tr.scheduler_patience <= 0 or not (0 < tr.scheduler_factor < 1):
        raise ValueError(f'{cfg.task}: bad scheduler settings')
    if not (0.0 < tr.p_floor < 0.01):
        raise ValueError(f'{cfg.task}: training.p_floor must be in (0, 0.01)')


def load_task(name_or_path, overrides=None):
    """Load a task config by name (tasks/<name>.yaml) or by explicit path.

    overrides: nested dict merged BEFORE validation, e.g.
        {'observation': {'obs_dist_max': 3}, 'training': {'num_sgd_steps': 5000}}
    """
    if os.path.exists(name_or_path):
        path = name_or_path
    else:
        path = os.path.join(TASKS_DIR, name_or_path + '.yaml')
        if not os.path.exists(path):
            raise FileNotFoundError(f'no task config at {path}')
    d = _read_yaml(path) or {}
    base = asdict(TaskConfig())          # full defaults; file values override them
    _deep_merge(base, d)
    if overrides:
        _deep_merge(base, overrides)
    cfg = _from_dict(base)
    cfg = TaskConfig(**cfg)
    _validate(cfg)
    return cfg


def list_tasks(tasks_dir=None):
    td = tasks_dir or TASKS_DIR
    if not os.path.isdir(td):
        return []
    return sorted(f[:-5] for f in os.listdir(td) if f.endswith('.yaml'))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        cfg = load_task(sys.argv[1])
        print(cfg.summary())
        rp = cfg.resolve_paths()
        for k in ('training_file', 'benchmark_file'):
            p = rp['data'][k]
            print(f'{k}: {p}  (exists: {os.path.exists(p)})')
    else:
        print(f'usage: python pulse/task_config.py <task_name>')
        print(f'configured tasks: {list_tasks()}')
