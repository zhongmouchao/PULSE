"""Activity models p_a(x) — registry pattern. A model maps a feature matrix
to probabilities in (0, 1) (sigmoid applied inside forward).

To add a new architecture:
    1. write the nn.Module here (hyperparameters come in via **kw);
    2. register it:  MODELS['my_net'] = MyNet
    3. add any new hyperparameter fields to ModelConfig in task_config.py;
    4. select per task:  model: {classifier: my_net}
"""
import math

import torch
import torch.nn as nn


class LR(nn.Module):
    """Linear + sigmoid (the linear baseline)."""
    def __init__(self, input_dim, **kw):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class NAM(nn.Module):
    """Neural additive model ('per-site MLP'). Each position i gets its OWN
    one-hidden-layer net f_i: R^3 -> R over its chemical3 triplet; the peptide
    logit is the sum of per-site contributions scaled by 1/sqrt(n_sites),
    then one sigmoid:

        p_a(x) = sigmoid( (1/sqrt(n_sites)) * sum_i f_i(x_i) ),
        f_i = Linear_i(3, h) -> ReLU -> Linear_i(h, 1)

    LR is the special case where every f_i is linear, so this strictly
    generalizes LR while keeping the per-site additivity that single-mutant
    DMS benchmarks reward; unlike a shared across-site net, every position
    has its own weights, so position-specific chemistry is learnable. All
    per-site linears are batched into two einsum contractions — one kernel
    call each, no Python loop over sites. Parameters: n_sites * (5h + 1)
    (201 sites, h=4 -> 4,221; LR: 604). The 1/sqrt(n_sites) scaling keeps
    the pre-sigmoid logit in the gradient-healthy zone when many per-site
    contributions align.

    The per-site output layers (W2, b2) are zero-initialized so p_a == 0.5
    for every peptide at init — no random initial ranking is baked in; the
    output weights learn first and the hidden layers follow from step 1.

    Requires input_dim divisible by 3 (the chemical3 triplet layout)."""
    def __init__(self, input_dim, hidden_dim=4, **kw):
        super().__init__()
        if input_dim % 3 != 0:
            raise ValueError(f'NAM expects chemical3 triplet features, got '
                             f'input_dim={input_dim}')
        self.input_dim = input_dim
        self.n_sites = input_dim // 3
        self.hidden_dim = hidden_dim
        self.W1 = nn.Parameter(torch.empty(self.n_sites, 3, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.n_sites, hidden_dim))
        self.W2 = nn.Parameter(torch.zeros(self.n_sites, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.n_sites))
        # xavier-normal for a 3 -> h linear, applied per site
        nn.init.normal_(self.W1, std=math.sqrt(2.0 / (3 + hidden_dim)))
        self.scale = 1.0 / math.sqrt(self.n_sites)

    def forward(self, x):
        sites = x.view(-1, self.n_sites, 3)
        hidden = torch.relu(torch.einsum('bsi,sih->bsh', sites, self.W1) + self.b1)
        contrib = torch.einsum('bsh,sh->bs', hidden, self.W2) + self.b2
        return torch.sigmoid(contrib.sum(dim=1, keepdim=True) * self.scale)


MODELS = {'LR': LR, 'NAM': NAM}


def build_model(cfg, input_dim):
    name = cfg.model.classifier
    cls = MODELS.get(name)
    if cls is None:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    kw = {'hidden_dim': cfg.model.hidden_dims[0]} if cls is NAM else {}
    return cls(input_dim, **kw)
