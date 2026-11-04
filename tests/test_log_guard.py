"""Boundary test for the float32 log-floor guard in unlabeled_term
(pulse/train_sgd.py).

Two independent float32 roundings can collide in the unlabeled loss term:

  * pa's clamp ceiling 1 - p_floor = 1 - 1e-8 ROUNDS TO EXACTLY 1.0 in
    float32 (eps near 1.0 is ~1.2e-7), so a confident model output pa = 1.0
    passes the "clamp" untouched;
  * p_un = -expm1(sum_log) saturates to exactly 1.0 for draws with many
    encodings (e.g. (1 - 0.107)^194 ~ 3e-10 -> 1 - 3e-10 -> 1.0 in float32).

The OLD expression log1p(-pa*p_un) then evaluates log(0) = -inf; the inf
gradient becomes NaN under gradient clipping (0 * inf) and poisons every
parameter in one step (the step-640 XBB15 / step-4100 Eta failures). The
guarded form log(clamp(1 - pa*p_un, min=p_floor)) is finite there and is
bit-comparable to log1p everywhere the argument is float32-representable.

Run:  python tests/test_log_guard.py
"""
import sys

import numpy as np
import torch

fails = []
def check(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fails.append(name)


pf = 1e-8

# the two roundings that collide
check('clamp ceiling 1 - 1e-8 rounds to 1.0 in float32',
      float(torch.tensor([2.0]).clamp(max=1 - pf)[0]) == 1.0)
sum_log = torch.tensor(194.0) * np.log(1 - 0.107)   # saturated-pool draw, p_o=0.107
check('p_un saturates to exactly 1.0 in float32',
      float(-torch.expm1(sum_log.float())) == 1.0)

# old vs guarded at the boundary
one = torch.ones(4)
old = torch.log1p(-one * one)
new = torch.log(torch.clamp(1 - one * one, min=pf))
check('old log1p(-pa*p_un) = -inf at pa=p_un=1 (the bug)',
      bool(torch.isinf(old).all()) and bool((old < 0).all()))
check('guarded form finite at the boundary', bool(torch.isfinite(new).all()))
check('guarded form equals log(p_floor) there',
      abs(float(new[0]) - np.log(pf)) < 1e-5)

# gradient behavior at the boundary
pa_g = torch.ones(1, requires_grad=True)
f_old = torch.log1p(-pa_g * 1.0)
f_old.backward()
g_old = pa_g.grad.clone()
pa_g2 = torch.ones(1, requires_grad=True)
f_new = torch.log(torch.clamp(1 - pa_g2 * 1.0, min=pf))
f_new.backward()
check('old backward gives non-finite grad', not bool(torch.isfinite(g_old).all()))
check('guarded backward gives finite grad', bool(torch.isfinite(pa_g2.grad).all()))

# identical to log1p wherever the argument is representable
pa = torch.tensor([0.7, 0.2, 0.95, 0.999])
pu = torch.tensor([0.4, 0.9, 0.1, 0.5])
ref = torch.log1p(-pa * pu)
got = torch.log(torch.clamp(1 - pa * pu, min=pf))
check('identical to log1p in the normal regime',
      bool(torch.allclose(ref, got, rtol=1e-6, atol=1e-8)))

print()
print('ALL PASS' if not fails else f'FAILURES: {fails}')
sys.exit(1 if fails else 0)
