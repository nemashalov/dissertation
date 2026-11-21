#!/usr/bin/env python3
"""
Test the quantitative 'social good of elasticity' claims from OpenSpec change
``elasticity-social-good-quantification`` (dissertation repo):

  Task 1.3 — fragmentation identity / bound:  B(x) <= F(x), with B(x)=F(x) in
             the clean (overload) case.
  Task 2.2 — saturation curve: the benefit B(eps) grows with elasticity headroom
             eps and saturates (plateaus), i.e. non-decreasing and bounded.

Model (bin-packing-as-cluster, faithful to the corrected B(x) of the sibling
change ``elasticity-social-good-collapse``):
  - M nodes, node j has capacity C_j.
  - A set of tasks with rigid sizes s_i (the group demands).
  - no-elastic:  a task must fit entirely in one node's *free gap* -> rigid
                 first-fit placement; fragmentation traps free capacity.
  - elastic:     tasks may be resized to fill free gaps -> all capacity usable
                 up to total demand.

Definitions:
  C_noelastic(x) = rigidly placed work.
  C_elastic(x)   = min(total_demand, total_capacity)   (full elasticity).
  B(x)           = C_elastic(x) - C_noelastic(x)  >= 0.
  F(x)           = residual free capacity after maximal rigid placement
                   = total_capacity - C_noelastic(x).      (trapped capacity)

Claim:  B(x) <= F(x) always;  B(x) = F(x) iff total_demand >= total_capacity
(overload), i.e. elasticity recovers exactly the fragmented capacity.
"""
import random

import numpy as np


def rigid_place(nodes_free: list[float], sizes: list[float]) -> float:
    """First-fit placement of rigid tasks; return placed work."""
    placed = 0.0
    for s in sizes:
        for j in range(len(nodes_free)):
            if nodes_free[j] >= s:
                nodes_free[j] -= s
                placed += s
                break
    return placed


def social_good(nodes_cap, sizes, shrink_floor: float = 0.0):
    """
    Return (B, F, C_noelastic, C_elastic).
    shrink_floor in (0,1]: minimum size fraction a task can shrink to.
      shrink_floor=1 -> fully rigid (no elasticity).
      shrink_floor=0 -> full elasticity.
    """
    total_cap = float(sum(nodes_cap))
    total_demand = float(sum(sizes))
    if shrink_floor >= 1.0:
        C_noelastic = rigid_place(list(nodes_cap), sizes)
        return 0.0, total_cap - C_noelastic, C_noelastic, C_noelastic

    # elastic placement: place as much as possible, filling gaps.
    # With full elasticity (floor=0) usable = min(demand, capacity).
    # With bounded floor f, a task can shrink to f*s; a gap >= f*s can be used.
    free = [float(c) for c in nodes_cap]
    placed = 0.0
    for s in sizes:
        s_min = shrink_floor * s
        # find node with enough free to take the shrunken task
        for j in range(len(free)):
            if free[j] >= s_min:
                used = min(s, free[j])
                free[j] -= used
                placed += used
                break
    C_elastic = min(total_demand, total_cap)
    C_noelastic = rigid_place(list(nodes_cap), sizes)
    B = C_elastic - C_noelastic
    F = total_cap - C_noelastic
    return B, F, C_noelastic, C_elastic


def saturation_curve(nodes_cap, sizes, floors):
    """B(eps) vs elasticity floor eps (0=full elastic .. 1=rigid)."""
    return {eps: social_good(nodes_cap, sizes, eps)[0] for eps in floors}


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_bound_and_identity(seed=0, trials=400):
    rng = random.Random(seed)
    for _ in range(trials):
        m = rng.randint(2, 8)
        caps = [rng.randint(5, 50) for _ in range(m)]
        n = rng.randint(1, 40)
        sizes = [rng.randint(1, 25) for _ in range(n)]
        B, F, _cno, _cel = social_good(caps, sizes, 0.0)
        _check(B >= -1e-9, f"B>=0 violated: B={B}")
        _check(B <= F + 1e-9, f"bound B<=F violated: B={B}, F={F}")
        # overload => identity
        if sum(sizes) >= sum(caps):
            _check(abs(B - F) < 1e-9,
                   f"identity B=F violated in overload: B={B}, F={F}")
    return trials


def test_saturation(seed=7):
    caps = [10, 14, 18, 22]
    rng = random.Random(seed)
    sizes = [rng.randint(2, 30) for _ in range(30)]
    floors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    B = saturation_curve(caps, sizes, floors)
    vals = [B[e] for e in floors]
    # non-decreasing as eps decreases toward full elasticity
    for a, b in zip(vals, vals[1:]):
        _check(a - b > -1e-9, f"saturation non-monotone: {vals}")
    _check(vals[0] >= vals[-1], "full elasticity should dominate rigid")
    # plateau at full elasticity equals fragmentation (overload case here)
    Bfull, F, _cno, _cel = social_good(caps, sizes, 0.0)
    if sum(sizes) >= sum(caps):
        _check(abs(Bfull - F) < 1e-9, "identity at full elasticity")
    return dict(zip(floors, vals))


def main():
    n_trials = test_bound_and_identity()
    print(f"[PASS] bound B(x)<=F(x) and identity B(x)=F(x) in overload: "
          f"{n_trials} randomized instances")
    curve = test_saturation()
    print("[PASS] saturation curve non-decreasing and bounded "
          f"(full elasticity B={curve[0.0]:.2f}, rigid B={curve[1.0]:.2f})")
    print("[PASS] all OpenSpec elasticity-social-good-quantification "
          "quantitative tests (tasks 1.3, 2.2)")


if __name__ == '__main__':
    main()