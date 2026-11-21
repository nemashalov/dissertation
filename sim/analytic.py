#!/usr/bin/env python3
"""
Analytical / fluid-approximation model for the gang-reserve scenario.

Complements the discrete simulator (gang_reserve.py). Derives closed-form
approximations for reserve utilization and mean waiting time and compares them
to simulation across the arrival-rate (load) axis, to study whether the
dependency can be approximated analytically.

Setting (simulation plan P1.1):
  - Reserve grows linearly C_avail(t) = R_gang*(1 - t/T) for t in [0,T], 0 else.
  - Tasks arrive as Poisson(lambda), work w_i ~ dist, run at x in [x_min,x_max],
    speed x^alpha (Prasanna-Musicus). Must finish by T (released at launch).
  - Total reserve area A = int_0^T C_avail(t) dt = R_gang*T/2 (linear).

Fluid model:
  - Work arrives at rate a = lambda*E[w] (work units per unit time).
  - A task arriving at time a is placeable iff it can finish by T at some
    x<=x_max, i.e. w_i <= x_max^alpha*(T-a). So the expected work a task can
    complete on the reserve is E[w ^ c(a)] with c(a)=x_max^alpha*(T-a).
  - Cumulative placeable work W(a) = int_0^a lambda*E[w ^ c(s)] ds.
  - Utilization_analytical = min(1, W(T)/A).
  - Backlog B(t) (fluid): work that arrived and is placeable but not yet served.
    Service rate = min(C_avail(t), placeable arrival rate) when backlog>0.
    B'(t) = a_p(t) - C_avail(t)  if B(t)>0,  else max(0, a_p(t)-C_avail(t)),
    where a_p(t)=lambda*E[w ^ c(t)] is the placeable arrival rate.
  - Mean wait approximation (Little's-law analog over horizon):
    Wbar = ( int_0^T B(t) dt + W_unplaced*(horizon-T) ) / (lambda*horizon)
    where W_unplaced = W(T) - (work actually served) is work that could not fit
    and waits for the main cluster until horizon.
"""
import argparse
import numpy as np


def e_w_cap(dist, base, c):
    """E[min(w, c)] for the given work distribution (base = mean scale)."""
    if c <= 0:
        return 0.0
    if dist == 'exp':
        # E[min(W,c)] with W~Exp(1/base): c - base*(1 - exp(-c/base))
        return c - base * (1.0 - np.exp(-c / base))
    if dist == 'lognorm':
        # numeric: W ~ lognormal(log(base), 0.8)
        rng = np.random.default_rng(0)
        w = rng.lognormal(np.log(base), 0.8, size=200000)
        return float(np.mean(np.minimum(w, c)))
    if dist == 'uniform':
        # W ~ U(0.2*base, 1.8*base)
        lo, hi = 0.2 * base, 1.8 * base
        if c <= lo:
            return 0.0
        if c >= hi:
            return 0.5 * (lo + hi)
        # E[min] over U(lo,hi)
        return (c * (c - lo) / (hi - lo)
                + 0.5 * (hi - c) * (hi + c) / (hi - lo))
    raise ValueError(dist)


def e_w_le(dist, base, c):
    """E[W * 1_{W<=c}] for the given distribution (base = mean scale)."""
    if c <= 0:
        return 0.0
    if dist == 'exp':
        # W~Exp(1/base): E[W 1_{W<=c}] = base - base e^{-c/base}(1+c/base)
        return base - base * np.exp(-c / base) * (1.0 + c / base)
    if dist == 'uniform':
        lo, hi = 0.2 * base, 1.8 * base
        if c <= lo:
            return 0.0
        if c >= hi:
            return 0.5 * (lo + hi)
        return (lo ** 3 * 0.0 + (c ** 3 - lo ** 3)) / (3 * (hi - lo))
    # lognorm: numeric
    rng = np.random.default_rng(0)
    w = rng.lognormal(np.log(base), 0.8, size=200000)
    return float(np.mean(w[w <= c]))


def analytic(R_gang, T, lam, work_base, x_min, x_max, alpha, dist,
             horizon=None, n=4000):
    """Utilization (processor-time based) and mean-wait approximation.

    Matches the simulator's greedy policy: a task arriving at t is placed at
    x_max if it fits by T (w <= x_max^alpha (T-t)), else at x_min if it fits
    (w <= x_min^alpha (T-t)). Processor-time consumed per task = w * x^(1-alpha).
    Utilization = min(1, total placeable processor-time / reserve area A).
    """
    horizon = horizon if horizon is not None else (T + 20.0 * work_base)
    A = R_gang * T / 2.0                      # total reserve area (linear)
    ts = np.linspace(0.0, T, n)
    dt = ts[1] - ts[0]
    pt = np.zeros_like(ts)                    # placeable processor-time rate
    for i, t in enumerate(ts):
        cmax = max(0.0, x_max ** alpha * (T - t))   # fits at x_max
        cmin = max(0.0, x_min ** alpha * (T - t))   # fits at x_min (deadline)
        # tasks placed at x_max: w <= cmax; at x_min: cmax < w <= cmin
        e_max = e_w_le(dist, work_base, cmax)
        e_min = e_w_le(dist, work_base, cmin) - e_max
        pt[i] = lam * (e_max * x_max ** (1 - alpha)
                       + e_min * x_min ** (1 - alpha))
    total_pt = pt.sum() * dt                  # total placeable processor-time
    util = min(1.0, total_pt / A)
    # backlog fluid ODE in processor-time: B'(t)=pt - C_avail when B>0
    B = np.zeros_like(ts)
    C = R_gang * (1.0 - ts / T)
    for i in range(1, n):
        d = pt[i - 1] - C[i - 1]
        if B[i - 1] > 0:
            B[i] = max(0.0, B[i - 1] + d * dt)
        else:
            B[i] = max(0.0, d * dt)
    area_B = B.sum() * dt                     # int_0^T B(t) dt (processor-time)
    served = min(total_pt, A)                 # capacity-capped served pt
    unplaced_pt = max(0.0, total_pt - served)  # could not fit by T
    # Little's law analog: total wait = area_B + unplaced_pt*(horizon-T)
    total_wait = area_B + unplaced_pt * (horizon - T)
    n_arr = lam * horizon
    mean_wait = total_wait / n_arr if n_arr > 0 else 0.0
    return util, mean_wait, total_pt, A, served


# ---------------------------------------------------------------------------
# Extreme Value Theory model for the gang-task straggler penalty.
#
# The completion time of a gang (brigade) task with N synchronous threads is
# the MAXIMUM of the N thread service times (all-reduce waits for the slowest).
# By Extreme Value Theory (EVT), as N grows the distribution of this maximum
# converges to one of three domains (Gumbel / Frechet / Weibull); for the
# light-tailed service times typical of computer systems it is Gumbel.
# This is the analytical counterpart of Barroso's "tail at scale": the same
# rare tail that is negligible at N=1 dominates at large N.
#
# For thread time X ~ Exp(1/theta) (mean theta):
#   E[max_i X_i] = theta * H_N ~ theta*(ln N + gamma),  gamma ~ 0.5772
#   P(max > q)   = 1 - (1 - e^{-q/theta})^N
# Gumbel approximation: location mu = theta*ln N, scale beta = theta.
# ---------------------------------------------------------------------------
_HARMONIC = None


def _h_n(N):
    global _HARMONIC
    if _HARMONIC is None or len(_HARMONIC) < N + 1:
        import math
        _HARMONIC = [0.0] * (N + 1)
        for i in range(1, N + 1):
            _HARMONIC[i] = _HARMONIC[i - 1] + 1.0 / i
    return _HARMONIC[N]


def evt_gang_max(N, theta, q=None):
    """Analytical EVT straggler metrics for a gang task with N threads.

    X_i ~ Exp(1/theta) (light-tailed, mean theta). Returns dict with
    E[max], the tail probability P(max>q) at scale N (if q given), and the
    Gumbel location/scale.
    """
    e_max = theta * _h_n(N)
    gumbel_loc = theta * np.log(N)
    gumbel_scale = theta
    tail = None
    if q is not None:
        tail = 1.0 - (1.0 - np.exp(-q / theta)) ** N
    return {'N': N, 'theta': theta, 'E_max': e_max,
            'gumbel_loc': gumbel_loc, 'gumbel_scale': gumbel_scale,
            'tail_at_q': tail, 'q': q}


def evt_tail_at_scale(q, theta, Ns):
    """P(max of N > q) as a function of N (Barroso's 1% -> 63% example)."""
    return {N: 1.0 - (1.0 - np.exp(-q / theta)) ** N for N in Ns}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--Rgang', type=float, default=16)
    ap.add_argument('--T', type=float, default=10.0)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--work-base', type=float, default=2.0)
    ap.add_argument('--xmin', type=float, default=1)
    ap.add_argument('--xmax', type=float, default=4)
    ap.add_argument('--alpha', type=float, default=0.8)
    ap.add_argument('--dist', default='exp', choices=['exp', 'lognorm', 'uniform'])
    ap.add_argument('--horizon', type=float, default=None)
    # EVT block
    ap.add_argument('--evt', action='store_true',
                    help='EVT straggler model: E[max] of gang task over N threads')
    ap.add_argument('--theta', type=float, default=1.0,
                    help='mean thread service time for EVT (Exponential)')
    ap.add_argument('--Ns', type=int, nargs='+', default=[1, 8, 16, 32, 64, 100],
                    help='thread counts N for EVT tail-at-scale')
    ap.add_argument('--tailq', type=float, default=None,
                    help='quantile q for P(max>q) at scale (default 5*theta)')
    args = ap.parse_args()

    if args.evt:
        theta = args.theta
        q = args.tailq if args.tailq is not None else 5.0 * theta
        print(f"=== Extreme Value Theory: gang-task straggler (theta={theta}) ===")
        print(f"EVT domain: Gumbel (light-tailed Exp service times)")
        print(f"{'N':>5}{'E[max]':>12}{'Gumbel loc':>12}{'Gumbel scale':>14}"
              f"{'P(max>q)':>12}")
        for N in args.Ns:
            r = evt_gang_max(N, theta, q)
            print(f"{N:>5}{r['E_max']:>12.3f}{r['gumbel_loc']:>12.3f}"
                  f"{r['gumbel_scale']:>14.3f}{r['tail_at_q']:>12.4f}")
        # Barroso's 1% -> 63% at-scale illustration
        ts = evt_tail_at_scale(q, theta, args.Ns)
        print(f"\nTail at scale P(max>{q:.1f}*theta) vs N  "
              f"(Barroso 8.4.4 example):")
        print(f"{'N':>5}{'P(max>q)':>12}{'growth vs N=1':>18}")
        base = ts.get(1, None)
        for N, p in ts.items():
            g = f"{(p/ts[1]):.1f}x" if base and base > 0 else "n/a"
            print(f"{N:>5}{p:>12.4f}{g:>18}")
        return

    util, mw, W_T, A, served = analytic(
        args.Rgang, args.T, args.lam, args.work_base, args.xmin, args.xmax,
        args.alpha, args.dist, args.horizon)
    print(f"analytic: R_gang={args.Rgang} T={args.T} lambda={args.lam} "
          f"work_base={args.work_base} x=[{args.xmin},{args.xmax}] "
          f"alpha={args.alpha} dist={args.dist}")
    print(f"  reserve area A={A:.1f}  placeable work W(T)={W_T:.2f}")
    print(f"  utilization(analytic)={util:.3f}  mean_wait(analytic)={mw:.3f}")


if __name__ == '__main__':
    main()