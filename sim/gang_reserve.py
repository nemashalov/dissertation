#!/usr/bin/env python3
"""
Gang-reserve simulator: elastic tasks utilize capacity gradually reserved for a
large gang (brigade) task, returning resources at its launch.

Follows the event-driven discrete-time methodology of ElastiSim [21] and the
setting formalized in the dissertation simulation plan (P1.1, hypothesis Г4).

Scenario (from annotation, second method):
  - Cluster of total capacity C.
  - A gang task J_gang arrives at t0, requires R_gang processors simultaneously
    (synchronous data parallelism + all-reduce). It launches at time T when the
    gradually reserved capacity r(t) reaches R_gang, r(T) = R_gang.
  - Until T the reserved-but-not-yet-needed capacity C_avail(t) = R_gang - r(t)
    would idle. Elastic tasks are placed on this growing reserve, run on the
    reserved processors, and are compressed to x_min / completed by time T
    (resources returned at launch). Stochastic allocation time T and growth
    profile r(t).

Comparison:
  - WITHOUT elasticity: reserved capacity idles until T; elastic tasks cannot
    use it (a rigid task could not be compressed/evicted by T).
  - WITH elasticity: elastic tasks occupy the reserve, are compressed/evicted
    at T, reducing mean waiting time and raising utilization.

Metrics: mean waiting time, utilization of reserved capacity, gang launch
not delayed (guaranteed by construction: all elastic tasks release by T).
"""
import argparse
import numpy as np


def rng_from_seed(seed):
    return np.random.default_rng(seed)


def draw_work(rng, dist, base):
    """Draw a task's total work w_i. dist in {exp, lognorm, uniform}."""
    if dist == 'exp':
        return rng.exponential(base)
    if dist == 'lognorm':
        return rng.lognormal(np.log(base), 0.8)
    if dist == 'uniform':
        return rng.uniform(base * 0.2, base * 1.8)
    raise ValueError(dist)


class Task:
    __slots__ = ('id', 'arrival', 'work', 'x_min', 'x_max', 'alpha',
                 'start', 'x_alloc', 'finish', 'evicted', 'placed_on_reserve')
    def __init__(self, tid, arrival, work, x_min, x_max, alpha):
        self.id = tid
        self.arrival = arrival
        self.work = work
        self.x_min = x_min
        self.x_max = x_max
        self.alpha = alpha
        self.start = None       # time execution began
        self.x_alloc = None     # processors actually allocated
        self.finish = None      # time execution completed
        self.evicted = False    # was running on reserve and released at T
        self.placed_on_reserve = False


def run_scenario(seed, C, R_gang, t0, T, lam, work_base, x_min, x_max, alpha,
                 dist, reserve_profile='linear', horizon=None, mode='elastic',
                 strategy='backfill'):
    """Run one replication. Returns (tasks, stats).

    mode='elastic'    : elastic tasks use the growing reserve, released at T.
    mode='no-elastic' : reserved capacity idles until T; tasks cannot use it
                        (rigid task cannot be compressed/evicted by launch).
    strategy (elastic only):
      'backfill' : newly arrived queued tasks are placed on new reserve capacity.
      'scale'    : free reserve capacity is first given to already-running
                   elastic tasks (up to x_max), speeding them up; the remainder
                   is backfilled with new tasks. Mirrors "scale current workload
                   vs backfill new ones"."""
    rng = rng_from_seed(seed)
    horizon = horizon if horizon is not None else (T + 20.0 * work_base)

    # ---- arrival process (Poisson) ----
    tasks = []
    t = 0.0
    tid = 0
    while True:
        t += rng.exponential(1.0 / lam) if lam > 0 else float('inf')
        if t > horizon:
            break
        tasks.append(Task(tid, t, draw_work(rng, dist, work_base),
                          x_min, x_max, alpha))
        tid += 1

    # ---- growth profile r(t): reserved capacity for the gang task ----
    def reserve(t):
        if t <= t0:
            return 0.0
        if t >= T:
            return R_gang
        if reserve_profile == 'linear':
            return R_gang * (t - t0) / (T - t0)
        if reserve_profile == 'sqrt':
            u = (t - t0) / (T - t0)
            return R_gang * np.sqrt(u)
        if reserve_profile == 'step':
            # reserve in k steps
            k = 5
            step_t = t0 + (T - t0) * np.floor(k * (t - t0) / (T - t0)) / k
            return R_gang * (step_t - t0) / (T - t0)
        raise ValueError(reserve_profile)

    # ---- event-driven discrete-time loop over task arrivals ----
    # Elastic tasks execute at speed x^alpha on x processors (Prasana-Musicus [29]).
    # Reserve budget at time t available to elastic tasks: C_avail(t) = R_gang - r(t).
    # Place a task on the reserve with the maximum feasible x in [x_min, x_max]
    # such that it can finish by T (resources returned at launch), else wait.
    waiting_time = []        # finish - arrival (or eviction release) for tasks
    util_reserve = 0.0       # area under (reserve used by elastic tasks) / dt
    total_reserve_area = 0.0 # area under C_avail(t) dt (idle capacity available)

    # Simple discrete stepping for the time-dependent reserve area.
    dt = 0.05
    tcur = 0.0
    # sort arrivals; maintain a simple greedy scheduler
    pending = sorted(tasks, key=lambda ta: ta.arrival)
    queue = []               # tasks waiting (arrival order, FCFS)
    active = []              # tasks currently executing on reserve
    pi = 0                   # pointer into pending
    # We step time and at each step schedule new arrivals + advance running tasks.
    # Continuous-time exact integration is overkill; stepping is adequate and
    # reproducible.
    while tcur <= horizon and (pi < len(pending) or queue or active):
        # admit arrivals
        while pi < len(pending) and pending[pi].arrival <= tcur + 1e-12:
            queue.append(pending[pi]); pi += 1
        # free capacity on reserve at current time
        cavail = max(0.0, R_gang - reserve(tcur))
        # processors currently used by active tasks
        used = sum(ta.x_alloc for ta in active)
        free = max(0.0, cavail - used)
        # greedy FCFS placement on reserve (only in elastic mode: a rigid task
        # could not be compressed/evicted by launch time T)
        if mode == 'elastic':
            # 'scale': first give free capacity to already-running tasks (up to
            # x_max), speeding them up, before backfilling new arrivals.
            if strategy == 'scale' and free > 0 and active:
                free_before = free
                active_sorted = sorted(active, key=lambda ta: -ta.x_alloc)
                for ta in active_sorted:
                    if free <= 0:
                        break
                    grow = min(free, ta.x_max - ta.x_alloc)
                    if grow > 0:
                        ta.x_alloc += grow
                        free -= grow
                if free_before != free:
                    used = sum(ta.x_alloc for ta in active)
                    free = max(0.0, cavail - used)
            placed_any = True
            while placed_any and queue and free > 0:
                placed_any = False
                for ta in queue[:]:
                    # choose max feasible x so the task can finish by T,
                    # else reduce to x_min
                    deadline = T  # must release at launch
                    # time available until T
                    dt_left = max(0.0, T - tcur)
                    for x in (ta.x_max, ta.x_min):
                        if x <= free and x >= ta.x_min:
                            dur = ta.work / (x ** ta.alpha)
                            if dur <= dt_left:
                                ta.start = tcur
                                ta.x_alloc = x
                                ta.placed_on_reserve = True
                                active.append(ta)
                                queue.remove(ta)
                                free -= x
                                placed_any = True
                                break
                    if not placed_any:
                        break
        # advance active tasks over dt; those that finish release their x
        finished = []
        for ta in active:
            dur = ta.work / (ta.x_alloc ** ta.alpha)
            if tcur + dt >= ta.start + dur:
                ta.finish = ta.start + dur
                finished.append(ta)
            elif tcur + dt >= T:
                # reached launch: evict (compress to x_min still needs CPU, but
                # the gang task must launch; model as forced release at T)
                ta.finish = T
                ta.evicted = True
                finished.append(ta)
        for ta in finished:
            active.remove(ta)
            waiting_time.append(ta.finish - ta.arrival)
        # accumulate reserve utilization
        used_now = sum(ta.x_alloc for ta in active)
        util_reserve += used_now * dt
        total_reserve_area += cavail * dt
        tcur += dt
        # safety
        if tcur > horizon * 2:
            break

    # tasks never placed: count their (large) waiting as horizon-finish
    for ta in queue:
        waiting_time.append(max(0.0, horizon - ta.arrival))

    stats = {
        'n_tasks': len(tasks),
        'n_placed': len([ta for ta in tasks if ta.placed_on_reserve]),
        'n_evicted': len([ta for ta in tasks if ta.evicted]),
        'n_unplaced': len(queue),
        'mean_wait': float(np.mean(waiting_time)) if waiting_time else 0.0,
        'util_reserve': float(util_reserve / total_reserve_area)
            if total_reserve_area > 0 else 0.0,
        'total_reserve_area': float(total_reserve_area),
    }
    return tasks, stats


def run_oracle(seed, C, R_gang, t0, T, lam, work_base, x_min, x_max, alpha,
               dist, reserve_profile='linear', horizon=None):
    """Theoretical lower bound via a clairvoyant, work-conserving oracle.

    The oracle knows all arrivals in advance and is work-conserving: it keeps
    the growing reserve as full as possible and, among admitted tasks, serves
    those with the smallest remaining work first (SRPT flavour). Because it
    never idles while work is placeable and always runs at the maximum feasible
    allocation, no online scheduler can achieve a strictly smaller mean waiting
    time. Comparing the greedy policy against this oracle yields a competitive
    ratio (upper bound) in the spirit of the dissertation's oracle analysis.
    """
    rng = rng_from_seed(seed)
    horizon = horizon if horizon is not None else (T + 20.0 * work_base)

    # ---- same arrival process as run_scenario (identical seed) ----
    tasks = []
    t = 0.0
    tid = 0
    while True:
        t += rng.exponential(1.0 / lam) if lam > 0 else float('inf')
        if t > horizon:
            break
        tasks.append(Task(tid, t, draw_work(rng, dist, work_base),
                          x_min, x_max, alpha))
        tid += 1

    def reserve(tt):
        if tt <= t0:
            return 0.0
        if tt >= T:
            return R_gang
        if reserve_profile == 'linear':
            return R_gang * (tt - t0) / (T - t0)
        if reserve_profile == 'sqrt':
            u = (tt - t0) / (T - t0)
            return R_gang * np.sqrt(u)
        if reserve_profile == 'step':
            k = 5
            step_t = t0 + (T - t0) * np.floor(k * (tt - t0) / (T - t0)) / k
            return R_gang * (step_t - t0) / (T - t0)
        raise ValueError(reserve_profile)

    # remaining work per task (runs at x <= x_max, speed x^alpha)
    remaining = {ta.id: ta.work for ta in tasks}
    placed = set()
    dt = 0.05
    tcur = 0.0
    active = []          # tasks currently running
    waiting = []         # finish - arrival per task that started
    unplaced = 0
    while tcur <= horizon:
        cavail = max(0.0, R_gang - reserve(tcur))
        used = sum(ta.x_alloc for ta in active)
        free = max(0.0, cavail - used)
        # admitted: arrived, not placed, remaining work > 0
        ready = [ta for ta in tasks
                 if ta.arrival <= tcur + 1e-12 and ta.id not in placed
                 and remaining[ta.id] > 0]
        # SRPT order: smallest remaining work first
        ready.sort(key=lambda ta: remaining[ta.id])
        for ta in ready:
            if free <= 0:
                break
            x = min(ta.x_max, free)
            if x < ta.x_min:
                x = ta.x_min
            if x > free:
                break
            dur = remaining[ta.id] / (x ** alpha)
            if dur <= max(0.0, T - tcur) + 1e-9:
                ta.start = tcur
                ta.x_alloc = x
                active.append(ta)
                placed.add(ta.id)
                free -= x
        # advance active tasks
        finished = []
        for ta in active:
            remaining[ta.id] -= (ta.x_alloc ** alpha) * dt
            if remaining[ta.id] <= 1e-12:
                ta.finish = tcur
                finished.append(ta)
            elif tcur + dt >= T:
                ta.finish = T
                ta.evicted = True
                finished.append(ta)
        for ta in finished:
            active.remove(ta)
            waiting.append(ta.finish - ta.arrival)
        tcur += dt
        if tcur > horizon * 2:
            break
    # any task that never started waits until horizon
    for ta in tasks:
        if ta.id not in placed:
            unplaced += 1
            waiting.append(max(0.0, horizon - ta.arrival))
    return tasks, {
        'mean_wait': float(np.mean(waiting)) if waiting else 0.0,
        'n_placed': len(placed),
        'n_unplaced': unplaced,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--reps', type=int, default=50, help='replications')
    ap.add_argument('--C', type=float, default=64, help='cluster capacity')
    ap.add_argument('--Rgang', type=float, default=16, help='gang task requirement')
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--T', type=float, default=10.0, help='gang launch time')
    ap.add_argument('--lam', type=float, default=1.0, help='arrival rate (1/time)')
    ap.add_argument('--work-base', type=float, default=2.0)
    ap.add_argument('--xmin', type=float, default=1)
    ap.add_argument('--xmax', type=float, default=4)
    ap.add_argument('--alpha', type=float, default=0.8, help='sublinear exponent')
    ap.add_argument('--dist', default='exp', choices=['exp', 'lognorm', 'uniform'])
    ap.add_argument('--profile', default='linear',
                    choices=['linear', 'sqrt', 'step'])
    ap.add_argument('--horizon', type=float, default=None)
    ap.add_argument('--mode', default='elastic',
                    choices=['elastic', 'no-elastic'],
                    help='elastic: tasks use reserve; no-elastic: reserve idles')
    ap.add_argument('--strategy', default='backfill',
                    choices=['backfill', 'scale', 'both'],
                    help='elastic placement: backfill new tasks vs scale current')
    ap.add_argument('--oracle', action='store_true',
                    help='also run the theoretical clairvoyant oracle lower bound '
                         'and print the greedy/oracle competitive ratio')
    ap.add_argument('--compare', action='store_true',
                    help='run both modes and print a side-by-side comparison')
    args = ap.parse_args()

    def run(mode, strategy):
        agg = {'mean_wait': [], 'util_reserve': [], 'n_evicted': [],
               'n_placed': []}
        for r in range(args.reps):
            _, st = run_scenario(
                seed=args.seed + r, C=args.C, R_gang=args.Rgang, t0=args.t0,
                T=args.T, lam=args.lam, work_base=args.work_base,
                x_min=args.xmin, x_max=args.xmax, alpha=args.alpha,
                dist=args.dist, reserve_profile=args.profile,
                horizon=args.horizon, mode=mode, strategy=strategy)
            agg['mean_wait'].append(st['mean_wait'])
            agg['util_reserve'].append(st['util_reserve'])
            agg['n_evicted'].append(st['n_evicted'])
            agg['n_placed'].append(st['n_placed'])
        se = 1.96 / np.sqrt(args.reps)
        return {
            'mean_wait': (np.mean(agg['mean_wait']),
                          se * np.std(agg['mean_wait'])),
            'util_reserve': (np.mean(agg['util_reserve']),
                             se * np.std(agg['util_reserve'])),
            'n_evicted': np.mean(agg['n_evicted']),
            'n_placed': np.mean(agg['n_placed']),
        }

    print(f"=== Gang-reserve simulation (Python/NumPy, ElastiSim-style) ===")
    print(f"params: C={args.C} R_gang={args.Rgang} t0={args.t0} T={args.T} "
          f"lambda={args.lam} work_base={args.work_base} "
          f"x=[{args.xmin},{args.xmax}] alpha={args.alpha} dist={args.dist} "
          f"profile={args.profile} reps={args.reps} strategy={args.strategy}")

    # ---- strategy study: scale current workload vs backfill new ones ----
    if args.strategy == 'both':
        strats = ['backfill', 'scale']
        sr = {s: run('elastic', s) for s in strats}
        print(f"\n--- strategy study (elastic reserve usage) ---")
        print(f"{'strategy':<12}{'mean wait':>14}{'reserve util':>14}"
              f"{'placed':>8}{'evicted':>9}")
        for s in strats:
            mw, mw_ci = sr[s]['mean_wait']
            ur, ur_ci = sr[s]['util_reserve']
            print(f"{s:<12}{mw:>8.3f}±{mw_ci:<6.3f}{ur:>8.3f}±{ur_ci:<6.3f}"
                  f"{sr[s]['n_placed']:>8.1f}{sr[s]['n_evicted']:>9.1f}")
        b, sc = sr['backfill'], sr['scale']
        print(f"scale vs backfill: mean_wait change "
              f"{(sc['mean_wait'][0]-b['mean_wait'][0])/b['mean_wait'][0]*100:+.1f}%, "
              f"util delta {(sc['util_reserve'][0]-b['util_reserve'][0])*100:+.1f} pp")
        if args.oracle:
            _finish_oracle(args)

    # ---- elastic vs no-elastic comparison ----
    if args.compare:
        e, ne = run('elastic', args.strategy if args.strategy != 'both'
                    else 'backfill'), run('no-elastic', args.strategy)
        mw_rel = (e['mean_wait'][0] - ne['mean_wait'][0]) / ne['mean_wait'][0] * 100
        ur_rel = (e['util_reserve'][0] - ne['util_reserve'][0]) * 100
        print(f"\n--- elastic vs no-elastic ---")
        print(f"{'mode':<12}{'mean wait':>14}{'reserve util':>14}"
              f"{'placed':>8}{'evicted':>9}")
        for lbl, r in (('elastic', e), ('no-elastic', ne)):
            mw, mw_ci = r['mean_wait']
            ur, ur_ci = r['util_reserve']
            print(f"{lbl:<12}{mw:>8.3f}±{mw_ci:<6.3f}{ur:>8.3f}±{ur_ci:<6.3f}"
                  f"{r['n_placed']:>8.1f}{r['n_evicted']:>9.1f}")
        print(f"elastic vs no-elastic: mean_wait change {mw_rel:+.1f}%, "
              f"reserve utilization delta {ur_rel:+.1f} pp")
        if args.oracle:
            _finish_oracle(args)

    # ---- default: single mode table ----
    if not args.compare and args.strategy != 'both':
        r = run(args.mode, args.strategy)
        print(f"\n--- {args.mode} (strategy={args.strategy}) ---")
        mw, mw_ci = r['mean_wait']
        ur, ur_ci = r['util_reserve']
        print(f"mean wait: {mw:.3f}±{mw_ci:.3f}  reserve util: {ur:.3f}±{ur_ci:.3f}  "
              f"placed: {r['n_placed']:.1f}  evicted: {r['n_evicted']:.1f}")
        if args.oracle:
            _finish_oracle(args)


def _finish_oracle(args):
    """Theoretical lower bound: clairvoyant work-conserving oracle."""
    o = {'mean_wait': [], 'n_placed': []}
    for r in range(args.reps):
        _, st = run_oracle(seed=args.seed + r, C=args.C, R_gang=args.Rgang,
                           t0=args.t0, T=args.T, lam=args.lam,
                           work_base=args.work_base, x_min=args.xmin,
                           x_max=args.xmax, alpha=args.alpha, dist=args.dist,
                           reserve_profile=args.profile, horizon=args.horizon)
        o['mean_wait'].append(st['mean_wait'])
        o['n_placed'].append(st['n_placed'])
    omw = np.mean(o['mean_wait'])
    se = 1.96 * np.std(o['mean_wait']) / np.sqrt(args.reps)
    print(f"\n--- theoretical lower bound (oracle) ---")
    print(f"oracle mean wait: {omw:.3f}±{se:.3f}  placed: {np.mean(o['n_placed']):.1f}")


if __name__ == '__main__':
    main()