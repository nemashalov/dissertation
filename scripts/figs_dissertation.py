#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация иллюстраций диссертации (блоки A — эксперименты, B — аналитика).
Выход: draft/figures/fig-*.svg.
Модель: однородный кластер ёмкости C, n владельцев (квоты Q_k, веса phi_k),
эластичные задачи (c_i, x_min, x_max). Алгоритм размещения — заметание по
приоритету (фазы заполнение/расширение) + правило минимальной невязки
относительно справедливой доли FS_k = (phi_k / sum_j phi_j) Q_k.
Согласуется с draft/10-simulation-plan.md и показателями определения 6.1.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 100,
})

OUT = "/Users/n.mashalov/Desktop/dissertation/draft/figures/"


# ----------------------------------------------------------------------
# Модель размещения
# ----------------------------------------------------------------------
def fair_share(phi, Q):
    """Справедливые доли FS_k = (phi_k / sum_j phi_j) Q_k."""
    S = sum(phi)
    return np.array([(p / S) * q for p, q in zip(phi, Q)])


def sweep_place(tasks, C, use_elastic=True):
    """
    Заметание по убыванию приоритета: фаза заполнения x_min, затем фаза
    расширения x_max. tasks: list[(c, xmin, xmax)]. Возвращает x_i.
    """
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i][0])
    x = [0.0] * len(tasks)
    remain = C
    # фаза заполнения
    for i in order:
        c, xmin, xmax = tasks[i]
        if remain < xmin:
            continue
        if use_elastic:
            xi = min(xmin, remain)
        else:  # жёсткая: только xmax
            if remain < xmax:
                continue
            xi = xmax
        x[i] = xi
        remain -= xi
    # фаза расширения (только эластичная)
    if use_elastic:
        for i in reversed(order):
            c, xmin, xmax = tasks[i]
            if x[i] > 0:
                grow = min(xmax - x[i], remain)
                x[i] += grow
                remain -= grow
    return np.array(x)


def allocated_volume(tasks, x):
    """C_elastic / C_no_elastic-стическая эффективная ёмкость размещения."""
    return sum(xi for xi in x)


# ----------------------------------------------------------------------
# Эксперимент 1 (рис 6.1): B(x), C_elastic vs C_no_elastic при росте ширины r
# ----------------------------------------------------------------------
def exp1():
    rng = np.random.default_rng(7)
    C = 100.0
    # фиксированный набор x_min; x_max = x_min * r. При жёстком размещении
    # задачи имеют размер x_max, растущий с r, поэтому C_no ниже C при r > 1.
    # Кривая C_no(r) НЕ монотонна: остаток жёсткой упаковки зависит от
    # конкретных размеров задач. Спрос по минимумам D ~ 165 > C = 100, поэтому
    # действует режим перегрузки и точный закон гл. 5 даёт B = F = C - C_no.
    n = 60
    xmins = rng.uniform(1.5, 4.0, n)
    c = rng.uniform(1, 100, n)
    widths = np.arange(1.0, 9.0, 0.5)
    reps = 60
    C_el = np.zeros(len(widths))
    C_no = np.zeros(len(widths))
    for wi, r in enumerate(widths):
        for _ in range(reps):
            tasks = list(zip(c, xmins, xmins * r))
            x_el = sweep_place(tasks, C, use_elastic=True)
            x_no = sweep_place(tasks, C, use_elastic=False)
            C_el[wi] += allocated_volume(tasks, x_el)
            C_no[wi] += allocated_volume(tasks, x_no)
    C_el /= reps
    C_no /= reps
    B = C_el - C_no

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(widths, C_el, marker="o", ms=4, lw=1.8, label=r"$C_{elastic}$")
    ax.plot(widths, C_no, marker="s", ms=4, lw=1.8, ls="--", label=r"$C_{no\,elastic}$")
    ax.plot(widths, B, marker="^", ms=4, lw=1.8, color="black", label=r"$B(x)=C_{el}-C_{no}\geq 0$")
    ax.set_xlabel(r"ширина диапазона $r=x_{\max}/x_{\min}$")
    ax.set_ylabel("эффективная ёмкость")
    ax.set_title("Общественное благо эластичности при росте $r$")
    ax.legend(frameon=False, loc="center right")
    ax.set_ylim(bottom=0)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-6-1-social-good.svg", format="svg")
    plt.close(fig)
    return widths, C_el, C_no, B


# ----------------------------------------------------------------------
# Эксперимент 1b (рис 6.2): утилизация vs степень фрагментации
# ----------------------------------------------------------------------
def exp2():
    rng = np.random.default_rng(11)
    C = 100.0
    frag_levels = np.arange(0.0, 1.0, 0.1)  # доля задач-"крупных", создающих дыры
    reps = 40
    U_el = np.zeros(len(frag_levels))
    U_no = np.zeros(len(frag_levels))
    for fi, f in enumerate(frag_levels):
        for _ in range(reps):
            n = 40
            c = rng.uniform(1, 100, n)
            xmins = rng.uniform(2, 6, n)
            # крупные задачи с широким диапазоном -> создают фрагментацию
            xmaxs = np.where(rng.uniform(0, 1, n) < f,
                             xmins * rng.uniform(4, 8),
                             xmins * rng.uniform(1.2, 2.0))
            tasks = list(zip(c, xmins, xmaxs))
            x_el = sweep_place(tasks, C, use_elastic=True)
            x_no = sweep_place(tasks, C, use_elastic=False)
            U_el[fi] += allocated_volume(tasks, x_el) / C
            U_no[fi] += allocated_volume(tasks, x_no) / C
    U_el /= reps
    U_no /= reps

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(frag_levels, U_el, marker="o", ms=4, lw=1.8, label="эластичное размещение")
    ax.plot(frag_levels, U_no, marker="s", ms=4, lw=1.8, ls="--", label="жёсткое размещение")
    ax.set_xlabel("доля крупных задач (степень фрагментации)")
    ax.set_ylabel(r"утилизация $U$")
    ax.set_title("Влияние фрагментации на утилизацию")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-6-2-util-fragmentation.svg", format="svg")
    plt.close(fig)
    return frag_levels, U_el, U_no


# ----------------------------------------------------------------------
# Эксперимент 2 (рис 6.3): A_k vs FS_k по владельцам, наивное vs алгоритм
# ----------------------------------------------------------------------
def fair_place(tasks_owners, phi, Q, C):
    """
    Размещение по алгоритму гл.4: каждый владелец получает свою справедливую долю
    FS_k (в пределах спроса), задачи владельца размещаются по приоритету в рамках
    его доли; затем остаток распределяется в порядке минимальной невязки
    Delta_k = FS_k - A_k. Возвращает A_k, U, allocations.
    """
    FS = fair_share(phi, Q)
    n_own = len(FS)
    # спрос владельца = сумма x_min его задач (нижняя граница обязательного размещения)
    demand = np.zeros(n_own)
    for c, xmin, xmax, own in tasks_owners:
        demand[own] += xmin
    # целевая доля фазы заполнения = min(FS_k, спрос_k) — «в пределах спроса»
    target = np.minimum(FS, demand)
    x = [0.0] * len(tasks_owners)
    A = np.zeros(n_own)
    # фаза заполнения: per-owner, задачи владельца по приоритету до его target_k.
    # Последняя задача может быть размещена частично (до target_k), т.к. в модели
    # x принимает любое значение из [x_min, x_max]. Гарантирует A_k = min(FS_k, спрос_k).
    for k in range(n_own):
        own_idx = [i for i, t in enumerate(tasks_owners) if t[3] == k]
        own_idx.sort(key=lambda i: -tasks_owners[i][0])
        for i in own_idx:
            if target[k] <= 0:
                break
            c, xmin, xmax, own = tasks_owners[i]
            if xmin > target[k]:
                continue  # задача не помещается в остаток доли владельца
            take = min(xmax, target[k])
            x[i] = take
            A[k] += take
            target[k] -= take
    # фаза расширения: (1) топ-ап до справедливой доли min(FS_k, спрос_k),
    # затем (2) распределение остатка как бонуса x_max в порядке невязки.
    # Это гарантирует A_k >= min(FS_k, demand_k) в пределах спроса.
    remain = C - sum(A)
    need = np.minimum(FS, demand)  # нижняя граница, которую гарантирует алгоритм
    # (1) топ-ап: довести каждого владельца до need_k
    for k in sorted(range(n_own), key=lambda k: need[k] - A[k], reverse=True):
        if A[k] >= need[k]:
            continue
        for i in range(len(tasks_owners)):
            if remain <= 0:
                break
            c, xmin, xmax, own = tasks_owners[i]
            if own != k:
                continue
            if x[i] == 0:
                if remain < xmin:
                    continue
                x[i] = xmin
                A[k] += xmin
                remain -= xmin
            grow = min(xmax - x[i], remain, max(0.0, need[k] - A[k]))
            x[i] += grow
            A[k] += grow
            remain -= grow
            if A[k] >= need[k]:
                break
    # (2) бонус: распределить оставшийся остаток как расширение до x_max
    order_owners = sorted(range(n_own), key=lambda k: FS[k] - A[k], reverse=True)
    for k in order_owners:
        for i in range(len(tasks_owners)):
            if remain <= 0:
                break
            c, xmin, xmax, own = tasks_owners[i]
            if own != k:
                continue
            if x[i] == 0:
                if remain >= xmin:
                    x[i] = xmin
                    A[k] += xmin
                    remain -= xmin
                else:
                    continue
            grow = min(xmax - x[i], remain)
            x[i] += grow
            A[k] += grow
            remain -= grow
    U = sum(x) / C
    return A, U, x


def naive_place(tasks_owners, C):
    """Наивное: заполнение по глобальному приоритету, без учёта справедливости."""
    idx = sorted(range(len(tasks_owners)), key=lambda i: -tasks_owners[i][0])
    x = [0.0] * len(tasks_owners)
    remain = C
    for i in idx:
        c, xmin, xmax, own = tasks_owners[i]
        if remain < xmin:
            continue
        xi = min(xmin, remain)
        x[i] = xi
        remain -= xi
    n_own = max(t[3] for t in tasks_owners) + 1
    A = np.zeros(n_own)
    for i, t in enumerate(tasks_owners):
        A[t[3]] += x[i]
    return A, x


def exp3():
    rng = np.random.default_rng(3)
    C = 100.0
    n_own = 4
    phi = [1, 2, 2, 4]
    Q = [C] * n_own
    FS = fair_share(phi, Q)
    tasks_owners = []
    for k in range(n_own):
        n_tasks = 12
        for _ in range(n_tasks):
            c = rng.uniform(1, 100)
            xmin = rng.uniform(2, 5)
            xmax = xmin * rng.uniform(1.5, 4)
            tasks_owners.append((c, xmin, xmax, k))
    A_algo, U_algo, _ = fair_place(tasks_owners, phi, Q, C)
    A_naive, _ = naive_place(tasks_owners, C)

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    xpos = np.arange(n_own)
    w = 0.27
    ax.bar(xpos - w, FS, w, label=r"$FS_k$", color="white", edgecolor="black", hatch="//")
    ax.bar(xpos, A_algo, w, label=r"алгоритм $A_k$", color="0.5", edgecolor="black")
    ax.bar(xpos + w, A_naive, w, label=r"наивное $A_k^{\mathrm{н}}$", color="0.85", edgecolor="black")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"владелец {k+1}" for k in range(n_own)])
    ax.set_ylabel("выделенная ёмкость")
    ax.set_title(r"Справедливая доля $FS_k$, алгоритм ($A_k$) и наивное размещение ($A_k^{\mathrm{н}}$)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-6-4-fairness.svg", format="svg")
    plt.close(fig)
    return FS, A_algo, A_naive, U_algo


# ----------------------------------------------------------------------
# Эксперимент 3 (рис 6.4): утилизация vs нагрузка, сравнение с DRF
# ----------------------------------------------------------------------
def drf_place(tasks_owners, phi, C):
    """Упрощённый DRF: max-min по доминирующему ресурсу (здесь 1 ресурс -> равная доля)."""
    n_own = max(t[3] for t in tasks_owners) + 1
    S = sum(phi)
    shares = np.array([p / S for p in phi])
    # каждой доле даём долю ёмкости
    caps = shares * C
    A = np.zeros(n_own)
    idx = sorted(range(len(tasks_owners)), key=lambda i: -tasks_owners[i][0])
    x = [0.0] * len(tasks_owners)
    for i in idx:
        c, xmin, xmax, own = tasks_owners[i]
        if caps[own] < xmin:
            continue
        xi = min(xmin, caps[own])
        x[i] = xi
        A[own] += xi
        caps[own] -= xi
    # расширение в порядке избытка доли
    order = sorted(range(n_own), key=lambda k: caps[k], reverse=True)
    for k in order:
        for i in range(len(tasks_owners)):
            if tasks_owners[i][3] != k or x[i] == 0 or caps[k] <= 0:
                continue
            grow = min(tasks_owners[i][2] - x[i], caps[k])
            x[i] += grow
            caps[k] -= grow
    return A, sum(x)


def exp4():
    rng = np.random.default_rng(5)
    C = 100.0
    n_own = 4
    phi = [1, 2, 2, 4]
    Q = [C] * n_own
    loads = np.arange(0.6, 1.21, 0.1)
    reps = 30
    U_algo = np.zeros(len(loads))
    U_drf = np.zeros(len(loads))
    U_her = np.zeros(len(loads))
    for li, rho in enumerate(loads):
        for _ in range(reps):
            # спрос по x_min суммарно = rho * C; x_max = x_min * r
            base = 30  # базовая генерация, затем нормируем на rho
            tasks_owners = []
            raw = []
            for k in range(n_own):
                for _ in range(base // n_own):
                    c = rng.uniform(1, 100)
                    xmin = rng.uniform(1.5, 4.0)
                    xmax = xmin * rng.uniform(2.0, 4.0)
                    raw.append((c, xmin, xmax, k))
            total_min = sum(t[1] for t in raw)
            # масштабируем x_min/x_max так, чтобы суммарный x_min-спрос = rho*C
            sc = rho * C / total_min
            tasks_owners = [(c, xmin * sc, xmax * sc, k) for (c, xmin, xmax, k) in raw]
            # разработанный алгоритм (эластичность + справедливость)
            A_a, U_a, _ = fair_place(tasks_owners, phi, Q, C)
            # DRF: жёсткое размещение в пределах справедливой доли (только x_min, без расширения)
            _, vol_d = drf_place(tasks_owners, phi, C)
            # Heracles-стиль: эластичность внутри фиксированных долей (без перераспределения остатка)
            _, vol_h = heracles_place(tasks_owners, phi, C)
            U_algo[li] += U_a
            U_drf[li] += vol_d / C
            U_her[li] += vol_h / C
    U_algo /= reps
    U_drf /= reps
    U_her /= reps

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.plot(loads, U_algo, marker="o", ms=4, lw=1.8, label="разработанный алгоритм")
    ax.plot(loads, U_drf, marker="s", ms=4, lw=1.8, ls="--", label="статические доли (DRF)")
    ax.plot(loads, U_her, marker="^", ms=4, lw=1.8, ls=":", label="Heracles [3]")
    ax.plot(loads, loads, color="0.5", lw=1, label=r"$U=\rho$ (эталон)")
    ax.set_xlabel(r"коэффициент загрузки $\rho$")
    ax.set_ylabel(r"утилизация $U$")
    ax.set_title("Сравнение утилизации при эластичной нагрузке")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-6-5-comparison.svg", format="svg")
    plt.close(fig)
    return loads, U_algo, U_drf, U_her


def heracles_place(tasks_owners, phi, C):
    """
    Heracles-стиль: каждый владелец жёстко ограничен своей долей shares_k*C
    и использует эластичность (расширение до x_max) внутри своей доли,
    но остаток доли другого владельца не перераспределяется.
    Возвращает A_k, суммарный объём.
    """
    n_own = max(t[3] for t in tasks_owners) + 1
    S = sum(phi)
    shares = np.array([p / S for p in phi])
    caps = shares * C
    A = np.zeros(n_own)
    idx = sorted(range(len(tasks_owners)), key=lambda i: -tasks_owners[i][0])
    x = [0.0] * len(tasks_owners)
    for k in range(n_own):
        own_idx = [i for i, t in enumerate(tasks_owners) if t[3] == k]
        own_idx.sort(key=lambda i: -tasks_owners[i][0])
        for i in own_idx:
            c, xmin, xmax, own = tasks_owners[i]
            if caps[own] < xmin:
                continue
            xi = min(xmin, caps[own])
            x[i] = xi
            A[own] += xi
            caps[own] -= xi
        # расширение внутри своей доли
        for i in own_idx:
            if caps[k] <= 0 or x[i] == 0:
                continue
            grow = min(tasks_owners[i][2] - x[i], caps[k])
            x[i] += grow
            caps[k] -= grow
            A[k] += grow
    return A, sum(x)


# ----------------------------------------------------------------------
# Эксперимент 3b (рис 6.5): время работы алгоритма vs N, линия O(N log N)
# ----------------------------------------------------------------------
def exp5():
    rng = np.random.default_rng(9)
    Ns = np.logspace(2, 4, 8).astype(int)
    times = []
    for N in Ns:
        acc = []
        for _ in range(20):
            tasks = [(rng.uniform(1, 100), 2.0, 8.0) for _ in range(N)]
            import time
            t0 = time.perf_counter()
            sweep_place(tasks, 100.0, use_elastic=True)
            acc.append((time.perf_counter() - t0) * 1e6)
        times.append(np.mean(acc))
    times = np.array(times)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.loglog(Ns, times, marker="o", ms=5, lw=1.8, label="замер времени")
    # аппроксимация ~ N log N
    fit = times[-1] * (Ns * np.log(Ns)) / (Ns[-1] * np.log(Ns[-1]))
    ax.loglog(Ns, fit, ls="--", color="0.4", lw=1.5, label=r"$O(N\log N)$")
    ax.set_xlabel(r"число задач $N$")
    ax.set_ylabel(r"время, $\mu$s")
    ax.set_title(r"Сложность алгоритма $O(N\log N)$")
    ax.legend(frameon=False)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-6-6-complexity.svg", format="svg")
    plt.close(fig)
    return Ns, times


# ----------------------------------------------------------------------
# Аналитические рисунки
# ----------------------------------------------------------------------
def analytic_usl():
    """Рис 3.1 — кривая USL C(N) = N/(1+a(N-1)+b N(N-1))."""
    N = np.linspace(1, 200, 400)
    for a, b, style, lab in [(0.05, 0.0006, "-", r"$\alpha=0.05,\ \beta=6\cdot10^{-4}$"),
                              (0.02, 0.0002, "--", r"$\alpha=0.02,\ \beta=2\cdot10^{-4}$")]:
        C = N / (1 + a * (N - 1) + b * N * (N - 1))
        peak = N[np.argmax(C)]
        plt.plot(N, C, style, lw=1.8, label=lab)
        plt.axvline(peak, color="0.5", lw=0.8, ls=":")
    plt.xlabel(r"число потоков-исполнителей $N$")
    plt.ylabel(r"пропускная способность $C(N)$")
    plt.title("Универсальный закон масштабируемости (USL)")
    plt.legend(frameon=False)
    plt.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT + "fig-3-1-usl.svg", format="svg")
    plt.close()


def analytic_counterexample():
    """Рис 4.2 — контрпример: алгоритм 9700 vs оптимум 10000."""
    labels = ["алгоритм\nзаметания", "оптимум"]
    values = [9700, 10000]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    bars = ax.bar(labels, values, width=0.5, color=["0.75", "0.3"], edgecolor="black")
    ax.set_ylabel(r"$\sum_i c_i x_i$")
    ax.set_ylim(0, 11000)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 150, f"{v}",
                ha="center", va="bottom", fontsize=10)
    ax.set_title(r"Контрпример §4.3: $C=100$")
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-4-2-counterexample.svg", format="svg")
    plt.close(fig)


def analytic_kleinrock():
    """Рис 5.1 — закон сохранения Клейнрока: перераспределение задержки."""
    # Два класса; суммарная взвешенная задержка постоянна = rho W0/(1-rho).
    rho = 0.7
    rho1, rho2 = 0.4, 0.3
    W0 = 1.0
    inv = rho * W0 / (1 - rho)
    # FCFS: обе равны W = W0/(1-rho)
    W_fcfs = W0 / (1 - rho)
    # Приоритет: класс 1 имеет задержку меньше, класс 2 — больше
    W1 = W_fcfs * 0.5
    W2 = (inv - rho1 * W1) / rho2
    rho_kW_fcfs = [rho1 * W_fcfs, rho2 * W_fcfs]
    rho_kW_pri = [rho1 * W1, rho2 * W2]

    xpos = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.bar(xpos - w/2, rho_kW_fcfs, w, label=r"FCFS ($\rho_k W_k$)", color="white", edgecolor="black", hatch="//")
    ax.bar(xpos + w/2, rho_kW_pri, w, label=r"приоритет ($\rho_k W_k$)", color="0.5", edgecolor="black")
    ax.axhline(inv, color="0.3", lw=1, ls=":")
    ax.text(1.45, inv + 0.01, r"$\Sigma_k\rho_k W_k=\frac{\rho W_0}{1-\rho}$", fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["класс 1", "класс 2"])
    ax.set_ylabel(r"взвешенная задержка $\rho_k W_k$")
    ax.set_title("Закон сохранения Клейнрока: инвариант суммы")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-5-1-kleinrock.svg", format="svg")
    plt.close(fig)
    return inv, rho1 * W1 + rho2 * W2, rho1 * W_fcfs + rho2 * W_fcfs


def analytic_social_good():
    """Рис 5.2 — B(x) >= 0 как следствие вложенности допустимых множеств.

    Эластичное размещение допустимо всюду, где допустимо жёсткое
    (x_max лежит в диапазоне [x_min, x_max]), поэтому C_elastic >= C_no_elastic.
    Иллюстрация: средние по 40 случайным инстансам при каждом r; жёсткое
    размещение теряет ёмкость на фрагментации с ростом ширины диапазона r,
    эластичное заполняет кластер целиком (режим перегрузки: B = F).
    """
    rng = np.random.default_rng(21)
    C = 100.0
    n = 60
    reps = 40
    r_grid = np.linspace(1.0, 8.0, 15)
    C_el = np.zeros(len(r_grid))
    C_no = np.zeros(len(r_grid))
    for ri, r in enumerate(r_grid):
        for _ in range(reps):
            xmins = rng.uniform(1.5, 4.0, n)
            c = rng.uniform(1, 100, n)
            tasks = list(zip(c, xmins, xmins * r))
            C_el[ri] += allocated_volume(tasks, sweep_place(tasks, C, use_elastic=True))
            C_no[ri] += allocated_volume(tasks, sweep_place(tasks, C, use_elastic=False))
    C_el /= reps
    C_no /= reps
    B = C_el - C_no

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(r_grid, C_el, lw=1.8, label=r"$C_{elastic}$")
    ax.plot(r_grid, C_no, lw=1.8, ls="--", label=r"$C_{no\,elastic}$")
    ax.plot(r_grid, B, lw=1.8, ls=":", color="black", label=r"$B(x)=C_{el}-C_{no}\geq 0$")
    ax.set_xlabel(r"эластичность $r=x_{\max}/x_{\min}$")
    ax.set_ylabel("эффективная ёмкость")
    ax.set_title(r"Общественное благо $B(x)\geq 0$ (среднее по 40 инстансам)")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-5-2-social-good.svg", format="svg")
    plt.close(fig)
    return r_grid, C_el, C_no, B


def main():
    r1 = exp1()
    r2 = exp2()
    r3 = exp3()
    r4 = exp4()
    r5 = exp5()
    analytic_usl()
    analytic_counterexample()
    rk = analytic_kleinrock()
    analytic_social_good()

    # Протокол численных результатов для гл.6
    print("=== Эксперимент 1 (6.2): ширина r -> C_el, C_no, B ===")
    for r, ce, cn, b in zip(*r1):
        print(f"  r={r:3.1f}: C_el={ce:6.2f} C_no={cn:6.2f} B={b:6.2f}")
    print("\n=== Эксперимент 1b (6.2): фрагментация -> U_el, U_no ===")
    for f, ue, un in zip(*r2):
        print(f"  frag={f:3.1f}: U_el={ue:.3f} U_no={un:.3f}")
    print("\n=== Эксперимент 2 (6.3): FS_k, A_algo, A_naive ===")
    for k, (fs, aa, an) in enumerate(zip(*r3[:3])):
        print(f"  owner{k+1}: FS={fs:.2f} A_algo={aa:.2f} A_naive={an:.2f}")
    print(f"  утилизация алгоритма U={r3[3]:.3f}")
    print("\n=== Эксперимент 3 (6.4): rho -> U_algo, U_drf, U_her ===")
    for rho, ua, ud, uh in zip(*r4):
        print(f"  rho={rho:.2f}: U_algo={ua:.3f} U_drf={ud:.3f} U_her={uh:.3f}")
    print("\n=== Эксперимент 3b (6.5): N -> время ===")
    for N, t in zip(*r5):
        print(f"  N={N}: t={t:.2f} mus")
    print("\n=== Клейнрок (5.1): инвариант ===")
    print(f"  инвариант={rk[0]:.4f}; приоритет sum={rk[1]:.4f}; FCFS sum={rk[2]:.4f}")
    print("\nВСЕ РИСУНКИ ЗАПИСАНЫ В draft/figures/")


if __name__ == "__main__":
    main()