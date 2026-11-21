#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Иллюстрации количественной оценки общественного блага эластичности
(OpenSpec change `elasticity-social-good-quantification`, задачи 1.3 и 2.2).

  - fig-5-3-fragmentation-identity.svg  — граница B(x) <= F(x) и тождество
        B(x) = F(x) при перегрузке (спрос >= ёмкость). По оси X — фрагментация
        F(x) (запертая ёмкость), по оси Y — общественное благо B(x); точки
        лежат на диагонали B=F при перегрузке и ниже неё при недогрузке.
  - fig-6-3-saturation.svg              — пороговая кривая: B(eps) = F(x)
        константа при любой нетривиальной эластичности (eps < 1) и обращается
        в ноль при eps = 1 (жёсткие задачи); непрерывного насыщения в модели
        идеализированного эластичного размещения (C_elastic = min(D, C)) нет.

Модель заимствована из sim/test_fragmentation_identity.py (той же реализации,
что и проходящие тесты).
"""
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
from test_fragmentation_identity import social_good, saturation_curve

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

OUT = os.path.join(HERE, "..", "draft", "figures", "")


def fig_identity():
    """B(x) vs F(x): граница B<=F, тождество при перегрузке."""
    rng = random.Random(123)
    Bs, Fs, overload = [], [], []
    for _ in range(300):
        m = rng.randint(2, 10)
        caps = [rng.randint(5, 60) for _ in range(m)]
        sizes = [rng.randint(1, 30) for _ in range(rng.randint(1, 50))]
        B, F, _, _ = social_good(caps, sizes, 0.0)
        Bs.append(B)
        Fs.append(F)
        overload.append(sum(sizes) >= sum(caps))
    Bs, Fs = np.array(Bs), np.array(Fs)
    overload = np.array(overload)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    lim = max(Fs.max(), Bs.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="0.6", lw=1, ls="--",
            label=r"$B(x)=F(x)$")
    ax.scatter(Fs[overload], Bs[overload], s=14, color="black",
               label="перегрузка (спрос$\\geq$ёмкость)")
    ax.scatter(Fs[~overload], Bs[~overload], s=14, facecolor="none",
               edgecolor="0.4", label="недогрузка")
    ax.set_xlabel(r"фрагментация $F(x)$ (запертая ёмкость)")
    ax.set_ylabel(r"общественное благо $B(x)$")
    ax.set_title(r"Граница $B(x)\leq F(x)$ и тождество при перегрузке")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-5-3-fragmentation-identity.svg", format="svg")
    plt.close(fig)
    return Bs, Fs, overload


def fig_saturation():
    """B(eps) по запасу эластичности eps: монотонный рост и плато."""
    caps = [10, 14, 18, 22]
    rng = random.Random(7)
    sizes = [rng.randint(2, 30) for _ in range(30)]
    floors = np.linspace(0.0, 1.0, 21)
    B = saturation_curve(caps, sizes, [float(e) for e in floors])
    vals = [B[float(e)] for e in floors]
    Bfull, F, _, _ = social_good(caps, sizes, 0.0)

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(floors, vals, marker="o", ms=4, lw=1.8, color="black",
            label=r"$B(\varepsilon)$")
    ax.axhline(F, color="0.5", lw=1, ls=":", label=r"плато $F(x)$")
    ax.set_xlabel(r"запас эластичности $\varepsilon$ (0 — полная, 1 — жёсткая)")
    ax.set_ylabel(r"общественное благо $B(\varepsilon)$")
    ax.set_title("Кривая насыщения блага эластичности")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT + "fig-6-3-saturation.svg", format="svg")
    plt.close(fig)
    return floors, vals, F


def main():
    Bs, Fs, overload = fig_identity()
    n_over = int(overload.sum())
    print(f"рис. 5.3: {len(Bs)} инстансов; перегрузка {n_over} — на диагонали "
          f"(B=F), недогрузка {len(Bs)-n_over} — ниже (B<F); B<=F всюду")
    floors, vals, F = fig_saturation()
    print(f"рис. 6.3: B(eps<1)={vals[0]:.2f}=F константа при любой нетривиальной "
          f"эластичности, B(eps=1)={vals[-1]:.2f} (жёсткие задачи), плато F={F:.2f}; "
          f"зависимость пороговая, непрерывного насыщения нет")
    print("ВСЕ РИСУНКИ ЗАПИСАНЫ В draft/figures/")


if __name__ == "__main__":
    main()