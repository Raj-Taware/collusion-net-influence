"""Fixed-alpha vs permutation-calibrated net influence on scripted 3-player RPS.

For each state definition, number of games and collusion probability (CP=0 = no colluders),
report how often each detector fires: the false-alarm rate when CP=0, otherwise the rate of
correctly returning the true pair. Costs $0. Usage:
    python experiments/calibration_rps.py --out results/calibration_rps
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from netinf import detect, detect_calibrated, influence_matrix, net_influence, permutation_pvalues
from netinf.games.rps import AssistantCollusion, simulate

N_GAMES = [50, 100, 200, 300, 500, 1000]
CPS = [0.0, 0.2, 0.3, 0.4]


def one_run(state: str, n_games: int, cp: float, n_perm: int, rng) -> tuple[bool, bool]:
    a, _ = simulate(3, n_games + 1, AssistantCollusion(0, 1, cp) if cp else None, rng)
    actions, states = (a[1:], a[:-1]) if state == "prev" else (a[1:], None)
    net = net_influence(influence_matrix(actions, states))
    p = permutation_pvalues(actions, states, n_perm, rng)
    target = (0, 1) if cp else None
    fixed, calib = detect(net), detect_calibrated(net, p)
    if target is None:  # any alarm is a false alarm
        return fixed is not None, calib is not None
    return fixed == target, calib == target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/calibration_rps"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for state in ("none", "prev"):
        for n in N_GAMES:
            for cp in CPS:
                hits = np.array([one_run(state, n, cp, args.perms, rng) for _ in range(args.iters)])
                rows.append({"state": state, "n_games": n, "cp": cp,
                             "fixed_alpha": 100 * hits[:, 0].mean(), "calibrated": 100 * hits[:, 1].mean()})
                print(rows[-1])

    with open(args.out / "calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(2, len(CPS), figsize=(13, 6), sharex=True, sharey=True)
    for r, state in enumerate(("none", "prev")):
        for c, cp in enumerate(CPS):
            ax = axes[r, c]
            sel = [x for x in rows if x["state"] == state and x["cp"] == cp]
            for key, style in (("fixed_alpha", "o-"), ("calibrated", "s--")):
                ax.plot([x["n_games"] for x in sel], [x[key] for x in sel], style, label=key)
            if cp == 0:
                ax.axhline(5, c="k", lw=0.6, ls=":")
            ax.set_title(f"state={state}, " + ("no collusion (false alarms)" if cp == 0 else f"CP={cp} (detection)"),
                         fontsize=9)
            ax.set_xscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("games")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of runs flagged")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Fixed alpha=0.05 vs permutation-calibrated net influence ({args.iters} runs/cell)")
    fig.tight_layout()
    fig.savefig(args.out / "calibration.png", dpi=150)


if __name__ == "__main__":
    main()
