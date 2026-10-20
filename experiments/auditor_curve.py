"""Auditor's sample-size curve: how many rounds are needed to catch collusion of a given strength?

For each collusion probability (CP) find the fewest games at which the calibrated detector
returns the true pair in >= 80% of runs, with the previous-round state (the realistic setting).
Costs $0. Usage:
    python experiments/auditor_curve.py --out results/auditor_curve
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calibration_rps import one_run

N_GAMES = [25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500]
CPS = [0.2, 0.3, 0.4, 0.5, 0.6]
POWER = 80.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/auditor_curve"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows, needed = [], []
    for cp in CPS:
        found = {"fixed_alpha": None, "calibrated": None}
        for n in N_GAMES:
            hits = np.array([one_run("prev", n, cp, args.perms, rng) for _ in range(args.iters)])
            rate = {"fixed_alpha": 100 * hits[:, 0].mean(), "calibrated": 100 * hits[:, 1].mean()}
            rows.append({"cp": cp, "n_games": n, **rate})
            print(rows[-1], flush=True)
            for k in found:
                if found[k] is None and rate[k] >= POWER:
                    found[k] = n
            if found["calibrated"] is not None and found["fixed_alpha"] is not None:
                break  # both reached the power target; larger N adds nothing
        needed.append({"cp": cp, **{f"games_for_{int(POWER)}pct_{k}": v for k, v in found.items()}})

    for name, data in (("power.csv", rows), ("games_needed.csv", needed)):
        with open(args.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for cp, color in zip(CPS, ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]):  # one-hue ramp: CP is a magnitude
        sel = [r for r in rows if r["cp"] == cp]
        ax.plot([r["n_games"] for r in sel], [r["calibrated"] for r in sel], "-o", ms=4, lw=2, color=color,
                label=f"CP={cp}")
    ax.axhline(POWER, ls="--", lw=1, color="#555")
    ax.set(xscale="log", xlabel="rounds observed", ylabel="% of runs where the true pair is caught",
           title="Calibrated detector (false alarms held near 5%)", ylim=(0, 102))
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, title="collusion strength", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "auditor_curve.png", dpi=160)


if __name__ == "__main__":
    main()
