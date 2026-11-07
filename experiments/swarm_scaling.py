"""Finding one colluding pair in a swarm: does calibrated net influence scale with the number of agents?

Scripted N-player RPS (no LLM, $0). One pair colludes (assist with probability CP), everyone else plays at random.
With N agents there are N(N-1)/2 candidate pairs, so the Bonferroni cutoff tightens and the null has more chances to
look dependent. For each N, rounds and CP we report:
  calibrated - calibrated test returns exactly the true pair (CP>0) / any pair (CP=0, false alarm)
  top1       - true pair has the highest min(Gamma(i;j), Gamma(j;i)) score (no test, just ranking)
State is empty: the previous-round joint state has 3^N values, too many for counts at swarm sizes.
Usage:  python experiments/swarm_scaling.py --out results/swarm_scaling
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from netinf import detect_calibrated, influence_matrix, net_influence, permutation_pvalues
from netinf.influence import pair_scores
from netinf.games.rps import AssistantCollusion, simulate

SIZES = [3, 5, 10, 20]
ROUNDS = [100, 300]
CPS = [0.0, 0.3, 0.6]


def one_run(args):
    n, rounds, cp, seed = args
    rng = np.random.default_rng(seed)
    pairs = n * (n - 1) // 2
    a, _ = simulate(n, rounds, AssistantCollusion(0, 1, cp) if cp else None, rng)
    net = net_influence(influence_matrix(a, None))
    p = permutation_pvalues(a, None, 20 * pairs + 100, rng)  # just enough perms for the Bonferroni cutoff
    hit = detect_calibrated(net, p)
    scores = pair_scores(net)
    top = max(scores, key=scores.get)
    if not cp:
        return hit is not None, False
    return hit == (0, 1), top == (0, 1)


def plot(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, rounds in zip(axes, ROUNDS):
        for cp, style in zip(CPS, ["k:", "C0-o", "C1-o"]):
            r = [x for x in rows if x["rounds"] == rounds and x["cp"] == cp]
            label = "false alarm (no colluders)" if not cp else f"true pair caught, CP={cp}"
            ax.plot([x["agents"] for x in r], [x["calibrated_pct"] for x in r], style, label=label)
        ax.set(title=f"{rounds} rounds", xlabel="agents in the swarm (log scale)", xscale="log")
        ax.minorticks_off()
        ax.set_xticks(SIZES, [str(n) for n in SIZES])
    axes[0].set_ylabel("% of runs")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "swarm_scaling.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("results/swarm_scaling"))
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from an existing CSV")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(args.out / "swarm_scaling.csv"))]
        return plot(rows, args.out)
    rows = []
    with ProcessPoolExecutor() as pool:
        for n in SIZES:
            iters = args.iters if n <= 10 else max(args.iters // 2, 10)  # n=20: 4,000 perms per run
            for rounds in ROUNDS:
                for cp in CPS:
                    jobs = [(n, rounds, cp, hash((n, rounds, cp, i)) % 2**32) for i in range(iters)]
                    hits = np.array(list(pool.map(one_run, jobs)))
                    rows.append({"agents": n, "pairs": n * (n - 1) // 2, "rounds": rounds, "cp": cp, "runs": iters,
                                 "calibrated_pct": 100 * hits[:, 0].mean(),
                                 "top1_pct": 100 * hits[:, 1].mean() if cp else float("nan")})
                    print(rows[-1], flush=True)
    with open(args.out / "swarm_scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    plot(rows, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
