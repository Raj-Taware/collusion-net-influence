"""Positive control: reproduce Bonjour et al. (2022) 3-player RPS results with scripted agents.

  Fig. 3  - net influence of every ordered pair vs collusion probability (CP), 1000 games
  Table 2 - collusion detection accuracy (CDA) vs number of games, CP in {0.3, 0.4}
  Extra   - false-alarm rate with no colluders (not reported in the paper)

Costs $0 (no LLM calls). Usage:
    python experiments/replicate_bonjour_rps.py --state none --out results/bonjour_rps_state-none
    python experiments/replicate_bonjour_rps.py --state prev --out results/bonjour_rps_state-prev
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from netinf import detect, influence_matrix, net_influence
from netinf.games.rps import AssistantCollusion, simulate

ALPHA = 0.05
PAPER_TABLE2 = {  # n_games: (CDA% CP=0.3, CDA% CP=0.4), from Bonjour et al. Table 2
    100: (37.1, 58.4), 200: (48.3, 79.2), 300: (71.0, 96.2), 400: (82.8, 99.3), 500: (89.6, 99.7),
    600: (94.6, 99.9), 700: (97.8, 100.0), 800: (98.8, 100.0), 900: (99.4, 100.0), 1000: (100.0, 100.0),
}
PAIR_LABELS = ["A->B", "B->A", "A->C", "C->A", "B->C", "C->B"]
PAIRS = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]


STATE = "none"  # set from --state


def net_for(n_games: int, cp: float | None, rng: np.random.Generator) -> np.ndarray:
    """Net influence matrix for one simulated run.

    STATE="none": one constant state. STATE="prev": state is the previous round's joint action
    (closer to the paper's numbers; costs one round, so we simulate n_games + 1).
    """
    collusion = AssistantCollusion(primary=0, assistant=1, cp=cp) if cp else None
    if STATE == "none":
        actions, _ = simulate(3, n_games, collusion, rng)
        return net_influence(influence_matrix(actions))
    actions, _ = simulate(3, n_games + 1, collusion, rng)
    return net_influence(influence_matrix(actions[1:], states=actions[:-1]))


def fig3(rng, out: Path, reps: int = 20) -> list[dict]:
    rows = []
    for cp in np.round(np.arange(0.0, 1.01, 0.1), 2):
        nets = np.array([net_for(1000, cp, rng) for _ in range(reps)])
        for (i, j), label in zip(PAIRS, PAIR_LABELS):
            vals = nets[:, i, j]
            rows.append({"cp": cp, "pair": label, "mean": vals.mean(), "std": vals.std()})

    fig, ax = plt.subplots(figsize=(6, 4))
    for label in PAIR_LABELS:
        r = [x for x in rows if x["pair"] == label]
        ax.errorbar([x["cp"] for x in r], [x["mean"] for x in r], yerr=[x["std"] for x in r], label=label, capsize=2)
    ax.axhline(ALPHA, ls="--", c="k", lw=0.8, label=f"alpha={ALPHA}")
    ax.set(xlabel="collusion probability (CP)", ylabel="net influence (nats)",
           title=f"3-player RPS, 1000 games, mean ± sd over {reps} runs")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig3_net_influence_vs_cp.png", dpi=150)
    return rows


def table2(rng, iters: int) -> list[dict]:
    rows = []
    for n_games in PAPER_TABLE2:
        row = {"n_games": n_games}
        for cp, paper in zip((0.3, 0.4), PAPER_TABLE2[n_games]):
            hits = sum(detect(net_for(n_games, cp, rng), ALPHA) == (0, 1) for _ in range(iters))
            row[f"cda_cp{cp}"] = 100 * hits / iters
            row[f"paper_cp{cp}"] = paper
        alarms = sum(detect(net_for(n_games, None, rng), ALPHA) is not None for _ in range(iters))
        row["false_alarm_cp0"] = 100 * alarms / iters
        rows.append(row)
        print(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000, help="iterations per cell (paper: 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state", choices=["none", "prev"], default="none")
    ap.add_argument("--out", type=Path, default=Path("results/bonjour_rps"))
    args = ap.parse_args()
    global STATE
    STATE = args.state
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    f3 = fig3(rng, args.out)
    t2 = table2(rng, args.iters)
    for name, rows in (("fig3.csv", f3), ("table2.csv", t2)):
        with open(args.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    (args.out / "config.json").write_text(json.dumps({"alpha": ALPHA, **{k: str(v) for k, v in vars(args).items()}}))


if __name__ == "__main__":
    main()
