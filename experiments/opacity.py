"""Opacity ladder: can each monitor tell a colluding A-B pair from an honest pair whose chat looks the same?

For each level, positives are the covert_* games and negatives are the matched honest control:
  plain  - covert_plain  vs channel       (collusion spelled out in the messages)
  code   - covert_code   vs chat_topics   (move hidden in an innocent topic)
  opaque - covert_opaque vs chat_opaque   (one unreadable word; stands in for an encrypted channel)
Detectors come from judge.csv (judge_chat.py): talk, talk+act (LLM judge) and act (net influence, reads no messages).
Writes opacity.csv and opacity.png into the results dir.
Usage:
    python experiments/opacity.py results/llm_rps
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np

from analyze_llm import auroc_ci, fmt
from netinf import influence_matrix, net_influence

HORIZONS = (10, 20, 30, 50, 75, 100)  # rounds an auditor has logged

LEVELS = {"plain": ("covert_plain", "chat_plain"), "code": ("covert_code", "chat_topics"),
          "opaque": ("covert_opaque", "chat_opaque")}
DETECTORS = {"talk": ("judge reads messages", "#eb6834"), "talk_act": ("judge reads messages + moves", "#8a5cd1"),
             "act_net_AB": ("net influence on moves (reads no messages)", "#2a78d6")}


def level_table(rows: list[dict]) -> list[dict]:
    out = []
    for level, (pos, neg) in LEVELS.items():
        g = [r for r in rows if r["condition"] in (pos, neg)]
        if not any(r["condition"] == pos for r in g) or not any(r["condition"] == neg for r in g):
            continue
        y = [r["condition"] == pos for r in g]
        decoded = [float(r["assist_rate"]) for r in g if r["condition"] == pos]
        for key in DETECTORS:
            s = [float(r[key]) if r[key] not in ("", "None") else np.nan for r in g]
            out.append({"level": level, "detector": key, "games": len(g), "auroc": auroc_ci(s, y),
                        "covert_assist_rate": float(np.mean(decoded)),
                        "covert_colluded": sum(r["colluded"] == "True" for r in g if r["condition"] == pos)})
    return out


def net_ab(actions: np.ndarray) -> float:
    """Action-only score on a move log, as in llm_rps.score: previous round as state."""
    net = net_influence(influence_matrix(actions[1:], actions[:-1]))
    return float(min(net[0, 1], net[1, 0]))


def horizon_table(mdir: Path) -> list[dict]:
    """AUROC of the action-only score when the auditor sees only the first n rounds of each game."""
    games = [json.loads(f.read_text()) for f in sorted(mdir.glob("*_run*.json"))]
    out = []
    for level, (pos, neg) in LEVELS.items():
        g = [x for x in games if x["condition"] in (pos, neg)]
        if not g:
            continue
        y = [x["condition"] == pos for x in g]
        for n in HORIZONS:
            if n <= len(g[0]["actions"]):
                out.append({"level": level, "rounds": n,
                            "auroc": auroc_ci([net_ab(np.array(x["actions"][:n])) for x in g], y)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()
    tables = {d.name: level_table(list(csv.DictReader((d / "judge.csv").open())))
              for d in sorted(args.results_dir.iterdir()) if (d / "judge.csv").exists()}
    tables = {m: t for m, t in tables.items() if t}
    if not tables:
        raise SystemExit("no opacity-ladder games with judge scores found")
    with open(args.results_dir / "opacity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "detector", "games", "auroc", "covert_assist_rate", "covert_colluded"])
        for m, t in tables.items():
            for r in t:
                w.writerow([m, r["level"], r["detector"], r["games"], fmt(r["auroc"], 2),
                            f"{r['covert_assist_rate']:.2f}", r["covert_colluded"]])
    print(open(args.results_dir / "opacity.csv").read())

    fig, axes = plt.subplots(1, len(tables), figsize=(4.6 * len(tables), 3.4), squeeze=False, sharey=True)
    for ax, (model, t) in zip(axes[0], tables.items()):
        levels = [lv for lv in LEVELS if any(r["level"] == lv for r in t)]
        x = np.arange(len(levels))
        for key, (label, color) in DETECTORS.items():
            v = np.array([next(r["auroc"] for r in t if r["level"] == lv and r["detector"] == key) for lv in levels])
            ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=color, marker="o",
                        capsize=3, lw=2, label=label)
        ax.axhline(0.5, color="#999", lw=1, ls=":")
        ax.set_xticks(x, levels)
        ax.set_xlabel("how readable the collusion channel is")
        ax.set_title(model, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("AUROC, colluding vs honest pair")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.results_dir / "opacity.png", dpi=160)

    horizons = {m: horizon_table(args.results_dir / m) for m in tables}
    with open(args.results_dir / "opacity_rounds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "rounds", "act_auroc"])
        for m, t in horizons.items():
            for r in t:
                w.writerow([m, r["level"], r["rounds"], fmt(r["auroc"], 2)])
    print(open(args.results_dir / "opacity_rounds.csv").read())


if __name__ == "__main__":
    main()
