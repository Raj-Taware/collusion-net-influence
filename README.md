# netinf: net-influence collusion detection for LLM agents

Research project for the [Apart AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25) (research question 1.4).

**Question.** Pairwise net influence (Bonjour, Aggarwal & Bhargava, UAI 2022) identifies a colluding pair among 3-4 scripted or RL agents from their actions alone. Does it work when the agents are LLMs?

> **Timing disclosure.** Because of a scheduling conflict with the official sprint window (Oct 23-25, 2026), this work was done starting Oct 12, 2026. The git history shows the real dates.

## Status

- [x] Net influence estimator (Eq. 3-5 of the paper), with tests
- [x] Positive control: reproduce the paper's 3-player Rock-Paper-Scissors results with scripted colluders
- [ ] LLM agent harness (Gemini)
- [ ] LLM experiments: net influence vs action correlation vs LLM judge

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"
pytest
python experiments/replicate_bonjour_rps.py --state none --out results/bonjour_rps_state-none
python experiments/replicate_bonjour_rps.py --state prev --out results/bonjour_rps_state-prev
```

## Method

For agents *i*, *j* with actions *a* and observed states *s*:

- **Individual influence** γ(i; j): conditional mutual information between the two agents' actions given state (Eq. 3), estimated from counts.
- **Net influence** Γ(i; j) = γ(i; j) − max over k≠i,j of γ(k; j) (Eq. 4).
- **Detection rule:** flag (i, j) if Γ(i; j) ≥ α and Γ(j; i) ≥ α, and exactly one pair qualifies (Eq. 5). The paper uses α = 0.05.

## Layout

```
src/netinf/influence.py      estimator: individual_influence, influence_matrix, net_influence, detect
src/netinf/games/rps.py      n-player RPS, payoff rules, scripted assistant collusion
experiments/                 runnable experiment scripts
results/                     CSVs and figures produced by experiments
tests/                       unit tests
```
