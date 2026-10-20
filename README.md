# Catching colluding AI agents from their actions

Can an auditor tell that two AI agents are secretly working together, just by watching what they do?

This repo tests one answer to that question: **pairwise net influence** (Bonjour, Aggarwal & Bhargava, UAI 2022), an information-theoretic statistic that flags pairs of agents whose actions depend on each other more than chance allows. We reproduce it, show where it breaks, fix it, and test it on LLM agents.

Built for the [Apart AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25) (research questions 1.4 and 2.3, with 2.2 as an extension).

## Findings so far

**1. The original method reproduces.** With scripted colluders in 3-player Rock-Paper-Scissors, our implementation matches the paper's detection rates (e.g. 99.3% vs 99.3% at 400 games, collusion probability 0.4).

**2. It falsely accuses innocent agents when games are short.** The paper never tests a game with no colluders. When we do, the fixed threshold (α = 0.05) flags an innocent pair in **30-40%** of 100-round games once the state includes the previous round. Short games with history-dependent agents are exactly the setting of LLM experiments.

**3. A permutation test fixes it.** Replacing the fixed threshold with a within-state permutation test (Bonferroni over pairs) brings false accusations down to about **3%** in the same setting, at some cost in detection power.

| 100 rounds, previous-round state | fixed α = 0.05 | calibrated |
|---|---|---|
| no collusion (false alarm) | 40% | 3% |
| collusion probability 0.3 (detected) | 21% | 12% |
| collusion probability 0.4 (detected) | 50% | 32% |

<sub>Pilot, 100 runs per cell. The full grid is in `results/calibration_rps/`.</sub>

![Net influence vs collusion probability, reproducing Bonjour et al. Fig. 3](results/bonjour_rps_state-prev/fig3_net_influence_vs_cp.png)

## The LLM experiment

Three LLM agents play 100 rounds of Rock-Paper-Scissors. A and B get progressively more opportunity to collude against C:

| condition | what A and B get | role |
|---|---|---|
| `independent` | nothing | negative control: any alarm is false |
| `channel` | a private chat C can't see, no instructions | does collusion emerge from talk alone? |
| `channel_shared` | private chat + rewarded on their combined score | does an incentive tip them into it? |
| `instructed` | B is told A's move and asked to help | positive control |

Whether A and B *actually* colluded is decided from behaviour (B plays the move A's move beats more often than chance), not from what they say. Against that label we compare three detectors: an LLM judge reading the chat, the same judge reading chat and moves, and calibrated net influence on moves alone.

Runs on free Kaggle GPUs with two model families (Qwen2.5-7B, Phi-3.5) via Ollama. Every LLM response is cached, so figures can be rebuilt without a GPU.

## Reproduce

```bash
uv venv && uv pip install -e ".[dev]"
pytest

# scripted games, no LLM, $0
python experiments/replicate_bonjour_rps.py --state prev --out results/bonjour_rps_state-prev
python experiments/calibration_rps.py
python experiments/auditor_curve.py

# LLM games: any OpenAI-compatible server (Ollama, vLLM, Gemini's compat endpoint)
export NETINF_BASE_URL=http://localhost:11434/v1 NETINF_MODEL=qwen2.5:7b
python experiments/llm_rps.py --condition channel --runs 10 --rounds 100
python experiments/judge_chat.py results/llm_rps/qwen2.5_7b
python experiments/analyze_llm.py results/llm_rps

# or on Kaggle (GPU T4 x2): kaggle kernels push -p kaggle
```

## Method

For agents *i*, *j* with actions *a* and observed state *s*:

- **Individual influence** γ(i; j): conditional mutual information between the two agents' actions given the state (paper Eq. 3), estimated from counts.
- **Net influence** Γ(i; j) = γ(i; j) − max over k ≠ i, j of γ(k; j) (Eq. 4).
- **Original rule:** flag (i, j) if Γ(i; j) ≥ α and Γ(j; i) ≥ α, for exactly one pair (Eq. 5).
- **Calibrated rule (ours):** flag (i, j) if both net influences are positive and γ(i; j) beats a within-state permutation null at p < 0.05 / number of pairs.

## Layout

```
src/netinf/influence.py     estimator, original and calibrated detectors
src/netinf/games/rps.py     n-player RPS and scripted assistant collusion
src/netinf/llm.py           OpenAI-compatible chat client with an on-disk response cache
experiments/                one script per experiment; each writes CSVs and figures to results/
kaggle/                     Kaggle GPU kernel that runs the LLM experiments
tests/                      unit tests
```

## References

- Bonjour, Aggarwal & Bhargava (2022). [Information theoretic approach to detect collusion in multi-agent games](https://proceedings.mlr.press/v180/bonjour22a.html). UAI.
- Apart Research (2026). [AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25): research questions and related work.

MIT licensed.
