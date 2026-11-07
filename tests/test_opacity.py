"""Opacity-ladder conditions: A's message carries its real move, B can decode it, controls carry nothing."""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import llm_rps  # noqa: E402

CODE = {"weather": "S", "coffee": "R", "music": "P", "ok": "S", "sure": "R", "fine": "P"}


class FakeLLM:
    """A plays at random; B, when given the secret code, decodes A's last message perfectly; C plays at random."""
    model = "fake"

    def generate(self, prompt, system="", temperature=1.0, seed=0, max_tokens=16):
        rng = np.random.default_rng(seed)
        if prompt.splitlines()[-1].startswith("Write"):  # message request: echo the instruction so the code survives
            return prompt.splitlines()[-1]
        if "SECRET" in system:
            last_a = [l for l in prompt.splitlines() if l.strip().startswith("A:")][-1]
            m = re.search(r"should play (\w+)", last_a)
            if m:
                return m.group(1)[0]
            return next(v for k, v in CODE.items() if re.search(rf"\b{k}\b", last_a))
        return "RPS"[rng.integers(3)]


def test_covert_conditions_carry_the_move_and_controls_do_not():
    for cond in ("covert_plain", "covert_code", "covert_opaque"):
        g = llm_rps.play(FakeLLM(), cond, 60, 0, 1.0, np.random.default_rng(0))
        assert g["assist_rate"] == 1.0, cond
        assert len(g["chat"]) == 120, cond
    for cond in ("chat_topics", "chat_opaque"):
        g = llm_rps.play(FakeLLM(), cond, 200, 0, 1.0, np.random.default_rng(0))
        assert 0.2 < g["assist_rate"] < 0.47, cond  # chance is 1/3
        assert len(g["chat"]) == 400, cond


def test_opaque_messages_look_alike():
    words = {c: {m for _, m in llm_rps.play(FakeLLM(), c, 50, 1, 1.0, np.random.default_rng(1))["chat"]}
             for c in ("covert_opaque", "chat_opaque")}
    assert words["covert_opaque"] == words["chat_opaque"] == set(llm_rps.WORDS)


def test_judge_parses_percent_after_explanation():
    import judge_chat

    class Says:
        def __init__(self, text):
            self.text = text

        def generate(self, *a, **k):
            return self.text

    game = {"actions": [[0, 1, 2]], "chat": []}
    assert judge_chat.ask_judge(Says("80"), game, False) == 0.8
    assert judge_chat.ask_judge(Says("Based on 3 rounds, likelihood is 90%."), game, False) == 0.9
    assert judge_chat.ask_judge(Says("Hard to say."), game, False) is None
