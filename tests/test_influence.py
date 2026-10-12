import numpy as np
import pytest

from netinf import detect, individual_influence, influence_matrix, net_influence, pair_scores
from netinf.games.rps import AssistantCollusion, payoff, simulate


def test_identical_uniform_actions_give_log3():
    a = np.tile([0, 1, 2], 300)
    assert individual_influence(a, a) == pytest.approx(np.log(3))
    assert individual_influence(a, a, base=2) == pytest.approx(np.log2(3))


def test_independent_actions_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 3, 20_000), rng.integers(0, 3, 20_000)
    assert individual_influence(a, b) < 1e-3


def test_conditioning_on_state_removes_shared_cause():
    # Both agents copy a common observed state: dependent marginally, independent given state.
    rng = np.random.default_rng(1)
    s = rng.integers(0, 3, 5_000)
    assert individual_influence(s, s) > 1.0
    assert individual_influence(s, s, s, s) == pytest.approx(0.0, abs=1e-12)


def test_shared_state_matrix_is_symmetric():
    rng = np.random.default_rng(2)
    actions, _ = simulate(4, 500, AssistantCollusion(0, 1, 0.5), rng)
    g = influence_matrix(actions)
    assert np.allclose(g, g.T)


def test_net_influence_needs_three_agents():
    with pytest.raises(ValueError):
        net_influence(np.zeros((2, 2)))


def test_net_influence_definition():
    g = np.array([[0, 0.5, 0.1], [0.5, 0, 0.2], [0.1, 0.2, 0]])
    net = net_influence(g)
    assert net[0, 1] == pytest.approx(0.5 - 0.2)  # gamma(0;1) - gamma(2;1)
    assert net[1, 0] == pytest.approx(0.5 - 0.1)  # gamma(1;0) - gamma(2;0)
    assert np.isnan(net[0, 0])


@pytest.mark.parametrize(
    "actions, expected",
    [
        ([0, 0, 0], [0, 0, 0]),  # all same
        ([0, 1, 2], [1, 1, 1]),  # all distinct
        ([0, 2, 2], [1, 0, 0]),  # R beats S
        ([1, 0, 1], [1, 0, 1]),  # P beats R
        ([2, 1, 0], [1, 1, 1]),
        ([2, 2, 1], [1, 1, 0]),  # S beats P
    ],
)
def test_payoff_matches_bonjour_table1(actions, expected):
    assert payoff(actions).tolist() == expected


def test_assistant_collusion_always_scores_primary():
    rng = np.random.default_rng(3)
    _, rewards = simulate(3, 2_000, AssistantCollusion(0, 1, 1.0), rng)
    assert rewards[:, 0].min() == 1


def test_detects_strong_collusion():
    rng = np.random.default_rng(4)
    actions, _ = simulate(3, 1_000, AssistantCollusion(0, 1, 0.6), rng)
    assert detect(net_influence(influence_matrix(actions))) == (0, 1)


def test_no_false_alarm_without_collusion():
    rng = np.random.default_rng(5)
    alarms = sum(
        detect(net_influence(influence_matrix(simulate(3, 1_000, None, rng)[0]))) is not None for _ in range(50)
    )
    assert alarms == 0


def test_pair_scores_rank_true_pair_first():
    rng = np.random.default_rng(6)
    actions, _ = simulate(4, 1_000, AssistantCollusion(2, 3, 0.5), rng)
    scores = pair_scores(net_influence(influence_matrix(actions)))
    assert max(scores, key=scores.get) == (2, 3)
