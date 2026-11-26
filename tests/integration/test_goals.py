from worth_reaching.planning.goals import GoalSequence, Stage


def equal(observation, target):
    return observation == target


def outcomes(goals):
    return [(event.index, event.end_step, event.outcome) for event in goals.events]


def test_arrival_on_the_deadline_counts_and_a_miss_abandons_later_stages():
    goals = GoalSequence([Stage("a", 2), Stage("b", 3), Stage("c", 9)], equal)
    assert goals.advance("x", 0).target == "a"
    assert goals.advance("a", 2).target == "b"
    assert goals.advance("x", 3) is None
    assert outcomes(goals) == [(0, 2, "arrived"), (1, 3, "timeout")]
    assert not goals.arrived_all
    assert goals.handoff_step == 3


def test_one_observation_can_satisfy_consecutive_stages():
    goals = GoalSequence([Stage("a"), Stage("a")], equal)
    assert goals.advance("a", 4) is None
    assert goals.arrived_all
    assert [event.start_step for event in goals.events] == [0, 4]


def test_termination_abandonment_and_suffix_replacement():
    goals = GoalSequence([Stage("a"), Stage("b")], equal)
    goals.advance("a", 1)
    goals.replace_remaining([Stage("c")])
    assert goals.active.target == "c"
    goals.finish("x", 5)
    assert outcomes(goals) == [(0, 1, "arrived"), (1, 5, "terminated")]

    abandoned = GoalSequence([Stage("a")], equal)
    abandoned.abandon(0)
    assert outcomes(abandoned) == [(0, 0, "abandoned")]
    assert abandoned.handoff_step == 0


def test_direct_execution_has_no_stages():
    goals = GoalSequence([], equal)
    goals.finish("x", 7)
    assert goals.arrived_all and goals.events == [] and goals.handoff_step == 0
