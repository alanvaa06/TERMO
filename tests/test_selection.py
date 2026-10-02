from __future__ import annotations

from termo.config import FticConfig
from termo.selection import (
    Candidate,
    candidate_from_record,
    ftic_states,
    is_eligible,
    pick_winner,
    simpler_alternative,
)

FTIC = FticConfig(k0=3, mean_phase_days=40, saturated_k=6, max_jump_fraction=0.4)


def candidate(
    n_states: int,
    jump_penalty: float,
    stability: float = 0.8,
    separation: float = 0.1,
    passes: bool = True,
    wcss: float = 5000.0,
    jumps: int = 20,
) -> Candidate:
    return Candidate(
        trial_id=f"jm_k{n_states}_lam{jump_penalty:g}",
        n_states=n_states,
        jump_penalty=jump_penalty,
        stability=stability,
        separation=separation,
        passes_duration=passes,
        wcss=wcss,
        jumps=jumps,
    )


def test_candidate_from_record() -> None:
    built = candidate_from_record(
        "jm_k3_lam80",
        {"model": "jump", "n_states": 3, "jump_penalty": 80.0},
        {
            "stability": 0.7,
            "excess_short": 0.2,
            "passes_duration": True,
            "wcss": 10.0,
            "jumps": 4,
        },
    )
    assert built == candidate(3, 80.0, 0.7, 0.2, True, 10.0, 4)
    assert built.score == 0.7 * 0.2


def test_winner_has_the_best_score_among_those_passing_duration() -> None:
    flickering = candidate(4, 5.0, stability=0.9, separation=0.5, passes=False)
    solid = candidate(3, 80.0, stability=0.8, separation=0.2)
    weak = candidate(2, 80.0, stability=0.9, separation=0.1)
    assert pick_winner([flickering, solid, weak]) == solid
    assert pick_winner([flickering]) is None
    assert pick_winner([]) is None


def test_winner_needs_positive_stability_and_separation() -> None:
    """Two negative factors multiply into a positive score; that must not win."""
    both_negative = candidate(3, 80.0, stability=-0.05, separation=-0.02)
    barely_good = candidate(2, 80.0, stability=0.8, separation=0.0005)
    assert both_negative.score > barely_good.score
    assert pick_winner([both_negative, barely_good]) == barely_good
    unseparated = candidate(2, 12.0, stability=0.9, separation=-0.01)
    assert pick_winner([both_negative, unseparated]) is None


def test_one_eligibility_rule_for_every_candidate() -> None:
    """The simpler model that FTIC proposes must clear the same bar as the winner."""
    assert is_eligible(candidate(2, 80.0))
    assert not is_eligible(candidate(2, 80.0, passes=False))
    assert not is_eligible(candidate(2, 80.0, stability=0.0))
    assert not is_eligible(candidate(2, 80.0, separation=0.0))
    assert not is_eligible(candidate(2, 80.0, separation=-0.01))


def test_ftic_compares_only_the_winning_penalty() -> None:
    candidates = [
        candidate(2, 80.0, wcss=9000.0),
        candidate(3, 80.0, wcss=5000.0),
        candidate(4, 80.0, wcss=4990.0),
        candidate(5, 12.0, wcss=10.0),  # other penalty: ignored
    ]
    kwargs = {"wcss_saturated": 3000.0, "n_obs": 2000, "n_features": 12, "config": FTIC}
    assert ftic_states(candidates, 80.0, **kwargs) == 3
    assert ftic_states(candidates, 999.0, **kwargs) is None


def test_ftic_excludes_models_that_jump_too_often() -> None:
    candidates = [candidate(2, 80.0, wcss=9000.0), candidate(3, 80.0, wcss=10.0, jumps=900)]
    kwargs = {"wcss_saturated": 3000.0, "n_obs": 2000, "n_features": 12, "config": FTIC}
    assert ftic_states(candidates, 80.0, **kwargs) == 2


def test_simpler_alternative_only_when_ftic_wants_fewer_states() -> None:
    two, three, four = candidate(2, 80.0), candidate(3, 80.0), candidate(4, 80.0)
    other_penalty = candidate(2, 12.0)
    pool = [other_penalty, two, three, four]
    assert simpler_alternative(pool, three, 2) == two
    assert simpler_alternative(pool, three, 3) is None
    assert simpler_alternative(pool, three, 4) is None
    assert simpler_alternative(pool, three, None) is None
    assert simpler_alternative([three, four], three, 2) is None
