"""Pick K and lambda by walk-forward score; FTIC gives a second opinion on K."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from termo.config import FticConfig
from termo.validation.ftic import ftic


@dataclass(frozen=True)
class Candidate:
    trial_id: str
    n_states: int
    jump_penalty: float
    stability: float
    separation: float  # excess eta-squared at the short horizon
    passes_duration: bool
    wcss: float
    jumps: int

    @property
    def score(self) -> float:
        return self.stability * self.separation


def candidate_from_record(
    trial_id: str, config: Mapping[str, Any], metrics: Mapping[str, Any]
) -> Candidate:
    return Candidate(
        trial_id=trial_id,
        n_states=int(config["n_states"]),
        jump_penalty=float(config["jump_penalty"]),
        stability=float(metrics["stability"]),
        separation=float(metrics["excess_short"]),
        passes_duration=bool(metrics["passes_duration"]),
        wcss=float(metrics["wcss"]),
        jumps=int(metrics["jumps"]),
    )


def is_eligible(candidate: Candidate) -> bool:
    """Can this configuration be the final model? Same rule for the winner and for any other."""
    return candidate.passes_duration and candidate.stability > 0 and candidate.separation > 0


def pick_winner(candidates: Sequence[Candidate]) -> Candidate | None:
    """Highest stability x separation among the eligible configurations.

    Eligible: passes the duration filter, and both factors are positive. The product of
    two negative numbers would otherwise rank an unstable, unseparated model first.
    """
    eligible = [c for c in candidates if is_eligible(c)]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c.score)


def ftic_states(
    candidates: Sequence[Candidate],
    jump_penalty: float,
    *,
    wcss_saturated: float,
    n_obs: int,
    n_features: int,
    config: FticConfig,
) -> int | None:
    """K with the lowest FTIC among the configurations that share `jump_penalty`."""
    prior_jumps = n_obs / config.mean_phase_days
    values: dict[int, float] = {}
    for c in candidates:
        if c.jump_penalty != jump_penalty or c.jumps > config.max_jump_fraction * n_obs:
            continue
        values[c.n_states] = ftic(
            c.n_states,
            c.wcss,
            c.jumps,
            wcss_saturated=wcss_saturated,
            saturated_states=config.saturated_k,
            n_obs=n_obs,
            n_features=n_features,
            prior_states=config.k0,
            prior_jumps=prior_jumps,
        )
    if not values:
        return None
    return min(values, key=lambda k: values[k])


def simpler_alternative(
    candidates: Sequence[Candidate], winner: Candidate, ftic_k: int | None
) -> Candidate | None:
    """When FTIC prefers fewer states than the winner: same lambda, the smaller K."""
    if ftic_k is None or ftic_k >= winner.n_states:
        return None
    for c in candidates:
        if c.n_states == ftic_k and c.jump_penalty == winner.jump_penalty:
            return c
    return None
