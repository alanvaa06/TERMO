"""Regime models behind one small interface: the jump model and the K-means baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from sklearn.cluster import KMeans

N_INIT = 10
RANDOM_STATE = 0


class RegimeModel(Protocol):
    @property
    def n_states(self) -> int: ...

    def insample_labels(self) -> np.ndarray:
        """Labels of the training rows, as fitted."""
        ...

    def online_labels(self, features: pd.DataFrame) -> np.ndarray:
        """Label of row t using rows up to t only."""
        ...

    def full_labels(self, features: pd.DataFrame) -> np.ndarray:
        """Labels decoded with the whole sequence in view."""
        ...


RegimeFitter = Callable[[pd.DataFrame], RegimeModel]


@dataclass(frozen=True, eq=False)
class JumpRegimeModel:
    model: JumpModel
    n_states: int

    def insample_labels(self) -> np.ndarray:
        return np.asarray(self.model.labels_, dtype=int)

    def online_labels(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_online(features), dtype=int)

    def full_labels(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=int)


@dataclass(frozen=True, eq=False)
class KMeansRegimeModel:
    model: KMeans
    n_states: int

    def insample_labels(self) -> np.ndarray:
        return np.asarray(self.model.labels_, dtype=int)

    def online_labels(self, features: pd.DataFrame) -> np.ndarray:
        # No memory: the nearest centroid today is the same with or without the future.
        return np.asarray(self.model.predict(features.to_numpy()), dtype=int)

    def full_labels(self, features: pd.DataFrame) -> np.ndarray:
        return self.online_labels(features)


def jump_fitter(n_states: int, jump_penalty: float) -> RegimeFitter:
    def fit(features: pd.DataFrame) -> RegimeModel:
        model = JumpModel(
            n_components=n_states,
            jump_penalty=jump_penalty,
            cont=False,
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        )
        model.fit(features)
        return JumpRegimeModel(model=model, n_states=n_states)

    return fit


def kmeans_fitter(n_states: int) -> RegimeFitter:
    """The jump model with zero penalty, fitted with scikit-learn because it is far faster."""

    def fit(features: pd.DataFrame) -> RegimeModel:
        model = KMeans(n_clusters=n_states, n_init=N_INIT, random_state=RANDOM_STATE)
        model.fit(features.to_numpy())
        return KMeansRegimeModel(model=model, n_states=n_states)

    return fit
