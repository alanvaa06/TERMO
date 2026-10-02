"""Do the same regimes come out when the model sees different years?"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from termo.features.pipeline import fit_pipeline
from termo.regime.model import RegimeFitter
from termo.validation.metrics import adjusted_rand


def halves_ari(
    curve: pd.DataFrame, columns: Sequence[str], burn_in: int, fitter: RegimeFitter
) -> float:
    """S2: fit one model per half of the sample, let both label the whole sample, compare.

    Each half has its own PCA, clipping and scaling, so nothing is shared but the method.
    """
    dates = curve.index[burn_in:]
    middle = len(dates) // 2
    first_end, second_start, last = dates[middle - 1], dates[middle], dates[-1]
    windows = (
        (fit_pipeline(curve, first_end, columns, burn_in), dates[0], first_end),
        (fit_pipeline(curve, last, columns, burn_in, train_start=second_start), second_start, last),
    )
    labelings: list[np.ndarray] = []
    for pipeline, train_start, train_end in windows:
        features = pipeline.transform(curve)
        model = fitter(features.loc[train_start:train_end])
        labelings.append(model.full_labels(features))
    return adjusted_rand(labelings[0], labelings[1])


def stability_score(consecutive_ari: Sequence[float], halves: float) -> float:
    """Average of S1 (mean ARI between consecutive refits) and S2 (halves)."""
    if not consecutive_ari:
        raise ValueError("stability needs at least two refits")
    return (float(np.mean(consecutive_ari)) + halves) / 2.0
