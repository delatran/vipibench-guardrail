from __future__ import annotations

import math


def temperature_scale_probability(score: float, temperature: float) -> float:
    if not 0.0 <= score <= 1.0:
        raise ValueError("prediction score must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    clipped = min(max(score, 1e-7), 1 - 1e-7)
    logit = math.log(clipped / (1 - clipped)) / temperature
    return 1 / (1 + math.exp(-logit))
