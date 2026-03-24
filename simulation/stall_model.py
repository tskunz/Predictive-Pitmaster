"""Stall hazard model — logistic probability function.

Physics-informed priors, not trained weights. Refinement happens in Phase 3
when empirical cook data is available.
"""

import math
from .constants import STALL_TEMP_LOW_F, STALL_TEMP_HIGH_F


def stall_probability(
    current_temp_f: float,
    humidity_pct: float = 50.0,
    wind_mph: float = 5.0,
    thickness_in: float = 5.0,
    temp_slope: float = 0.5,
) -> float:
    """Compute probability of stall onset at current conditions.

    Temperature-gated: returns 0.0 outside [140, 185]°F.
    Uses a logistic hazard function with physics-informed coefficients.

    Args:
        current_temp_f: Current internal meat temperature in °F.
        humidity_pct: Relative humidity in percent (0–100).
        wind_mph: Wind speed in mph.
        thickness_in: Meat thickness at thickest point in inches.
        temp_slope: Current rate of temperature rise in °F/min.

    Returns:
        Stall onset probability (0.0–1.0).
    """
    if current_temp_f < STALL_TEMP_LOW_F or current_temp_f > STALL_TEMP_HIGH_F:
        return 0.0

    temp_distance = -abs(current_temp_f - 160.0)  # °F — peaks near 160°F

    z = (
        -3.0                         # intercept: stall is not the default state
        + 0.02  * humidity_pct       # higher humidity → more evaporative surface cooling
        + 0.10  * wind_mph           # more airflow → more evaporation
        + 0.50  * thickness_in       # thicker cut → larger moisture reservoir
        - 4.00  * temp_slope         # fast-rising temp → not stalling
        + 0.15  * temp_distance      # closer to 160°F → higher probability
    )

    return 1.0 / (1.0 + math.exp(-z))
