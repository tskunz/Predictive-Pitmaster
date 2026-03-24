"""Rest phase model — Newton's Law of Cooling.

Models the temperature drop of rested meat after hitting target temp.
Safety floor: flag if temp drops below 145°F USDA minimum.
"""

import math
from .constants import COOLING_RATE, REST_SAFETY_FLOOR_F


def rest_temperature(
    pull_temp_f: float,
    ambient_temp_f: float,
    time_minutes: float,
    rest_method: str = "cooler",
) -> float:
    """Compute meat temperature after resting using Newton's Law of Cooling.

    T(t) = T_ambient + (T_initial - T_ambient) × e^(-k × t)

    Args:
        pull_temp_f: Internal temp when pulled off the smoker (°F).
        ambient_temp_f: Ambient/room temperature (°F).
        time_minutes: Rest duration in minutes.
        rest_method: One of "none", "oven", "cooler".

    Returns:
        Meat temperature after resting in °F.
    """
    k = COOLING_RATE.get(rest_method, COOLING_RATE["cooler"])  # 1/min
    temp = ambient_temp_f + (pull_temp_f - ambient_temp_f) * math.exp(-k * time_minutes)  # °F
    return temp


def rest_is_safe(temp_f: float) -> bool:
    """Return True if rested temperature is above the USDA safety floor (145°F)."""
    return temp_f >= REST_SAFETY_FLOOR_F  # °F
