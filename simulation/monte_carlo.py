"""Monte Carlo simulation engine — vectorized batch implementation.

All N iterations run simultaneously through a batched FD solver.
One Python loop over time steps; each step processes all iterations
in parallel via NumPy broadcasting on shape (n_iter, n_nodes+1) arrays.

Target: < 15 seconds for 5,000 iterations.
"""

from dataclasses import dataclass
import numpy as np
from .constants import (
    THERMAL_DIFFUSIVITY, BIOLOGICAL_NOISE_FRACTION, EQUIPMENT_PROFILES,
)
from .heat_diffusion import solve_heat, solve_heat_batch
from .stall_model import stall_probability


@dataclass
class SimInputs:
    """All inputs required for the Monte Carlo simulation."""
    protein: str
    grade: str
    thickness_inches: float        # inches
    smoker_temp_f: float           # °F
    initial_temp_f: float          # °F (fridge temp, typically ~40°F)
    target_temp_f: float           # °F
    equipment: str
    wrap_type: str = "none"
    wrap_temp_f: float | None = None  # °F
    elevation_ft: float = 0.0     # feet
    is_stuffed: bool = False
    humidity_pct: float = 50.0    # percent
    wind_mph: float = 5.0         # mph
    ambient_temp_f: float = 70.0  # °F


@dataclass
class SimResult:
    """Monte Carlo simulation output."""
    p10_minutes: float            # minutes — optimistic (10th percentile) finish
    p50_minutes: float            # minutes — median finish
    p90_minutes: float            # minutes — conservative (90th percentile) finish
    confidence: str               # "HIGH", "MEDIUM", or "LOW"
    stall_probability: float      # 0.0–1.0 — estimated probability of stall during cook
    sensitivity: dict             # fractional contribution of each noise source to spread
    all_finish_times: np.ndarray  # minutes — raw MC finish times for histogram


def run_simulation(
    inputs: SimInputs,
    n_iterations: int = 5000,
    seed: int | None = None,
) -> SimResult:
    """Run Monte Carlo simulation and return P10/P50/P90 finish times.

    Uses a vectorized batch FD solver — all iterations advance simultaneously.

    Args:
        inputs: All cook parameters (see SimInputs).
        n_iterations: Number of Monte Carlo iterations (default 5,000).
        seed: Optional random seed for reproducibility.

    Returns:
        SimResult with percentiles, confidence tier, stall probability, and sensitivity.
    """
    rng = np.random.default_rng(seed)

    base_alpha = THERMAL_DIFFUSIVITY.get(inputs.protein, {}).get(inputs.grade, 1.28e-7)  # m²/s
    temp_std = EQUIPMENT_PROFILES.get(inputs.equipment, {}).get("temp_std_f", 15.0)   # °F

    # --- Sample noise sources ---
    # Biological variability: ±10% uniform on thermal diffusivity
    alpha_multipliers = rng.uniform(
        1.0 - BIOLOGICAL_NOISE_FRACTION,
        1.0 + BIOLOGICAL_NOISE_FRACTION,
        size=n_iterations,
    )  # dimensionless
    alpha_mm2s_arr = base_alpha * alpha_multipliers * 1e6  # mm²/s — per iteration

    # Weather perturbations: ±15% around condition-adjusted baselines
    wind_base = max(0.5, 1.0 + (inputs.wind_mph - 5.0) * 0.02)          # dimensionless
    humid_base = max(0.5, 1.0 + (inputs.humidity_pct - 50.0) * 0.005)   # dimensionless

    wind_factors = np.clip(
        rng.uniform(0.85, 1.15, size=n_iterations) * wind_base, 0.3, 2.0
    )  # dimensionless
    humidity_factors = np.clip(
        rng.uniform(0.85, 1.15, size=n_iterations) * humid_base, 0.3, 2.0
    )  # dimensionless

    # --- High-fidelity anchor: single solve at nominal inputs (n_nodes=50) ---
    # This gives the calibrated P50 (best estimate). The batch MC is used only
    # for the *spread* (uncertainty range). Combining them gives accurate absolute
    # times without paying the O(n³) cost of running n_nodes=50 for all iterations.
    _, t_anchor = solve_heat(
        protein=inputs.protein,
        grade=inputs.grade,
        thickness_inches=inputs.thickness_inches,
        smoker_temp_f=inputs.smoker_temp_f,
        initial_temp_f=inputs.initial_temp_f,
        target_temp_f=inputs.target_temp_f,
        wrap_type=inputs.wrap_type,
        wrap_temp_f=inputs.wrap_temp_f,
        elevation_ft=inputs.elevation_ft,
        is_stuffed=inputs.is_stuffed,
        wind_factor=1.0,
        humidity_factor=1.0,
        max_minutes=1800,
    )  # minutes — nominal-condition reference

    # --- Vectorized batch MC: captures spread from all noise sources ---
    finish_times = solve_heat_batch(
        protein=inputs.protein,
        grade=inputs.grade,
        thickness_inches=inputs.thickness_inches,
        smoker_temp_f=inputs.smoker_temp_f,
        initial_temp_f=inputs.initial_temp_f,
        target_temp_f=inputs.target_temp_f,
        alpha_mm2s_arr=alpha_mm2s_arr,
        temp_std=temp_std,
        wind_factors=wind_factors,
        humidity_factors=humidity_factors,
        rng=rng,
        wrap_type=inputs.wrap_type,
        wrap_temp_f=inputs.wrap_temp_f,
        elevation_ft=inputs.elevation_ft,
        is_stuffed=inputs.is_stuffed,
        n_nodes=8,
        max_minutes=1800,
    )

    valid = finish_times[np.isfinite(finish_times)]
    n_valid = len(valid)

    if n_valid < n_iterations * 0.3:
        p10 = float(np.percentile(valid, 10)) if n_valid > 0 else 1800.0
        p50 = t_anchor if np.isfinite(t_anchor) else 1800.0
        p90 = 1800.0
        confidence = "LOW"
    else:
        # Re-center the batch distribution around the high-fidelity anchor.
        # The batch P50 has systematic spatial error; the anchor corrects it.
        # The spread (P90-P10) is preserved from the batch.
        batch_p50 = float(np.percentile(valid, 50))  # minutes — biased center
        batch_p10 = float(np.percentile(valid, 10))  # minutes
        batch_p90 = float(np.percentile(valid, 90))  # minutes

        half_spread_lo = batch_p50 - batch_p10  # minutes — lower half-spread
        half_spread_hi = batch_p90 - batch_p50  # minutes — upper half-spread

        p50 = t_anchor if np.isfinite(t_anchor) else batch_p50
        p10 = p50 - half_spread_lo   # minutes
        p90 = p50 + half_spread_hi   # minutes

        spread = p90 - p10  # minutes
        if spread < 90:
            confidence = "HIGH"
        elif spread < 150:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

    # --- Stall probability ---
    # Estimate at the representative stall-zone temperature (160°F midpoint)
    stall_prob = stall_probability(
        current_temp_f=160.0,             # °F — midpoint of stall zone
        humidity_pct=inputs.humidity_pct,
        wind_mph=inputs.wind_mph,
        thickness_in=inputs.thickness_inches,
        temp_slope=0.3,                   # °F/min — typical rising slope
    )

    # --- Sensitivity analysis ---
    sensitivity = _sensitivity_analysis(inputs, base_alpha, temp_std, rng)

    return SimResult(
        p10_minutes=round(p10, 1),
        p50_minutes=round(p50, 1),
        p90_minutes=round(p90, 1),
        confidence=confidence,
        stall_probability=round(stall_prob, 3),
        sensitivity=sensitivity,
        all_finish_times=valid,
    )


def _sensitivity_analysis(
    inputs: SimInputs,
    base_alpha: float,
    temp_std: float,
    rng: np.random.Generator,
    n_small: int = 500,
) -> dict:
    """Estimate fractional contribution of each noise source to prediction spread.

    Runs 3 mini batches, each with only one noise source active.
    Returns fractions normalized to sum to 1.0.

    Args:
        inputs: Cook parameters.
        base_alpha: Base thermal diffusivity in m²/s.
        temp_std: Equipment smoker temp std dev in °F.
        rng: Random generator.
        n_small: Iterations per sensitivity batch.

    Returns:
        Dict mapping source name → fractional spread contribution (0.0–1.0).
    """
    def _batch_spread(alpha_mult_arr, t_std, wind_var, humid_var) -> float:
        """Run a mini batch and return P90-P10 spread in minutes."""
        alpha_arr = base_alpha * alpha_mult_arr * 1e6  # mm²/s
        wf = np.clip(rng.uniform(1.0 - wind_var, 1.0 + wind_var, size=n_small), 0.3, 2.0)
        hf = np.clip(rng.uniform(1.0 - humid_var, 1.0 + humid_var, size=n_small), 0.3, 2.0)
        ft = solve_heat_batch(
            protein=inputs.protein, grade=inputs.grade,
            thickness_inches=inputs.thickness_inches,
            smoker_temp_f=inputs.smoker_temp_f,
            initial_temp_f=inputs.initial_temp_f,
            target_temp_f=inputs.target_temp_f,
            alpha_mm2s_arr=alpha_arr,
            temp_std=t_std, wind_factors=wf, humidity_factors=hf, rng=rng,
            wrap_type=inputs.wrap_type, wrap_temp_f=inputs.wrap_temp_f,
            elevation_ft=inputs.elevation_ft, is_stuffed=inputs.is_stuffed,
            n_nodes=8, max_minutes=1800,
        )
        v = ft[np.isfinite(ft)]
        return float(np.percentile(v, 90) - np.percentile(v, 10)) if len(v) > 1 else 0.0

    # Biological only: alpha ±10%, no equipment noise, no weather noise
    bio_spread = _batch_spread(
        rng.uniform(1.0 - BIOLOGICAL_NOISE_FRACTION, 1.0 + BIOLOGICAL_NOISE_FRACTION, n_small),
        t_std=0.0, wind_var=0.0, humid_var=0.0,
    )

    # Equipment only: flat alpha, full equipment noise, no weather noise
    equip_spread = _batch_spread(
        np.ones(n_small),
        t_std=temp_std, wind_var=0.0, humid_var=0.0,
    )

    # Weather only: flat alpha, no equipment noise, weather ±15%
    weather_spread = _batch_spread(
        np.ones(n_small),
        t_std=0.0, wind_var=0.15, humid_var=0.15,
    )

    total = bio_spread + equip_spread + weather_spread or 1.0  # avoid divide-by-zero
    return {
        "Biological variability": round(bio_spread / total, 3),
        "Equipment variance":     round(equip_spread / total, 3),
        "Weather variability":    round(weather_spread / total, 3),
    }
