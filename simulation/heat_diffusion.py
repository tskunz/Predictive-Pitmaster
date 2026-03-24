"""1D heat diffusion solver — Fourier's Law, explicit finite differences.

The meat is modeled as a 1D slab (surface to center, half-thickness).
Explicit FD scheme with convective (Robin) boundary conditions.
Evaporative cooling from the stall is applied per-timestep (coupled, not post-hoc).

All numeric values have unit comments.
"""

import math
import numpy as np
from .constants import (
    THERMAL_DIFFUSIVITY, BIOT_NUMBER, N_NODES,
    WRAP_EVAP_REDUCTION, STALL_TEMP_LOW_F, STALL_TEMP_HIGH_F,
)

# Base evaporative cooling rate at stall peak (°F/min of effective surface cooling)
BASE_EVAP_RATE = 1.0  # °F/min


def boiling_point_f(elevation_ft: float) -> float:
    """Compute boiling point of water at given elevation.

    Args:
        elevation_ft: Elevation above sea level in feet.

    Returns:
        Boiling point in °F.
    """
    return 212.0 - (1.5 * elevation_ft / 1000.0)  # °F


def solve_heat(
    protein: str,
    grade: str,
    thickness_inches: float,
    smoker_temp_f: float,
    initial_temp_f: float,
    target_temp_f: float,
    wrap_type: str = "none",
    wrap_temp_f: float | None = None,
    elevation_ft: float = 0.0,
    is_stuffed: bool = False,
    smoker_temp_noise: np.ndarray | None = None,
    wind_factor: float = 1.0,
    humidity_factor: float = 1.0,
    dt_minutes: float = 1.0,
    max_minutes: int = 1800,
    alpha_override_mm2s: float | None = None,
    n_nodes: int | None = None,
) -> tuple[np.ndarray, float]:
    """Solve 1D heat equation forward in time.

    Uses explicit finite differences with Robin boundary conditions:
      Surface: T[0]_new = T[0] + Fo*(T[1]-T[0]) + Fo*Bi*(T_smoker-T[0])
      Interior: T[i]_new = T[i] + Fo*(T[i-1] - 2*T[i] + T[i+1])
    where Fo = alpha*dt/dx² (Fourier number). Stability: Fo*(1+Bi) <= 0.5.

    Args:
        protein: Protein type key (e.g. "beef").
        grade: Grade key (e.g. "choice").
        thickness_inches: Full thickness of meat at thickest point, in inches.
        smoker_temp_f: Target smoker temperature in °F.
        initial_temp_f: Starting internal temperature of meat in °F.
        target_temp_f: Target internal temperature (done) in °F.
        wrap_type: Wrap strategy key from WRAP_EVAP_REDUCTION.
        wrap_temp_f: Temperature at which wrap is applied (°F). None = no planned wrap.
        elevation_ft: Elevation above sea level in feet.
        is_stuffed: Whether the cut is stuffed (reduces effective alpha by 20%).
        smoker_temp_noise: Array of per-minute smoker temp fluctuations (°F). Length >= max_minutes.
        wind_factor: Multiplier on Biot number (>1 = more convective cooling, longer cook).
        humidity_factor: Multiplier on evaporative cooling rate (>1 = more evaporation).
        dt_minutes: Time step in minutes.
        max_minutes: Maximum simulation time in minutes.
        alpha_override_mm2s: If provided, use this thermal diffusivity (mm²/s) instead of lookup.
        n_nodes: Number of spatial nodes. Defaults to N_NODES (50). Use smaller value (e.g. 15)
            for faster Monte Carlo iterations at slightly reduced spatial resolution.

    Returns:
        Tuple of (center_temp_history: np.ndarray at 1-min intervals, finish_time_minutes: float).
        finish_time_minutes is np.inf if meat never reached target temp.
    """
    # Half-thickness: model surface to center (symmetry condition)
    nn = n_nodes if n_nodes is not None else N_NODES  # number of spatial nodes
    L_mm = (thickness_inches / 2.0) * 25.4  # mm — half-thickness in mm
    dx = L_mm / nn  # mm — spatial step

    if alpha_override_mm2s is not None:
        alpha_mm2s = alpha_override_mm2s  # mm²/s — directly provided
    else:
        alpha = THERMAL_DIFFUSIVITY.get(protein, {}).get(grade, 1.28e-7)  # m²/s
        alpha_mm2s = alpha * 1e6  # mm²/s — convert for mm-based grid

    if is_stuffed:
        alpha_mm2s *= 0.80  # mm²/s — 20% reduction for stuffed cuts

    dt_s = dt_minutes * 60.0  # seconds
    Bi = BIOT_NUMBER * wind_factor  # dimensionless

    # Compute Fourier number; enforce stability
    Fo = alpha_mm2s * dt_s / (dx ** 2)  # dimensionless
    max_Fo = 0.45 / (1.0 + Bi)  # dimensionless — stability limit
    if Fo > max_Fo:
        dt_s = max_Fo * dx ** 2 / alpha_mm2s  # seconds — reduce time step
        Fo = max_Fo  # dimensionless

    dt_min_actual = dt_s / 60.0  # minutes
    n_steps = int(max_minutes / dt_min_actual) + 1
    bp = boiling_point_f(elevation_ft)  # °F — surface temp ceiling

    # Initialize temperature field: uniform at initial temp
    T = np.full(nn + 1, initial_temp_f, dtype=np.float64)  # °F
    center_idx = nn  # center node (half-slab, center = far end from surface)

    # Output at 1-minute intervals
    output_interval = max(1, int(1.0 / dt_min_actual))
    n_output = int(max_minutes) + 1
    temp_history = np.zeros(n_output, dtype=np.float64)  # °F
    output_idx = 0

    evap_base = BASE_EVAP_RATE * humidity_factor  # °F/min
    stall_high = min(STALL_TEMP_HIGH_F, bp)  # °F — altitude-corrected upper gate
    finish_time = np.inf  # minutes

    # Evaporative weight vector for this node count (surface=1.0, center=0.3)
    evap_weights = np.array(
        [1.0 - 0.7 * (i / nn) for i in range(nn + 1)], dtype=np.float64
    )  # dimensionless

    for step in range(n_steps):
        current_time_min = step * dt_min_actual  # minutes

        # Smoker temperature with noise
        smoker_eff = smoker_temp_f  # °F
        if smoker_temp_noise is not None:
            noise_idx = min(int(current_time_min), len(smoker_temp_noise) - 1)
            smoker_eff += smoker_temp_noise[noise_idx]

        # Explicit FD update
        T_new = T.copy()
        # Interior nodes
        T_new[1:nn] = (
            T[1:nn]
            + Fo * (T[0:nn-1] - 2.0 * T[1:nn] + T[2:nn+1])
        )
        # Surface (x=0): Robin BC — convective heat from smoker
        T_new[0] = T[0] + Fo * (T[1] - T[0]) + Fo * Bi * (smoker_eff - T[0])
        # Center (x=L): symmetry — no flux, mirror from adjacent node
        T_new[nn] = T[nn] + Fo * (T[nn-1] - T[nn])

        # Evaporative cooling in stall zone (coupled, not post-hoc)
        surface_temp = T_new[0]  # °F
        if STALL_TEMP_LOW_F <= surface_temp <= stall_high:
            is_wrapped = wrap_type != "none"
            if wrap_temp_f is not None:
                is_wrapped = is_wrapped and surface_temp >= wrap_temp_f

            reduction = WRAP_EVAP_REDUCTION.get(wrap_type, 0.0) if is_wrapped else 0.0  # dimensionless
            effective_evap = evap_base * (1.0 - reduction)  # °F/min

            # Logistic ramp: peaks near midpoint of stall zone
            midpoint = (STALL_TEMP_LOW_F + stall_high) / 2.0  # °F
            spread = (stall_high - STALL_TEMP_LOW_F) / 6.0  # °F
            logistic = 1.0 / (1.0 + math.exp(-(surface_temp - midpoint) / spread))  # dimensionless

            # Scale by driving delta (more heat → more evaporation, capped at 100°F delta)
            driving_delta = max(smoker_eff - surface_temp, 0.0)  # °F
            evap_scale = min(driving_delta / 100.0, 1.0)  # dimensionless

            evap_cooling = effective_evap * logistic * evap_scale * dt_min_actual  # °F

            # Apply with surface weighting (vectorized): full at surface, 30% at center
            T_new -= evap_cooling * evap_weights  # °F — vectorized over all nodes

        T = T_new
        np.minimum(T, bp, out=T)  # °F — cap at boiling point

        # Record output
        if step % output_interval == 0 and output_idx < n_output:
            temp_history[output_idx] = T[center_idx]
            output_idx += 1

        # Check finish — early exit once center reaches target (no need to continue)
        if T[center_idx] >= target_temp_f and finish_time == np.inf:
            finish_time = current_time_min  # minutes
            break  # early exit: finish time found, no need to simulate further

    return temp_history[:output_idx], finish_time


def solve_heat_batch(
    protein: str,
    grade: str,
    thickness_inches: float,
    smoker_temp_f: float,
    initial_temp_f: float,
    target_temp_f: float,
    alpha_mm2s_arr: np.ndarray,
    temp_std: float,
    wind_factors: np.ndarray,
    humidity_factors: np.ndarray,
    rng: np.random.Generator,
    wrap_type: str = "none",
    wrap_temp_f: float | None = None,
    elevation_ft: float = 0.0,
    is_stuffed: bool = False,
    n_nodes: int = 5,
    max_minutes: int = 1800,
) -> np.ndarray:
    """Vectorized batch FD solver: all N iterations advance simultaneously.

    Instead of running N serial solves, this runs ONE time-stepping loop where
    each step updates all N iterations in parallel via NumPy broadcasting.
    Shape convention: (n_iter, n_nodes+1) for the temperature field.

    Args:
        protein: Protein key for lookup fallback.
        grade: Grade key for lookup fallback.
        thickness_inches: Full thickness in inches (half-slab modeled).
        smoker_temp_f: Smoker setpoint in °F (noise applied per-step).
        initial_temp_f: Starting meat temperature in °F.
        target_temp_f: Target internal temperature (done condition) in °F.
        alpha_mm2s_arr: Thermal diffusivity per iteration in mm²/s, shape (n_iter,).
        temp_std: Smoker temperature noise std dev in °F (same for all iterations).
        wind_factors: Per-iteration Biot multipliers, shape (n_iter,).
        humidity_factors: Per-iteration evap cooling multipliers, shape (n_iter,).
        rng: NumPy random generator (noise sampled on-the-fly to save memory).
        wrap_type: Wrap strategy key.
        wrap_temp_f: Temperature at which wrap is applied in °F.
        elevation_ft: Elevation above sea level in feet.
        is_stuffed: Reduces effective alpha by 20%.
        n_nodes: Spatial nodes (5 is sufficient for MC; use 50 for display solve).
        max_minutes: Maximum simulation time in minutes.

    Returns:
        finish_times: np.ndarray of shape (n_iter,) in minutes. np.inf if not finished.
    """
    n_iter = len(alpha_mm2s_arr)
    nn = n_nodes  # spatial nodes

    L_mm = (thickness_inches / 2.0) * 25.4  # mm — half-thickness
    dx = L_mm / nn  # mm — spatial step

    alpha_arr = alpha_mm2s_arr.copy()  # mm²/s — per iteration
    if is_stuffed:
        alpha_arr *= 0.80  # mm²/s — stuffed modifier

    Bi_arr = BIOT_NUMBER * wind_factors  # dimensionless — per iteration

    # Time step: use max alpha + max Bi for strictest stability across all iterations
    alpha_max = float(np.max(alpha_arr))  # mm²/s
    Bi_max = float(np.max(Bi_arr))        # dimensionless
    max_Fo = 0.45 / (1.0 + Bi_max)       # dimensionless — stability limit

    # Always use the maximum stable time step (Fo = max_Fo).
    # This maximises accuracy-per-step and minimises total step count.
    # With n_nodes=8 and typical α≈0.128 mm²/s the step is ~3–4 min,
    # giving ~450 loop iterations per 1800-min sim — fast and accurate.
    dt_s = max_Fo * dx**2 / alpha_max  # seconds

    dt_min_actual = dt_s / 60.0  # minutes
    n_steps = int(max_minutes / dt_min_actual) + 1

    bp = boiling_point_f(elevation_ft)  # °F
    stall_high = min(STALL_TEMP_HIGH_F, bp)  # °F

    # Per-iteration Fourier numbers — shape (n_iter,)
    Fo_arr = alpha_arr * dt_s / dx**2  # dimensionless

    # Temperature field — shape (n_iter, nn+1)
    T = np.full((n_iter, nn + 1), initial_temp_f, dtype=np.float64)  # °F
    center_idx = nn  # center node index

    finish_times = np.full(n_iter, np.inf)  # minutes
    done = np.zeros(n_iter, dtype=bool)

    # Evaporative weight profile: 1.0 at surface, 0.3 at center — shape (nn+1,)
    evap_weights = np.linspace(1.0, 0.3, nn + 1)  # dimensionless

    # Wrap reduction scalar
    wrap_reduction = WRAP_EVAP_REDUCTION.get(wrap_type, 0.0)  # dimensionless

    for step in range(n_steps):
        if np.all(done):
            break

        current_time_min = step * dt_min_actual  # minutes

        # Smoker noise sampled on-the-fly (avoids (n_iter, max_minutes) allocation)
        if temp_std > 0.0:
            smoker_eff = smoker_temp_f + rng.normal(0.0, temp_std, size=n_iter)  # °F
        else:
            smoker_eff = np.full(n_iter, smoker_temp_f)  # °F

        # --- Explicit FD update (all iterations in parallel) ---
        T_new = T.copy()

        # Interior nodes: shape (n_iter, nn-1)
        T_new[:, 1:nn] = (
            T[:, 1:nn]
            + Fo_arr[:, None] * (T[:, 0:nn-1] - 2.0 * T[:, 1:nn] + T[:, 2:nn+1])
        )

        # Surface BC (Robin — convective): shape (n_iter,)
        T_new[:, 0] = (
            T[:, 0]
            + Fo_arr * (T[:, 1] - T[:, 0])
            + Fo_arr * Bi_arr * (smoker_eff - T[:, 0])
        )

        # Center BC (symmetry — zero flux): shape (n_iter,)
        T_new[:, nn] = T[:, nn] + Fo_arr * (T[:, nn - 1] - T[:, nn])

        # --- Evaporative cooling in stall zone ---
        surface_temps = T_new[:, 0]  # °F — shape (n_iter,)
        in_stall = (surface_temps >= STALL_TEMP_LOW_F) & (surface_temps <= stall_high)

        if np.any(in_stall):
            midpoint = (STALL_TEMP_LOW_F + stall_high) / 2.0  # °F
            spread_val = (stall_high - STALL_TEMP_LOW_F) / 6.0  # °F

            # Determine which iterations are actively wrapped
            is_wrapped = np.zeros(n_iter, dtype=bool)
            if wrap_type != "none":
                if wrap_temp_f is not None:
                    is_wrapped = surface_temps >= wrap_temp_f
                else:
                    is_wrapped = np.ones(n_iter, dtype=bool)

            reduction_arr = np.where(is_wrapped, wrap_reduction, 0.0)  # dimensionless

            evap_base_arr = BASE_EVAP_RATE * humidity_factors * (1.0 - reduction_arr)  # °F/min

            logistic = 1.0 / (1.0 + np.exp(
                -(surface_temps - midpoint) / spread_val
            ))  # dimensionless
            driving_delta = np.maximum(smoker_eff - surface_temps, 0.0)  # °F
            evap_scale = np.minimum(driving_delta / 100.0, 1.0)  # dimensionless

            # Total evap cooling this step per iteration — shape (n_iter,)
            evap_cooling = evap_base_arr * logistic * evap_scale * dt_min_actual  # °F

            # Surface-only evap cooling for the batch solver.
            # All-node weighting is numerically unstable at coarse resolution (n_nodes=8)
            # because large dt means large per-step extraction that outpaces conduction.
            # Surface-only is physically correct (evaporation happens at the surface);
            # the stall effect on the interior is captured naturally via the reduced gradient.
            # The absolute time bias from this approximation is corrected by the high-fidelity
            # anchor in run_simulation(), so only the spread matters here.
            T_new[in_stall, 0] -= evap_cooling[in_stall]

        np.clip(T_new, None, bp, out=T_new)  # °F — cap at boiling point
        T = T_new

        # Track finish times: first step where center reaches target
        newly_done = (~done) & (T[:, center_idx] >= target_temp_f)
        if np.any(newly_done):
            finish_times[newly_done] = current_time_min  # minutes
            done |= newly_done

    return finish_times
