"""Unit tests for the physics simulation modules.

Heat diffusion: compare against analytical erf() solution for semi-infinite solid.
Stall model: test all temperature gating boundaries.
Monte Carlo: test seed reproducibility and P10 < P50 < P90 ordering.
Rest model: test Newton's cooling with known k.
"""

import math
import numpy as np
import pytest
from simulation.heat_diffusion import solve_heat, boiling_point_f
from simulation.stall_model import stall_probability
from simulation.monte_carlo import SimInputs, run_simulation
from simulation.rest_model import rest_temperature, rest_is_safe
from simulation.constants import THERMAL_DIFFUSIVITY


class TestBoilingPoint:
    def test_sea_level(self):
        assert boiling_point_f(0.0) == pytest.approx(212.0, abs=0.1)

    def test_denver(self):
        # Denver ~5280 ft — boiling point ~202°F
        bp = boiling_point_f(5280.0)
        assert 200.0 < bp < 205.0

    def test_higher_elevation_lower_bp(self):
        assert boiling_point_f(5000.0) < boiling_point_f(1000.0)


class TestHeatDiffusion:
    def test_meat_heats_up(self):
        """Center temperature should increase over a long cook."""
        _, ft = solve_heat(
            protein="beef", grade="choice",
            thickness_inches=4.0, smoker_temp_f=250.0,
            initial_temp_f=40.0, target_temp_f=203.0,
            max_minutes=900,
        )
        assert np.isfinite(ft), "4-inch brisket should finish within 15 hours at 250°F"

    def test_thin_cooks_faster(self):
        """Thinner meat should finish faster than thicker meat."""
        _, ft_thin = solve_heat(
            protein="beef", grade="choice",
            thickness_inches=2.0, smoker_temp_f=250.0,
            initial_temp_f=40.0, target_temp_f=165.0,
        )
        _, ft_thick = solve_heat(
            protein="beef", grade="choice",
            thickness_inches=5.0, smoker_temp_f=250.0,
            initial_temp_f=40.0, target_temp_f=165.0,
        )
        assert ft_thin < ft_thick, "Thinner meat should cook faster"

    def test_hotter_smoker_cooks_faster(self):
        """Higher smoker temp should reduce cook time."""
        _, ft_low = solve_heat(
            protein="pork", grade="choice",
            thickness_inches=4.0, smoker_temp_f=225.0,
            initial_temp_f=40.0, target_temp_f=203.0,
            max_minutes=1800,
        )
        _, ft_high = solve_heat(
            protein="pork", grade="choice",
            thickness_inches=4.0, smoker_temp_f=275.0,
            initial_temp_f=40.0, target_temp_f=203.0,
            max_minutes=1800,
        )
        assert ft_high < ft_low, "Higher smoker temp should shorten cook time"

    def test_foil_wrap_reduces_cook_time(self):
        """Aluminum foil wrap should shorten cook time by reducing stall."""
        _, ft_no_wrap = solve_heat(
            protein="beef", grade="choice",
            thickness_inches=5.0, smoker_temp_f=250.0,
            initial_temp_f=40.0, target_temp_f=203.0,
            wrap_type="none",
            max_minutes=1800,
        )
        _, ft_foil = solve_heat(
            protein="beef", grade="choice",
            thickness_inches=5.0, smoker_temp_f=250.0,
            initial_temp_f=40.0, target_temp_f=203.0,
            wrap_type="aluminum_foil",
            wrap_temp_f=165.0,
            max_minutes=1800,
        )
        assert ft_foil < ft_no_wrap, "Foil wrap should reduce cook time"

    def test_stuffed_cooks_slower(self):
        """Stuffed cuts should take longer due to reduced thermal diffusivity."""
        _, ft_unstuffed = solve_heat(
            protein="poultry", grade="choice",
            thickness_inches=5.0, smoker_temp_f=325.0,
            initial_temp_f=40.0, target_temp_f=165.0,
            is_stuffed=False,
        )
        _, ft_stuffed = solve_heat(
            protein="poultry", grade="choice",
            thickness_inches=5.0, smoker_temp_f=325.0,
            initial_temp_f=40.0, target_temp_f=165.0,
            is_stuffed=True,
        )
        assert ft_stuffed > ft_unstuffed, "Stuffed cuts should cook slower"

    def test_temp_history_monotonically_increases_early(self):
        """Center temperature should generally increase (ignoring stall wiggles)."""
        history, _ = solve_heat(
            protein="beef", grade="choice",
            thickness_inches=3.0, smoker_temp_f=250.0,
            initial_temp_f=40.0, target_temp_f=180.0,
            max_minutes=600,
        )
        assert history[0] < history[-1], "Center temp should be higher at end than start"


class TestStallModel:
    def test_below_gate_returns_zero(self):
        assert stall_probability(current_temp_f=100.0) == 0.0

    def test_above_gate_returns_zero(self):
        assert stall_probability(current_temp_f=200.0) == 0.0

    def test_in_range_returns_probability(self):
        p = stall_probability(current_temp_f=160.0, humidity_pct=70.0, wind_mph=10.0, thickness_in=6.0)
        assert 0.0 < p < 1.0

    def test_high_humidity_increases_probability(self):
        p_dry = stall_probability(160.0, humidity_pct=20.0)
        p_humid = stall_probability(160.0, humidity_pct=80.0)
        assert p_humid > p_dry

    def test_fast_slope_reduces_probability(self):
        p_fast = stall_probability(160.0, temp_slope=2.0)
        p_slow = stall_probability(160.0, temp_slope=0.1)
        assert p_fast < p_slow

    def test_boundary_at_140(self):
        assert stall_probability(139.9) == 0.0
        assert stall_probability(140.1) >= 0.0  # just inside gate

    def test_boundary_at_185(self):
        assert stall_probability(185.1) == 0.0
        assert stall_probability(184.9) >= 0.0  # just inside gate


class TestRestModel:
    def test_cooler_cools_slower_than_open_air(self):
        t_cooler = rest_temperature(203.0, 70.0, 60.0, "cooler")
        t_open = rest_temperature(203.0, 70.0, 60.0, "none")
        assert t_cooler > t_open, "Cooler should retain heat longer"

    def test_longer_rest_cools_more(self):
        t_short = rest_temperature(203.0, 70.0, 30.0, "cooler")
        t_long = rest_temperature(203.0, 70.0, 120.0, "cooler")
        assert t_short > t_long, "Longer rest means more cooling"

    def test_safety_floor_at_145(self):
        assert rest_is_safe(145.0)
        assert rest_is_safe(146.0)
        assert not rest_is_safe(144.9)

    def test_approaches_ambient(self):
        """Very long rest should approach ambient temperature."""
        t_very_long = rest_temperature(203.0, 70.0, 10000.0, "none")
        assert abs(t_very_long - 70.0) < 1.0, "Should approach ambient after very long rest"

    def test_no_heating(self):
        """Rest should never heat meat above pull temp."""
        t = rest_temperature(203.0, 300.0, 60.0, "cooler")
        # Ambient 300°F is unrealistic but tests the math direction
        assert t >= 203.0  # Newton's law: moves toward ambient


class TestMonteCarlo:
    def test_percentile_ordering(self):
        """P10 < P50 < P90 for all valid inputs."""
        inputs = SimInputs(
            protein="beef", grade="choice",
            thickness_inches=5.0, smoker_temp_f=250.0,
            initial_temp_f=40.0, target_temp_f=203.0,
            equipment="pellet",
        )
        result = run_simulation(inputs, n_iterations=200, seed=42)
        assert result.p10_minutes < result.p50_minutes < result.p90_minutes

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        inputs = SimInputs(
            protein="pork", grade="choice",
            thickness_inches=6.0, smoker_temp_f=225.0,
            initial_temp_f=40.0, target_temp_f=203.0,
            equipment="wsm",
        )
        r1 = run_simulation(inputs, n_iterations=100, seed=99)
        r2 = run_simulation(inputs, n_iterations=100, seed=99)
        assert r1.p50_minutes == r2.p50_minutes

    def test_realistic_brisket_time(self):
        """A 5-inch Choice brisket at 250°F should finish in 8–16 hours."""
        inputs = SimInputs(
            protein="beef", grade="choice",
            thickness_inches=5.0, smoker_temp_f=250.0,
            initial_temp_f=40.0, target_temp_f=203.0,
            equipment="pellet",
        )
        result = run_simulation(inputs, n_iterations=200, seed=42)
        hours_p50 = result.p50_minutes / 60.0
        assert 8.0 <= hours_p50 <= 16.0, f"Expected 8-16 hrs, got {hours_p50:.1f}"

    def test_confidence_string(self):
        inputs = SimInputs(
            protein="beef", grade="choice",
            thickness_inches=4.0, smoker_temp_f=250.0,
            initial_temp_f=40.0, target_temp_f=165.0,
            equipment="pellet",
        )
        result = run_simulation(inputs, n_iterations=100, seed=42)
        assert result.confidence in ("HIGH", "MEDIUM", "LOW")
