"""The Predictive Pitmaster — Streamlit app entry point.

Know when your BBQ will be done, with quantified uncertainty.
Physics-informed Monte Carlo simulation: 5,000 iterations per prediction.
"""

import streamlit as st
from datetime import datetime, timedelta

from simulation.monte_carlo import SimInputs, run_simulation
from simulation.rest_model import rest_temperature, rest_is_safe
from ui.inputs import render_sidebar
from ui.outputs import render_outputs
from ui.charts import render_phase_timeline, render_distribution, render_sensitivity


st.set_page_config(
    page_title="The Predictive Pitmaster",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: dark background, readable on mobile
st.markdown("""
<style>
    .stApp { background-color: #1a1a1a; color: #f0f0f0; }
    .stMetric { background-color: #2a2a2a; border-radius: 8px; padding: 12px; }
    .stButton>button { background-color: #CD853F; color: white; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #B8732A; }
    section[data-testid="stSidebar"] { background-color: #2a2a2a; }
    h1, h2, h3 { color: #F5A623; }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("The Predictive Pitmaster")
    st.caption("Physics-informed BBQ cook time estimator — we quantify uncertainty so you can plan dinner.")

    inputs = render_sidebar()

    if inputs is None:
        # Landing state: show instructions
        st.markdown("""
        ### How it works

        1. **Fill in the sidebar** — protein, cut, smoker type, weather, wrap plan
        2. **Hit Run Prediction** — 5,000 physics simulations run in seconds
        3. **Get three times** — Optimistic (P10), Most Likely (P50), Conservative (P90)
        4. **Plan your dinner** — the app back-calculates when to light your fire

        ---

        **The physics:**
        Heat moves through meat via the Fourier heat equation. The stall (that 150–165°F
        plateau) is caused by evaporative surface cooling matching heat input from the smoker.
        Monte Carlo sampling captures biological variation in the meat, smoker temperature
        fluctuations, and weather effects — giving you a realistic uncertainty range rather
        than a single guess.

        ---
        """)
        return

    # Build SimInputs from sidebar values
    sim_inputs = SimInputs(
        protein=inputs["protein"],
        grade=inputs["grade"],
        thickness_inches=inputs["thickness_inches"],
        smoker_temp_f=inputs["smoker_temp_f"],
        initial_temp_f=inputs["initial_temp_f"],
        target_temp_f=inputs["target_temp_f"],
        equipment=inputs["equipment"],
        wrap_type=inputs["wrap_type"],
        wrap_temp_f=inputs["wrap_temp_f"],
        elevation_ft=inputs["elevation_ft"],
        is_stuffed=inputs["is_stuffed"],
        humidity_pct=inputs["humidity_pct"],
        wind_mph=inputs["wind_mph"],
        ambient_temp_f=inputs["ambient_temp_f"],
    )

    # Run simulation with spinner
    with st.spinner("Running 5,000 simulations..."):
        result = run_simulation(sim_inputs, n_iterations=5000)

    # Compute fire start time
    dinner_time = inputs.get("dinner_time")
    rest_min = inputs.get("rest_duration_min", 60)

    if dinner_time:
        # Backward plan: dinner - P90 cook time - rest = fire start
        fire_start = dinner_time - timedelta(minutes=result.p90_minutes + rest_min)
    else:
        fire_start = datetime.now()

    # Render outputs
    render_outputs(result, inputs, fire_start)

    st.markdown("---")
    st.subheader("Phase Timeline")
    render_phase_timeline(result, inputs)

    # Rest phase safety check
    rest_temp = rest_temperature(
        pull_temp_f=inputs["target_temp_f"],
        ambient_temp_f=inputs["ambient_temp_f"],
        time_minutes=rest_min,
        rest_method=inputs["rest_method"],
    )
    if not rest_is_safe(rest_temp):
        st.error(
            f"**Food safety warning:** After {rest_min} min rest "
            f"({inputs['rest_method'].replace('_', ' ')}), meat may drop to "
            f"{rest_temp:.0f}°F — below the 145°F USDA safe serving floor. "
            "Consider a shorter rest or switch to cooler/oven method."
        )
    else:
        st.caption(
            f"After {rest_min} min rest ({inputs['rest_method'].replace('_', ' ')}): "
            f"estimated ~{rest_temp:.0f}°F — safe to serve."
        )

    # Expandable charts
    with st.expander("Show Distribution of 5,000 Simulations"):
        render_distribution(result, inputs)

    with st.expander("Show Sensitivity Analysis"):
        st.caption("Which inputs contribute most to your uncertainty window?")
        render_sensitivity(result)
        for source, fraction in result.sensitivity.items():
            st.write(f"**{source}** contributes **{fraction:.0%}** of your prediction spread")

    with st.expander("About the Model"):
        st.markdown("""
        **Physics Engine:** 1D Fourier heat equation solved via explicit finite differences.
        The meat is modeled as a slab of uniform composition. Heat conducts from surface to center.

        **Stall Model:** The BBQ stall (plateau around 150–165°F) is caused by evaporative cooling
        from surface moisture, which temporarily balances heat input from the smoker.
        A logistic hazard function models onset probability as a function of humidity, wind, and thickness.

        **Monte Carlo:** 5,000 iterations sample biological variability in thermal diffusivity (+/-10%),
        smoker temperature fluctuations (per equipment profile), and weather perturbations (+/-15%).

        **Constants:** Thermal diffusivity values from Singh & Heldman, *Introduction to Food Engineering*, Table 4.2.
        These are physics-informed priors. Calibration against real cook data happens in future phases.

        **What this is NOT:** A trained ML model. A guarantee. A replacement for a thermometer.
        """)


if __name__ == "__main__":
    main()
