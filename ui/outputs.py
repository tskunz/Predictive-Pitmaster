"""Main area output display components."""

import streamlit as st
from datetime import datetime, timedelta
from simulation.monte_carlo import SimResult
from simulation.constants import PROTEIN_CUTS


def _fmt_time(dt: datetime) -> str:
    """Format datetime as 12-hour time without leading zero, cross-platform."""
    return dt.strftime("%I:%M %p").lstrip("0") or dt.strftime("%I:%M %p")


def render_outputs(result: SimResult, inputs: dict, start_time: datetime) -> None:
    """Render primary outputs: start time, eat time, confidence, percentiles.

    Args:
        result: Monte Carlo simulation result.
        inputs: Raw input dict from sidebar.
        start_time: When the fire should be lit (datetime).
    """
    dinner_time = inputs.get("dinner_time")
    rest_min = inputs.get("rest_duration_min", 60)

    # Compute finish times as clock times
    fire_start = start_time
    eat_time = dinner_time if dinner_time else fire_start + timedelta(minutes=result.p50_minutes + rest_min)

    p10_finish = fire_start + timedelta(minutes=result.p10_minutes)
    p50_finish = fire_start + timedelta(minutes=result.p50_minutes)
    p90_finish = fire_start + timedelta(minutes=result.p90_minutes)

    # Confidence badge color
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(result.confidence, "gray")

    # --- Hero section ---
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Light Fire At", _fmt_time(fire_start) if fire_start else "—")
    with col2:
        st.metric("Eat At", _fmt_time(eat_time))
    with col3:
        st.markdown(
            f"**Confidence**  \n"
            f"<span style='color:{conf_color}; font-size:1.4em; font-weight:bold'>{result.confidence}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Percentile display ---
    st.subheader("Finish Time Estimates")
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimistic (P10)", _fmt_time(p10_finish),
              delta=f"{result.p10_minutes/60:.1f} hrs from fire")
    c2.metric("Most Likely (P50)", _fmt_time(p50_finish),
              delta=f"{result.p50_minutes/60:.1f} hrs from fire")
    c3.metric("Conservative (P90)", _fmt_time(p90_finish),
              delta=f"{result.p90_minutes/60:.1f} hrs from fire")

    spread_hrs = (result.p90_minutes - result.p10_minutes) / 60.0
    st.caption(f"Total cook time range: {result.p10_minutes/60:.1f} – {result.p90_minutes/60:.1f} hours "
               f"(+/-{spread_hrs/2:.1f} hr uncertainty window)")

    # --- Stall warning ---
    if result.stall_probability > 0.3:
        st.warning(
            f"**Stall likely** — probability {result.stall_probability:.0%}. "
            "Your cook will plateau around 150–165°F for 2–4 hours. This is normal physics, not a problem. "
            "The P90 estimate accounts for it."
        )
    elif result.stall_probability > 0.1:
        st.info(
            f"**Stall possible** — probability {result.stall_probability:.0%}. "
            "Watch for a plateau around 150–165°F."
        )
