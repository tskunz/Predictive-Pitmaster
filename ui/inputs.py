"""Sidebar input components for the Predictive Pitmaster.

All inputs live in st.sidebar. Returns a dict of validated inputs.
"""

import streamlit as st
from datetime import datetime, time
from simulation.constants import PROTEIN_CUTS, GRADE_OPTIONS, EQUIPMENT_PROFILES, WRAP_EVAP_REDUCTION


def render_sidebar() -> dict | None:
    """Render all sidebar inputs and return a dict of values, or None if not submitted.

    Returns:
        Dict with keys matching SimInputs fields plus dinner_time, rest_method, rest_duration_min.
        Returns None until the Run Prediction button is pressed.
    """
    st.sidebar.title("Cook Setup")

    # --- Step 1: Protein & Geometry ---
    st.sidebar.header("1. Protein & Geometry")

    protein = st.sidebar.selectbox(
        "Protein",
        options=list(PROTEIN_CUTS.keys()),
        format_func=str.capitalize,
    )

    cuts = PROTEIN_CUTS[protein]
    cut_key = st.sidebar.selectbox(
        "Cut",
        options=list(cuts.keys()),
        format_func=lambda k: cuts[k]["name"],
    )
    cut_info = cuts[cut_key]

    grades = GRADE_OPTIONS[protein]
    default_grade_idx = grades.index("choice") if "choice" in grades else 0
    grade = st.sidebar.selectbox(
        "Grade / Fat Level",
        options=grades,
        index=default_grade_idx,
        format_func=str.capitalize,
    )

    # Default thickness: midpoint of typical range
    t_lo, t_hi = cut_info["typical_thickness"]
    default_thickness = round((t_lo + t_hi) / 2.0, 1)
    thickness_inches = st.sidebar.number_input(
        "Thickness (inches)",
        min_value=0.5,
        max_value=15.0,
        value=default_thickness,
        step=0.25,
        help="Measure at thickest point, as it will sit on the grill (after any prep).",
    )

    bone_in = st.sidebar.checkbox("Bone-in", value=False)

    is_stuffed = False
    if cut_info.get("stuffable", False):
        is_stuffed = st.sidebar.checkbox("Stuffed", value=False)

    # --- Step 2: Smoker Setup ---
    st.sidebar.header("2. Smoker Setup")

    equipment = st.sidebar.selectbox(
        "Smoker Type",
        options=list(EQUIPMENT_PROFILES.keys()),
        format_func=lambda k: EQUIPMENT_PROFILES[k]["name"],
    )

    smoker_temp_f = st.sidebar.number_input(
        "Smoker Temperature (°F)",
        min_value=150,
        max_value=400,
        value=225,
        step=5,
    )

    default_target = cut_info["default_target_temp"]
    target_temp_f = st.sidebar.number_input(
        "Target Internal Temp (°F)",
        min_value=130,
        max_value=215,
        value=default_target,
        step=1,
    )

    initial_temp_f = st.sidebar.number_input(
        "Starting Meat Temp (°F)",
        min_value=32,
        max_value=80,
        value=40,
        step=1,
        help="Typical refrigerator temp is 38–42°F.",
    )

    # --- Step 3: Wrap Plan ---
    st.sidebar.header("3. Wrap Plan")

    wrap_options = list(WRAP_EVAP_REDUCTION.keys())
    wrap_labels = {
        "none": "No Wrap",
        "foil_boat": "Foil Boat",
        "butcher_paper": "Butcher Paper",
        "aluminum_foil": "Aluminum Foil",
    }
    wrap_type = st.sidebar.selectbox(
        "Wrap Type",
        options=wrap_options,
        format_func=lambda k: wrap_labels.get(k, k),
    )

    wrap_temp_f = None
    if wrap_type != "none":
        wrap_temp_f = st.sidebar.number_input(
            "Wrap at Internal Temp (°F)",
            min_value=140,
            max_value=185,
            value=165,
            step=5,
            help="Wrap when meat hits this internal temperature.",
        )

    # --- Step 4: Weather ---
    st.sidebar.header("4. Weather Conditions")
    st.sidebar.info("Check your local weather for current conditions.")

    ambient_temp_f = st.sidebar.number_input(
        "Outside Air Temp (°F)",
        min_value=-20,
        max_value=120,
        value=75,
        step=1,
    )

    humidity_pct = st.sidebar.slider(
        "Humidity (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
    )

    wind_mph = st.sidebar.number_input(
        "Wind Speed (mph)",
        min_value=0,
        max_value=60,
        value=5,
        step=1,
    )

    elevation_ft = st.sidebar.number_input(
        "Elevation (feet above sea level)",
        min_value=0,
        max_value=14000,
        value=0,
        step=100,
        help="Affects boiling point ceiling on surface temperature.",
    )

    # --- Step 5: Dinner Planning (optional) ---
    st.sidebar.header("5. Dinner Planning (Optional)")

    plan_dinner = st.sidebar.checkbox("Plan for a specific dinner time", value=False)
    dinner_time = None
    if plan_dinner:
        dinner_date = st.sidebar.date_input("Dinner date", value=datetime.today())
        dinner_clock = st.sidebar.time_input("Dinner time", value=time(18, 0))
        dinner_time = datetime.combine(dinner_date, dinner_clock)

    rest_method_labels = {
        "none": "Open Air (no wrap)",
        "oven": "Low-Temp Oven Hold",
        "cooler": "Insulated Cooler / Cambro",
    }
    rest_method = st.sidebar.selectbox(
        "Rest Method",
        options=list(rest_method_labels.keys()),
        index=2,
        format_func=lambda k: rest_method_labels[k],
    )

    rest_duration_min = st.sidebar.number_input(
        "Rest Duration (minutes)",
        min_value=0,
        max_value=240,
        value=60,
        step=15,
    )

    # --- Run button ---
    run = st.sidebar.button("Run Prediction", type="primary", use_container_width=True)

    if not run:
        return None

    return {
        "protein": protein,
        "grade": grade,
        "thickness_inches": float(thickness_inches),
        "smoker_temp_f": float(smoker_temp_f),
        "initial_temp_f": float(initial_temp_f),
        "target_temp_f": float(target_temp_f),
        "equipment": equipment,
        "wrap_type": wrap_type,
        "wrap_temp_f": float(wrap_temp_f) if wrap_temp_f else None,
        "elevation_ft": float(elevation_ft),
        "is_stuffed": is_stuffed,
        "humidity_pct": float(humidity_pct),
        "wind_mph": float(wind_mph),
        "ambient_temp_f": float(ambient_temp_f),
        "bone_in": bone_in,
        "dinner_time": dinner_time,
        "rest_method": rest_method,
        "rest_duration_min": int(rest_duration_min),
    }
