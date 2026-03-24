"""Charts for the Predictive Pitmaster UI.

All charts use matplotlib (compatible with HuggingFace Spaces free tier).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from simulation.monte_carlo import SimResult


def render_phase_timeline(result: SimResult, inputs: dict) -> None:
    """Render a Gantt-style horizontal bar chart of cook phases.

    Phases: Smoke -> Stall -> Post-Stall/Wrap -> Rest
    Stall phase shown with uncertainty band.
    """
    p10 = result.p10_minutes
    p50 = result.p50_minutes
    p90 = result.p90_minutes

    # Estimate phase breakpoints (rough heuristics for visualization)
    rest_min = inputs.get("rest_duration_min", 60)

    smoke_end = p50 * 0.40
    stall_start = smoke_end
    stall_end_p50 = p50 * 0.70
    stall_width = (p90 - p10) * 0.5  # uncertainty band

    phases = [
        ("Smoke Phase",     0,             smoke_end,      "#8B4513"),
        ("Stall (est.)",    stall_start,   stall_end_p50,  "#DAA520"),
        ("Post-Stall",      stall_end_p50, p50,            "#CD853F"),
        ("Rest",            p50,           p50 + rest_min, "#6B8E6B"),
    ]

    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    for label, start, end, color in phases:
        ax.barh(0, end - start, left=start, height=0.5, color=color, edgecolor="#333", linewidth=0.5)

    # Uncertainty band on stall
    ax.barh(0, stall_width, left=stall_start - stall_width / 2, height=0.7,
            color="gold", alpha=0.2, edgecolor="gold", linewidth=1, linestyle="--")

    # P10/P50/P90 lines
    for val, label, ls in [(p10, "P10", "--"), (p50, "P50", "-"), (p90, "P90", "--")]:
        ax.axvline(val, color="white", linestyle=ls, linewidth=1.0, alpha=0.7)
        ax.text(val, 0.35, label, color="white", fontsize=7, ha="center", va="bottom")

    # Labels
    for label, start, end, color in phases:
        mid = (start + end) / 2.0
        ax.text(mid, 0, label, color="white", fontsize=8, ha="center", va="center", fontweight="bold")

    ax.set_xlim(0, p90 + rest_min + 30)
    ax.set_yticks([])
    ax.set_xlabel("Minutes from fire start", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # X-axis in hours
    max_x = p90 + rest_min + 30
    hour_ticks = list(range(0, int(max_x) + 60, 60))
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f"{h//60}h" for h in hour_ticks], color="white", fontsize=8)

    patches = [mpatches.Patch(color=c, label=l) for l, _, _, c in phases]
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              facecolor="#2a2a2a", edgecolor="#555", labelcolor="white")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_distribution(result: SimResult, inputs: dict) -> None:
    """Render histogram of Monte Carlo finish times with P10/P50/P90 lines."""
    times = result.all_finish_times
    if len(times) == 0:
        st.warning("No valid finish times to display.")
        return

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    ax.hist(times / 60.0, bins=60, color="#CD853F", edgecolor="#1a1a1a", alpha=0.8)

    for val, label, color in [
        (result.p10_minutes, "P10 (Optimistic)", "#4CAF50"),
        (result.p50_minutes, "P50 (Most Likely)", "#FFC107"),
        (result.p90_minutes, "P90 (Conservative)", "#F44336"),
    ]:
        ax.axvline(val / 60.0, color=color, linestyle="--", linewidth=1.5, label=label)

    ax.set_xlabel("Cook Time (hours from fire start)", color="white", fontsize=10)
    ax.set_ylabel("Simulations", color="white", fontsize=10)
    ax.set_title(f"Distribution of {len(times):,} Simulated Cook Times", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="white", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_sensitivity(result: SimResult) -> None:
    """Render horizontal bar chart showing which inputs drive the most uncertainty."""
    sens = result.sensitivity
    if not sens:
        return

    labels = list(sens.keys())
    values = [sens[k] * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    colors = ["#CD853F", "#DAA520", "#8B6914"]
    bars = ax.barh(labels, values, color=colors[:len(labels)], edgecolor="#333")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", color="white", va="center", fontsize=10)

    ax.set_xlabel("% of Total Uncertainty", color="white", fontsize=9)
    ax.set_xlim(0, 110)
    ax.tick_params(colors="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
