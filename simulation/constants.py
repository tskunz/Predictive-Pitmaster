"""Physics constants, protein library, and equipment profiles.

All constants must be traced to source. Values from Singh & Heldman unless noted.
Do NOT hardcode these values in simulation modules — import from here.
"""

# Thermal diffusivity (m²/s) by protein and grade
# Source: Singh & Heldman, Introduction to Food Engineering, Table 4.2
# Higher fat content → lower diffusivity → slower heat transfer
THERMAL_DIFFUSIVITY = {
    "beef": {
        "select":  1.33e-7,   # m²/s — lean
        "choice":  1.28e-7,   # m²/s — moderate fat (Choice brisket default)
        "prime":   1.23e-7,   # m²/s — high fat marbling
        "wagyu":   1.18e-7,   # m²/s — very high fat (extrapolated from prime trend)
    },
    "pork": {
        "select":  1.38e-7,   # m²/s
        "choice":  1.30e-7,   # m²/s
        "prime":   1.22e-7,   # m²/s
    },
    "poultry": {
        "select":  1.47e-7,   # m²/s — lean breast meat
        "choice":  1.40e-7,   # m²/s — mixed dark/white
        "prime":   1.35e-7,   # m²/s — heavier, higher fat birds
    },
    "lamb": {
        "select":  1.30e-7,   # m²/s
        "choice":  1.25e-7,   # m²/s
        "prime":   1.20e-7,   # m²/s
    },
}

# Biological noise: ±10% uniform distribution around base alpha per MC iteration
BIOLOGICAL_NOISE_FRACTION = 0.10  # dimensionless

# Stuffed modifier: reduce effective alpha by 20% when stuffed
# Stuffing has lower thermal conductivity and higher air content than muscle
STUFFED_ALPHA_REDUCTION = 0.20  # dimensionless

# Equipment temperature variance profiles (standard deviation in °F)
# Used to sample smoker temp noise in Monte Carlo
EQUIPMENT_PROFILES = {
    "offset":    {"name": "Offset Stick-Burner",              "temp_std_f": 25.0},  # °F std dev
    "pellet":    {"name": "Pellet Grill (Traeger, RecTeq…)",  "temp_std_f": 8.0},   # °F std dev
    "kamado":    {"name": "Kamado (BGE, Kamado Joe…)",        "temp_std_f": 10.0},  # °F std dev
    "wsm":       {"name": "Weber Smokey Mountain",            "temp_std_f": 15.0},  # °F std dev
    "kettle":    {"name": "Weber Kettle / Charcoal",          "temp_std_f": 20.0},  # °F std dev
    "electric":  {"name": "Electric Smoker (Masterbuilt…)",   "temp_std_f": 5.0},   # °F std dev
}

# Wrap evaporative cooling reduction factors (dimensionless, 0–1)
# 1.0 = fully eliminates evaporative cooling; 0.0 = no effect
WRAP_EVAP_REDUCTION = {
    "none":           0.00,  # dimensionless
    "foil_boat":      0.45,  # dimensionless — partial bottom coverage
    "butcher_paper":  0.60,  # dimensionless — breathable, moderate reduction
    "aluminum_foil":  0.95,  # dimensionless — fully sealed, nearly eliminates evaporation
}

# Rest phase cooling rate constants (1/min) for Newton's Law of Cooling
# Higher k → faster temperature drop
COOLING_RATE = {
    "none":   0.025,  # 1/min — open air on counter
    "oven":   0.005,  # 1/min — low-temp oven hold (~150°F)
    "cooler": 0.003,  # 1/min — insulated cooler / Cambro
}

# Food safety floor for rest phase
REST_SAFETY_FLOOR_F = 145.0  # °F — USDA minimum safe serving temperature

# Stall temperature gate — stall only physically possible in this range
STALL_TEMP_LOW_F  = 140.0  # °F
STALL_TEMP_HIGH_F = 185.0  # °F

# Biot number: ratio of surface convection to internal conduction
# Small Bi (~0.3) means surface heats slowly relative to internal conduction
# Physically correct for BBQ (hot air → meat surface at low velocity)
BIOT_NUMBER = 0.3  # dimensionless

# Spatial discretization for finite difference solver
N_NODES = 50  # dimensionless — number of spatial nodes (surface to center)

# Protein & cut library
# Each cut: default target temp, typical thickness range, typical weight range, stall_prone flag
PROTEIN_CUTS = {
    "beef": {
        "brisket_packer": {
            "name": "Brisket (Packer)",
            "default_target_temp": 203,    # °F
            "typical_thickness": (4.0, 7.0),  # inches
            "typical_weight": (10, 18),     # lbs
            "stall_prone": True,
            "stuffable": False,
        },
        "brisket_flat": {
            "name": "Brisket (Flat Only)",
            "default_target_temp": 200,    # °F
            "typical_thickness": (2.5, 4.5),
            "typical_weight": (5, 10),
            "stall_prone": True,
            "stuffable": False,
        },
        "chuck_roast": {
            "name": "Chuck Roast",
            "default_target_temp": 200,    # °F
            "typical_thickness": (3.0, 5.0),
            "typical_weight": (3, 6),
            "stall_prone": True,
            "stuffable": False,
        },
        "beef_ribs": {
            "name": "Beef Short Ribs",
            "default_target_temp": 203,    # °F
            "typical_thickness": (2.0, 3.5),
            "typical_weight": (3, 6),
            "stall_prone": True,
            "stuffable": False,
        },
    },
    "pork": {
        "pork_butt": {
            "name": "Pork Butt (Boston Butt)",
            "default_target_temp": 203,    # °F
            "typical_thickness": (5.0, 8.0),
            "typical_weight": (6, 10),
            "stall_prone": True,
            "stuffable": False,
        },
        "pork_ribs_spare": {
            "name": "Spare Ribs",
            "default_target_temp": 203,    # °F
            "typical_thickness": (1.0, 2.0),
            "typical_weight": (3, 5),
            "stall_prone": False,
            "stuffable": False,
        },
        "pork_ribs_baby_back": {
            "name": "Baby Back Ribs",
            "default_target_temp": 197,    # °F
            "typical_thickness": (0.75, 1.5),
            "typical_weight": (1.5, 3),
            "stall_prone": False,
            "stuffable": False,
        },
        "pork_loin": {
            "name": "Pork Loin",
            "default_target_temp": 145,    # °F
            "typical_thickness": (3.0, 5.0),
            "typical_weight": (3, 6),
            "stall_prone": False,
            "stuffable": True,
        },
    },
    "poultry": {
        "whole_chicken": {
            "name": "Whole Chicken",
            "default_target_temp": 165,    # °F
            "typical_thickness": (4.0, 6.0),
            "typical_weight": (3, 6),
            "stall_prone": False,
            "stuffable": True,
        },
        "turkey_breast": {
            "name": "Turkey Breast",
            "default_target_temp": 165,    # °F
            "typical_thickness": (3.0, 5.0),
            "typical_weight": (4, 8),
            "stall_prone": False,
            "stuffable": True,
        },
        "whole_turkey": {
            "name": "Whole Turkey",
            "default_target_temp": 165,    # °F
            "typical_thickness": (6.0, 10.0),
            "typical_weight": (10, 22),
            "stall_prone": False,
            "stuffable": True,
        },
    },
    "lamb": {
        "lamb_shoulder": {
            "name": "Lamb Shoulder",
            "default_target_temp": 203,    # °F
            "typical_thickness": (4.0, 6.0),
            "typical_weight": (4, 8),
            "stall_prone": True,
            "stuffable": False,
        },
        "lamb_leg": {
            "name": "Leg of Lamb",
            "default_target_temp": 145,    # °F
            "typical_thickness": (4.0, 7.0),
            "typical_weight": (5, 10),
            "stall_prone": False,
            "stuffable": False,
        },
    },
}

# Grade options per protein
GRADE_OPTIONS = {
    "beef":    ["select", "choice", "prime", "wagyu"],
    "pork":    ["select", "choice", "prime"],
    "poultry": ["select", "choice", "prime"],
    "lamb":    ["select", "choice", "prime"],
}
