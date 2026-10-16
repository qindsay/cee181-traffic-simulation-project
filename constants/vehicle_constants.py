VEHICLE_TYPES = ("car", "truck")
VEHICLE_TYPE_PROBABILITIES = (0.85, 0.15)
DRIVER_Z_SCORE_LIMIT = 2.5
IDM_DELTA = 4.0
FORCED_BRAKE_ACCELERATION_MPS2 = -4.5
MIN_IDM_ACCELERATION_MPS2 = -9.0

VEHICLE_DISTRIBUTION_PARAMETERS = {
    "car": {
        "length_m": (4.5, 0.3, 3.8, 5.5),
        "v0_mps": (30.0, 3.0, 22.0, 38.0),
        "T_s": (1.2, 0.25, 0.6, 2.2),
        "a_mps2": (1.6, 0.3, 0.8, 2.6),
        "b_mps2": (2.0, 0.4, 1.2, 3.5),
        "s0_m": (2.0, 0.3, 1.0, 3.5),
        "politeness": (0.25, 0.1, 0.0, 0.6),
    },
    "truck": {
        "length_m": (12.5, 1.5, 9.5, 16.5),
        "v0_mps": (24.5, 1.8, 20.0, 28.0),
        "T_s": (1.7, 0.3, 1.1, 2.8),
        "a_mps2": (0.9, 0.15, 0.5, 1.4),
        "b_mps2": (1.5, 0.25, 1.0, 2.5),
        "s0_m": (3.0, 0.5, 2.0, 5.0),
        "politeness": (0.35, 0.1, 0.1, 0.7),
    },
}
