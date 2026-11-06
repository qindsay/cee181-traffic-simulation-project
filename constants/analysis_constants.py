M_PER_MI = 1609.344
MPS_TO_MPH = 3600.0 / M_PER_MI

# --- settings ---------------------------------------------------------------
SAMPLE_INTERVAL_S = 1.0      # trajectory sampling for export / Edie cells
CELL_LENGTH_M = 100.0        # Edie cell length
CELL_DURATION_S = 30.0       # Edie cell duration
WARMUP_S = 120.0             # ignored in statistics (road starts empty)
MAINLINE_LANES = (0, 1)      # lane 2 is the on-ramp
# (station position m, lanes) where crossings are recorded like a spot-speed study
STATIONS = {"upstream": (220.0, (0, 1)), "downstream": (850.0, (0, 1)), "ramp": (200.0, (2,))}