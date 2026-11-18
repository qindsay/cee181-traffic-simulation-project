M_PER_MI = 1609.344
MPS_TO_MPH = 3600.0 / M_PER_MI

from constants import simulation_constants as _sim

# --- settings ---------------------------------------------------------------
SAMPLE_INTERVAL_S = 1.0      # trajectory sampling for export / Edie cells
CELL_LENGTH_M = 100.0        # Edie cell length
CELL_DURATION_S = 30.0       # Edie cell duration
WARMUP_S = 120.0             # ignored in statistics (road starts empty)
MAINLINE_LANES = (0, 1)      # lane 2 is the on-ramp
# (station position m, lanes) where crossings are recorded like a spot-speed study
# Placed relative to the road geometry so they move with it.
STATIONS = {
    "upstream": (_sim.MERGE_ZONE_START_M - 130.0, (0, 1)),   # before the merge
    "downstream": (_sim.MERGE_ZONE_END_M + 100.0, (0, 1)),   # after the merge
    "ramp": (_sim.RAMP_START_M + 50.0, (2,)),                # on the ramp
}
# Lane-1 stretch where congestion waves from the merge are measured (wave speed)
WAVE_WINDOW_M = (_sim.MERGE_ZONE_START_M - 150.0, _sim.MERGE_ZONE_START_M + 250.0)