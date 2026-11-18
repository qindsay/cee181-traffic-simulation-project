TIME_STEP_S = 0.1
RANDOM_SEED = 42

HIGHWAY_LENGTH_M = 1000.0
RAMP_START_M = 225.0                       # 225 m of mainline before the ramp starts
MERGE_ZONE_START_M = RAMP_START_M + 200.0  # ramp is 200 m long -> merge zone starts at 425 m
MERGE_ZONE_END_M = MERGE_ZONE_START_M + 400.0  # 400 m merge zone -> ends at 825 m
METER_STOP_LINE_M = MERGE_ZONE_START_M
# [B] braking disturbance: a lane-1 vehicle in this stretch (end of the merge zone) brakes hard
DISTURBANCE_ZONE_M = (MERGE_ZONE_END_M - 200.0, MERGE_ZONE_END_M + 75.0)

# veh/h per lane (0 = left, 1 = right, 2 = on-ramp). Change these to test congestion or free flow.
# Ramp at 450: at 600 the ramp backed up, because lane 1 + ramp (~2,050 veh/h) is more than one
# lane carries in this model and nobody moves left to make room for merging cars.
DEMAND_VPH = {0: 1450.0, 1: 1450.0, 2: 450.0}
RAMP_METER_RATE_VPH = 450.0
CAV_PENETRATION = 0.2   # window only ([C] key): share of arriving passenger cars that are automated (0.0 - 1.0)
LEAD_DISTURBANCE_DURATION_S = 3.5
DETECTOR_SAMPLING_INTERVAL_S = 10.0
SWITCH_THRESHOLD = 0.1

DETECTORS = (
    ("DET_UP_L0", 0, MERGE_ZONE_START_M - 100.0),
    ("DET_UP_L1", 1, MERGE_ZONE_START_M - 100.0),
    ("DET_MERGE_L0", 0, MERGE_ZONE_END_M + 20.0),
    ("DET_MERGE_L1", 1, MERGE_ZONE_END_M + 20.0),
    ("DET_DOWN_L0", 0, HIGHWAY_LENGTH_M - 50.0),
    ("DET_DOWN_L1", 1, HIGHWAY_LENGTH_M - 50.0),
)

MAINLINE_INITIAL_SPEED_MPS = 27.0
RAMP_INITIAL_SPEED_MPS = 20.0
