# truck = single-unit (box) truck, semi = tractor-trailer, bus = transit bus / coach.
# Heavy vehicles together are still 15% of traffic, as before.
VEHICLE_TYPES = ("car", "truck", "semi", "bus")
VEHICLE_TYPE_PROBABILITIES = (0.85, 0.05, 0.08, 0.02)
# On-ramp mix: fewer heavy vehicles (8% instead of 15%), mostly fewer semis, which
# need very long gaps to merge and otherwise back the ramp up.
RAMP_VEHICLE_TYPE_PROBABILITIES = (0.92, 0.04, 0.03, 0.01)
DRIVER_Z_SCORE_LIMIT = 2.5
IDM_DELTA = 4.0
FORCED_BRAKE_ACCELERATION_MPS2 = -4.5
MIN_IDM_ACCELERATION_MPS2 = -9.0

# ---------------------------------------------------------------------------
# Human reaction time: each human driver sees the vehicle ahead as it was
# reaction_time_s ago (drawn per driver, see VEHICLE_DISTRIBUTION_PARAMETERS).
# Emergency reflex: if a collision is imminent by the TRUE situation (gap below
# the jam gap, or time-to-collision below this), the driver reacts to the true
# situation right away. Safety net only, so nobody crashes.
HUMAN_REFLEX_TTC_S = 1.5
# (Reaction time averages 0.7 s, close to the 2/3 s of Gipps' car-following model.
#  With a 1.0 s average, stop-and-go waves turned into chains of emergency stops.)

# ---------------------------------------------------------------------------
# CAV controller (our design). Runs every CAV_UPDATE_INTERVAL_S and holds its
# command in between. Each update:
#   1. measure gap, own speed, leader speed (radar, range CAV_SENSOR_RANGE_M)
#   2. target gap      s_des = s0 + h*v
#   3. normal command  min( k_cruise*(v0 - v),  k_gap*(gap - s_des) + k_speed*(v_lead - v) )
#                      clipped to the comfort range [-COMFORT_DECEL, +COMFORT_ACCEL]
#   4. safety check    RSS minimum safe gap; if the gap is smaller, brake at least
#                      COMFORT_DECEL and as hard as needed (up to CAV_MAX_BRAKE)
#   5. comfort jerk limit on the change between updates (not in safety braking)
CAV_UPDATE_INTERVAL_S = 0.1        # 10 updates per second (= the CAV's reaction time)
CAV_SENSOR_RANGE_M = 150.0         # radar range; beyond it the CAV just cruises
CAV_TIME_GAP_S = 1.0               # h, target time gap (humans: 1.2 s on average)
CAV_STANDSTILL_GAP_M = 2.0         # s0, same as the average car
CAV_DESIRED_SPEED_MPS = 30.0       # v0, same as the average car driver (67 mph)
CAV_K_CRUISE = 0.4                 # 1/s   (Milanes & Shladover 2014 / SUMO speed control)
CAV_K_GAP = 0.25                   # 1/s²  gap-error gain
CAV_K_SPEED = 1.0                  # 1/s   speed-difference gain
# String stability (a slowdown cannot grow down a line of CAVs) needs
#   k_gap*h^2 + 2*k_speed*h >= 2   ->  0.25*1 + 2*1*1 = 2.25  (OK)
CAV_COMFORT_ACCEL_MPS2 = 1.5
CAV_COMFORT_DECEL_MPS2 = 3.0       # ISO 15622 (ACC standard) allows 3.5-5 depending on speed
CAV_COMFORT_JERK_MPS3 = 2.5        # max increase of acceleration per second (speeding up / easing off the brake)
CAV_BRAKE_JERK_MPS3 = 5.0          # max increase of braking per second in normal driving
CAV_MAX_BRAKE_MPS2 = 9.0           # physical emergency braking limit (same as any vehicle here)
# RSS safe distance (Shalev-Shwartz et al. 2017, Mobileye):
#   d_safe = v*rho + a_acc*rho^2/2 + (v + rho*a_acc)^2/(2*b_cav) - v_lead^2/(2*b_lead) + s0
RSS_RESPONSE_TIME_S = CAV_UPDATE_INTERVAL_S   # rho
RSS_ACCEL_DURING_RESPONSE_MPS2 = CAV_COMFORT_ACCEL_MPS2   # a_acc
RSS_CAV_BRAKE_MPS2 = 6.0           # b_cav, braking the CAV can always apply
RSS_LEAD_MAX_BRAKE_MPS2 = 9.0      # b_lead, worst-case braking of the vehicle ahead (sim maximum)

# Safety limits for lane changes and merges: nobody (the mover or the new
# follower) may be forced to brake harder than this. Merges near the taper end
# accept up to MERGE_SAFE_DECEL_END_MPS2 (urgency).
SAFE_DECEL_MPS2 = 4.0
MERGE_SAFE_DECEL_END_MPS2 = 6.0

# Lane changes / merges are decided on a fixed clock instead of every time step,
# so results do not change when the time step changes.
LANE_CHANGE_DECISION_INTERVAL_S = 1.0
MERGE_DECISION_INTERVAL_S = 0.2

VEHICLE_DISTRIBUTION_PARAMETERS = {
    "car": {
        "length_m": (4.5, 0.3, 3.8, 5.5),
        "v0_mps": (30.0, 3.0, 22.0, 38.0),
        "T_s": (1.2, 0.25, 0.6, 2.2),
        "a_mps2": (1.6, 0.3, 0.8, 2.6),
        "b_mps2": (2.0, 0.4, 1.2, 3.5),
        "s0_m": (2.0, 0.3, 1.0, 3.5),
        "politeness": (0.25, 0.1, 0.0, 0.6),
        "reaction_time_s": (0.7, 0.15, 0.4, 1.1),
    },
    "truck": {
        "length_m": (12.5, 1.5, 9.5, 16.5),
        "v0_mps": (24.5, 1.8, 20.0, 28.0),
        "T_s": (1.7, 0.3, 1.1, 2.8),
        "a_mps2": (0.9, 0.15, 0.5, 1.4),
        "b_mps2": (1.5, 0.25, 1.0, 2.5),
        "s0_m": (3.0, 0.5, 2.0, 5.0),
        "politeness": (0.35, 0.1, 0.1, 0.7),
        "reaction_time_s": (0.7, 0.12, 0.45, 1.0),
    },
    # Tractor-trailer: ~21 m long (WB-62/WB-67 design vehicles are 21-22.4 m),
    # often speed-governed / 55 mph truck limit in CA, slow to accelerate when loaded.
    "semi": {
        "length_m": (21.0, 1.0, 18.5, 23.0),
        "v0_mps": (25.0, 1.5, 21.0, 29.0),
        "T_s": (1.8, 0.3, 1.2, 2.8),
        "a_mps2": (0.6, 0.1, 0.4, 0.9),
        "b_mps2": (1.4, 0.2, 1.0, 2.0),
        "s0_m": (3.5, 0.5, 2.5, 5.0),
        "politeness": (0.4, 0.1, 0.15, 0.7),
        "reaction_time_s": (0.7, 0.12, 0.45, 1.0),
    },
    # Bus: 40-45 ft (12.2-13.7 m); gentle acceleration and braking because
    # passengers may be standing.
    "bus": {
        "length_m": (12.8, 0.6, 12.0, 14.0),
        "v0_mps": (27.0, 1.5, 23.0, 30.0),
        "T_s": (1.5, 0.25, 1.0, 2.2),
        "a_mps2": (1.0, 0.15, 0.6, 1.4),
        "b_mps2": (1.5, 0.2, 1.0, 2.0),
        "s0_m": (2.5, 0.4, 1.5, 4.0),
        "politeness": (0.4, 0.1, 0.15, 0.7),
        "reaction_time_s": (0.7, 0.12, 0.45, 1.0),
    },
}
