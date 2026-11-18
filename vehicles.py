from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from constants import vehicle_constants
from models import StatParam


@dataclass
class Vehicle:
    """Represents a single vehicle and its longitudinal IDM dynamics."""
    vehicle_id: int
    veh_type: str
    lane_id: int          # 0 = Left Mainline, 1 = Right Mainline, 2 = Merging Ramp
    visual_lane: float    # Smoothly interpolates during lane changes for Pygame visuals
    position_m: float
    speed_mps: float
    length_m: float
    v0_mps: float
    T_s: float
    a_mps2: float
    b_mps2: float
    s0_m: float
    politeness: float
    driver_z_score: float
    acceleration_mps2: float = 0.0
    delta: float = vehicle_constants.IDM_DELTA
    forced_brake_timer_s: float = 0.0  # Used for interactive disturbance testing
    meter_released: bool = False       # FIX: got its green at the ramp meter
    reaction_time_s: float = 0.0       # human: sees the vehicle ahead this late (CAV: update interval)
    # (position, speed) after each step, newest last; lets followers see this
    # vehicle "as it was" one reaction time ago
    history: deque = field(default_factory=lambda: deque(maxlen=80), repr=False, compare=False)

    def compute_acceleration(self, lead_veh: Optional['Vehicle'], max_gap_m: Optional[float] = None) -> float:
        """Calculates longitudinal acceleration using the Intelligent Driver Model (IDM).

        FIX: when there is both a leader and a fixed obstacle (red meter, taper
        end), the more restrictive one wins (before, the obstacle was ignored
        whenever a leader existed). A forced disturbance can only make the driver
        brake harder, never ignore a leader it is about to hit.
        """
        acc = self._idm_single(None, None)
        if lead_veh is not None:
            acc = min(acc, self._idm_single(lead_veh, None))
        if max_gap_m is not None:
            acc = min(acc, self._idm_single(None, max_gap_m))
        if self.forced_brake_timer_s > 0.0:
            acc = min(acc, vehicle_constants.FORCED_BRAKE_ACCELERATION_MPS2)
        return acc

    def _idm_single(self, lead_veh: Optional['Vehicle'], max_gap_m: Optional[float]) -> float:
        """IDM against one leader, one stationary obstacle, or a free road."""
        """ acceleration = a · [ 1 − (v / v0)^δ − (s* / s)² ]

            s* = s0 + max(0, v·T + v·Δv / (2·√(a·b)))   ← desired gap"""
        
        free_road_term = (self.speed_mps / self.v0_mps) ** self.delta

        if lead_veh is None and max_gap_m is None:
            return self.a_mps2 * (1.0 - free_road_term)

        if lead_veh is not None:
            s = lead_veh.position_m - self.position_m - lead_veh.length_m
            dv = self.speed_mps - lead_veh.speed_mps
        else:
            # Approaching a physical dead-end (e.g., end of merge taper or red ramp meter)
            s = max_gap_m
            dv = self.speed_mps

        s = max(s, 0.01)  # Guard against division by zero if gap is ever 0
        s_star = self.s0_m + max(
            0.0,
            self.speed_mps * self.T_s + (self.speed_mps * dv) / (2.0 * np.sqrt(self.a_mps2 * self.b_mps2))
        )
        #The max(0, …) keeps s* from dropping below s0 when the car ahead pulls away quickly (negative dv).
        interaction_term = (s_star / s) ** 2
        acc = self.a_mps2 * (1.0 - free_road_term - interaction_term)
        return float(
            np.clip(acc, vehicle_constants.MIN_IDM_ACCELERATION_MPS2, self.a_mps2)
        )


@dataclass
class CAVVehicle(Vehicle):
    """Automated vehicle with our CAV controller (see the CAV block in
    constants/vehicle_constants.py for the equations and parameter sources).

    compute_acceleration() returns the controller's command for the current
    situation (steps 1-4). The simulator calls it every CAV_UPDATE_INTERVAL_S,
    applies the comfort jerk limit (step 5) and holds the command in between.
    Lane-change and merge checks call it too, so it has no side effects.
    """
    acc_command_mps2: float = 0.0       # last command (held between updates)
    next_acc_update_s: float = 0.0      # sim time of the next controller update

    def compute_acceleration(self, lead_veh: Optional['Vehicle'], max_gap_m: Optional[float] = None) -> float:
        acc = self._control(None, 0.0, 0.0)                  # free road: cruise
        if lead_veh is not None:
            gap = lead_veh.position_m - self.position_m - lead_veh.length_m
            # radar measures the leader's speed and (from its change) its braking
            acc = min(acc, self._control(gap, lead_veh.speed_mps,
                                         getattr(lead_veh, "acceleration_mps2", 0.0)))
        if max_gap_m is not None:                             # red meter / taper end = stopped "leader"
            acc = min(acc, self._control(max_gap_m, 0.0, 0.0))
        if self.forced_brake_timer_s > 0.0:                   # [B] disturbance applies to CAVs too
            acc = min(acc, vehicle_constants.FORCED_BRAKE_ACCELERATION_MPS2)
        return acc

    def rss_safe_gap(self, lead_speed_mps: float) -> float:
        """Step 4: RSS minimum safe gap - no collision even if the vehicle ahead
        brakes as hard as possible right now."""
        v, rho = self.speed_mps, vehicle_constants.RSS_RESPONSE_TIME_S
        a_acc = vehicle_constants.RSS_ACCEL_DURING_RESPONSE_MPS2
        d = (v * rho + 0.5 * a_acc * rho ** 2
             + (v + rho * a_acc) ** 2 / (2.0 * vehicle_constants.RSS_CAV_BRAKE_MPS2)
             - lead_speed_mps ** 2 / (2.0 * vehicle_constants.RSS_LEAD_MAX_BRAKE_MPS2))
        return max(d, 0.0) + self.s0_m

    def _control(self, gap_m: Optional[float], lead_speed_mps: float, lead_accel_mps2: float) -> float:
        v = self.speed_mps
        # Step 3: normal driving = the smaller of cruise and follow, within comfort limits
        acc = vehicle_constants.CAV_K_CRUISE * (self.v0_mps - v)
        if gap_m is not None and gap_m <= vehicle_constants.CAV_SENSOR_RANGE_M:
            desired_gap = self.s0_m + vehicle_constants.CAV_TIME_GAP_S * v          # step 2
            follow = (vehicle_constants.CAV_K_GAP * (gap_m - desired_gap)
                      + vehicle_constants.CAV_K_SPEED * (lead_speed_mps - v))
            acc = min(acc, follow)
        acc = float(np.clip(acc, -vehicle_constants.CAV_COMFORT_DECEL_MPS2,
                            vehicle_constants.CAV_COMFORT_ACCEL_MPS2))

        # Step 4: safety check (overrides comfort)
        if gap_m is not None and gap_m < self.rss_safe_gap(lead_speed_mps):
            closing = v - lead_speed_mps
            if closing > 0.0:
                # closing in: brake as hard as the leader brakes, plus enough to
                # cancel the closing speed before the gap drops below 1 m
                needed = (max(0.0, -lead_accel_mps2)
                          + closing ** 2 / (2.0 * max(gap_m - 1.0, 0.1)))
                brake = min(vehicle_constants.CAV_MAX_BRAKE_MPS2,
                            max(vehicle_constants.CAV_COMFORT_DECEL_MPS2, needed))
                acc = min(acc, -brake)
            else:
                # gap is opening: do not speed up until the safe gap is restored,
                # and brake at least as hard as the leader does
                acc = min(acc, 0.0, lead_accel_mps2)
        return acc


class DriverVehicleFactory:
    """Generates heterogeneous vehicles using empirical distributions and Z-scores."""
    def __init__(self, seed: int = 42, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.next_id = 1
        self.distributions = {
            veh_type: {
                name: StatParam(*parameters)
                for name, parameters in distributions.items()
            }
            for veh_type, distributions
            in vehicle_constants.VEHICLE_DISTRIBUTION_PARAMETERS.items()
        }
        self.veh_types = vehicle_constants.VEHICLE_TYPES
        self.veh_probs = vehicle_constants.VEHICLE_TYPE_PROBABILITIES
        self.ramp_veh_probs = vehicle_constants.RAMP_VEHICLE_TYPE_PROBABILITIES

    def create_vehicle(self, lane_id: int, position_m: float, initial_speed_mps: float,
                       rng: Optional[np.random.Generator] = None, is_cav: bool = False) -> Vehicle:
        # FIX: the simulator passes one random stream per lane, so a vehicle's
        # parameters do not depend on what happened in the other lanes.
        rng = rng if rng is not None else self.rng
        probs = self.ramp_veh_probs if lane_id == 2 else self.veh_probs
        v_type = str(rng.choice(self.veh_types, p=probs))
        dist = self.distributions[v_type]
        driver_z = float(
            np.clip(
                rng.standard_normal(),
                -vehicle_constants.DRIVER_Z_SCORE_LIMIT,
                vehicle_constants.DRIVER_Z_SCORE_LIMIT,
            )
        )

        params = dict(
            reaction_time_s=dist["reaction_time_s"].sample(rng, z_override=-driver_z),
            length_m=dist["length_m"].sample(rng),
            v0_mps=dist["v0_mps"].sample(rng, z_override=driver_z),
            T_s=dist["T_s"].sample(rng, z_override=-driver_z),
            a_mps2=dist["a_mps2"].sample(rng, z_override=driver_z),
            b_mps2=dist["b_mps2"].sample(rng),
            s0_m=dist["s0_m"].sample(rng, z_override=-0.5 * driver_z),
            politeness=dist["politeness"].sample(rng, z_override=-driver_z),
        )
        # The same random numbers are drawn either way, so the human drivers in
        # the stream are identical whatever the CAV penetration is.
        # Only passenger cars can be automated (trucks, semis, buses stay human).
        is_cav = is_cav and v_type == "car"
        if is_cav:
            # Automated driving is not randomized: fixed controller settings
            # (vehicle length is physical, so it stays sampled).
            params.update(
                v0_mps=vehicle_constants.CAV_DESIRED_SPEED_MPS,
                T_s=vehicle_constants.CAV_TIME_GAP_S,
                a_mps2=vehicle_constants.CAV_COMFORT_ACCEL_MPS2,
                b_mps2=vehicle_constants.CAV_COMFORT_DECEL_MPS2,
                s0_m=vehicle_constants.CAV_STANDSTILL_GAP_M,
                politeness=dist["politeness"].mean,
                reaction_time_s=vehicle_constants.CAV_UPDATE_INTERVAL_S,
            )
            driver_z = 0.0
            v_type = "cav"

        veh = (CAVVehicle if is_cav else Vehicle)(
            vehicle_id=self.next_id,
            veh_type=v_type,
            lane_id=lane_id,
            visual_lane=float(lane_id),
            position_m=position_m,
            speed_mps=initial_speed_mps,
            driver_z_score=driver_z,
            **params,
        )
        self.next_id += 1
        return veh
