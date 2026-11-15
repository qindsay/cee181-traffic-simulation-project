from dataclasses import dataclass
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
    """Automated vehicle: linear ACC (constant time-gap) law + IDM safety fallback.

        a_ACC = k1 · (gap − s0 − t_hw · v) + k2 · (v_lead − v)

    * Inputs are only the current gap to the vehicle ahead and its current
      speed (what an on-board radar would measure) — no future positions.
    * a_ACC is clipped to [ACC_MIN_ACCEL, ACC_MAX_ACCEL] (about −4 / +2 m/s²).
    * Safety fallback: if IDM is in its emergency regime (it wants to brake
      harder than its comfortable deceleration b) and that is stronger than
      the ACC command, the IDM braking is used instead — this may exceed −4.
    * The simulator holds the command for ACC_UPDATE_INTERVAL_S between updates.
    """ 
    k1: float = vehicle_constants.ACC_K1
    k2: float = vehicle_constants.ACC_K2
    t_hw_s: float = vehicle_constants.ACC_TIME_HEADWAY_S
    k_speed: float = vehicle_constants.ACC_K_SPEED
    acc_command_mps2: float = 0.0       # last controller output (zero-order hold)
    next_acc_update_s: float = 0.0      # sim time of the next controller update

    def _acc_law(self, lead_veh: Optional['Vehicle'], max_gap_m: Optional[float]) -> float:
        # Cruise mode: speed control toward the set speed v0 (the ACC law has no
        # free-road term, so a lone CAV would otherwise never accelerate).
        cruise_acc = self.k_speed * (self.v0_mps - self.speed_mps)
        if lead_veh is None and max_gap_m is None:
            acc = cruise_acc
        else:
            if lead_veh is not None:
                gap = lead_veh.position_m - self.position_m - lead_veh.length_m
                lead_speed = lead_veh.speed_mps
            else:
                gap = max_gap_m     # fixed obstacle (red meter / taper end) = stopped leader
                lead_speed = 0.0
            spacing_error = gap - self.s0_m - self.t_hw_s * self.speed_mps
            acc = min(cruise_acc, self.k1 * spacing_error + self.k2 * (lead_speed - self.speed_mps))
        return float(np.clip(acc, vehicle_constants.ACC_MIN_ACCEL_MPS2, vehicle_constants.ACC_MAX_ACCEL_MPS2))

    def _idm_single(self, lead_veh: Optional['Vehicle'], max_gap_m: Optional[float]) -> float:
        acc_cmd = self._acc_law(lead_veh, max_gap_m)
        if lead_veh is None and max_gap_m is None:
            return acc_cmd
        idm_acc = Vehicle._idm_single(self, lead_veh, max_gap_m)
        if idm_acc < -self.b_mps2 and idm_acc < acc_cmd:
            return idm_acc          # safety fallback: emergency IDM braking
        return acc_cmd


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

    def create_vehicle(self, lane_id: int, position_m: float, initial_speed_mps: float,
                       rng: Optional[np.random.Generator] = None, is_cav: bool = False) -> Vehicle:
        # FIX: the simulator passes one random stream per lane, so a vehicle's
        # parameters do not depend on what happened in the other lanes.
        rng = rng if rng is not None else self.rng
        v_type = str(rng.choice(self.veh_types, p=self.veh_probs))
        dist = self.distributions[v_type]
        driver_z = float(
            np.clip(
                rng.standard_normal(),
                -vehicle_constants.DRIVER_Z_SCORE_LIMIT,
                vehicle_constants.DRIVER_Z_SCORE_LIMIT,
            )
        )

        params = dict(
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
        if is_cav:
            # Automated driving is not randomized: use the mean parameters
            # (vehicle length is physical, so it stays sampled).
            for name in ("v0_mps", "T_s", "a_mps2", "b_mps2", "s0_m", "politeness"):
                params[name] = dist[name].mean
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
