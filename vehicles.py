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

    def compute_idm_acceleration(self, lead_veh: Optional['Vehicle'], max_gap_m: Optional[float] = None) -> float:
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

        s = max(s, 0.01)  # Guard against division by zero
        s_star = self.s0_m + max(
            0.0,
            self.speed_mps * self.T_s + (self.speed_mps * dv) / (2.0 * np.sqrt(self.a_mps2 * self.b_mps2))
        )
        interaction_term = (s_star / s) ** 2
        acc = self.a_mps2 * (1.0 - free_road_term - interaction_term)
        return float(
            np.clip(acc, vehicle_constants.MIN_IDM_ACCELERATION_MPS2, self.a_mps2)
        )


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
                       rng: Optional[np.random.Generator] = None) -> Vehicle:
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

        veh = Vehicle(
            vehicle_id=self.next_id,
            veh_type=v_type,
            lane_id=lane_id,
            visual_lane=float(lane_id),
            position_m=position_m,
            speed_mps=initial_speed_mps,
            length_m=dist["length_m"].sample(rng),
            v0_mps=dist["v0_mps"].sample(rng, z_override=driver_z),
            T_s=dist["T_s"].sample(rng, z_override=-driver_z),
            a_mps2=dist["a_mps2"].sample(rng, z_override=driver_z),
            b_mps2=dist["b_mps2"].sample(rng),
            s0_m=dist["s0_m"].sample(rng, z_override=-0.5 * driver_z),
            politeness=dist["politeness"].sample(rng, z_override=-driver_z),
            driver_z_score=driver_z
        )
        self.next_id += 1
        return veh