from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class StatParam:
    """Defines a parameter distribution with mean, std, and hard physical bounds."""
    mean: float
    std: float
    min_val: float
    max_val: float

    def sample(self, rng: np.random.Generator, z_override: Optional[float] = None) -> float:
        z = z_override if z_override is not None else rng.standard_normal()
        val = self.mean + z * self.std
        return float(np.clip(val, self.min_val, self.max_val))

    @staticmethod
    def z_to_percentile(z: float) -> float:
        """Approximates normal CDF percentile (0-100) from Z-score."""
        return float(0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3))) * 100.0)


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
    delta: float = 4.0
    forced_brake_timer_s: float = 0.0  # Used for interactive disturbance testing

    def compute_idm_acceleration(self, lead_veh: Optional['Vehicle'], max_gap_m: Optional[float] = None) -> float:
        """Calculates longitudinal acceleration using the Intelligent Driver Model (IDM)."""
        if self.forced_brake_timer_s > 0.0:
            return -4.5  # Controlled lead-vehicle braking disturbance

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
        return float(np.clip(acc, -9.0, self.a_mps2))


class DriverVehicleFactory:
    """Generates heterogeneous vehicles using empirical distributions and Z-scores."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.next_id = 1
        self.veh_types = ["car", "truck"]
        self.veh_probs = [0.85, 0.15]

        # Plug your empirical averages and standard deviations here (SI units: m, s)
        self.distributions = {
            "car": {
                "length_m": StatParam(4.5, 0.3, 3.8, 5.5),
                "v0_mps": StatParam(30.0, 3.0, 22.0, 38.0),
                "T_s": StatParam(1.2, 0.25, 0.6, 2.2),
                "a_mps2": StatParam(1.6, 0.3, 0.8, 2.6),
                "b_mps2": StatParam(2.0, 0.4, 1.2, 3.5),
                "s0_m": StatParam(2.0, 0.3, 1.0, 3.5),
                "politeness": StatParam(0.25, 0.1, 0.0, 0.6),
            },
            "truck": {
                "length_m": StatParam(12.5, 1.5, 9.5, 16.5),
                "v0_mps": StatParam(24.5, 1.8, 20.0, 28.0),
                "T_s": StatParam(1.7, 0.3, 1.1, 2.8),
                "a_mps2": StatParam(0.9, 0.15, 0.5, 1.4),
                "b_mps2": StatParam(1.5, 0.25, 1.0, 2.5),
                "s0_m": StatParam(3.0, 0.5, 2.0, 5.0),
                "politeness": StatParam(0.35, 0.1, 0.1, 0.7),
            }
        }

    def create_vehicle(self, lane_id: int, position_m: float, initial_speed_mps: float) -> Vehicle:
        v_type = str(self.rng.choice(self.veh_types, p=self.veh_probs))
        dist = self.distributions[v_type]
        driver_z = float(np.clip(self.rng.standard_normal(), -2.5, 2.5))

        veh = Vehicle(
            vehicle_id=self.next_id,
            veh_type=v_type,
            lane_id=lane_id,
            visual_lane=float(lane_id),
            position_m=position_m,
            speed_mps=initial_speed_mps,
            length_m=dist["length_m"].sample(self.rng),
            v0_mps=dist["v0_mps"].sample(self.rng, z_override=driver_z),
            T_s=dist["T_s"].sample(self.rng, z_override=-driver_z),
            a_mps2=dist["a_mps2"].sample(self.rng, z_override=driver_z),
            b_mps2=dist["b_mps2"].sample(self.rng),
            s0_m=dist["s0_m"].sample(self.rng, z_override=-0.5 * driver_z),
            politeness=dist["politeness"].sample(self.rng, z_override=-driver_z),
            driver_z_score=driver_z
        )
        self.next_id += 1
        return veh