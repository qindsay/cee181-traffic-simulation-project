from typing import List, Dict, Optional, Tuple
import numpy as np
from constants import simulation_constants
from vehicles import DriverVehicleFactory, Vehicle


class VirtualDetector:
    """
    Matches the CEE 181/281 Observation Exchange Format:
    sensor_id, interval_start_s, interval_end_s, count, mean_speed_mps, quality_flag
    """
    def __init__(
        self,
        sensor_id: str,
        lane_id: int,
        position_m: float,
        sampling_interval_s: float = simulation_constants.DETECTOR_SAMPLING_INTERVAL_S,
    ):
        self.sensor_id = sensor_id
        self.lane_id = lane_id
        self.position_m = position_m
        self.sampling_interval_s = sampling_interval_s
        self.interval_start_s = 0.0
        self._crossed_speeds: List[float] = []
        self.last_flash_time_s = -10.0

    def check_crossing(self, prev_pos: float, curr_pos: float, speed_mps: float, sim_time_s: float):
        if prev_pos < self.position_m <= curr_pos:
            self._crossed_speeds.append(speed_mps)
            self.last_flash_time_s = sim_time_s

    def step_interval(self, current_time_s: float) -> Optional[Dict]:
        if current_time_s - self.interval_start_s >= self.sampling_interval_s - 1e-6:
            count = len(self._crossed_speeds)
            mean_speed = float(np.mean(self._crossed_speeds)) if count > 0 else np.nan
            record = {
                "sensor_id": self.sensor_id,
                "interval_start_s": round(self.interval_start_s, 2),
                "interval_end_s": round(current_time_s, 2),
                "count": count,
                "mean_speed_mps": round(mean_speed, 2) if not np.isnan(mean_speed) else None,
                "quality_flag": "OK"
            }
            self.interval_start_s = current_time_s
            self._crossed_speeds.clear()
            return record
        return None


class HighwaySimulator:
    """Manages the 2 mainline lanes + 1 merging ramp, queues, detectors, and step logic."""
    def __init__(self, dt_s: float = 0.1, seed: int = 42):
        self.dt_s = dt_s
        self.time_s = 0.0
        self.rng = np.random.default_rng(seed)
        self.factory = DriverVehicleFactory(seed=seed)

        # Highway & Ramp Geometry (meters)
        self.length_m = simulation_constants.HIGHWAY_LENGTH_M
        self.ramp_start_m = simulation_constants.RAMP_START_M
        self.merge_zone_start_m = simulation_constants.MERGE_ZONE_START_M
        self.merge_zone_end_m = simulation_constants.MERGE_ZONE_END_M
        self.meter_stop_line_m = simulation_constants.METER_STOP_LINE_M

        # Active vehicles per lane (0: Left, 1: Right, 2: Ramp)
        self.lanes: Dict[int, List[Vehicle]] = {0: [], 1: [], 2: []}
        # Unmet boundary entry queues
        self.entry_queues: Dict[int, List[Vehicle]] = {0: [], 1: [], 2: []}
        self.demand_vph = simulation_constants.DEMAND_VPH.copy()

        # Ramp Metering Control
        self.ramp_meter_enabled = False
        self.meter_rate_vph = simulation_constants.RAMP_METER_RATE_VPH
        self.last_release_time_s = -999.0
        self.meter_is_green = True

        self.detectors = [
            VirtualDetector(sensor_id, lane_id, position_m)
            for sensor_id, lane_id, position_m in simulation_constants.DETECTORS
        ]

        # Standard Exchange Logs
        self.vehicle_state_log: List[Dict] = []
        self.observation_log: List[Dict] = []
        self.total_exited_vehicles = 0

    def _sort_lanes(self):
        for lane_id in self.lanes:
            self.lanes[lane_id].sort(key=lambda v: v.position_m, reverse=True)

    def trigger_lead_disturbance(self) -> Optional[int]:
        """Forces a downstream vehicle in Lane 1 to brake temporarily."""
        candidates = [v for v in self.lanes[1] if 500.0 <= v.position_m <= 800.0]
        if not candidates and self.lanes[1]:
            candidates = [self.lanes[1][0]]
        if candidates:
            target = candidates[0]
            target.forced_brake_timer_s = simulation_constants.LEAD_DISTURBANCE_DURATION_S
            return target.vehicle_id
        return None

    def _spawn_and_discharge_queues(self):
        for lane_id, vph in self.demand_vph.items():
            prob_arrival = (vph / 3600.0) * self.dt_s
            if self.rng.random() < prob_arrival:
                spawn_pos = self.ramp_start_m if lane_id == 2 else 0.0
                init_speed = (
                    simulation_constants.RAMP_INITIAL_SPEED_MPS
                    if lane_id == 2
                    else simulation_constants.MAINLINE_INITIAL_SPEED_MPS
                )
                self.entry_queues[lane_id].append(
                    self.factory.create_vehicle(lane_id, spawn_pos, init_speed)
                )

            if self.entry_queues[lane_id]:
                cand_veh = self.entry_queues[lane_id][0]
                lane_vehs = self.lanes[lane_id]
                if not lane_vehs:
                    self.lanes[lane_id].append(self.entry_queues[lane_id].pop(0))
                else:
                    last_veh = lane_vehs[-1]
                    gap = last_veh.position_m - cand_veh.position_m - last_veh.length_m
                    min_entry_gap = cand_veh.s0_m + cand_veh.speed_mps * 0.75
                    if gap >= min_entry_gap:
                        cand_veh.speed_mps = min(cand_veh.v0_mps, last_veh.speed_mps)
                        self.lanes[lane_id].append(self.entry_queues[lane_id].pop(0))

    def _find_neighbors(self, target_lane_id: int, pos_m: float) -> Tuple[Optional[Vehicle], Optional[Vehicle]]:
        leader, follower = None, None
        for v in self.lanes[target_lane_id]:
            if v.position_m > pos_m:
                leader = v
            else:
                follower = v
                break
        return leader, follower

    def _process_merging_and_lane_changes(self):
        self._sort_lanes()

        # 1. Mandatory Merging: Ramp (Lane 2) -> Right Mainline (Lane 1)
        remaining_ramp = []
        for veh in self.lanes[2]:
            if self.merge_zone_start_m <= veh.position_m <= self.merge_zone_end_m:
                leader_1, follower_1 = self._find_neighbors(1, veh.position_m)
                gap_ahead = (leader_1.position_m - veh.position_m - leader_1.length_m) if leader_1 else 999.0
                gap_behind = (veh.position_m - follower_1.position_m - veh.length_m) if follower_1 else 999.0

                urgency = (veh.position_m - self.merge_zone_start_m) / (self.merge_zone_end_m - self.merge_zone_start_m)
                min_safe_decel = -3.8 - 3.2 * urgency
                new_follower_acc = follower_1.compute_idm_acceleration(veh) if follower_1 else 0.0

                if gap_ahead > veh.s0_m * 0.75 and gap_behind > (follower_1.s0_m * 0.75 if follower_1 else 2.0) and new_follower_acc >= min_safe_decel:
                    veh.lane_id = 1
                    self.lanes[1].append(veh)
                    self._sort_lanes()
                else:
                    remaining_ramp.append(veh)
            else:
                remaining_ramp.append(veh)
        self.lanes[2] = remaining_ramp

        # 2. Discretionary MOBIL Lane Change: Right Mainline (Lane 1) -> Left Mainline (Lane 0)
        remaining_l1 = []
        for idx, veh in enumerate(self.lanes[1]):
            curr_leader = self.lanes[1][idx - 1] if idx > 0 else None
            acc_curr = veh.compute_idm_acceleration(curr_leader)

            if acc_curr < -0.3 or veh.speed_mps < 0.85 * veh.v0_mps:
                leader_0, follower_0 = self._find_neighbors(0, veh.position_m)
                gap_ahead = (leader_0.position_m - veh.position_m - leader_0.length_m) if leader_0 else 999.0
                gap_behind = (veh.position_m - follower_0.position_m - veh.length_m) if follower_0 else 999.0

                acc_new = veh.compute_idm_acceleration(leader_0)
                f0_old_acc = follower_0.compute_idm_acceleration(leader_0) if follower_0 else 0.0
                f0_new_acc = follower_0.compute_idm_acceleration(veh) if follower_0 else 0.0

                incentive = (acc_new - acc_curr) + veh.politeness * (f0_new_acc - f0_old_acc)
                if gap_ahead > veh.s0_m and gap_behind > (follower_0.s0_m if follower_0 else 2.5) and f0_new_acc > -3.0 and incentive > 0.35:
                    veh.lane_id = 0
                    self.lanes[0].append(veh)
                    self._sort_lanes()
                    continue
            remaining_l1.append(veh)
        self.lanes[1] = remaining_l1

    def step(self, log_states: bool = False):
        """Advances simulation by dt_s. Set log_states=True when exporting trajectories."""
        self._spawn_and_discharge_queues()
        self._process_merging_and_lane_changes()
        self._sort_lanes()

        headway_s = 3600.0 / max(self.meter_rate_vph, 1.0)
        self.meter_is_green = not self.ramp_meter_enabled or ((self.time_s - self.last_release_time_s) >= headway_s)

        for lane_id, veh_list in self.lanes.items():
            for i, veh in enumerate(veh_list):
                if veh.forced_brake_timer_s > 0.0:
                    veh.forced_brake_timer_s = max(0.0, veh.forced_brake_timer_s - self.dt_s)

                leader = veh_list[i - 1] if i > 0 else None
                max_gap = None

                if lane_id == 2:
                    if self.ramp_meter_enabled and veh.position_m <= self.meter_stop_line_m:
                        is_first_at_meter = (leader is None or leader.position_m > self.meter_stop_line_m)
                        if self.meter_is_green and is_first_at_meter and (self.meter_stop_line_m - veh.position_m) < 6.0:
                            self.last_release_time_s = self.time_s
                            self.meter_is_green = False
                        elif not self.meter_is_green and is_first_at_meter:
                            max_gap = max(0.01, self.meter_stop_line_m - veh.position_m)

                    if max_gap is None:
                        dist_to_taper_end = self.merge_zone_end_m - veh.position_m
                        if leader is None or (leader.position_m - veh.position_m - leader.length_m) > dist_to_taper_end:
                            max_gap = max(0.01, dist_to_taper_end)

                veh.acceleration_mps2 = veh.compute_idm_acceleration(leader, max_gap_m=max_gap)

        for lane_id in [0, 1, 2]:
            surviving = []
            for veh in self.lanes[lane_id]:
                prev_pos = veh.position_m
                new_speed = max(0.0, veh.speed_mps + veh.acceleration_mps2 * self.dt_s)
                dx = max(0.0, 0.5 * (veh.speed_mps + new_speed) * self.dt_s)
                veh.position_m += dx
                veh.speed_mps = new_speed

                veh.visual_lane += (float(veh.lane_id) - veh.visual_lane) * min(1.0, self.dt_s * 6.0)

                for det in self.detectors:
                    if det.lane_id == lane_id:
                        det.check_crossing(prev_pos, veh.position_m, veh.speed_mps, self.time_s)

                if log_states:
                    self.vehicle_state_log.append({
                        "time_s": round(self.time_s, 2),
                        "vehicle_id": veh.vehicle_id,
                        "lane_id": veh.lane_id,
                        "position_m": round(veh.position_m, 2),
                        "speed_mps": round(veh.speed_mps, 2),
                        "acceleration_mps2": round(veh.acceleration_mps2, 2)
                    })

                if veh.position_m <= self.length_m:
                    surviving.append(veh)
                else:
                    self.total_exited_vehicles += 1
            self.lanes[lane_id] = surviving

        self.time_s += self.dt_s
        for det in self.detectors:
            obs = det.step_interval(self.time_s)
            if obs is not None:
                self.observation_log.append(obs)