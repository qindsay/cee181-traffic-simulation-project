from typing import List, Dict, Optional, Tuple
import numpy as np
from constants import simulation_constants, vehicle_constants
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
        self.step_count = 0  # FIX: time = step_count * dt (no floating-point drift)
        # FIX: independent random streams (before, arrivals and vehicle parameters
        # both used default_rng(seed) and drew the SAME numbers). One stream per
        # lane for arrival times and one per lane for vehicle parameters.
        arrival_ss, vehicle_ss = np.random.SeedSequence(seed).spawn(2)
        self.arrival_rngs = {lane: np.random.default_rng(s) for lane, s in zip((0, 1, 2), arrival_ss.spawn(3))}
        self.vehicle_rngs = {lane: np.random.default_rng(s) for lane, s in zip((0, 1, 2), vehicle_ss.spawn(3))}
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
        # FIX: arrival times are drawn as exponential headways (Poisson arrivals)
        # in continuous time, so they are the same whatever the time step.
        self.next_arrival_s = {lane: self._next_headway_s(lane) for lane in self.demand_vph}

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

    def _next_headway_s(self, lane_id: int) -> float:
        vph = self.demand_vph.get(lane_id, 0.0)
        if vph <= 0.0:
            return float("inf")
        return float(self.arrival_rngs[lane_id].exponential(3600.0 / vph))

    def _on_clock(self, period_s: float) -> bool:
        """True when the current time is a multiple of period_s."""
        ratio = self.time_s / period_s
        return abs(ratio - round(ratio)) < 1e-6

    def _spawn_and_discharge_queues(self):
        for lane_id, vph in self.demand_vph.items():
            if self.next_arrival_s[lane_id] == float("inf") and vph > 0.0:
                self.next_arrival_s[lane_id] = self.time_s + self._next_headway_s(lane_id)
            while self.next_arrival_s[lane_id] <= self.time_s + 1e-9:
                spawn_pos = self.ramp_start_m if lane_id == 2 else 0.0
                init_speed = (
                    simulation_constants.RAMP_INITIAL_SPEED_MPS
                    if lane_id == 2
                    else simulation_constants.MAINLINE_INITIAL_SPEED_MPS
                )
                self.entry_queues[lane_id].append(
                    self.factory.create_vehicle(lane_id, spawn_pos, init_speed, rng=self.vehicle_rngs[lane_id])
                )
                self.next_arrival_s[lane_id] += self._next_headway_s(lane_id)

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

    def _process_merging_and_lane_changes(self) -> None:
        """Apply mandatory ramp merges, then discretionary lane changes.

        FIX: decisions run on a fixed clock (merges every 0.2 s, lane changes
        every 1 s) instead of every step, so a smaller dt no longer means more
        lane changes. Use a dt that divides 0.2 s (0.05, 0.1 or 0.2).
        """
        self._sort_lanes()
        if self._on_clock(vehicle_constants.MERGE_DECISION_INTERVAL_S):
            self._process_ramp_merges()
        if self._on_clock(vehicle_constants.LANE_CHANGE_DECISION_INTERVAL_S):
            self._process_discretionary_lane_changes()

    def _process_ramp_merges(self) -> None:
        """Move ramp vehicles into lane 1 when a safe merge gap is available."""
        remaining_ramp: List[Vehicle] = []
        for veh in self.lanes[2]:
            if self.merge_zone_start_m <= veh.position_m <= self.merge_zone_end_m:
                if self._can_merge_from_ramp(veh):
                    veh.lane_id = 1
                    self.lanes[1].append(veh)
                    self._sort_lanes()
                else:
                    remaining_ramp.append(veh)
            else:
                remaining_ramp.append(veh)
        self.lanes[2] = remaining_ramp

    def _can_merge_from_ramp(self, veh: Vehicle) -> bool:
        leader, follower = self._find_neighbors(1, veh.position_m)
        gap_ahead = (
            leader.position_m - veh.position_m - leader.length_m
            if leader
            else 999.0
        )
        gap_behind = (
            veh.position_m - follower.position_m - veh.length_m
            if follower
            else 999.0
        )

        merge_distance = self.merge_zone_end_m - self.merge_zone_start_m
        urgency = (veh.position_m - self.merge_zone_start_m) / merge_distance
        # Drivers accept harder braking the closer they are to the taper end.
        minimum_safe_deceleration = -(
            vehicle_constants.SAFE_DECEL_MPS2
            + (vehicle_constants.MERGE_SAFE_DECEL_END_MPS2 - vehicle_constants.SAFE_DECEL_MPS2) * urgency
        )
        follower_acceleration = (
            follower.compute_idm_acceleration(veh)
            if follower
            else 0.0
        )
        # FIX: the merging vehicle must also be able to follow its NEW leader.
        # Before, only the follower was checked, so fast ramp vehicles merged a
        # metre behind slow mainline cars and drove through them.
        own_acceleration = veh.compute_idm_acceleration(leader)
        minimum_rear_gap = follower.s0_m if follower else 2.0

        return (
            gap_ahead > veh.s0_m
            and gap_behind > minimum_rear_gap
            and follower_acceleration >= minimum_safe_deceleration
            and own_acceleration >= minimum_safe_deceleration
        )

    def _process_discretionary_lane_changes(self) -> None:
        """Apply MOBIL lane changes from lane 1 into lane 0."""
        remaining_lane: List[Vehicle] = []
        for index, veh in enumerate(self.lanes[1]):
            current_leader = self.lanes[1][index - 1] if index > 0 else None
            if self._should_change_to_left_lane(veh, current_leader):
                veh.lane_id = 0
                self.lanes[0].append(veh)
                self._sort_lanes()
            else:
                remaining_lane.append(veh)
        self.lanes[1] = remaining_lane

    def _should_change_to_left_lane(
        self,
        veh: Vehicle,
        current_leader: Optional[Vehicle],
    ) -> bool:
        current_acceleration = veh.compute_idm_acceleration(current_leader)
        if current_acceleration >= -0.3 and veh.speed_mps >= 0.85 * veh.v0_mps:
            return False

        target_leader, target_follower = self._find_neighbors(0, veh.position_m)
        gap_ahead = (
            target_leader.position_m - veh.position_m - target_leader.length_m
            if target_leader
            else 999.0
        )
        gap_behind = (
            veh.position_m - target_follower.position_m - veh.length_m
            if target_follower
            else 999.0
        )

        target_acceleration = veh.compute_idm_acceleration(target_leader)
        follower_old_acceleration = (
            target_follower.compute_idm_acceleration(target_leader)
            if target_follower
            else 0.0
        )
        follower_new_acceleration = (
            target_follower.compute_idm_acceleration(veh)
            if target_follower
            else 0.0
        )
        incentive = (
            target_acceleration
            - current_acceleration
            + veh.politeness
            * (follower_new_acceleration - follower_old_acceleration)
        )
        minimum_rear_gap = target_follower.s0_m if target_follower else 2.5

        return (
            gap_ahead > veh.s0_m
            and gap_behind > minimum_rear_gap
            and follower_new_acceleration > -3.0
            # FIX: the lane changer itself must not need to brake hard
            and target_acceleration > -vehicle_constants.SAFE_DECEL_MPS2
            and incentive > 0.35
        )

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
                    obstacles = [max(0.01, self.merge_zone_end_m - veh.position_m)]  # taper end
                    if (self.ramp_meter_enabled and veh.position_m <= self.meter_stop_line_m
                            and not veh.meter_released):
                        is_first_at_meter = (leader is None or leader.position_m > self.meter_stop_line_m)
                        dist_to_line = self.meter_stop_line_m - veh.position_m
                        if self.meter_is_green and is_first_at_meter and dist_to_line < 6.0:
                            # FIX: remember the release; before, the car was stopped
                            # again on the next (red) step and released a second time.
                            veh.meter_released = True
                            self.last_release_time_s = self.time_s
                            self.meter_is_green = False
                        elif not self.meter_is_green and is_first_at_meter:
                            if veh.speed_mps ** 2 / (2.0 * max(dist_to_line, 0.1)) > -vehicle_constants.MIN_IDM_ACCELERATION_MPS2:
                                # physically cannot stop before the line: goes through (like an amber light)
                                veh.meter_released = True
                                self.last_release_time_s = self.time_s
                            else:
                                obstacles.append(max(0.01, dist_to_line))
                    max_gap = min(obstacles)

                veh.acceleration_mps2 = veh.compute_idm_acceleration(leader, max_gap_m=max_gap)

                if lane_id == 2 and veh.position_m >= self.merge_zone_start_m:
                    # FIX: in the merge zone a ramp driver also adjusts to the
                    # mainline car just ahead (at most comfortable braking b), so
                    # it drifts back into a gap instead of driving alongside
                    # mainline traffic until it is stuck at the taper end.
                    main_leader, _ = self._find_neighbors(1, veh.position_m)
                    if main_leader is not None:
                        veh.acceleration_mps2 = min(
                            veh.acceleration_mps2,
                            max(veh.compute_idm_acceleration(main_leader), -veh.b_mps2),
                        )

        for lane_id in [0, 1, 2]:
            surviving = []
            for veh in self.lanes[lane_id]:
                prev_pos = veh.position_m
                # FIX: exact stop inside the step, and store the acceleration that
                # was actually APPLIED. Before, a car standing at the meter kept
                # reporting -9 m/s^2, which corrupts any braking statistic.
                v_old, acc = veh.speed_mps, veh.acceleration_mps2
                new_speed = v_old + acc * self.dt_s
                if new_speed >= 0.0:
                    dx = v_old * self.dt_s + 0.5 * acc * self.dt_s ** 2
                else:
                    dx = -v_old ** 2 / (2.0 * acc) if acc < 0.0 else 0.0
                    new_speed = 0.0
                veh.position_m += dx
                veh.speed_mps = new_speed
                veh.acceleration_mps2 = (new_speed - v_old) / self.dt_s

                veh.visual_lane += (float(veh.lane_id) - veh.visual_lane) * min(1.0, self.dt_s * 6.0)

                for det in self.detectors:
                    if det.lane_id == lane_id:
                        det.check_crossing(prev_pos, veh.position_m, veh.speed_mps, self.time_s)

                if log_states:
                    self.vehicle_state_log.append({
                        # FIX: the state is AFTER the step, so stamp it with that time
                        "time_s": round(self.time_s + self.dt_s, 2),
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

        self.step_count += 1
        self.time_s = self.step_count * self.dt_s
        for det in self.detectors:
            obs = det.step_interval(self.time_s)
            if obs is not None:
                self.observation_log.append(obs)