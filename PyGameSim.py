import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import deque
import numpy as np
import pygame

# =====================================================================
# 1. STATISTICAL & PROBABILITY CURVE ENGINE (Z-SCORE & PERCENTILES)
# =====================================================================

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


class DriverVehicleFactory:
    """Generates heterogeneous vehicles using empirical distributions and Z-scores."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.next_id = 1
        self.veh_types = ["car", "truck"]
        self.veh_probs = [0.85, 0.15]

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

    def create_vehicle(self, lane_id: int, position_m: float, initial_speed_mps: float) -> 'Vehicle':
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


# =====================================================================
# 2. CORE SIMULATION ENTITIES (VEHICLE, DETECTOR, SIMULATOR)
# =====================================================================

@dataclass
class Vehicle:
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
        if self.forced_brake_timer_s > 0.0:
            return -4.5  # Controlled braking disturbance test

        free_road_term = (self.speed_mps / self.v0_mps) ** self.delta

        if lead_veh is None and max_gap_m is None:
            return self.a_mps2 * (1.0 - free_road_term)

        if lead_veh is not None:
            s = lead_veh.position_m - self.position_m - lead_veh.length_m
            dv = self.speed_mps - lead_veh.speed_mps
        else:
            s = max_gap_m
            dv = self.speed_mps

        s = max(s, 0.01)
        s_star = self.s0_m + max(
            0.0,
            self.speed_mps * self.T_s + (self.speed_mps * dv) / (2.0 * np.sqrt(self.a_mps2 * self.b_mps2))
        )
        interaction_term = (s_star / s) ** 2
        acc = self.a_mps2 * (1.0 - free_road_term - interaction_term)
        return float(np.clip(acc, -9.0, self.a_mps2))


class VirtualDetector:
    """CEE 181/281 Observation Exchange Format Sensor."""
    def __init__(self, sensor_id: str, lane_id: int, position_m: float, sampling_interval_s: float = 10.0):
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
    def __init__(self, dt_s: float = 0.1, seed: int = 42):
        self.dt_s = dt_s
        self.time_s = 0.0
        self.rng = np.random.default_rng(seed)
        self.factory = DriverVehicleFactory(seed=seed)

        # Highway & Ramp Geometry (meters)
        self.length_m = 1000.0
        self.ramp_start_m = 150.0
        self.merge_zone_start_m = 350.0
        self.merge_zone_end_m = 620.0
        self.meter_stop_line_m = 310.0

        self.lanes: Dict[int, List[Vehicle]] = {0: [], 1: [], 2: []}
        self.entry_queues: Dict[int, List[Vehicle]] = {0: [], 1: [], 2: []}
        self.demand_vph = {0: 1450.0, 1: 1450.0, 2: 600.0}

        # Ramp Metering State
        self.ramp_meter_enabled = False
        self.meter_rate_vph = 450.0
        self.last_release_time_s = -999.0
        self.meter_is_green = True

        self.detectors = [
            VirtualDetector("DET_UP_L0", lane_id=0, position_m=220.0),
            VirtualDetector("DET_UP_L1", lane_id=1, position_m=220.0),
            VirtualDetector("DET_MERGE_L0", lane_id=0, position_m=550.0),
            VirtualDetector("DET_MERGE_L1", lane_id=1, position_m=550.0),
            VirtualDetector("DET_DOWN_L0", lane_id=0, position_m=850.0),
            VirtualDetector("DET_DOWN_L1", lane_id=1, position_m=850.0),
        ]

        self.observation_log: List[Dict] = []
        self.total_exited_vehicles = 0

    def _sort_lanes(self):
        for lane_id in self.lanes:
            self.lanes[lane_id].sort(key=lambda v: v.position_m, reverse=True)

    def trigger_lead_disturbance(self):
        """Forces a downstream vehicle in Lane 1 to brake for 3.5 seconds."""
        candidates = [v for v in self.lanes[1] if 500.0 <= v.position_m <= 800.0]
        if not candidates and self.lanes[1]:
            candidates = [self.lanes[1][0]]
        if candidates:
            target = candidates[0]
            target.forced_brake_timer_s = 3.5
            return target.vehicle_id
        return None

    def _spawn_and_discharge_queues(self):
        for lane_id, vph in self.demand_vph.items():
            prob_arrival = (vph / 3600.0) * self.dt_s
            if self.rng.random() < prob_arrival:
                spawn_pos = self.ramp_start_m if lane_id == 2 else 0.0
                init_speed = 20.0 if lane_id == 2 else 27.0
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

    def step(self):
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

                # Smooth visual lane interpolation for lane-changing animations
                veh.visual_lane += (float(veh.lane_id) - veh.visual_lane) * min(1.0, self.dt_s * 6.0)

                for det in self.detectors:
                    if det.lane_id == lane_id:
                        det.check_crossing(prev_pos, veh.position_m, veh.speed_mps, self.time_s)

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


# =====================================================================
# 3. PYGAME GRAPHICAL ENGINE & LIVE TELEMETRY DASHBOARD
# =====================================================================

def speed_to_color(speed_mps: float) -> Tuple[int, int, int]:
    """Maps vehicle speed (0 to 32 m/s) to a smooth Red -> Yellow -> Green gradient."""
    ratio = float(np.clip(speed_mps / 30.0, 0.0, 1.0))
    if ratio < 0.5:
        r = 235
        g = int(60 + (ratio * 2.0) * 180)
        b = 50
    else:
        r = int(235 - ((ratio - 0.5) * 2.0) * 195)
        g = 225
        b = 75
    return (r, g, b)


def run_pygame_sim():
    pygame.init()
    WIDTH, HEIGHT = 1400, 760
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CEE 181/281 Freeway Simulator - IDM + Z-Score Heterogeneity + TMC Metering")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Segoe UI", 20, bold=True)
    font_ui = pygame.font.SysFont("Consolas", 14)
    font_small = pygame.font.SysFont("Segoe UI", 12)

    sim = HighwaySimulator(dt_s=0.1, seed=42)

    # Coordinate Mapping: 1000m maps across 1300px
    MARGIN_X = 50
    ROAD_PX_WIDTH = WIDTH - 2 * MARGIN_X
    SCALE_X = ROAD_PX_WIDTH / sim.length_m  # 1.3 pixels per meter

    LANE_Y = {
        0: 200,   # Left Mainline
        1: 245,   # Right Mainline
        2: 305    # Merging Ramp
    }
    LANE_WIDTH_PX = 38

    paused = False
    sim_speed_mult = 1
    selected_veh: Optional[Vehicle] = None

    # Rolling telemetry buffers for live bottom chart
    MAX_PTS = 350
    hist_l0_speed = deque(maxlen=MAX_PTS)
    hist_l1_speed = deque(maxlen=MAX_PTS)
    hist_queue = deque(maxlen=MAX_PTS)

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_m:
                    sim.ramp_meter_enabled = not sim.ramp_meter_enabled
                elif event.key == pygame.K_b:
                    sim.trigger_lead_disturbance()
                elif event.key == pygame.K_UP:
                    sim_speed_mult = min(10, sim_speed_mult + 1)
                elif event.key == pygame.K_DOWN:
                    sim_speed_mult = max(1, sim_speed_mult - 1)
                elif event.key == pygame.K_r:
                    sim = HighwaySimulator(dt_s=0.1, seed=42)
                    hist_l0_speed.clear()
                    hist_l1_speed.clear()
                    hist_queue.clear()
                    selected_veh = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                selected_veh = None
                for lane_vehs in sim.lanes.values():
                    for v in lane_vehs:
                        vx = MARGIN_X + v.position_m * SCALE_X
                        vy = LANE_Y[0] + v.visual_lane * 45 if v.visual_lane <= 1.0 else LANE_Y[1] + (v.visual_lane - 1.0) * 60
                        if abs(mx - vx) <= max(12, v.length_m * SCALE_X) and abs(my - vy) <= 14:
                            selected_veh = v
                            break

        # 2. Physics Update
        if not paused:
            for _ in range(sim_speed_mult):
                sim.step()

            l0_s = float(np.mean([v.speed_mps for v in sim.lanes[0]])) if sim.lanes[0] else 30.0
            l1_s = float(np.mean([v.speed_mps for v in sim.lanes[1]])) if sim.lanes[1] else 30.0
            tot_q = sum(len(q) for q in sim.entry_queues.values())
            hist_l0_speed.append(l0_s)
            hist_l1_speed.append(l1_s)
            hist_queue.append(tot_q)

        # 3. Rendering
        screen.fill((24, 28, 34))  # Dark slate background

        # --- Top Header & Controls Bar ---
        pygame.draw.rect(screen, (35, 40, 50), (0, 0, WIDTH, 95))
        title_surf = font_title.render("CEE 181/281 Freeway & Merging Simulator (Live Pygame Engine)", True, (240, 245, 255))
        screen.blit(title_surf, (20, 12))

        ctrl_text = (
            f"[SPACE] {'Resume' if paused else 'Pause'}   |   "
            f"[M] Ramp Meter: {'ON (450 vph)' if sim.ramp_meter_enabled else 'OFF'}   |   "
            f"[B] Trigger Lead Braking Disturbance   |   "
            f"[UP/DOWN] Speed: {sim_speed_mult}x   |   [R] Reset   |   [Click Car] Inspect"
        )
        screen.blit(font_ui.render(ctrl_text, True, (180, 210, 255)), (20, 44))

        active_count = sum(len(v) for v in sim.lanes.values())
        stats_line = (
            f"Sim Time: {sim.time_s:6.1f}s   |   Active Vehs: {active_count:3d}   |   "
            f"Completed Trips: {sim.total_exited_vehicles:4d}   |   "
            f"Boundary Entry Queues -> L0: {len(sim.entry_queues[0])}  L1: {len(sim.entry_queues[1])}  Ramp: {len(sim.entry_queues[2])}"
        )
        screen.blit(font_ui.render(stats_line, True, (220, 220, 220)), (20, 68))

        # --- Draw Highway Asphalt & Geometry ---
        # Mainline Asphalt (Lanes 0 and 1)
        mainline_top = LANE_Y[0] - LANE_WIDTH_PX // 2
        pygame.draw.rect(screen, (52, 56, 64), (MARGIN_X, mainline_top, ROAD_PX_WIDTH, LANE_WIDTH_PX * 2 + 10))

        # Ramp Asphalt (Lane 2)
        ramp_x1 = MARGIN_X + int(sim.ramp_start_m * SCALE_X)
        ramp_x2 = MARGIN_X + int(sim.merge_zone_start_m * SCALE_X)
        ramp_x3 = MARGIN_X + int(sim.merge_zone_end_m * SCALE_X)
        ramp_y = LANE_Y[2]

        ramp_poly = [
            (ramp_x1, ramp_y - 16),
            (ramp_x2, mainline_top + LANE_WIDTH_PX * 2 + 8),
            (ramp_x3, mainline_top + LANE_WIDTH_PX * 2 + 8),
            (ramp_x3 + 25, mainline_top + LANE_WIDTH_PX * 2),
            (ramp_x2, ramp_y + 18),
            (ramp_x1, ramp_y + 18)
        ]
        pygame.draw.polygon(screen, (52, 56, 64), ramp_poly)

        # Merge Zone Highlight
        merge_w = ramp_x3 - ramp_x2
        merge_surf = pygame.Surface((merge_w, LANE_WIDTH_PX + 35), pygame.SRCALPHA)
        merge_surf.fill((255, 165, 0, 35))
        screen.blit(merge_surf, (ramp_x2, LANE_Y[1]))
        screen.blit(font_small.render("ACCELERATION & MERGE ZONE (350m - 620m)", True, (255, 190, 90)), (ramp_x2 + 10, LANE_Y[2] + 24))

        # Lane Markings
        pygame.draw.line(screen, (230, 230, 230), (MARGIN_X, mainline_top), (MARGIN_X + ROAD_PX_WIDTH, mainline_top), 3)
        # Dashed centerline between Lane 0 and Lane 1
        for x_dash in range(MARGIN_X, MARGIN_X + ROAD_PX_WIDTH, 36):
            pygame.draw.line(screen, (200, 200, 200), (x_dash, LANE_Y[0] + 22), (x_dash + 18, LANE_Y[0] + 22), 2)

        # Bottom solid line (breaks open at merge zone)
        bot_y = mainline_top + LANE_WIDTH_PX * 2 + 8
        pygame.draw.line(screen, (230, 230, 230), (MARGIN_X, bot_y), (ramp_x2, bot_y), 3)
        pygame.draw.line(screen, (230, 230, 230), (ramp_x3, bot_y), (MARGIN_X + ROAD_PX_WIDTH, bot_y), 3)
        for x_dash in range(ramp_x2, ramp_x3, 24):
            pygame.draw.line(screen, (255, 200, 80), (x_dash, bot_y), (x_dash + 12, bot_y), 2)

        # Distance Ruler Ticks
        for m in range(0, 1001, 100):
            tx = MARGIN_X + int(m * SCALE_X)
            pygame.draw.line(screen, (120, 130, 145), (tx, mainline_top - 8), (tx, mainline_top), 2)
            screen.blit(font_small.render(f"{m}m", True, (150, 160, 175)), (tx - 14, mainline_top - 24))

        # --- Draw Virtual Loop Detectors ---
        for det in sim.detectors:
            dx = MARGIN_X + int(det.position_m * SCALE_X)
            dy = LANE_Y[det.lane_id]
            is_flashing = (sim.time_s - det.last_flash_time_s) < 0.25
            det_color = (0, 255, 255) if is_flashing else (70, 130, 200)
            pygame.draw.rect(screen, det_color, (dx - 3, dy - 12, 6, 24), 2 if not is_flashing else 0)
            if det.lane_id == 0:
                screen.blit(font_small.render(det.sensor_id, True, (120, 175, 230)), (dx - 28, mainline_top - 42))

        # --- Draw Ramp Meter Stop Line & Signal Light ---
        meter_x = MARGIN_X + int(sim.meter_stop_line_m * SCALE_X)
        pygame.draw.line(screen, (255, 255, 255), (meter_x, LANE_Y[2] - 15), (meter_x, LANE_Y[2] + 15), 3)
        light_color = (60, 235, 80) if sim.meter_is_green else (245, 50, 50)
        if not sim.ramp_meter_enabled:
            light_color = (120, 120, 120)
        pygame.draw.circle(screen, (20, 20, 20), (meter_x, LANE_Y[2] - 28), 10)
        pygame.draw.circle(screen, light_color, (meter_x, LANE_Y[2] - 28), 7)
        screen.blit(font_small.render("METER", True, (200, 200, 200)), (meter_x - 18, LANE_Y[2] - 52))

        # --- Draw Vehicles ---
        for lane_id, vehs in sim.lanes.items():
            for v in vehs:
                vx = MARGIN_X + v.position_m * SCALE_X
                # Interpolate vertical position smoothly
                if v.visual_lane <= 1.0:
                    vy = LANE_Y[0] + v.visual_lane * (LANE_Y[1] - LANE_Y[0])
                else:
                    # On ramp or merging into Lane 1
                    taper_Ratio = 0.0
                    if v.lane_id == 2 and v.position_m > sim.merge_zone_start_m:
                        taper_Ratio = (v.position_m - sim.merge_zone_start_m) / (sim.merge_zone_end_m - sim.merge_zone_start_m)
                    vy = LANE_Y[2] - taper_Ratio * 22.0 if v.lane_id == 2 else LANE_Y[1] + (v.visual_lane - 1.0) * (LANE_Y[2] - LANE_Y[1])

                vw = max(10, int(v.length_m * SCALE_X))
                vh = 16 if v.veh_type == "truck" else 12

                color = (255, 60, 255) if v.forced_brake_timer_s > 0 else speed_to_color(v.speed_mps)
                rect = pygame.Rect(int(vx - vw), int(vy - vh // 2), vw, vh)

                pygame.draw.rect(screen, color, rect, border_radius=3)
                pygame.draw.rect(screen, (15, 15, 15), rect, 1, border_radius=3)

                # Brake lights when decelerating noticeably
                if v.acceleration_mps2 < -1.0:
                    pygame.draw.rect(screen, (255, 20, 20), (rect.left, rect.top + 2, 3, vh - 4))

                # Highlight if clicked/selected
                if selected_veh is not None and v.vehicle_id == selected_veh.vehicle_id:
                    pygame.draw.rect(screen, (0, 255, 255), rect.inflate(6, 6), 2, border_radius=4)

        # --- Bottom Panel: Live Telemetry Chart & Selected Vehicle Inspector ---
        chart_rect = pygame.Rect(50, 410, 860, 310)
        pygame.draw.rect(screen, (32, 37, 46), chart_rect, border_radius=6)
        pygame.draw.rect(screen, (70, 80, 95), chart_rect, 1, border_radius=6)

        screen.blit(font_ui.render("LIVE TELEMETRY: Mainline Mean Speeds (m/s) & Unmet Boundary Queue (vehs)", True, (230, 235, 245)), (65, 420))

        # Horizontal Grid lines inside chart (0 to 35 scale)
        for val in [0, 10, 20, 30]:
            gy = chart_rect.bottom - 25 - int((val / 35.0) * (chart_rect.height - 65))
            pygame.draw.line(screen, (50, 56, 68), (chart_rect.left + 40, gy), (chart_rect.right - 15, gy), 1)
            screen.blit(font_small.render(f"{val}", True, (140, 150, 165)), (chart_rect.left + 12, gy - 8))

        def draw_series(data_deque, color, max_y=35.0):
            if len(data_deque) < 2:
                return
            pts = []
            plot_w = chart_rect.width - 65
            plot_h = chart_rect.height - 65
            for i, val in enumerate(data_deque):
                px = chart_rect.left + 45 + int((i / MAX_PTS) * plot_w)
                py = chart_rect.bottom - 25 - int((np.clip(val, 0.0, max_y) / max_y) * plot_h)
                pts.append((px, py))
            pygame.draw.lines(screen, color, False, pts, 2)

        draw_series(hist_l0_speed, (80, 180, 255))
        draw_series(hist_l1_speed, (255, 165, 60))
        draw_series(hist_queue, (255, 80, 80))

        # Legend
        screen.blit(font_small.render("── Lane 0 Speed (m/s)", True, (80, 180, 255)), (500, 422))
        screen.blit(font_small.render("── Lane 1 Speed (m/s)", True, (255, 165, 60)), (640, 422))
        screen.blit(font_small.render("── Boundary Queue (vehs)", True, (255, 80, 80)), (775, 422))

        # --- Right Bottom Panel: Vehicle Z-Score & IDM Inspector ---
        insp_rect = pygame.Rect(935, 410, 415, 310)
        pygame.draw.rect(screen, (32, 37, 46), insp_rect, border_radius=6)
        pygame.draw.rect(screen, (70, 80, 95), insp_rect, 1, border_radius=6)
        screen.blit(font_ui.render("VEHICLE & Z-SCORE INSPECTOR", True, (0, 230, 230)), (950, 420))

        if selected_veh is not None:
            pct = StatParam.z_to_percentile(selected_veh.driver_z_score)
            lines = [
                f"Vehicle ID       : #{selected_veh.vehicle_id} ({selected_veh.veh_type.upper()})",
                f"Current Lane     : Lane {selected_veh.lane_id}",
                f"Position (x)     : {selected_veh.position_m:6.1f} m",
                f"Current Speed (v): {selected_veh.speed_mps:6.2f} m/s ({selected_veh.speed_mps * 2.237:.1f} mph)",
                f"IDM Accel (dv/dt): {selected_veh.acceleration_mps2:+6.2f} m/s²",
                "-----------------------------------------",
                f"Driver Z-Score   : {selected_veh.driver_z_score:+5.2f} ({pct:.1f}th Percentile)",
                f"Desired Speed v0 : {selected_veh.v0_mps:5.2f} m/s",
                f"Safe Headway T   : {selected_veh.T_s:5.2f} s",
                f"Max Accel (a)    : {selected_veh.a_mps2:5.2f} m/s²",
                f"Comfort Decel (b): {selected_veh.b_mps2:5.2f} m/s²",
                f"Min Jam Gap (s0) : {selected_veh.s0_m:5.2f} m",
                f"MOBIL Politeness : {selected_veh.politeness:5.2f}",
            ]
            for idx, text_line in enumerate(lines):
                screen.blit(font_ui.render(text_line, True, (220, 230, 240)), (950, 452 + idx * 19))
        else:
            screen.blit(font_ui.render("Click on any moving vehicle on the", True, (150, 160, 175)), (950, 470))
            screen.blit(font_ui.render("highway to inspect its live IDM state,", True, (150, 160, 175)), (950, 492))
            screen.blit(font_ui.render("sampled Z-score, and percentile traits.", True, (150, 160, 175)), (950, 514))

            if sim.observation_log:
                last_obs = sim.observation_log[-1]
                screen.blit(font_ui.render("Latest Virtual Detector Reading:", True, (180, 210, 255)), (950, 575))
                screen.blit(font_ui.render(f"Sensor : {last_obs['sensor_id']} ({last_obs['interval_start_s']}s-{last_obs['interval_end_s']}s)", True, (210, 215, 225)), (950, 600))
                screen.blit(font_ui.render(f"Count  : {last_obs['count']} vehs | Mean Speed: {last_obs['mean_speed_mps']} m/s", True, (210, 215, 225)), (950, 622))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_pygame_sim()