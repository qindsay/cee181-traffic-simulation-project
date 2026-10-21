from collections import deque
import sys
from typing import Optional, Tuple

import numpy as np
import pygame

from constants import render_constants, simulation_constants
from models import StatParam
from simulator import HighwaySimulator
from vehicles import Vehicle
import traffic_analysis


def speed_to_color(speed_mps: float) -> Tuple[int, int, int]:
    """Map vehicle speed to a red-yellow-green color."""
    ratio = float(np.clip(speed_mps / 30.0, 0.0, 1.0))
    if ratio < 0.5:
        r = 235
        g = int(60 + (ratio * 2.0) * 180)
        b = 50
    else:
        r = int(235 - ((ratio - 0.5) * 2.0) * 195)
        g = 225
        b = 75
    return r, g, b


class Simulation:
    def __init__(self):
        pygame.init()

        self.window = pygame.display.set_mode(
            (render_constants.WINDOW_WIDTH, render_constants.WINDOW_HEIGHT),
            pygame.RESIZABLE,
        )
        self.screen = pygame.Surface(
            (render_constants.WINDOW_WIDTH, render_constants.WINDOW_HEIGHT)
        )
        self.viewport = pygame.Rect(
            0,
            0,
            render_constants.WINDOW_WIDTH,
            render_constants.WINDOW_HEIGHT,
        )
        pygame.display.set_caption("CEE 181/281 Modular Freeway Simulator (Pygame)")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont(
            "Segoe UI",
            render_constants.FONT_TITLE_SIZE,
            bold=True,
        )
        self.font_ui = pygame.font.SysFont("Consolas", render_constants.FONT_UI_SIZE)
        self.font_small = pygame.font.SysFont("Segoe UI", render_constants.FONT_SMALL_SIZE)

        self.sim = HighwaySimulator(
            dt_s=simulation_constants.TIME_STEP_S,
            seed=simulation_constants.RANDOM_SEED,
        )

        self.ROAD_PX_WIDTH = (
            render_constants.WINDOW_WIDTH - 2 * render_constants.MARGIN_X
        )
        self.SCALE_X = self.ROAD_PX_WIDTH / self.sim.length_m

        self.paused = False
        self.running = False
        self.sim_speed_mult = render_constants.INITIAL_SIM_SPEED_MULTIPLIER
        self.selected_veh: Optional[Vehicle] = None

        self.hist_l0_speed = deque(maxlen=render_constants.MAX_PTS)
        self.hist_l1_speed = deque(maxlen=render_constants.MAX_PTS)
        self.hist_queue = deque(maxlen=render_constants.MAX_PTS)

        # Records trajectories/crossings for the traffic-theory plots ([P] key)
        self.recorder = traffic_analysis.TrafficRecorder()
        self.status_message = ""

    def window_to_canvas_position(
        self, position: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        window_width, window_height = self.window.get_size()
        scale = min(
            window_width / render_constants.WINDOW_WIDTH,
            window_height / render_constants.WINDOW_HEIGHT,
        )
        viewport_width = int(render_constants.WINDOW_WIDTH * scale)
        viewport_height = int(render_constants.WINDOW_HEIGHT * scale)
        viewport = pygame.Rect(
            (window_width - viewport_width) // 2,
            (window_height - viewport_height) // 2,
            viewport_width,
            viewport_height,
        )
        if not viewport.collidepoint(position):
            return None

        return (
            int((position[0] - viewport.left) / scale),
            int((position[1] - viewport.top) / scale),
        )

    def present(self) -> None:
        window_width, window_height = self.window.get_size()
        scale = min(
            window_width / render_constants.WINDOW_WIDTH,
            window_height / render_constants.WINDOW_HEIGHT,
        )
        viewport_width = int(render_constants.WINDOW_WIDTH * scale)
        viewport_height = int(render_constants.WINDOW_HEIGHT * scale)
        self.viewport = pygame.Rect(
            (window_width - viewport_width) // 2,
            (window_height - viewport_height) // 2,
            viewport_width,
            viewport_height,
        )

        self.window.fill((0, 0, 0))
        scaled_screen = pygame.transform.smoothscale(
            self.screen,
            self.viewport.size,
        )
        self.window.blit(scaled_screen, self.viewport)
        pygame.display.flip()
    
    def render_highway(self) -> None:
        mainline_top = render_constants.LANE_Y[0] - render_constants.LANE_WIDTH_PX // 2
        pygame.draw.rect(
            self.screen,
            (52, 56, 64),
            (
                render_constants.MARGIN_X,
                mainline_top,
                self.ROAD_PX_WIDTH,
                render_constants.LANE_WIDTH_PX * 2 + 10,
            ),
        )

        ramp_x1 = render_constants.MARGIN_X + int(self.sim.ramp_start_m * self.SCALE_X)
        ramp_x2 = render_constants.MARGIN_X + int(self.sim.merge_zone_start_m * self.SCALE_X)
        ramp_x3 = render_constants.MARGIN_X + int(self.sim.merge_zone_end_m * self.SCALE_X)
        ramp_y = render_constants.LANE_Y[2]

        ramp_poly = [
            (ramp_x1, ramp_y - 16),
            (ramp_x2, mainline_top + render_constants.LANE_WIDTH_PX * 2 + 8),
            (ramp_x3, mainline_top + render_constants.LANE_WIDTH_PX * 2 + 8),
            (ramp_x3 + 25, mainline_top + render_constants.LANE_WIDTH_PX * 2),
            (ramp_x2, ramp_y + 18),
            (ramp_x1, ramp_y + 18)
        ]
        pygame.draw.polygon(self.screen, (52, 56, 64), ramp_poly)

        merge_w = ramp_x3 - ramp_x2
        merge_surf = pygame.Surface(
            (merge_w, render_constants.LANE_WIDTH_PX + 35),
            pygame.SRCALPHA,
        )
        merge_surf.fill((255, 165, 0, 35))
        self.screen.blit(merge_surf, (ramp_x2, render_constants.LANE_Y[1]))
        self.screen.blit(
            self.font_small.render(
                "ACCELERATION & MERGE ZONE (350m - 620m)",
                True,
                (255, 190, 90),
            ),
            (ramp_x2 + 10, render_constants.LANE_Y[2] + 24),
        )

        self.render_lane_markings(mainline_top, ramp_x2, ramp_x3)

    def render_lane_markings(
        self,
        mainline_top: int,
        merge_start_x: int,
        merge_end_x: int,
    ) -> None:
        """Draw lane boundaries and dashed road markings."""
        road_left = render_constants.MARGIN_X
        road_right = road_left + self.ROAD_PX_WIDTH
        lane_bottom = mainline_top + render_constants.LANE_WIDTH_PX * 2 + 8

        pygame.draw.line(
            self.screen,
            (230, 230, 230),
            (road_left, mainline_top),
            (road_right, mainline_top),
            3,
        )
        for x_dash in range(
            road_left,
            road_right,
            36,
        ):
            pygame.draw.line(
                self.screen,
                (200, 200, 200),
                (x_dash, render_constants.LANE_Y[0] + 22),
                (x_dash + 18, render_constants.LANE_Y[0] + 22),
                2,
            )

        pygame.draw.line(
            self.screen,
            (230, 230, 230),
            (road_left, lane_bottom),
            (merge_start_x, lane_bottom),
            3,
        )
        pygame.draw.line(
            self.screen,
            (230, 230, 230),
            (merge_end_x, lane_bottom),
            (road_right, lane_bottom),
            3,
        )
        for x_dash in range(merge_start_x, merge_end_x, 24):
            pygame.draw.line(
                self.screen,
                (255, 200, 80),
                (x_dash, lane_bottom),
                (x_dash + 12, lane_bottom),
                2,
            )

        self.render_distance_markers(mainline_top)
        self.render_detectors(mainline_top)

    def render_distance_markers(self, mainline_top: int) -> None:
        """Draw 100-meter markers above the mainline."""
        for distance_m in range(0, 1001, 100):
            marker_x = render_constants.MARGIN_X + int(
                distance_m * self.SCALE_X
            )
            pygame.draw.line(
                self.screen,
                (120, 130, 145),
                (marker_x, mainline_top - 8),
                (marker_x, mainline_top),
                2,
            )
            label = self.font_small.render(
                f"{distance_m}m",
                True,
                (150, 160, 175),
            )
            self.screen.blit(label, (marker_x - 14, mainline_top - 24))

    def render_detectors(self, mainline_top: int) -> None:
        """Draw detector markers and sensor labels."""
        for det in self.sim.detectors:
            dx = render_constants.MARGIN_X + int(det.position_m * self.SCALE_X)
            dy = render_constants.LANE_Y[det.lane_id]
            is_flashing = (self.sim.time_s - det.last_flash_time_s) < 0.25
            det_color = (0, 255, 255) if is_flashing else (70, 130, 200)
            border_width = 0 if is_flashing else 2
            pygame.draw.rect(
                self.screen,
                det_color,
                (dx - 3, dy - 12, 6, 24),
                border_width,
            )
            if det.lane_id == 0:
                label = self.font_small.render(
                    det.sensor_id,
                    True,
                    (120, 175, 230),
                )
                self.screen.blit(label, (dx - 28, mainline_top - 42))

    def render_series(
        self,
        chart_rect: pygame.Rect,
        data: deque[int | float],
        color: Tuple[int, int, int],
        max_y: float = 35.0,
    ) -> None:
        if len(data) < 2:
            return
        pts = []
        plot_w = chart_rect.width - 65
        plot_h = chart_rect.height - 65
        for i, val in enumerate(data):
            px = chart_rect.left + 45 + int(
                (i / render_constants.MAX_PTS) * plot_w
            )
            py = chart_rect.bottom - 25 - int((np.clip(val, 0.0, max_y) / max_y) * plot_h)
            pts.append((px, py))
        pygame.draw.lines(self.screen, color, False, pts, 2)

    def render_telemetry_chart(self) -> None:
        chart_rect = pygame.Rect(50, 410, 860, 310)
        pygame.draw.rect(self.screen, (32, 37, 46), chart_rect, border_radius=6)
        pygame.draw.rect(self.screen, (70, 80, 95), chart_rect, 1, border_radius=6)
        self.screen.blit(
            self.font_ui.render("LIVE TELEMETRY", True, (230, 235, 245)),
            (65, 420),
        )

        for val in [0, 10, 20, 30]:
            gy = chart_rect.bottom - 25 - int((val / 35.0) * (chart_rect.height - 65))
            pygame.draw.line(self.screen, (50, 56, 68), (chart_rect.left + 40, gy), (chart_rect.right - 15, gy), 1)
            self.screen.blit(self.font_small.render(f"{val}", True, (140, 150, 165)), (chart_rect.left + 12, gy - 8))

        self.render_series(chart_rect, self.hist_l0_speed, (80, 180, 255))
        self.render_series(chart_rect, self.hist_l1_speed, (255, 165, 60))
        self.render_series(chart_rect, self.hist_queue, (255, 80, 80))

        legend_items = (
            (310, "L0 speed (m/s)", (80, 180, 255)),
            (465, "L1 speed (m/s)", (255, 165, 60)),
            (620, "Queue", (255, 80, 80)),
        )
        for x, label, color in legend_items:
            pygame.draw.line(self.screen, color, (x, 428), (x + 16, 428), 2)
            self.screen.blit(
                self.font_small.render(label, True, color),
                (x + 22, 420),
            )

    def vehicle_screen_position(self, vehicle: Vehicle) -> Tuple[float, float]:
        x = render_constants.MARGIN_X + vehicle.position_m * self.SCALE_X

        if vehicle.visual_lane <= 1.0:
            y = render_constants.LANE_Y[0] + vehicle.visual_lane * (
                render_constants.LANE_Y[1] - render_constants.LANE_Y[0]
            )
        elif vehicle.lane_id == 2:
            taper_ratio = 0.0
            if vehicle.position_m > self.sim.merge_zone_start_m:
                taper_ratio = (
                    (vehicle.position_m - self.sim.merge_zone_start_m)
                    / (self.sim.merge_zone_end_m - self.sim.merge_zone_start_m)
                )
            y = render_constants.LANE_Y[2] - taper_ratio * 22.0
        else:
            y = render_constants.LANE_Y[1] + (vehicle.visual_lane - 1.0) * (
                render_constants.LANE_Y[2] - render_constants.LANE_Y[1]
            )

        return x, y

    def render_vehicle(self, vehicle: Vehicle) -> None:
        x, y = self.vehicle_screen_position(vehicle)
        width = max(10, int(vehicle.length_m * self.SCALE_X))
        height = 16 if vehicle.veh_type == "truck" else 12
        color = (
            (255, 60, 255)
            if vehicle.forced_brake_timer_s > 0
            else speed_to_color(vehicle.speed_mps)
        )
        rect = pygame.Rect(
            int(x - width),
            int(y - height // 2),
            width,
            height,
        )

        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        pygame.draw.rect(self.screen, (15, 15, 15), rect, 1, border_radius=3)

        if vehicle.acceleration_mps2 < -1.0:
            pygame.draw.rect(
                self.screen,
                (255, 20, 20),
                (rect.left, rect.top + 2, 3, height - 4),
            )

        if (
            self.selected_veh is not None
            and vehicle.vehicle_id == self.selected_veh.vehicle_id
        ):
            pygame.draw.rect(
                self.screen,
                (0, 255, 255),
                rect.inflate(6, 6),
                2,
                border_radius=4,
            )

    def render_vehicles(self) -> None:
        for vehs in self.sim.lanes.values():
            for vehicle in vehs:
                self.render_vehicle(vehicle)

    def render_header(self) -> None:
        """Draw the title, controls, and current simulation statistics."""
        pygame.draw.rect(
            self.screen,
            (35, 40, 50),
            (0, 0, render_constants.WINDOW_WIDTH, 95),
        )
        title = "CEE 181/281 Freeway & Merging Simulator (Modular OOP)"
        self.screen.blit(
            self.font_title.render(title, True, (240, 245, 255)),
            (20, 12),
        )

        controls = (
            f"[SPACE] {'Resume' if self.paused else 'Pause'}  |  "
            f"[M] Ramp Meter: {'ON (450 vph)' if self.sim.ramp_meter_enabled else 'OFF'}  |  "
            "[B] Braking Disturbance  |  "
            f"[UP/DOWN] Speed: {self.sim_speed_mult}x  |  "
            "[R] Reset  |  [P] Export Plots  |  [Click Car] Inspect"
        )
        self.screen.blit(
            self.font_ui.render(controls, True, (180, 210, 255)),
            (20, 44),
        )

        active_vehicles = sum(len(vehicles) for vehicles in self.sim.lanes.values())
        queue_counts = self.sim.entry_queues
        stats = (
            f"Sim Time: {self.sim.time_s:6.1f}s   |   "
            f"Active Vehs: {active_vehicles:3d}   |   "
            f"Completed Trips: {self.sim.total_exited_vehicles:4d}   |   "
            f"Boundary Entry Queues -> L0: {len(queue_counts[0])}  "
            f"L1: {len(queue_counts[1])}  Ramp: {len(queue_counts[2])}"
        )
        self.screen.blit(
            self.font_ui.render(stats, True, (220, 220, 220)),
            (20, 68),
        )
        if self.status_message:
            self.screen.blit(
                self.font_small.render(self.status_message, True, (120, 230, 140)),
                (20, 100),
            )

    def render_ramp_meter(self) -> None:
        """Draw the ramp meter stop line and signal."""
        meter_x = render_constants.MARGIN_X + int(
            self.sim.meter_stop_line_m * self.SCALE_X
        )
        lane_y = render_constants.LANE_Y[2]
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (meter_x, lane_y - 15),
            (meter_x, lane_y + 15),
            3,
        )

        if not self.sim.ramp_meter_enabled:
            light_color = (120, 120, 120)
        elif self.sim.meter_is_green:
            light_color = (60, 235, 80)
        else:
            light_color = (245, 50, 50)

        light_position = (meter_x, lane_y - 28)
        pygame.draw.circle(self.screen, (20, 20, 20), light_position, 10)
        pygame.draw.circle(self.screen, light_color, light_position, 7)
        self.screen.blit(
            self.font_small.render("METER", True, (200, 200, 200)),
            (meter_x - 18, lane_y - 52),
        )

    def render_vehicle_inspector(self) -> None:
        """Draw details for the selected vehicle or the latest detector reading."""
        inspector_rect = pygame.Rect(935, 410, 415, 310)
        pygame.draw.rect(
            self.screen,
            (32, 37, 46),
            inspector_rect,
            border_radius=6,
        )
        pygame.draw.rect(
            self.screen,
            (70, 80, 95),
            inspector_rect,
            1,
            border_radius=6,
        )
        self.screen.blit(
            self.font_ui.render(
                "VEHICLE & Z-SCORE INSPECTOR",
                True,
                (0, 230, 230),
            ),
            (950, 420),
        )

        if self.selected_veh is not None:
            self.render_selected_vehicle_details()
        else:
            self.render_detector_reading()

    def render_selected_vehicle_details(self) -> None:
        vehicle = self.selected_veh
        if vehicle is None:
            return

        percentile = StatParam.z_to_percentile(vehicle.driver_z_score)
        details = (
            f"Vehicle ID       : #{vehicle.vehicle_id} ({vehicle.veh_type.upper()})",
            f"Current Lane     : Lane {vehicle.lane_id}",
            f"Position (x)     : {vehicle.position_m:6.1f} m",
            f"Current Speed (v): {vehicle.speed_mps:6.2f} m/s "
            f"({vehicle.speed_mps * 2.237:.1f} mph)",
            f"IDM Accel (dv/dt): {vehicle.acceleration_mps2:+6.2f} m/s²",
            "-----------------------------------------",
            f"Driver Z-Score   : {vehicle.driver_z_score:+5.2f} "
            f"({percentile:.1f}th Percentile)",
            f"Desired Speed v0 : {vehicle.v0_mps:5.2f} m/s",
            f"Safe Headway T   : {vehicle.T_s:5.2f} s",
            f"Max Accel (a)    : {vehicle.a_mps2:5.2f} m/s²",
            f"Comfort Decel (b): {vehicle.b_mps2:5.2f} m/s²",
            f"Min Jam Gap (s0) : {vehicle.s0_m:5.2f} m",
            f"MOBIL Politeness : {vehicle.politeness:5.2f}",
        )
        for index, text in enumerate(details):
            self.screen.blit(
                self.font_ui.render(text, True, (220, 230, 240)),
                (950, 452 + index * 19),
            )

    def render_detector_reading(self) -> None:
        """Show selection instructions and the latest detector sample."""
        instructions = (
            "Click on any moving vehicle on the",
            "highway to inspect its live IDM state,",
            "sampled Z-score, and percentile traits.",
        )
        for index, text in enumerate(instructions):
            self.screen.blit(
                self.font_ui.render(text, True, (150, 160, 175)),
                (950, 470 + index * 22),
            )

        if not self.sim.observation_log:
            return

        observation = self.sim.observation_log[-1]
        self.screen.blit(
            self.font_ui.render(
                "Latest Virtual Detector Reading:",
                True,
                (180, 210, 255),
            ),
            (950, 575),
        )
        self.screen.blit(
            self.font_ui.render(
                f"Sensor : {observation['sensor_id']} "
                f"({observation['interval_start_s']}s-"
                f"{observation['interval_end_s']}s)",
                True,
                (210, 215, 225),
            ),
            (950, 600),
        )
        self.screen.blit(
            self.font_ui.render(
                f"Count  : {observation['count']} vehs | "
                f"Mean Speed: {observation['mean_speed_mps']} m/s",
                True,
                (210, 215, 225),
            ),
            (950, 622),
        )

    def render(self) -> None:
        self.screen.fill((24, 28, 34))
        self.render_header()
        self.render_highway()
        self.render_ramp_meter()
        self.render_vehicles()
        self.render_telemetry_chart()
        self.render_vehicle_inspector()
        self.present()

    def reset(self) -> None:
        """Restore the simulation and clear the display history."""
        self.sim = HighwaySimulator(
            dt_s=simulation_constants.TIME_STEP_S,
            seed=simulation_constants.RANDOM_SEED,
        )
        self.hist_l0_speed.clear()
        self.hist_l1_speed.clear()
        self.hist_queue.clear()
        self.selected_veh = None
        self.recorder = traffic_analysis.TrafficRecorder()
        self.status_message = ""

    def handle_keydown(self, key: int) -> None:
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_m:
            self.sim.ramp_meter_enabled = not self.sim.ramp_meter_enabled
        elif key == pygame.K_b:
            # same as before, but also remembered for the plots
            traffic_analysis.trigger_disturbance(self.sim, self.recorder)
        elif key == pygame.K_p:
            self.export_plots()
        elif key == pygame.K_UP:
            self.sim_speed_mult = min(
                render_constants.MAX_SIM_SPEED_MULTIPLIER,
                self.sim_speed_mult + 1,
            )
        elif key == pygame.K_DOWN:
            self.sim_speed_mult = max(
                render_constants.MIN_SIM_SPEED_MULTIPLIER,
                self.sim_speed_mult - 1,
            )
        elif key == pygame.K_r:
            self.reset()

    def export_plots(self) -> None:
        """[P]: save CSV data + traffic-theory plots for everything simulated so far."""
        if self.sim.time_s < traffic_analysis.WARMUP_S + 60:
            self.status_message = (f"Run at least {traffic_analysis.WARMUP_S + 60:.0f} s of "
                                   "simulated time before exporting plots.")
            return
        self.status_message = "Exporting data and plots (the window pauses a few seconds)..."
        self.render()
        out = traffic_analysis.export(self.sim, self.recorder)
        self.status_message = f"Saved data + plots to {out}"
        print(f"Saved data + plots to {out}")

    def find_vehicle_at(self, position: Tuple[int, int]) -> Optional[Vehicle]:
        mouse_x, mouse_y = position
        for lane_vehicles in self.sim.lanes.values():
            for vehicle in lane_vehicles:
                vehicle_x, vehicle_y = self.vehicle_screen_position(vehicle)
                hitbox_width = max(12, vehicle.length_m * self.SCALE_X)
                if (
                    abs(mouse_x - vehicle_x) <= hitbox_width
                    and abs(mouse_y - vehicle_y) <= 14
                ):
                    return vehicle
        return None

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self.window = pygame.display.set_mode(event.size, pygame.RESIZABLE)
        elif event.type == pygame.KEYDOWN:
            self.handle_keydown(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            canvas_position = self.window_to_canvas_position(event.pos)
            if canvas_position is not None:
                self.selected_veh = self.find_vehicle_at(canvas_position)

    def update_telemetry(self) -> None:
        lane_speeds = []
        for lane_id in (0, 1):
            vehicles = self.sim.lanes[lane_id]
            mean_speed = (
                float(np.mean([vehicle.speed_mps for vehicle in vehicles]))
                if vehicles
                else 30.0
            )
            lane_speeds.append(mean_speed)

        queue_count = sum(len(queue) for queue in self.sim.entry_queues.values())
        self.hist_l0_speed.append(lane_speeds[0])
        self.hist_l1_speed.append(lane_speeds[1])
        self.hist_queue.append(queue_count)

    def run(self) -> None:
        self.running = True

        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)

            if not self.paused:
                for _ in range(self.sim_speed_mult):
                    self.sim.step()
                    self.recorder.record(self.sim)
                self.update_telemetry()

            self.render()
            
            self.clock.tick(render_constants.FRAME_RATE)


def run_pygame_sim() -> None:
    simulation = Simulation()
    try:
        simulation.run()
    finally:
        pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_pygame_sim()
