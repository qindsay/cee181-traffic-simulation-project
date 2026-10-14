import sys
from typing import Optional, Tuple
from collections import deque
import numpy as np
import pygame

from models import StatParam, Vehicle
from simulator import HighwaySimulator


def speed_to_color(speed_mps: float) -> Tuple[int, int, int]:
    """Maps vehicle speed (0 to 30 m/s) to a smooth Red -> Yellow -> Green gradient."""
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
    pygame.display.set_caption("CEE 181/281 Modular Freeway Simulator (Pygame)")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Segoe UI", 20, bold=True)
    font_ui = pygame.font.SysFont("Consolas", 14)
    font_small = pygame.font.SysFont("Segoe UI", 12)

    sim = HighwaySimulator(dt_s=0.1, seed=42)

    MARGIN_X = 50
    ROAD_PX_WIDTH = WIDTH - 2 * MARGIN_X
    SCALE_X = ROAD_PX_WIDTH / sim.length_m

    LANE_Y = {0: 200, 1: 245, 2: 305}
    LANE_WIDTH_PX = 38

    paused = False
    sim_speed_mult = 1
    selected_veh: Optional[Vehicle] = None

    MAX_PTS = 350
    hist_l0_speed = deque(maxlen=MAX_PTS)
    hist_l1_speed = deque(maxlen=MAX_PTS)
    hist_queue = deque(maxlen=MAX_PTS)

    running = True
    while running:
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

        if not paused:
            for _ in range(sim_speed_mult):
                sim.step()

            l0_s = float(np.mean([v.speed_mps for v in sim.lanes[0]])) if sim.lanes[0] else 30.0
            l1_s = float(np.mean([v.speed_mps for v in sim.lanes[1]])) if sim.lanes[1] else 30.0
            tot_q = sum(len(q) for q in sim.entry_queues.values())
            hist_l0_speed.append(l0_s)
            hist_l1_speed.append(l1_s)
            hist_queue.append(tot_q)

        screen.fill((24, 28, 34))

        # --- Top Header & Controls Bar ---
        pygame.draw.rect(screen, (35, 40, 50), (0, 0, WIDTH, 95))
        screen.blit(font_title.render("CEE 181/281 Freeway & Merging Simulator (Modular OOP)", True, (240, 245, 255)), (20, 12))

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
        mainline_top = LANE_Y[0] - LANE_WIDTH_PX // 2
        pygame.draw.rect(screen, (52, 56, 64), (MARGIN_X, mainline_top, ROAD_PX_WIDTH, LANE_WIDTH_PX * 2 + 10))

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

        merge_w = ramp_x3 - ramp_x2
        merge_surf = pygame.Surface((merge_w, LANE_WIDTH_PX + 35), pygame.SRCALPHA)
        merge_surf.fill((255, 165, 0, 35))
        screen.blit(merge_surf, (ramp_x2, LANE_Y[1]))
        screen.blit(font_small.render("ACCELERATION & MERGE ZONE (350m - 620m)", True, (255, 190, 90)), (ramp_x2 + 10, LANE_Y[2] + 24))

        pygame.draw.line(screen, (230, 230, 230), (MARGIN_X, mainline_top), (MARGIN_X + ROAD_PX_WIDTH, mainline_top), 3)
        for x_dash in range(MARGIN_X, MARGIN_X + ROAD_PX_WIDTH, 36):
            pygame.draw.line(screen, (200, 200, 200), (x_dash, LANE_Y[0] + 22), (x_dash + 18, LANE_Y[0] + 22), 2)

        bot_y = mainline_top + LANE_WIDTH_PX * 2 + 8
        pygame.draw.line(screen, (230, 230, 230), (MARGIN_X, bot_y), (ramp_x2, bot_y), 3)
        pygame.draw.line(screen, (230, 230, 230), (ramp_x3, bot_y), (MARGIN_X + ROAD_PX_WIDTH, bot_y), 3)
        for x_dash in range(ramp_x2, ramp_x3, 24):
            pygame.draw.line(screen, (255, 200, 80), (x_dash, bot_y), (x_dash + 12, bot_y), 2)

        for m in range(0, 1001, 100):
            tx = MARGIN_X + int(m * SCALE_X)
            pygame.draw.line(screen, (120, 130, 145), (tx, mainline_top - 8), (tx, mainline_top), 2)
            screen.blit(font_small.render(f"{m}m", True, (150, 160, 175)), (tx - 14, mainline_top - 24))

        # --- Virtual Loop Detectors ---
        for det in sim.detectors:
            dx = MARGIN_X + int(det.position_m * SCALE_X)
            dy = LANE_Y[det.lane_id]
            is_flashing = (sim.time_s - det.last_flash_time_s) < 0.25
            det_color = (0, 255, 255) if is_flashing else (70, 130, 200)
            pygame.draw.rect(screen, det_color, (dx - 3, dy - 12, 6, 24), 2 if not is_flashing else 0)
            if det.lane_id == 0:
                screen.blit(font_small.render(det.sensor_id, True, (120, 175, 230)), (dx - 28, mainline_top - 42))

        # --- Ramp Meter Signal ---
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
                if v.visual_lane <= 1.0:
                    vy = LANE_Y[0] + v.visual_lane * (LANE_Y[1] - LANE_Y[0])
                else:
                    taper_ratio = 0.0
                    if v.lane_id == 2 and v.position_m > sim.merge_zone_start_m:
                        taper_ratio = (v.position_m - sim.merge_zone_start_m) / (sim.merge_zone_end_m - sim.merge_zone_start_m)
                    vy = LANE_Y[2] - taper_ratio * 22.0 if v.lane_id == 2 else LANE_Y[1] + (v.visual_lane - 1.0) * (LANE_Y[2] - LANE_Y[1])

                vw = max(10, int(v.length_m * SCALE_X))
                vh = 16 if v.veh_type == "truck" else 12

                color = (255, 60, 255) if v.forced_brake_timer_s > 0 else speed_to_color(v.speed_mps)
                rect = pygame.Rect(int(vx - vw), int(vy - vh // 2), vw, vh)

                pygame.draw.rect(screen, color, rect, border_radius=3)
                pygame.draw.rect(screen, (15, 15, 15), rect, 1, border_radius=3)

                if v.acceleration_mps2 < -1.0:
                    pygame.draw.rect(screen, (255, 20, 20), (rect.left, rect.top + 2, 3, vh - 4))

                if selected_veh is not None and v.vehicle_id == selected_veh.vehicle_id:
                    pygame.draw.rect(screen, (0, 255, 255), rect.inflate(6, 6), 2, border_radius=4)

        # --- Live Telemetry Chart ---
        chart_rect = pygame.Rect(50, 410, 860, 310)
        pygame.draw.rect(screen, (32, 37, 46), chart_rect, border_radius=6)
        pygame.draw.rect(screen, (70, 80, 95), chart_rect, 1, border_radius=6)
        screen.blit(font_ui.render("LIVE TELEMETRY: Mainline Mean Speeds (m/s) & Unmet Boundary Queue (vehs)", True, (230, 235, 245)), (65, 420))

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

        screen.blit(font_small.render("── Lane 0 Speed (m/s)", True, (80, 180, 255)), (500, 422))
        screen.blit(font_small.render("── Lane 1 Speed (m/s)", True, (255, 165, 60)), (640, 422))
        screen.blit(font_small.render("── Boundary Queue (vehs)", True, (255, 80, 80)), (775, 422))

        # --- Vehicle Z-Score & IDM Inspector ---
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