"""Figures and tables for the mid-report, made from the 9 comparison runs.

Run the 9 simulations first (README), then:
    python -m analysis.report_figures
Reads output/cav{0,0.25,0.5}_seed{42,1,7}/ and writes output/report/:
    fig1_layout.png, fig2_headline.png, fig3_time_space.png, fig4_fundamental.png,
    fig5_spot_speeds.png, fig6_string_stability.png, table_runs.csv, table_fd_fits.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from analysis import analyze_traffic
from constants import simulation_constants as sc, vehicle_constants as vc

LEVELS = ("0", "0.25", "0.5")
LABEL = {"0": "0% CAVs", "0.25": "25% CAVs", "0.5": "50% CAVs"}
SEEDS = (42, 1, 7)
WARMUP_S = 120.0
MPH = 2.23694
CONGESTED_MPS = 15.0        # a 10 s detector interval is slow below 15 m/s (33.6 mph) ...
CONGESTED_RUN = 3           # ... and congested if it is part of >= 3 slow intervals in a row (30 s)
INK, INK2, GRID, GREY, BLUE = "#1f2328", "#57606a", "#e6e8eb", "#9aa0a6", "#2f6fdb"
SHADES = {"0": "#b7cdf3", "0.25": "#6f9be6", "0.5": "#2f6fdb"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False})


def run_dir(base, cav, seed):
    return base / f"cav{cav}_seed{seed}"


def congested_share(det: pd.DataFrame) -> float:
    """Share of detector intervals (all mainline detectors, after warm-up) that are congested."""
    shares = []
    for _, g in det.groupby("sensor_id"):
        g = g[g.interval_start_s >= WARMUP_S].sort_values("interval_start_s")
        slow = (g.mean_speed_mps < CONGESTED_MPS).fillna(False).to_numpy()
        flag = np.zeros(len(slow), bool)
        run = 0
        for i, s in enumerate(slow):
            run = run + 1 if s else 0
            if run >= CONGESTED_RUN:
                flag[i - CONGESTED_RUN + 1:i + 1] = True
        shares.append(flag.mean() if len(flag) else 0.0)
    return float(np.mean(shares))


def run_table(base: Path) -> pd.DataFrame:
    rows = []
    for cav in LEVELS:
        for seed in SEEDS:
            d = run_dir(base, cav, seed)
            t = pd.read_csv(d / "trajectories.csv")
            v = pd.read_csv(d / "vehicles.csv")
            c = pd.read_csv(d / "detector_crossings.csv")
            det = pd.read_csv(d / "detector_observations.csv")
            main = t[(t.time_s > WARMUP_S) & (t.lane_id < 2)]
            allv = t[t.time_s > WARMUP_S]
            down = c[(c.station == "downstream") & (c.time_s > WARMUP_S)]
            cars = v.veh_type.isin(["car", "cav"]).sum()
            rows.append({
                "cav_share": float(cav), "seed": seed,
                "cav_share_of_cars": round((v.veh_type == "cav").sum() / cars, 3),
                "mean_speed_mph": round(main.speed_mps.mean() * MPH, 2),
                "time_stopped_veh_s": int((main.speed_mps < 2.0).sum()),          # 1 s samples
                "hard_braking_veh_s": int((allv.acceleration_mps2 < -3.0).sum()),
                "throughput_vph": round(len(down) / ((t.time_s.max() - WARMUP_S) / 3600.0)),
                "congested_detector_share": round(congested_share(det), 4),
                "min_gap_m": round(t.gap_to_leader_m.min(), 2),
            })
    return pd.DataFrame(rows)


def cells(base, cav):
    e = pd.concat([pd.read_csv(run_dir(base, cav, s) / "edie_cells.csv") for s in SEEDS])
    return e[(e.t_start_s >= WARMUP_S) & (e.density_vpm > 1)]


def fd_fits(base: Path) -> pd.DataFrame:
    rows = []
    for cav in LEVELS:
        e = cells(base, cav)
        k, q, u = e.density_vpm.to_numpy(), e.flow_vph.to_numpy(), e.speed_mph.to_numpy()
        gs = analyze_traffic.fit_greenshields(k, u)
        cong = k >= gs["kc_vpm"]
        gb = analyze_traffic.fit_greenberg(k[cong], u[cong])
        sm = analyze_traffic.fit_smooth_curve(k, q)
        top = analyze_traffic.top_flow_capacity(k, q)
        rows.append({"cav_share": float(cav), "cells": len(k),
                     "congested_cells_share": round(float((u < 30).mean()), 3),
                     "gs_uf_mph": round(gs["uf_mph"], 1), "gs_kj_vpm": round(gs["kj_vpm"]),
                     "gs_qmax_vph": round(gs["qmax_vph"]), "gs_r2": round(gs["r2"], 2),
                     "gb_c_mph": round(gb["c_mph"], 1), "gb_kj_vpm": round(gb["kj_vpm"]),
                     "gb_qmax_vph": round(gb["c_mph"] * gb["kj_vpm"] / np.e), "gb_r2": round(gb["r2"], 2),
                     "smooth_qmax_vph": round(sm["qmax_vph"]), "smooth_kc_vpm": round(sm["kc_vpm"]),
                     "p95_cell_flow_vph": round(float(np.percentile(q, 95))),
                     "top10_qmax_vph": round(top["qmax_vph"]), "top10_k_vpm": round(top["kc_vpm"])})
    return pd.DataFrame(rows)


def fig_simulator(out, seconds=240):
    """Screenshot of our Pygame simulator (run without a window), with a few labels."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    import pygame_main
    app = pygame_main.Simulation()
    app.cav_enabled = True
    app.sim.cav_enabled = True
    for i in range(int(seconds / app.sim.dt_s)):
        app.sim.step()
        app.recorder.record(app.sim)
        if i % 10 == 0:
            app.update_telemetry()
    app.render()
    img = pygame.surfarray.array3d(app.screen).swapaxes(0, 1)[:395]
    pygame.quit()
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    lab = dict(color="white", fontsize=12, fontweight="bold")
    ax.annotate("on-ramp (lane 2)\nstarts at 225 m", xy=(345, 300), xytext=(80, 300), va="center",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5), **lab)
    ax.text(1350, 300, "top lane = lane 0 (left)\nbottom lane = lane 1 (right)", ha="right", va="center", **lab)
    fig.savefig(out / "fig1_simulator.png", dpi=200)
    plt.close(fig)


def fig_layout(out):
    fig, ax = plt.subplots(figsize=(10, 2.6), constrained_layout=True)
    L = sc.HIGHWAY_LENGTH_M
    for y in (0, 1, 2):
        ax.plot([0, L], [y, y], color=GRID, lw=1)
    ax.fill_between([0, L], 0, 2, color="#eef1f4")
    ax.fill_between([sc.RAMP_START_M, sc.MERGE_ZONE_START_M], -1, 0, color="#f3efe6")
    ax.fill_between([sc.MERGE_ZONE_START_M, sc.MERGE_ZONE_END_M], -1, 0, color="#e8dcc4")
    ax.text(L / 2, 1.5, "lane 0 (left)", ha="center", va="center", color=INK2)
    ax.text(L / 2, 0.5, "lane 1 (right)", ha="center", va="center", color=INK2)
    ax.text((sc.RAMP_START_M + sc.MERGE_ZONE_START_M) / 2, -0.5, "on-ramp (lane 2)", ha="center", va="center", fontsize=9)
    ax.text((sc.MERGE_ZONE_START_M + sc.MERGE_ZONE_END_M) / 2, -0.5, "merge zone (400 m)", ha="center", va="center", fontsize=9)
    ax.plot([sc.METER_STOP_LINE_M] * 2, [-1, 0], color="#c0392b", lw=2)
    ax.text(sc.METER_STOP_LINE_M + 8, -1.12, "meter stop line", ha="left", va="top", fontsize=8, color="#c0392b")
    for name, (x, lanes) in {"upstream": (sc.MERGE_ZONE_START_M - 130, (0, 1)),
                             "downstream": (sc.MERGE_ZONE_END_M + 100, (0, 1)),
                             "ramp": (sc.RAMP_START_M + 50, (2,))}.items():
        y0, y1 = (-1, 0) if lanes == (2,) else (0, 2)
        ax.plot([x, x], [y0, y1], color=BLUE, lw=1.5, ls="--")
        if lanes == (2,):
            ax.text(x, -1.12, f"{name} station {x:.0f} m", ha="center", va="top", fontsize=8, color=BLUE)
        else:
            ax.text(x, 2.15, f"{name} station\n{x:.0f} m", ha="center", va="bottom", fontsize=8, color=BLUE)
    ax.annotate("", xy=(L, 2.9), xytext=(0, 2.9), arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(L / 2, 3.0, "direction of travel, x (m)", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xlim(-10, L + 10)
    ax.set_ylim(-1.6, 3.5)
    ax.set_yticks([])
    ax.set_xticks([0, sc.RAMP_START_M, sc.MERGE_ZONE_START_M, sc.MERGE_ZONE_END_M, L])
    ax.spines["left"].set_visible(False)
    fig.savefig(out / "fig1_layout.png", dpi=200)
    plt.close(fig)


def fig_headline(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    panels = (("mean_speed_mph", "Average mainline speed (mph)", 1),
              ("time_stopped_veh_s", "Time stopped per run (veh-s)", 0),
              ("congested_detector_share", "Congested detector intervals (%)", 1))
    for ax, (col, title, dec) in zip(axes, panels):
        scale = 100 if col == "congested_detector_share" else 1
        for i, cav in enumerate(LEVELS):
            vals = runs[runs.cav_share == float(cav)][col].to_numpy() * scale
            m = vals.mean()
            ax.bar(i, m, width=0.6, color=SHADES[cav])
            ax.plot([i, i], [vals.min(), vals.max()], color=INK, lw=1.3)
            ax.scatter([i] * len(vals), vals, color=INK, s=12, zorder=3)
            ax.text(i + 0.33, m, f"{m:,.{dec}f}", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(3), [LABEL[c] for c in LEVELS])
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.savefig(out / "fig2_headline.png", dpi=200)
    plt.close(fig)


def fig_time_space(base, out):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, constrained_layout=True)
    norm, cmap = plt.Normalize(0, 70), plt.get_cmap("Blues_r")
    for ax, cav in zip(axes, ("0", "0.5")):
        t = pd.read_csv(run_dir(base, cav, 42) / "trajectories.csv")
        t = t[t.lane_id == 1].sort_values(["vehicle_id", "time_s"])
        segs, cols = [], []
        for _, g in t.groupby("vehicle_id"):
            tt, xx, vv = g.time_s.to_numpy(), g.position_m.to_numpy(), g.speed_mps.to_numpy() * MPH
            brk = np.where(np.diff(tt) > 1.5)[0] + 1
            for a, b in zip(np.r_[0, brk], np.r_[brk, len(tt)]):
                if b - a < 2:
                    continue
                p = np.column_stack([tt[a:b], xx[a:b]])
                segs += list(np.stack([p[:-1], p[1:]], axis=1))
                cols += list(vv[a:b - 1])
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=0.5)
        lc.set_array(np.array(cols))
        ax.add_collection(lc)
        ax.axhspan(sc.MERGE_ZONE_START_M, sc.MERGE_ZONE_END_M, color="#f3efe6", zorder=0)
        for td in (400, 800):
            ax.annotate("", xy=(td, 1000), xytext=(td, 1065),
                        arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.4), annotation_clip=False)
        ax.set(xlim=(0, 1200), ylim=(0, 1000), ylabel="position x (m)")
        ax.set_title(f"{LABEL[cav]}, lane 1, seed 42", loc="left", fontsize=10)
    axes[1].set_xlabel("time t (s)")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("speed (mph), darker = slower")
    fig.savefig(out / "fig3_time_space.png", dpi=170)
    plt.close(fig)


def fig_fundamental(base, fits, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, cav in zip(axes, ("0", "0.5")):
        e = cells(base, cav)
        k, q = e.density_vpm.to_numpy(), e.flow_vph.to_numpy()
        f = fits[fits.cav_share == float(cav)].iloc[0]
        ax.scatter(k, q, s=6, lw=0, alpha=0.35, color=GREY if cav == "0" else BLUE,
                   label="Edie cells (3 seeds)")
        kk = np.linspace(1, 220, 400)
        ax.plot(kk, np.clip(f.gs_uf_mph * (1 - kk / f.gs_kj_vpm), 0, None) * kk, color=INK, lw=1.1, ls="--",
                label=f"Greenshields (q_max {f.gs_qmax_vph:,.0f})")
        kg = np.linspace(f.gs_kj_vpm / 2 * 0.6, f.gb_kj_vpm, 200)
        ax.plot(kg, np.clip(f.gb_c_mph * np.log(f.gb_kj_vpm / kg), 0, None) * kg, color="#6250d6", lw=1.1, ls=":",
                label=f"Greenberg (q_max {f.gb_qmax_vph:,.0f})")
        sm = analyze_traffic.fit_smooth_curve(k, q)
        ax.plot(sm["k_vpm"], sm["q_vph"], color="#e8853d", lw=2.4, label="smooth trend line along the top of our data")
        top = analyze_traffic.top_flow_capacity(k, q)
        ax.scatter(top["k_top"], top["q_top"], s=46, facecolor="none", edgecolor="#c0392b", lw=1.6, zorder=6,
                   label="10 highest-flow cells")
        ax.axhline(top["qmax_vph"], color="#c0392b", lw=1.6, zorder=5,
                   label=f"capacity = average of top 10 = {top['qmax_vph']:,.0f}")
        ax.set(xlim=(0, 220), ylim=(0, 3600), xlabel="density k (veh/mi/lane)")
        ax.set_title(f"{LABEL[cav]}: {100 * f.congested_cells_share:.0f}% of cells below 30 mph",
                     loc="left", fontsize=10)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92, edgecolor="none", markerscale=1.5)
    axes[0].set_ylabel("flow q (veh/h/lane)")
    fig.savefig(out / "fig4_fundamental.png", dpi=200)
    plt.close(fig)


def fig_spot_speeds(base, out):
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    bins = np.arange(0, 80, 2.5)
    for cav, col in (("0", GREY), ("0.5", BLUE)):
        c = pd.concat([pd.read_csv(run_dir(base, cav, s) / "detector_crossings.csv") for s in SEEDS])
        sp = c[(c.station == "upstream") & (c.time_s > WARMUP_S)].speed_mps * MPH
        p15, p50, p85 = np.percentile(sp, [15, 50, 85])
        w = np.ones(len(sp)) * 100 / len(sp)
        ax.hist(sp, bins=bins, weights=w, histtype="stepfilled", color=col, alpha=0.3, lw=0)
        ax.hist(sp, bins=bins, weights=w, histtype="step", color=col, lw=2,
                label=f"{LABEL[cav]}: 15th / 50th / 85th = {p15:.0f} / {p50:.0f} / {p85:.0f} mph")
    ax.set(xlabel="spot speed at the upstream station, 295 m (mph)", ylabel="% of vehicles per 2.5 mph")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.savefig(out / "fig5_spot_speeds.png", dpi=200)
    plt.close(fig)


def platoon_dips(follower: str, v_lead=15.0, brake_s=2.0):
    """10 followers behind a leader that brakes at -4.5 m/s^2 for brake_s seconds.
    Returns the speed dip (m/s) of each vehicle, leader first."""
    from simulator import HighwaySimulator
    from vehicles import Vehicle, CAVVehicle
    sim = HighwaySimulator(dt_s=0.1, seed=1)
    sim.demand_vph = {0: 0.0, 1: 0.0, 2: 0.0}
    sim.length_m = 1e9
    sim._process_discretionary_lane_changes = lambda: None
    vs = []
    for i in range(11):
        kw = dict(vehicle_id=i, veh_type="car", lane_id=1, visual_lane=1.0, position_m=5000 - i * 30.0,
                  speed_mps=v_lead, length_m=4.5, v0_mps=30.0, T_s=1.2, a_mps2=1.6, b_mps2=2.0, s0_m=2.0,
                  politeness=0.25, driver_z_score=0.0, reaction_time_s=0.7)
        if i > 0 and follower == "cav":
            kw.update(T_s=1.0, a_mps2=1.5, b_mps2=3.0, reaction_time_s=0.1)
            vs.append(CAVVehicle(**kw))
        else:
            vs.append(Vehicle(**kw))
    vs[0].v0_mps, vs[0].a_mps2 = v_lead, 3.0
    sim.lanes[1] = vs
    base, mins = None, [99.0] * 11
    for n in range(3000):
        t = n * 0.1
        if 200 <= t < 200 + brake_s:
            vs[0].forced_brake_timer_s = 0.2
        sim.step()
        if abs(t - 199.9) < 1e-6:
            base = [x.speed_mps for x in vs]
        if t >= 200:
            mins = [min(m, x.speed_mps) for m, x in zip(mins, vs)]
    return [b - m for b, m in zip(base, mins)]


def fig_string_stability(out):
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    x = np.arange(11)
    res = {}
    for name, col, lab in (("human", GREY, "human drivers (IDM, 0.7 s reaction)"),
                           ("cav", BLUE, "CAVs (our controller)")):
        d = platoon_dips(name)
        res[name] = d
        ax.plot(x, d, "-o", color=col, lw=2, ms=4, label=lab)
    ax.set(xlabel="position in the line (0 = leader that brakes)", ylabel="speed dip (m/s)", ylim=(0, None))
    ax.set_xticks(x)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / "fig6_string_stability.png", dpi=200)
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="output", help="folder holding the cav*_seed* run folders")
    ap.add_argument("--out", default="output/report")
    args = ap.parse_args()
    base, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = run_table(base)
    runs.to_csv(out / "table_runs.csv", index=False)
    fits = fd_fits(base)
    fits.to_csv(out / "table_fd_fits.csv", index=False)
    fig_layout(out)
    try:
        fig_simulator(out)
    except Exception as exc:  # pygame missing or no display driver
        print("simulator screenshot skipped:", exc)
    fig_headline(runs, out)
    fig_time_space(base, out)
    fig_fundamental(base, fits, out)
    fig_spot_speeds(base, out)
    dips = fig_string_stability(out)
    pd.DataFrame(dips).round(2).to_csv(out / "table_string_stability.csv", index_label="position")
    print(runs.groupby("cav_share").mean(numeric_only=True).round(3).to_string())
    print(fits.to_string())
    print("string stability dips (m/s):", {k: [round(v, 1) for v in d] for k, d in dips.items()})
    print(f"wrote figures and tables to {out}")


if __name__ == "__main__":
    main()
