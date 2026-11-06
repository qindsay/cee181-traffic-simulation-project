import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from analysis import analyze_traffic
from constants import analysis_constants, vehicle_constants

# Colors (validated categorical palette, light background)
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948"]
DPI = 140


# ---------------------------------------------------------------- plots
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 1.8})


def _r(x, n=1):
    """Round for summary.txt; None for missing values."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def _normal_pdf(x, mean, sd):
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def _save(plt, fig, path: Path, title: Optional[str] = None) -> None:
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)

def make_plots(sim, rec: analyze_traffic.TrafficRecorder, cells: List[dict], out: Path) -> dict:
    """Draw all figures into `out`; returns the numbers for summary.txt (in order)."""
    import matplotlib.pyplot as plt
    _style(plt)
    summary = {}
    run_label = f"(simulated {sim.time_s / 60:.0f} min, seed-based run)"

    greenshields = plot_fundamental_diagrams(plt, cells, out, run_label, summary)
    states = analyze_traffic.regime_states(cells, greenshields["uf_mph"]) if greenshields else None
    plot_time_space(plt, sim, rec, states, out, run_label, summary)
    plot_cumulative_counts(plt, sim, rec, out)
    plot_headways(plt, rec, out, summary)
    plot_spot_speeds(plt, rec, out, summary)
    plot_micro_fundamental(plt, rec, out)
    plot_driver_parameters(plt, rec, out, summary)
    return summary


def plot_fundamental_diagrams(plt, cells, out, run_label, summary) -> Optional[dict]:
    """Figure 1: q-k, u-k, u-q scatter of Edie cells with Greenshields and Greenberg
    fits. Returns the Greenshields fit (None if there are too few cells)."""
    occupied = analyze_traffic.occupied_cells(cells)
    if len(occupied) < 5:
        return None
    k = np.array([c["density_vpm"] for c in occupied])
    q = np.array([c["flow_vph"] for c in occupied])
    u = np.array([c["speed_mph"] for c in occupied])
    gs = analyze_traffic.fit_greenshields(k, u)
    # Greenberg describes congested traffic, so it is fitted to cells denser
    # than the Greenshields critical density only.
    congested = k >= gs["kc_vpm"]
    gb = analyze_traffic.fit_greenberg(k[congested], u[congested]) if congested.sum() >= 10 else None

    # model curves as (speed array, density array, color, label)
    k_gs = np.linspace(1, max(gs["kj_vpm"], k.max() * 1.1), 300)
    curves = [(np.clip(gs["uf_mph"] * (1 - k_gs / gs["kj_vpm"]), 0, None), k_gs, INK, "Greenshields (all cells)")]
    if gb is not None:
        k_gb = np.linspace(gs["kc_vpm"] * 0.6, max(gb["kj_vpm"], k.max()), 200)
        curves.append((np.clip(gb["c_mph"] * np.log(gb["kj_vpm"] / k_gb), 0, None), k_gb, C[5],
                       "Greenberg (congested cells)"))

    fig, (ax_qk, ax_uk, ax_uq) = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for lane in analysis_constants.MAINLINE_LANES:
        sel = [c for c in occupied if c["lane_id"] == lane]
        lane_k = [c["density_vpm"] for c in sel]
        lane_q = [c["flow_vph"] for c in sel]
        lane_u = [c["speed_mph"] for c in sel]
        kw = dict(s=10, alpha=0.45, lw=0, color=C[lane], label=f"lane {lane} cells")
        ax_qk.scatter(lane_k, lane_q, **kw)
        ax_uk.scatter(lane_k, lane_u, **kw)
        ax_uq.scatter(lane_q, lane_u, **kw)
    for cu, ck, col, name in curves:
        ax_qk.plot(ck, cu * ck, color=col, label=name)
        ax_uk.plot(ck, cu, color=col, label=name)
        ax_uq.plot(cu * ck, cu, color=col, label=name)

    ax_qk.scatter([gs["kc_vpm"]], [gs["qmax_vph"]], marker="D", s=40, color=INK, zorder=4)
    ax_qk.annotate(f"Greenshields capacity\nq_max = {gs['qmax_vph']:.0f} veh/h/lane\n"
                   f"at k_c = {gs['kc_vpm']:.0f} veh/mi/lane",
                   (gs["kc_vpm"], gs["qmax_vph"]), xytext=(28, -70), textcoords="offset points",
                   fontsize=8, ha="left", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax_qk.set(xlabel="density k (veh/mi/lane)", ylabel="flow q (veh/h/lane)", title="Flow - density")
    ax_uk.set(xlabel="density k (veh/mi/lane)", ylabel="space-mean speed u (mph)", title="Speed - density")
    ax_uq.set(xlabel="flow q (veh/h/lane)", ylabel="space-mean speed u (mph)", title="Speed - flow")
    equations = f"Greenshields: u = {gs['uf_mph']:.1f} {gs['slope']:+.3f} k  (R² {gs['r2']:.2f})"
    if gb is not None:
        equations += f"\nGreenberg: u = {gb['c_mph']:.1f} ln({gb['kj_vpm']:.0f} / k)  (R² {gb['r2']:.2f})"
    ax_uk.text(0.98, 0.97, equations, transform=ax_uk.transAxes, ha="right", va="top", fontsize=8)
    for ax in (ax_qk, ax_uk, ax_uq):
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    ax_qk.set_ylim(0, max(q.max(), gs["qmax_vph"]) * 1.25)
    ax_qk.legend(loc="upper left", fontsize=8)
    _save(plt, fig, out / "1_fundamental_diagrams.png",
          f"Fundamental diagrams from simulated trajectories (Edie cells "
          f"{analysis_constants.CELL_LENGTH_M:.0f} m x {analysis_constants.CELL_DURATION_S:.0f} s) {run_label}")

    summary.update({
        "Greenshields free-flow speed uf (mph)": _r(gs["uf_mph"]),
        "Greenshields jam density kj (veh/mi/lane)": _r(gs["kj_vpm"]),
        "Greenshields capacity qmax = uf*kj/4 (veh/h/lane)": _r(gs["qmax_vph"], 0),
        "Greenshields critical density kc = kj/2 (veh/mi/lane)": _r(gs["kc_vpm"]),
        "Greenshields R^2": _r(gs["r2"], 3),
        "Highest observed cell flow (veh/h/lane)": _r(q.max(), 0),
        "Density at highest observed flow (veh/mi/lane)": _r(k[int(np.argmax(q))]),
    })
    if gb is not None:
        summary.update({
            "Greenberg (congested cells) c (mph)": _r(gb["c_mph"]),
            "Greenberg jam density kj (veh/mi/lane)": _r(gb["kj_vpm"]),
            "Greenberg capacity qmax = c*kj/e (veh/h/lane)": _r(gb["qmax_vph"], 0),
            "Greenberg R^2": _r(gb["r2"], 3)})
    return gs


def plot_time_space(plt, sim, rec, states, out, run_label, summary) -> None:
    """Figure 2: trajectories of both mainline lanes colored by speed, with
    reference lines for the measured, Δq/Δk and IDM wave speeds on lane 1."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    wave = analyze_traffic.measure_wave_speed(rec)
    w_idm = analyze_traffic.idm_wave_speed_mps()
    cmap = LinearSegmentedColormap.from_list(
        "spd", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"])
    norm = Normalize(0, 35 * analysis_constants.MPS_TO_MPH)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
    for ax, lane in zip(axes, analysis_constants.MAINLINE_LANES):
        segments, segment_speeds = _trajectory_segments(rec, lane)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=0.6)
        lc.set_array(np.array(segment_speeds))
        ax.add_collection(lc)
        ax.set_xlim(0, sim.time_s)
        ax.set_ylim(0, sim.length_m)
        ax.grid(False)
        ax.axhspan(sim.merge_zone_start_m, sim.merge_zone_end_m, color=C[3], alpha=0.07, lw=0)
        ax.set_ylabel("position x (m)")
        ax.set_title(f"Lane {lane} ({'left' if lane == 0 else 'right, next to the on-ramp'}); "
                     "shaded = merge zone", loc="left")
        for e in rec.events:
            if e["lane_id"] == lane:
                ax.plot(e["time_s"], e["position_m"], "v", color=C[6], ms=9, mec=BG, mew=1.5)
        if lane == 1:
            _draw_wave_lines(ax, sim, rec, wave, states, w_idm)
    axes[1].set_xlabel("time t (s)")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("speed (mph), darker = slower")
    _save(plt, fig, out / "2_time_space.png", f"Time-space diagram (vehicle trajectories) {run_label}")

    if wave is not None:
        summary["Congestion wave speed measured, lane 1, 200-600 m (mph, negative = moving upstream)"] = _r(wave["w_mph"])
        summary["  fit quality (average correlation, 0-1)"] = _r(wave["score"], 2)
    else:
        summary["Congestion wave speed measured"] = ("not clear enough to measure automatically in this run; "
                                                     "compare the stripes in 2_time_space.png with the reference lines")
    if states is not None:
        summary["Free-flow state A, median (q veh/h/lane, k veh/mi/lane)"] = f"({states['qA']:.0f}, {states['kA']:.1f}) from {states['nA']} cells"
        summary["Congested state B, median (q veh/h/lane, k veh/mi/lane)"] = f"({states['qB']:.0f}, {states['kB']:.1f}) from {states['nB']} cells"
        summary["Shockwave A->B predicted w = (qA-qB)/(kA-kB) (mph)"] = _r(analyze_traffic.shockwave_speed_mph(states))
    summary["IDM theory: congested wave speed -(length + s0)/T for the mean car (mph)"] = _r(w_idm * analysis_constants.MPS_TO_MPH)


def _trajectory_segments(rec, lane):
    """Line segments (t, x) -> (t, x) between consecutive samples of each vehicle
    in `lane`, and the speed (mph) at the start of each segment."""
    by_vehicle: Dict[int, list] = {}
    for s in rec.traj:
        if s.lane_id == lane:
            by_vehicle.setdefault(s.vehicle_id, []).append(s)
    segments, speeds = [], []
    for samples in by_vehicle.values():
        for a, b in zip(samples, samples[1:]):
            if b.time_s - a.time_s <= analysis_constants.SAMPLE_INTERVAL_S * 1.01:   # skip jumps (lane change away and back)
                segments.append([(a.time_s, a.position_m), (b.time_s, b.position_m)])
                speeds.append(a.speed_mps * analysis_constants.MPS_TO_MPH)
    return segments, speeds


def _draw_wave_lines(ax, sim, rec, wave, states, w_idm) -> None:
    """Reference wave lines, drawn from x = 600 m at the moment lane 1 is
    slowest there: compare their slopes with the dark stripes."""
    x_start, x_stop = 600.0, 200.0   # the stretch where the merge waves are measured
    near_start = [s for s in rec.traj
                  if s.lane_id == 1 and analysis_constants.WARMUP_S < s.time_s < sim.time_s - 120
                  and abs(s.position_m - x_start) < 25]
    lines = []   # (wave speed m/s, line style, label)
    if wave is not None:
        lines.append((wave["w_mps"], "-", f"measured {wave['w_mph']:.1f} mph"))
    if states is not None:
        w_dqdk = analyze_traffic.shockwave_speed_mph(states) / analysis_constants.MPS_TO_MPH
        lines.append((w_dqdk, "--", f"Δq/Δk {w_dqdk * analysis_constants.MPS_TO_MPH:.1f} mph"))
    lines.append((w_idm, ":", f"IDM -(l+s0)/T {w_idm * analysis_constants.MPS_TO_MPH:.1f} mph"))
    if not near_start:
        return
    t_start = min(near_start, key=lambda s: s.speed_mps).time_s
    for w, ls, label in lines:
        if w >= -0.2:
            continue  # not an upstream-moving wave; nothing to draw
        t_end = min(t_start + (x_stop - x_start) / w, sim.time_s)
        ax.plot([t_start, t_end], [x_start, x_start + w * (t_end - t_start)], color=C[6], lw=2.0, ls=ls,
                label=label)
    ax.legend(loc="upper right", fontsize=8, title="wave speed lines (start at 600 m)", title_fontsize=8,
              framealpha=0.9, frameon=True, facecolor=BG, edgecolor=GRID)


def plot_cumulative_counts(plt, sim, rec, out) -> None:
    """Figure 3: N-curves at the upstream (+ ramp) and downstream stations and
    the number of vehicles between them."""
    up = sorted(c.time_s for c in rec.crossings if c.station in ("upstream", "ramp"))
    down = sorted(c.time_s for c in rec.crossings if c.station == "downstream")
    if not (up and down):
        return
    fig, (ax_n, ax_inside) = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
    ax_n.step(up, np.arange(1, len(up) + 1), where="post", color=C[0],
              label="N upstream (x = 220 m + ramp entrance)")
    ax_n.step(down, np.arange(1, len(down) + 1), where="post", color=C[1], label="N downstream (x = 850 m)")
    ax_n.set(xlabel="time t (s)", ylabel="cumulative vehicles N(t)",
             title="Cumulative counts: vertical gap = vehicles in between")
    ax_n.legend(loc="upper left", fontsize=8)
    t_grid = np.arange(0, sim.time_s, 5.0)
    inside = np.searchsorted(up, t_grid, side="right") - np.searchsorted(down, t_grid, side="right")
    ax_inside.plot(t_grid, inside, color=C[2])
    ax_inside.set(xlabel="time t (s)", ylabel="vehicles between 220 m and 850 m",
                  title="Accumulation N_up(t) - N_down(t)")
    _save(plt, fig, out / "3_cumulative_counts.png", "Cumulative vehicle counts (queueing view)")


def plot_headways(plt, rec, out, summary) -> None:
    """Figure 4: time headways at each mainline station vs exponential and
    shifted-exponential distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, station in zip(axes, ("upstream", "downstream")):
        h = []
        for lane in analysis_constants.MAINLINE_LANES:   # headways are measured per lane, then pooled
            times = sorted(c.time_s for c in rec.crossings
                           if c.station == station and c.lane_id == lane and c.time_s > analysis_constants.WARMUP_S)
            h += list(np.diff(times))
        h = np.array(h)
        if len(h) < 10:
            continue
        mean, sd = h.mean(), h.std(ddof=1)
        min_headway = np.percentile(h, 1)
        bins = np.arange(0, np.percentile(h, 98) + 1, 0.5)
        ax.hist(h, bins=bins, density=True, color=C[0], alpha=0.75, edgecolor=BG)
        hh = np.linspace(0, bins[-1], 300)
        ax.plot(hh, np.exp(-hh / mean) / mean, color=INK, label=f"exponential (Poisson), mean {mean:.2f} s")
        lam = 1 / (mean - min_headway)
        ax.plot(hh, np.where(hh >= min_headway, lam * np.exp(-lam * (hh - min_headway)), 0), color=C[1], ls="--",
                label=f"shifted exponential, min headway {min_headway:.2f} s")
        ax.set(xlabel="time headway h (s)", ylabel="probability density",
               title=f"{station.capitalize()} station (x = {analysis_constants.STATIONS[station][0]:.0f} m), lanes 0+1")
        ax.legend(fontsize=8)
        ax.text(0.98, 0.6, f"n = {len(h)}\nmean = {mean:.2f} s\nSD = {sd:.2f} s\nCV = {sd / mean:.2f}\n"
                f"flow = {3600 / mean:.0f} veh/h/lane", transform=ax.transAxes, ha="right", fontsize=8)
        summary[f"Mean time headway {station} (s)"] = _r(mean, 2)
        summary[f"Headway coefficient of variation {station} (1.0 = Poisson)"] = _r(sd / mean, 2)
    _save(plt, fig, out / "4_headways.png", "Time-headway distributions")


def plot_spot_speeds(plt, rec, out, summary) -> None:
    """Figure 5: spot speeds at each mainline station with a normal fit and
    the 15th / 50th / 85th percentiles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, station in zip(axes, ("upstream", "downstream")):
        sp = np.array([c.speed_mps * analysis_constants.MPS_TO_MPH for c in rec.crossings
                       if c.station == station and c.time_s > analysis_constants.WARMUP_S])
        if len(sp) < 10:
            continue
        mu, sd = sp.mean(), sp.std(ddof=1)
        p15, p50, p85 = np.percentile(sp, [15, 50, 85])
        ax.hist(sp, bins=30, density=True, color=C[2], alpha=0.75, edgecolor=BG)
        xx = np.linspace(sp.min(), sp.max(), 300)
        ax.plot(xx, _normal_pdf(xx, mu, sd), color=INK, label=f"normal fit: mean {mu:.1f}, SD {sd:.1f} mph")
        for p, name, y_frac in ((p15, "15th", 0.97), (p50, "50th", 0.80), (p85, "85th", 0.97)):
            ax.axvline(p, color=C[1] if name == "85th" else INK2, lw=1.2, ls="--")
            ax.text(p, ax.get_ylim()[1] * y_frac, f" {name}\n {p:.1f}", fontsize=8, va="top",
                    bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none", alpha=0.8))
        ax.set(xlabel="spot speed (mph)", ylabel="probability density",
               title=f"{station.capitalize()} station (x = {analysis_constants.STATIONS[station][0]:.0f} m), n = {len(sp)}")
        ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.55))
        summary[f"Spot speed {station}: mean / SD (mph)"] = f"{mu:.1f} / {sd:.1f}"
        summary[f"Spot speed {station}: 15th / 50th / 85th percentile (mph)"] = f"{p15:.1f} / {p50:.1f} / {p85:.1f}"
    _save(plt, fig, out / "5_spot_speeds.png", "Spot-speed study (time-mean speeds at a point)")


def plot_micro_fundamental(plt, rec, out) -> None:
    """Figure 6: simulated (gap, speed) states vs the IDM equilibrium curve
    s_e(v) = (s0 + v T) / sqrt(1 - (v / v0)^delta) of the mean car and truck."""
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for veh_type, col in (("car", C[0]), ("truck", C[1])):
        states = np.array([(s.gap_to_leader_m, s.speed_mps) for s in rec.traj
                           if s.veh_type == veh_type and s.lane_id in analysis_constants.MAINLINE_LANES
                           and not math.isnan(s.gap_to_leader_m) and s.time_s > analysis_constants.WARMUP_S
                           and s.gap_to_leader_m < 150])
        if len(states):
            ax.scatter(states[:, 0], states[:, 1] * analysis_constants.MPS_TO_MPH, s=4, alpha=0.25, lw=0, color=col,
                       label=f"{veh_type}s (simulated)")
        p = {n: vals[0] for n, vals in vehicle_constants.VEHICLE_DISTRIBUTION_PARAMETERS[veh_type].items()}
        vv = np.linspace(0, p["v0_mps"] * 0.995, 300)
        s_eq = (p["s0_m"] + vv * p["T_s"]) / np.sqrt(1 - (vv / p["v0_mps"]) ** vehicle_constants.IDM_DELTA)
        ax.plot(s_eq, vv * analysis_constants.MPS_TO_MPH, color=col, lw=2.2, label=f"IDM equilibrium, mean {veh_type}")
    ax.set(xlim=(0, 150), ylim=(0, None), xlabel="gap to leader s (m)", ylabel="speed v (mph)",
           title="Microscopic fundamental diagram: equilibrium speed v_e(s) vs simulated states")
    ax.legend(fontsize=8, loc="lower right", markerscale=3)
    _save(plt, fig, out / "6_micro_fundamental.png")


# How strongly each car parameter follows the driver z-score: the design SD of
# the parameter is multiplied by this factor (b is drawn independently).
Z_SCORE_SD_FACTOR = {"v0_mps": 1.0, "T_s": 1.0, "a_mps2": 1.0, "b_mps2": 1.0, "s0_m": 0.5}


def plot_driver_parameters(plt, rec, out, summary) -> None:
    """Figure 7: sampled car-driver parameters vs their Gaussian design."""
    cars = [v for v in rec.vehicles.values() if v["veh_type"] == "car"]
    if len(cars) < 20:
        return
    params = [("v0_mps", "desired speed v0 (m/s)"), ("T_s", "time headway T (s)"),
              ("a_mps2", "max acceleration a (m/s²)"), ("b_mps2", "comfortable decel b (m/s²)"),
              ("s0_m", "jam gap s0 (m)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)

    ax = axes[0, 0]
    z = np.array([v["driver_z_score"] for v in cars])
    ax.hist(z, bins=25, density=True, color=C[5], alpha=0.75, edgecolor=BG)
    zz = np.linspace(-3, 3, 200)
    ax.plot(zz, _normal_pdf(zz, 0, 1), color=INK, label="N(0, 1)")
    ax.set(title="Driver z-score (all cars)", xlabel="z")
    ax.legend(fontsize=8)

    for ax, (key, label) in zip(axes.flat[1:], params):
        vals = np.array([v[key] for v in cars])
        mean, std, _lo, _hi = vehicle_constants.VEHICLE_DISTRIBUTION_PARAMETERS["car"][key]
        design_sd = std * Z_SCORE_SD_FACTOR[key]
        ax.hist(vals, bins=25, density=True, color=C[0], alpha=0.75, edgecolor=BG)
        xx = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xx, _normal_pdf(xx, mean, design_sd), color=INK, label=f"design N({mean}, {design_sd:.2f}²)")
        ax.set(title=label, xlabel=label.split(" (")[0])
        ax.text(0.98, 0.95, f"sample mean {vals.mean():.2f}\nsample SD {vals.std(ddof=1):.2f}\n"
                f"85th pct {np.percentile(vals, 85):.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8)
        ax.legend(fontsize=7.5, loc="upper left")
    _save(plt, fig, out / "7_driver_parameters.png",
          f"Generated car drivers (n = {len(cars)}): samples vs the Gaussian design "
          "(v0, T, a, s0 all follow one driver z-score)")
    summary["Cars generated"] = len(cars)