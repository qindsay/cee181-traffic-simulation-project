"""Traffic-theory data export and presentation plots.

Two ways to use it:
  1. In the Pygame window press [P]: everything recorded so far is exported.
  2. Without a window:
         python traffic_analysis.py --minutes 20
         python traffic_analysis.py --minutes 20 --meter --disturbance 300 600
     (--disturbance = times in seconds when a lane-1 car brakes hard, like [B])

Output folder output/run_<date_time>/ contains
  CSV data:  trajectories.csv, detector_crossings.csv, edie_cells.csv,
             vehicles.csv, detector_observations.csv
  summary.txt with the fitted / measured numbers
  plots:
    1_fundamental_diagrams.png  q-k, u-k, u-q with Greenshields and Greenberg fits
    2_time_space.png            trajectories colored by speed (+ measured shockwave)
    3_cumulative_counts.png     N-curves upstream vs downstream (queueing / delay)
    4_headways.png              time-headway distributions vs Poisson (exponential)
    5_spot_speeds.png           spot-speed distribution, normal fit, 15/50/85th pct
    6_micro_fundamental.png     gap vs speed against the IDM equilibrium curve
    7_driver_parameters.png     sampled driver parameters vs their Gaussian design

Units: the simulator works in SI (m, s, m/s). Macroscopic plots use the units of
the lectures (veh/h, veh/mi, mph), with conversions defined below.

How density is measured: point detectors cannot measure density, so q, k and u
come from Edie's generalized definitions on space-time cells of each lane
(cell = CELL_LENGTH_M x CELL_DURATION_S):
    q = total distance travelled in the cell / cell area
    k = total time spent in the cell       / cell area
    u = q / k  (space-mean speed)
using the trajectories sampled every SAMPLE_INTERVAL_S.
"""
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from constants import vehicle_constants

# --- settings ---------------------------------------------------------------
SAMPLE_INTERVAL_S = 1.0      # trajectory sampling for export / Edie cells
CELL_LENGTH_M = 100.0        # Edie cell length
CELL_DURATION_S = 30.0       # Edie cell duration
WARMUP_S = 120.0             # ignored in statistics (road starts empty)
# (station position m, lanes) where crossings are recorded like a spot-speed study
STATIONS = {"upstream": (220.0, (0, 1)), "downstream": (850.0, (0, 1)), "ramp": (200.0, (2,))}

M_PER_MI = 1609.344
MPS_TO_MPH = 3600.0 / M_PER_MI

# Colors (validated categorical palette, light background)
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948"]


class TrafficRecorder:
    """Call record(sim) after every sim.step(). Cheap enough to leave on."""

    def __init__(self):
        self.traj: List[tuple] = []        # (t, id, lane, x, v, a, gap_to_leader or nan, type)
        self.crossings: List[tuple] = []   # (station, lane, t, id, v)
        self.vehicles: Dict[int, dict] = {}
        self.events: List[dict] = []
        self._last: Dict[int, tuple] = {}
        self.dt = None

    def mark_event(self, t: float, kind: str, lane: int, x: float):
        self.events.append({"time_s": t, "event": kind, "lane_id": lane, "position_m": x})

    def record(self, sim) -> None:
        t = sim.time_s
        self.dt = sim.dt_s
        ratio = t / SAMPLE_INTERVAL_S
        sample = abs(ratio - round(ratio)) < 1e-6
        current = {}
        for lane, vehs in sim.lanes.items():
            for i, v in enumerate(vehs):
                if v.vehicle_id not in self.vehicles:
                    self.vehicles[v.vehicle_id] = {
                        "vehicle_id": v.vehicle_id, "veh_type": v.veh_type, "origin_lane": lane,
                        "first_seen_s": round(t, 2), "driver_z_score": v.driver_z_score,
                        "v0_mps": v.v0_mps, "T_s": v.T_s, "a_mps2": v.a_mps2, "b_mps2": v.b_mps2,
                        "s0_m": v.s0_m, "politeness": v.politeness, "length_m": v.length_m}
                prev = self._last.get(v.vehicle_id)
                if prev is not None and prev[2] == lane and v.position_m > prev[0]:
                    for name, (x, lanes) in STATIONS.items():
                        if lane in lanes and prev[0] < x <= v.position_m:
                            f = (x - prev[0]) / (v.position_m - prev[0])
                            self.crossings.append((name, lane, t - sim.dt_s + f * sim.dt_s, v.vehicle_id,
                                                   prev[1] + f * (v.speed_mps - prev[1])))
                current[v.vehicle_id] = (v.position_m, v.speed_mps, lane)
                if sample:
                    gap = (vehs[i - 1].position_m - vehs[i - 1].length_m - v.position_m) if i > 0 else math.nan
                    self.traj.append((round(t, 3), v.vehicle_id, lane, v.position_m, v.speed_mps,
                                      v.acceleration_mps2, gap, v.veh_type))
        self._last = current


# ---------------------------------------------------------------- analysis
def edie_cells(rec: TrafficRecorder, road_length_m: float, lanes=(0, 1)) -> List[dict]:
    """Edie's q, k, u per lane on a CELL_LENGTH_M x CELL_DURATION_S grid."""
    acc: Dict[tuple, list] = {}
    for t, vid, lane, x, v, a, gap, vt in rec.traj:
        if lane not in lanes or t < WARMUP_S or x < 0 or x >= road_length_m:
            continue
        key = (lane, int(x // CELL_LENGTH_M), int(t // CELL_DURATION_S))
        cell = acc.setdefault(key, [0.0, 0.0])
        cell[0] += v * SAMPLE_INTERVAL_S          # distance travelled (m)
        cell[1] += SAMPLE_INTERVAL_S              # time spent (s)
    rows = []
    area = CELL_LENGTH_M * CELL_DURATION_S       # m*s
    for (lane, ix, it), (dist, tt) in sorted(acc.items()):
        q = dist / area                             # veh/s
        k = tt / area                               # veh/m
        rows.append({"lane_id": lane, "x_start_m": ix * CELL_LENGTH_M, "t_start_s": it * CELL_DURATION_S,
                     "flow_vph": q * 3600.0, "density_vpm": k * M_PER_MI,
                     "speed_mph": (q / k) * MPS_TO_MPH if k > 0 else math.nan})
    return rows


def fit_greenshields(k, u):
    """Linear u = uf - (uf/kj) k (lecture: regression of speed on density)."""
    slope, intercept = np.polyfit(k, u, 1)
    pred = intercept + slope * k
    r2 = 1 - np.sum((u - pred) ** 2) / np.sum((u - np.mean(u)) ** 2)
    uf, kj = intercept, -intercept / slope
    return {"uf_mph": uf, "kj_vpm": kj, "qmax_vph": uf * kj / 4, "kc_vpm": kj / 2,
            "uc_mph": uf / 2, "slope": slope, "r2": r2}


def fit_greenberg(k, u):
    """u = c ln(kj / k): linear regression of u on ln k."""
    slope, intercept = np.polyfit(np.log(k), u, 1)
    c = -slope
    kj = math.exp(intercept / c)
    pred = intercept + slope * np.log(k)
    r2 = 1 - np.sum((u - pred) ** 2) / np.sum((u - np.mean(u)) ** 2)
    return {"c_mph": c, "kj_vpm": kj, "kc_vpm": kj / math.e, "qmax_vph": c * kj / math.e, "r2": r2}


def measure_wave_speed(rec: TrafficRecorder, lane: int = 1, x_min: float = 200.0, x_max: float = 600.0,
                       bin_m: float = 25.0):
    """Average speed of the congestion (stop-and-go) waves, from the trajectories.

    1. Average speeds of lane `lane` in 25 m x 1 s bins between x_min and x_max
       (where the merge waves travel), smoothed over 10 s.
    2. For every candidate wave speed w between -1 and -15 m/s, shift each
       downstream speed series by the travel time of a wave to each upstream
       bin (100-250 m apart) and compute the average correlation.
    3. The w that lines the speed field up best is the measured wave speed
       (what you would read off the dark stripes in the time-space diagram).
    """
    t_vals = sorted({t for t, *_ in rec.traj if t > WARMUP_S})
    if len(t_vals) < 180:
        return None
    t_index = {t: i for i, t in enumerate(t_vals)}
    nb = int((x_max - x_min) // bin_m)
    sums = np.zeros((len(t_vals), nb))
    cnts = np.zeros((len(t_vals), nb))
    for t, vid, l, x, v, a, g, vt in rec.traj:
        if l == lane and t in t_index and x_min <= x < x_min + nb * bin_m:
            j = int((x - x_min) // bin_m)
            sums[t_index[t], j] += v
            cnts[t_index[t], j] += 1
    V = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    for j in range(nb):  # empty bin -> last seen speed
        col = V[:, j]
        last = np.nanmean(col) if np.any(~np.isnan(col)) else 0.0
        for i in range(len(col)):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]
    V = np.apply_along_axis(lambda c: np.convolve(c, np.ones(10) / 10, mode="same"), 0, V)
    A = V - V.mean(axis=0)
    candidates = np.arange(1.0, 15.01, 0.25)
    scores = []
    for w in candidates:
        rs = []
        for i in range(nb):
            for j in range(i + 1, nb):
                sep = (j - i) * bin_m
                if not 100.0 <= sep <= 250.0:
                    continue
                lag = int(round(sep / w))
                if lag < 1 or lag >= len(A) - 60:
                    continue
                x1, x2 = A[lag:, i], A[:-lag, j]
                den = np.sqrt(np.sum(x1 * x1) * np.sum(x2 * x2))
                if den > 0:
                    rs.append(np.sum(x1 * x2) / den)
        scores.append(np.mean(rs) if rs else -1.0)
    scores = np.array(scores)
    best = int(np.argmax(scores))
    if scores[best] < 0.3 or best in (0, len(candidates) - 1):
        return None  # no clear waves, or the best fit is at the edge of the search range
    w = -candidates[best]
    return {"w_mps": float(w), "w_mph": float(w) * MPS_TO_MPH, "score": float(scores[best])}


def regime_states(cells: List[dict], uf_mph: float):
    """Typical free-flowing (A) and congested (B) states from the Edie cells of
    both mainline lanes: A = cells faster than 80% of uf, B = slower than 40%."""
    sel = [c for c in cells if c["density_vpm"] > 1.0]
    A = [c for c in sel if c["speed_mph"] >= 0.8 * uf_mph]
    B = [c for c in sel if c["speed_mph"] <= 0.4 * uf_mph]
    if len(A) < 3 or len(B) < 3:
        return None
    qA, kA = np.median([c["flow_vph"] for c in A]), np.median([c["density_vpm"] for c in A])
    qB, kB = np.median([c["flow_vph"] for c in B]), np.median([c["density_vpm"] for c in B])
    return {"qA": qA, "kA": kA, "qB": qB, "kB": kB, "nA": len(A), "nB": len(B)}


# ---------------------------------------------------------------- export
def _write_csv(path: Path, header: List[str], rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def export(sim, rec: TrafficRecorder, out_dir: Optional[str] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")  # draw to files only (safe next to the Pygame window)
    out = Path(out_dir) if out_dir else Path("output") / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(out / "trajectories.csv",
               ["time_s", "vehicle_id", "lane_id", "position_m", "speed_mps", "acceleration_mps2",
                "gap_to_leader_m", "veh_type"],
               [(t, i, l, round(x, 3), round(v, 3), round(a, 3), "" if math.isnan(g) else round(g, 3), vt)
                for t, i, l, x, v, a, g, vt in rec.traj])
    _write_csv(out / "detector_crossings.csv", ["station", "lane_id", "time_s", "vehicle_id", "speed_mps"],
               [(s, l, round(t, 3), i, round(v, 3)) for s, l, t, i, v in rec.crossings])
    cells = edie_cells(rec, sim.length_m)
    _write_csv(out / "edie_cells.csv", list(cells[0].keys()) if cells else ["empty"],
               [[round(v, 3) if isinstance(v, float) else v for v in r.values()] for r in cells])
    vrows = list(rec.vehicles.values())
    if vrows:
        _write_csv(out / "vehicles.csv", list(vrows[0].keys()),
                   [[round(v, 4) if isinstance(v, float) else v for v in r.values()] for r in vrows])
    obs = sim.observation_log
    if obs:
        _write_csv(out / "detector_observations.csv", list(obs[0].keys()), [list(r.values()) for r in obs])

    summary = make_plots(sim, rec, cells, out)
    lines = [f"Traffic analysis summary - simulated time {sim.time_s:.0f} s "
             f"(first {WARMUP_S:.0f} s excluded), dt {sim.dt_s} s", ""]
    lines += [f"{k}: {v}" for k, v in summary.items()]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------- plots
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 1.8})


def _r(x, n=1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def make_plots(sim, rec: TrafficRecorder, cells: List[dict], out: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    _style(plt)
    summary = {}
    title_tail = f"(simulated {sim.time_s / 60:.0f} min, seed-based run)"

    # 1 ---- fundamental diagrams --------------------------------------------
    good = [c for c in cells if c["density_vpm"] > 1.0]
    if len(good) >= 5:
        k = np.array([c["density_vpm"] for c in good])
        q = np.array([c["flow_vph"] for c in good])
        u = np.array([c["speed_mph"] for c in good])
        gs = fit_greenshields(k, u)
        # Greenberg describes congested traffic, so it is fitted to cells denser
        # than the Greenshields critical density only.
        cong = k >= gs["kc_vpm"]
        gb = fit_greenberg(k[cong], u[cong]) if cong.sum() >= 10 else None
        kk = np.linspace(1, max(gs["kj_vpm"], k.max() * 1.1), 300)
        u_gs = np.clip(gs["uf_mph"] * (1 - kk / gs["kj_vpm"]), 0, None)
        curves = [(u_gs, kk, INK, "Greenshields (all cells)")]
        if gb is not None:
            kg = np.linspace(gs["kc_vpm"] * 0.6, max(gb["kj_vpm"], k.max()) , 200)
            curves.append((np.clip(gb["c_mph"] * np.log(gb["kj_vpm"] / kg), 0, None), kg, C[5],
                           "Greenberg (congested cells)"))
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
        lane_cols = {0: C[0], 1: C[1]}
        for lane in (0, 1):
            sel = [c for c in good if c["lane_id"] == lane]
            kw = dict(s=10, alpha=0.45, lw=0, color=lane_cols[lane], label=f"lane {lane} cells")
            ax[0].scatter([c["density_vpm"] for c in sel], [c["flow_vph"] for c in sel], **kw)
            ax[1].scatter([c["density_vpm"] for c in sel], [c["speed_mph"] for c in sel], **kw)
            ax[2].scatter([c["flow_vph"] for c in sel], [c["speed_mph"] for c in sel], **kw)
        for uu, kx, col, name in curves:
            ax[0].plot(kx, uu * kx, color=col, label=name)
            ax[1].plot(kx, uu, color=col, label=name)
            ax[2].plot(uu * kx, uu, color=col, label=name)
        ax[0].scatter([gs["kc_vpm"]], [gs["qmax_vph"]], marker="D", s=40, color=INK, zorder=4)
        ax[0].annotate(f"Greenshields capacity\nq_max = {gs['qmax_vph']:.0f} veh/h/lane\n"
                       f"at k_c = {gs['kc_vpm']:.0f} veh/mi/lane",
                       (gs["kc_vpm"], gs["qmax_vph"]), xytext=(28, -70), textcoords="offset points",
                       fontsize=8, ha="left", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax[0].set(xlabel="density k (veh/mi/lane)", ylabel="flow q (veh/h/lane)", title="Flow - density")
        ax[1].set(xlabel="density k (veh/mi/lane)", ylabel="space-mean speed u (mph)", title="Speed - density")
        ax[2].set(xlabel="flow q (veh/h/lane)", ylabel="space-mean speed u (mph)", title="Speed - flow")
        eq = f"Greenshields: u = {gs['uf_mph']:.1f} {gs['slope']:+.3f} k  (R² {gs['r2']:.2f})"
        if gb is not None:
            eq += f"\nGreenberg: u = {gb['c_mph']:.1f} ln({gb['kj_vpm']:.0f} / k)  (R² {gb['r2']:.2f})"
        ax[1].text(0.98, 0.97, eq, transform=ax[1].transAxes, ha="right", va="top", fontsize=8)
        for a in ax:
            a.set_xlim(left=0)
            a.set_ylim(bottom=0)
        ax[0].set_ylim(0, max(q.max(), gs["qmax_vph"]) * 1.25)
        ax[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Fundamental diagrams from simulated trajectories (Edie cells "
                     f"{CELL_LENGTH_M:.0f} m x {CELL_DURATION_S:.0f} s) {title_tail}",
                     x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.savefig(out / "1_fundamental_diagrams.png", dpi=140)
        plt.close(fig)
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

    # 2 ---- time-space + shockwaves -----------------------------------------
    wave = measure_wave_speed(rec)
    states = regime_states(cells, gs["uf_mph"]) if len(good) >= 5 else None
    cars = vehicle_constants.VEHICLE_DISTRIBUTION_PARAMETERS["car"]
    w_idm = -(cars["length_m"][0] + cars["s0_m"][0]) / cars["T_s"][0]   # m/s
    cmap = LinearSegmentedColormap.from_list(
        "spd", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"])
    norm = Normalize(0, 35 * MPS_TO_MPH)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
    for ax, lane in zip(axes, (0, 1)):
        by_v: Dict[int, list] = {}
        for t, vid, l, x, v, a, g, vt in rec.traj:
            if l == lane:
                by_v.setdefault(vid, []).append((t, x, v))
        segs, cols = [], []
        for pts in by_v.values():
            for p, n in zip(pts, pts[1:]):
                if n[0] - p[0] <= SAMPLE_INTERVAL_S * 1.01:
                    segs.append([(p[0], p[1]), (n[0], n[1])])
                    cols.append(p[2] * MPS_TO_MPH)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=0.6)
        lc.set_array(np.array(cols))
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
            # Reference wave lines, drawn from x = 600 m at the moment lane 1 is
            # slowest there: compare their slopes with the dark stripes.
            x0, x_stop = 600.0, 200.0   # the stretch where the merge waves are measured
            near = [(t, v) for t, vid, l, x, v, a, g, vt in rec.traj
                    if l == 1 and WARMUP_S < t < sim.time_s - 120 and abs(x - x0) < 25]
            lines = []
            if wave is not None:
                lines.append((wave["w_mps"], "-", f"measured {wave['w_mph']:.1f} mph"))
            if states is not None:
                w_dqdk = (states["qA"] - states["qB"]) / (states["kA"] - states["kB"]) / MPS_TO_MPH
                lines.append((w_dqdk, "--", f"Δq/Δk {w_dqdk * MPS_TO_MPH:.1f} mph"))
            lines.append((w_idm, ":", f"IDM -(l+s0)/T {w_idm * MPS_TO_MPH:.1f} mph"))
            if near:
                t_slow = min(near, key=lambda p: p[1])[0]
                for w, ls, label in lines:
                    if w >= -0.2:
                        continue  # not an upstream-moving wave; nothing to draw
                    t_end = min(t_slow + (x_stop - x0) / w, sim.time_s)
                    ax.plot([t_slow, t_end], [x0, x0 + w * (t_end - t_slow)], color=C[6], lw=2.0, ls=ls,
                            label=label)
                ax.legend(loc="upper right", fontsize=8, title="wave speed lines (start at 600 m)", title_fontsize=8,
                          framealpha=0.9, frameon=True, facecolor=BG, edgecolor=GRID)
    axes[1].set_xlabel("time t (s)")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("speed (mph), darker = slower")
    fig.suptitle(f"Time-space diagram (vehicle trajectories) {title_tail}", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.savefig(out / "2_time_space.png", dpi=140)
    plt.close(fig)
    if wave is not None:
        summary["Congestion wave speed measured, lane 1, 200-600 m (mph, negative = moving upstream)"] = _r(wave["w_mph"])
        summary["  fit quality (average correlation, 0-1)"] = _r(wave["score"], 2)
    else:
        summary["Congestion wave speed measured"] = ("not clear enough to measure automatically in this run; "
                                                     "compare the stripes in 2_time_space.png with the reference lines")
    if states is not None:
        w_dqdk = (states["qA"] - states["qB"]) / (states["kA"] - states["kB"])
        summary["Free-flow state A, median (q veh/h/lane, k veh/mi/lane)"] = f"({states['qA']:.0f}, {states['kA']:.1f}) from {states['nA']} cells"
        summary["Congested state B, median (q veh/h/lane, k veh/mi/lane)"] = f"({states['qB']:.0f}, {states['kB']:.1f}) from {states['nB']} cells"
        summary["Shockwave A->B predicted w = (qA-qB)/(kA-kB) (mph)"] = _r(w_dqdk)
    summary["IDM theory: congested wave speed -(length + s0)/T for the mean car (mph)"] = _r(w_idm * MPS_TO_MPH)

    # 3 ---- cumulative counts -----------------------------------------------
    up = sorted(t for s, l, t, i, v in rec.crossings if s in ("upstream", "ramp"))
    down = sorted(t for s, l, t, i, v in rec.crossings if s == "downstream")
    if up and down:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
        ax[0].step(up, np.arange(1, len(up) + 1), where="post", color=C[0],
                   label="N upstream (x = 220 m + ramp entrance)")
        ax[0].step(down, np.arange(1, len(down) + 1), where="post", color=C[1], label="N downstream (x = 850 m)")
        ax[0].set(xlabel="time t (s)", ylabel="cumulative vehicles N(t)",
                  title="Cumulative counts: vertical gap = vehicles in between")
        ax[0].legend(loc="upper left", fontsize=8)
        tg = np.arange(0, sim.time_s, 5.0)
        inside = np.searchsorted(up, tg, side="right") - np.searchsorted(down, tg, side="right")
        ax[1].plot(tg, inside, color=C[2])
        ax[1].set(xlabel="time t (s)", ylabel="vehicles between 220 m and 850 m",
                  title="Accumulation N_up(t) - N_down(t)")
        fig.suptitle("Cumulative vehicle counts (queueing view)", x=0.01, ha="left", fontsize=11,
                     fontweight="bold")
        fig.savefig(out / "3_cumulative_counts.png", dpi=140)
        plt.close(fig)

    # 4 ---- headways --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, station in zip(axes, ("upstream", "downstream")):
        h = []
        for lane in (0, 1):
            ts = sorted(t for s, l, t, i, v in rec.crossings if s == station and l == lane and t > WARMUP_S)
            h += list(np.diff(ts))
        h = np.array(h)
        if len(h) < 10:
            continue
        mean, sd = h.mean(), h.std(ddof=1)
        tau = np.percentile(h, 1)
        bins = np.arange(0, np.percentile(h, 98) + 1, 0.5)
        ax.hist(h, bins=bins, density=True, color=C[0], alpha=0.75, edgecolor=BG)
        hh = np.linspace(0, bins[-1], 300)
        ax.plot(hh, np.exp(-hh / mean) / mean, color=INK, label=f"exponential (Poisson), mean {mean:.2f} s")
        lam = 1 / (mean - tau)
        ax.plot(hh, np.where(hh >= tau, lam * np.exp(-lam * (hh - tau)), 0), color=C[1], ls="--",
                label=f"shifted exponential, min headway {tau:.2f} s")
        ax.set(xlabel="time headway h (s)", ylabel="probability density",
               title=f"{station.capitalize()} station (x = {STATIONS[station][0]:.0f} m), lanes 0+1")
        ax.legend(fontsize=8)
        ax.text(0.98, 0.6, f"n = {len(h)}\nmean = {mean:.2f} s\nSD = {sd:.2f} s\nCV = {sd / mean:.2f}\n"
                f"flow = {3600 / mean:.0f} veh/h/lane", transform=ax.transAxes, ha="right", fontsize=8)
        summary[f"Mean time headway {station} (s)"] = _r(mean, 2)
        summary[f"Headway coefficient of variation {station} (1.0 = Poisson)"] = _r(sd / mean, 2)
    fig.suptitle("Time-headway distributions", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(out / "4_headways.png", dpi=140)
    plt.close(fig)

    # 5 ---- spot speeds -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, station in zip(axes, ("upstream", "downstream")):
        sp = np.array([v * MPS_TO_MPH for s, l, t, i, v in rec.crossings if s == station and t > WARMUP_S])
        if len(sp) < 10:
            continue
        mu, sd = sp.mean(), sp.std(ddof=1)
        p15, p50, p85 = np.percentile(sp, [15, 50, 85])
        ax.hist(sp, bins=30, density=True, color=C[2], alpha=0.75, edgecolor=BG)
        xx = np.linspace(sp.min(), sp.max(), 300)
        ax.plot(xx, np.exp(-0.5 * ((xx - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)), color=INK,
                label=f"normal fit: mean {mu:.1f}, SD {sd:.1f} mph")
        for p, name, yfrac in ((p15, "15th", 0.97), (p50, "50th", 0.80), (p85, "85th", 0.97)):
            ax.axvline(p, color=C[1] if name == "85th" else INK2, lw=1.2, ls="--")
            ax.text(p, ax.get_ylim()[1] * yfrac, f" {name}\n {p:.1f}", fontsize=8, va="top",
                    bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none", alpha=0.8))
        ax.set(xlabel="spot speed (mph)", ylabel="probability density",
               title=f"{station.capitalize()} station (x = {STATIONS[station][0]:.0f} m), n = {len(sp)}")
        ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.55))
        summary[f"Spot speed {station}: mean / SD (mph)"] = f"{mu:.1f} / {sd:.1f}"
        summary[f"Spot speed {station}: 15th / 50th / 85th percentile (mph)"] = f"{p15:.1f} / {p50:.1f} / {p85:.1f}"
    fig.suptitle("Spot-speed study (time-mean speeds at a point)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.savefig(out / "5_spot_speeds.png", dpi=140)
    plt.close(fig)

    # 6 ---- microscopic fundamental diagram ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for vt, col in (("car", C[0]), ("truck", C[1])):
        g = np.array([(gap, v) for t, i, l, x, v, a, gap, typ in rec.traj
                      if typ == vt and l in (0, 1) and not math.isnan(gap) and t > WARMUP_S and gap < 150])
        if len(g):
            ax.scatter(g[:, 0], g[:, 1] * MPS_TO_MPH, s=4, alpha=0.25, lw=0, color=col, label=f"{vt}s (simulated)")
        p = {n: vals[0] for n, vals in vehicle_constants.VEHICLE_DISTRIBUTION_PARAMETERS[vt].items()}
        vv = np.linspace(0, p["v0_mps"] * 0.995, 300)
        se = (p["s0_m"] + vv * p["T_s"]) / np.sqrt(1 - (vv / p["v0_mps"]) ** vehicle_constants.IDM_DELTA)
        ax.plot(se, vv * MPS_TO_MPH, color=col, lw=2.2, label=f"IDM equilibrium, mean {vt}")
    ax.set(xlim=(0, 150), ylim=(0, None), xlabel="gap to leader s (m)", ylabel="speed v (mph)",
           title="Microscopic fundamental diagram: equilibrium speed v_e(s) vs simulated states")
    ax.legend(fontsize=8, loc="lower right", markerscale=3)
    fig.savefig(out / "6_micro_fundamental.png", dpi=140)
    plt.close(fig)

    # 7 ---- driver parameter distributions (z-score design) ----------------
    vehs = [v for v in rec.vehicles.values() if v["veh_type"] == "car"]
    if len(vehs) >= 20:
        names = [("v0_mps", "desired speed v0 (m/s)"), ("T_s", "time headway T (s)"),
                 ("a_mps2", "max acceleration a (m/s²)"), ("b_mps2", "comfortable decel b (m/s²)"),
                 ("s0_m", "jam gap s0 (m)")]
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
        z = np.array([v["driver_z_score"] for v in vehs])
        ax = axes[0, 0]
        ax.hist(z, bins=25, density=True, color=C[5], alpha=0.75, edgecolor=BG)
        zz = np.linspace(-3, 3, 200)
        ax.plot(zz, np.exp(-zz ** 2 / 2) / math.sqrt(2 * math.pi), color=INK, label="N(0, 1)")
        ax.set(title="Driver z-score (all cars)", xlabel="z")
        ax.legend(fontsize=8)
        weights = {"v0_mps": 1, "T_s": -1, "a_mps2": 1, "b_mps2": None, "s0_m": -0.5}
        for ax, (key, label) in zip(axes.flat[1:], names):
            vals = np.array([v[key] for v in vehs])
            mean, std, lo, hi = vehicle_constants.VEHICLE_DISTRIBUTION_PARAMETERS["car"][key]
            w = weights[key]
            eff_std = std * (abs(w) if w is not None else 1)
            ax.hist(vals, bins=25, density=True, color=C[0], alpha=0.75, edgecolor=BG)
            xx = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(xx, np.exp(-0.5 * ((xx - mean) / eff_std) ** 2) / (eff_std * math.sqrt(2 * math.pi)),
                    color=INK, label=f"design N({mean}, {eff_std:.2f}²)")
            ax.set(title=label, xlabel=label.split(" (")[0])
            ax.text(0.98, 0.95, f"sample mean {vals.mean():.2f}\nsample SD {vals.std(ddof=1):.2f}\n"
                    f"85th pct {np.percentile(vals, 85):.2f}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=8)
            ax.legend(fontsize=7.5, loc="upper left")
        fig.suptitle(f"Generated car drivers (n = {len(vehs)}): samples vs the Gaussian design "
                     "(v0, T, a, s0 all follow one driver z-score)", x=0.01, ha="left", fontsize=11,
                     fontweight="bold")
        fig.savefig(out / "7_driver_parameters.png", dpi=140)
        plt.close(fig)
        summary["Cars generated"] = len(vehs)
    return summary


# ---------------------------------------------------------------- headless run
def main():
    from simulator import HighwaySimulator
    from constants import simulation_constants
    ap = argparse.ArgumentParser(description="Run the simulator without a window and export plots/data.")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=simulation_constants.RANDOM_SEED)
    ap.add_argument("--dt", type=float, default=simulation_constants.TIME_STEP_S)
    ap.add_argument("--meter", action="store_true", help="turn the ramp meter on")
    ap.add_argument("--disturbance", type=float, nargs="*", default=[],
                    help="times (s) when a lane-1 vehicle brakes hard, like pressing [B]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sim = HighwaySimulator(dt_s=args.dt, seed=args.seed)
    sim.ramp_meter_enabled = args.meter
    rec = TrafficRecorder()
    pending = sorted(args.disturbance)
    n_steps = int(round(args.minutes * 60 / args.dt))
    for k in range(n_steps):
        if pending and sim.time_s >= pending[0] - 1e-9:
            pending.pop(0)
            trigger_disturbance(sim, rec)
        sim.step()
        rec.record(sim)
    out = export(sim, rec, args.out)
    print(f"Exported data and plots to {out}/")
    print((out / "summary.txt").read_text())


def trigger_disturbance(sim, rec: TrafficRecorder) -> Optional[int]:
    """Same as pressing [B]; also remembers where/when for the shockwave analysis."""
    vid = sim.trigger_lead_disturbance()
    if vid is not None:
        veh = next(v for v in sim.lanes[1] if v.vehicle_id == vid)
        rec.mark_event(sim.time_s, "disturbance", 1, veh.position_m)
    return vid


if __name__ == "__main__":
    main()
