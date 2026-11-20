
import math
from typing import Dict, List, NamedTuple, Optional
import numpy as np

from constants import vehicle_constants, analysis_constants

# ---------------------------------------------------------------- recorder
class TrajectorySample(NamedTuple):
    """One vehicle at one sampled time (one row of trajectories.csv)."""
    time_s: float
    vehicle_id: int
    lane_id: int
    position_m: float
    speed_mps: float
    acceleration_mps2: float
    gap_to_leader_m: float     # nan for the first vehicle in its lane
    veh_type: str


class Crossing(NamedTuple):
    """A vehicle passing a detector station (one row of detector_crossings.csv)."""
    station: str
    lane_id: int
    time_s: float
    vehicle_id: int
    speed_mps: float


class TrafficRecorder:
    """Call record(sim) after every sim.step(). Cheap enough to leave on."""

    def __init__(self):
        self.traj: List[TrajectorySample] = []
        self.crossings: List[Crossing] = []
        self.vehicles: Dict[int, dict] = {}   # vehicle_id -> parameters at first sight
        self.events: List[dict] = []          # disturbances, for marking the plots
        self._last: Dict[int, tuple] = {}     # vehicle_id -> (position, speed, lane) last step
        self.dt = None

    def mark_event(self, t: float, kind: str, lane: int, x: float):
        self.events.append({"time_s": t, "event": kind, "lane_id": lane, "position_m": x})

    def record(self, sim) -> None:
        """Record the simulator state; call once after every sim.step().

        Every step, vehicles seen for the first time are added to
        self.vehicles, and any detector station a vehicle passed during the
        step is logged in self.crossings. When the time is a multiple of
        analysis_constants.SAMPLE_INTERVAL_S, one TrajectorySample per vehicle (including the gap
        to its leader in the same lane) is also appended to self.traj.

        Args:
            sim (HighwaySimulator): the simulator, read right after its step;
                uses time_s, dt_s and lanes (lane id -> vehicles ordered from
                the front of the lane to the back).
        """
        t = sim.time_s
        self.dt = sim.dt_s
        ratio = t / analysis_constants.SAMPLE_INTERVAL_S
        is_sample_time = abs(ratio - round(ratio)) < 1e-6
        current = {}
        for lane, vehs in sim.lanes.items():
            for i, v in enumerate(vehs):
                if v.vehicle_id not in self.vehicles:
                    self._register_vehicle(v, lane, t)
                prev = self._last.get(v.vehicle_id)
                if prev is not None:
                    self._record_crossings(v, lane, prev, t, sim.dt_s)
                current[v.vehicle_id] = (v.position_m, v.speed_mps, lane)
                if is_sample_time:
                    leader = vehs[i - 1] if i > 0 else None
                    gap = leader.position_m - leader.length_m - v.position_m if leader else math.nan
                    self.traj.append(
                        TrajectorySample(
                            round(t, 3), 
                            v.vehicle_id, 
                            lane, 
                            v.position_m,
                            v.speed_mps, 
                            v.acceleration_mps2, 
                            gap, 
                            v.veh_type))
        self._last = current

    def _register_vehicle(self, v, lane: int, t: float) -> None:
        self.vehicles[v.vehicle_id] = {
            "vehicle_id": v.vehicle_id, "veh_type": v.veh_type, "origin_lane": lane,
            "first_seen_s": round(t, 2), "driver_z_score": v.driver_z_score,
            "v0_mps": v.v0_mps, "T_s": v.T_s, "a_mps2": v.a_mps2, "b_mps2": v.b_mps2,
            "s0_m": v.s0_m, "politeness": v.politeness, "length_m": v.length_m,
            "reaction_time_s": v.reaction_time_s}

    def _record_crossings(self, v, lane: int, prev: tuple, t: float, dt: float) -> None:
        """Log every station passed during the last step, interpolating the
        crossing time and speed linearly within the step."""
        prev_x, prev_v, prev_lane = prev
        if prev_lane != lane or v.position_m <= prev_x:
            return
        for name, (station_x, lanes) in analysis_constants.STATIONS.items():
            if lane in lanes and prev_x < station_x <= v.position_m:
                frac = (station_x - prev_x) / (v.position_m - prev_x)
                self.crossings.append(Crossing(name, lane, t - dt + frac * dt, v.vehicle_id,
                                               prev_v + frac * (v.speed_mps - prev_v)))


def trigger_disturbance(sim, rec: TrafficRecorder) -> Optional[int]:
    """Same as pressing [B]; also remembers where/when for the shockwave analysis."""
    vid = sim.trigger_lead_disturbance()
    if vid is not None:
        veh = next(v for v in sim.lanes[1] if v.vehicle_id == vid)
        rec.mark_event(sim.time_s, "disturbance", 1, veh.position_m)
    return vid


# ---------------------------------------------------------------- analysis
def edie_cells(rec: TrafficRecorder, road_length_m: float, lanes=analysis_constants.MAINLINE_LANES) -> List[dict]:
    """Edie's q, k, u per lane on a analysis_constants.CELL_LENGTH_M x analysis_constants.CELL_DURATION_S grid."""
    # (lane, cell index along x, cell index along t) -> [distance travelled m, time spent s]
    totals: Dict[tuple, list] = {}
    for s in rec.traj:
        if s.lane_id not in lanes or s.time_s < analysis_constants.WARMUP_S or not 0 <= s.position_m < road_length_m:
            continue
        key = (s.lane_id, int(s.position_m // analysis_constants.CELL_LENGTH_M), int(s.time_s // analysis_constants.CELL_DURATION_S))
        cell = totals.setdefault(key, [0.0, 0.0])
        cell[0] += s.speed_mps * analysis_constants.SAMPLE_INTERVAL_S
        cell[1] += analysis_constants.SAMPLE_INTERVAL_S
    area = analysis_constants.CELL_LENGTH_M * analysis_constants.CELL_DURATION_S       # m*s
    rows = []
    for (lane, ix, it), (dist, time_spent) in sorted(totals.items()):
        q = dist / area                          # veh/s
        k = time_spent / area                    # veh/m
        rows.append({"lane_id": lane, "x_start_m": ix * analysis_constants.CELL_LENGTH_M, "t_start_s": it * analysis_constants.CELL_DURATION_S,
                     "flow_vph": q * 3600.0, "density_vpm": k * analysis_constants.M_PER_MI,
                     "speed_mph": (q / k) * analysis_constants.MPS_TO_MPH if k > 0 else math.nan})
    return rows


def occupied_cells(cells: List[dict]) -> List[dict]:
    """Cells with real traffic in them (almost-empty cells give noisy speeds)."""
    return [c for c in cells if c["density_vpm"] > 1.0]


def _r_squared(y, pred) -> float:
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


def fit_greenshields(k, u):
    """Linear u = uf - (uf/kj) k (lecture: regression of speed on density)."""
    slope, intercept = np.polyfit(k, u, 1)
    uf, kj = intercept, -intercept / slope
    return {"uf_mph": uf, "kj_vpm": kj, "qmax_vph": uf * kj / 4, "kc_vpm": kj / 2,
            "uc_mph": uf / 2, "slope": slope, "r2": _r_squared(u, intercept + slope * k)}


def fit_greenberg(k, u):
    """u = c ln(kj / k): linear regression of u on ln k."""
    slope, intercept = np.polyfit(np.log(k), u, 1)
    c = -slope
    kj = math.exp(intercept / c)
    return {"c_mph": c, "kj_vpm": kj, "kc_vpm": kj / math.e, "qmax_vph": c * kj / math.e,
            "r2": _r_squared(u, intercept + slope * np.log(k))}


def top_flow_capacity(k, q, n: int = 10) -> Optional[dict]:
    """Capacity from the data itself: the average flow of the n highest-flow
    cells (n = 10), and the average density of those cells."""
    k = np.asarray(k, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(q) < n:
        return None
    idx = np.argsort(q)[-n:]
    return {"qmax_vph": float(q[idx].mean()), "kc_vpm": float(k[idx].mean()),
            "k_top": k[idx], "q_top": q[idx]}


def fit_smooth_curve(k, q, bin_vpm: float = 5.0, min_cells: int = 5,
                     percentile: float = 90.0, bandwidth_vpm: float = 7.0) -> Optional[dict]:
    """Our own smooth flow-density trend line, drawn along the upper part of
    the data (no assumed shape like Greenshields or Greenberg).

    1. Group the cells into density bins 5 veh/mi/lane wide.
    2. In every bin with at least 5 cells, take the 90th-percentile flow
       (the flow that only 10% of the cells in that bin exceed).
    3. Draw a smooth line through those points (plus the origin) with a
       bell-curve weighted local straight-line fit (LOESS-style).
    Using the 90th percentile puts the line along the top of the data cloud,
    where the road is working at its best, instead of through the middle."""
    k = np.asarray(k, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(k) < 20:
        return None
    edges = np.arange(0.0, k.max() + bin_vpm, bin_vpm)
    pk, pq, pn = [0.0], [0.0], [float(min_cells)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (k >= lo) & (k < hi)
        if sel.sum() >= min_cells:
            pk.append(float(k[sel].mean()))
            pq.append(float(np.percentile(q[sel], percentile)))
            pn.append(float(sel.sum()))
    pk, pq, pn = np.array(pk), np.array(pq), np.sqrt(np.array(pn))
    # stop where the data runs out (a gap of more than 3 empty bins)
    jump = np.where(np.diff(pk) > 3 * bin_vpm)[0]
    if len(jump):
        pk, pq, pn = pk[:jump[0] + 1], pq[:jump[0] + 1], pn[:jump[0] + 1]
    if len(pk) < 4:
        return None
    grid = np.arange(0.0, pk.max() + 0.25, 0.5)
    curve = []
    for g in grid:
        w = pn * np.exp(-0.5 * ((pk - g) / bandwidth_vpm) ** 2)
        X = np.vstack([np.ones_like(pk), pk - g]).T
        A = X.T @ (X * w[:, None])
        curve.append(np.linalg.solve(A, X.T @ (w * pq))[0])
    curve = np.clip(np.array(curve), 0, None)
    i = int(np.argmax(curve))
    return {"k_vpm": grid, "q_vph": curve, "qmax_vph": float(curve[i]), "kc_vpm": float(grid[i]),
            "points_k": pk, "points_q": pq, "percentile": percentile, "r2": float("nan")}


def measure_wave_speed(rec: TrafficRecorder, lane: int = 1,
                       x_min: float = analysis_constants.WAVE_WINDOW_M[0],
                       x_max: float = analysis_constants.WAVE_WINDOW_M[1],
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
    speeds = _speed_field(rec, lane, x_min, x_max, bin_m)
    if speeds is None:
        return None
    anomalies = speeds - speeds.mean(axis=0)   # speed above/below each bin's average
    candidates = np.arange(1.0, 15.01, 0.25)   # upstream wave speeds to try (m/s)
    scores = np.array([_wave_alignment(anomalies, w, bin_m) for w in candidates])
    best = int(np.argmax(scores))
    if scores[best] < 0.3 or best in (0, len(candidates) - 1):
        return None  # no clear waves, or the best fit is at the edge of the search range
    w = -candidates[best]
    return {"w_mps": float(w), "w_mph": float(w) * analysis_constants.MPS_TO_MPH, "score": float(scores[best])}


def _speed_field(rec: TrafficRecorder, lane: int, x_min: float, x_max: float, bin_m: float):
    """Mean speed per (sample time, space bin) after warm-up, gaps filled and
    smoothed over 10 s. Rows = time, columns = space bins. None if the run is
    too short."""
    times = sorted({s.time_s for s in rec.traj if s.time_s > analysis_constants.WARMUP_S})
    if len(times) < 180:
        return None
    row_of = {t: i for i, t in enumerate(times)}
    n_bins = int((x_max - x_min) // bin_m)
    sums = np.zeros((len(times), n_bins))
    counts = np.zeros((len(times), n_bins))
    for s in rec.traj:
        if s.lane_id == lane and s.time_s in row_of and x_min <= s.position_m < x_min + n_bins * bin_m:
            j = int((s.position_m - x_min) // bin_m)
            sums[row_of[s.time_s], j] += s.speed_mps
            counts[row_of[s.time_s], j] += 1
    V = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    for j in range(n_bins):  # empty bin -> last seen speed (column mean before the first one)
        col = V[:, j]
        last = np.nanmean(col) if np.any(~np.isnan(col)) else 0.0
        for i in range(len(col)):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]
    return np.apply_along_axis(lambda c: np.convolve(c, np.ones(10) / 10, mode="same"), 0, V)


def _wave_alignment(anomalies, w_mps: float, bin_m: float) -> float:
    """Average correlation between bin pairs 100-250 m apart when the
    upstream series is shifted by the travel time of a wave moving upstream
    at w_mps. -1 if no pair could be compared."""
    n_bins = anomalies.shape[1]
    correlations = []
    for i in range(n_bins):
        for j in range(i + 1, n_bins):
            sep = (j - i) * bin_m
            if not 100.0 <= sep <= 250.0:
                continue
            lag = int(round(sep / w_mps))   # samples a wave needs from bin j back to bin i
            if lag < 1 or lag >= len(anomalies) - 60:
                continue
            upstream, downstream = anomalies[lag:, i], anomalies[:-lag, j]
            den = np.sqrt(np.sum(upstream ** 2) * np.sum(downstream ** 2))
            if den > 0:
                correlations.append(np.sum(upstream * downstream) / den)
    return np.mean(correlations) if correlations else -1.0


def regime_states(cells: List[dict], uf_mph: float):
    """Typical free-flowing (A) and congested (B) states from the Edie cells of
    both mainline lanes: A = cells faster than 80% of uf, B = slower than 40%."""
    occupied = occupied_cells(cells)
    A = [c for c in occupied if c["speed_mph"] >= 0.8 * uf_mph]
    B = [c for c in occupied if c["speed_mph"] <= 0.4 * uf_mph]
    if len(A) < 3 or len(B) < 3:
        return None
    qA, kA = np.median([c["flow_vph"] for c in A]), np.median([c["density_vpm"] for c in A])
    qB, kB = np.median([c["flow_vph"] for c in B]), np.median([c["density_vpm"] for c in B])
    return {"qA": qA, "kA": kA, "qB": qB, "kB": kB, "nA": len(A), "nB": len(B)}


def shockwave_speed_mph(states: dict) -> float:
    """Shockwave between states A and B: w = (qA - qB) / (kA - kB)."""
    return (states["qA"] - states["qB"]) / (states["kA"] - states["kB"])


def idm_wave_speed_mps() -> float:
    """Congested-branch wave speed of the IDM for the mean car: -(length + s0) / T."""
    cars = vehicle_constants.VEHICLE_DISTRIBUTION_PARAMETERS["car"]
    return -(cars["length_m"][0] + cars["s0_m"][0]) / cars["T_s"][0]
