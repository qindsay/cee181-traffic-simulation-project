"""Traffic-theory data export and presentation plots.

Two ways to use it:
  1. In the Pygame window press [P]: everything recorded so far is exported.
  2. Without a window:
         python -m analysis.traffic_analysis_main --minutes 20
         python -m analysis.traffic_analysis_main --minutes 20 --meter --disturbance 300 600
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

File layout:
  settings -> recorder (collects data during the run) -> analysis (numbers)
  -> export (CSV + summary) -> plots (one function per figure) -> headless run
"""
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
from typing import List, Optional

from analysis import analyze_traffic, plot
from constants import analysis_constants

# ---------------------------------------------------------------- export
def _write_csv(path: Path, header: List[str], rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _rounded_values(row: dict, digits: int) -> list:
    return [round(v, digits) if isinstance(v, float) else v for v in row.values()]


def export(sim, rec: analyze_traffic.TrafficRecorder, out_dir: Optional[str] = None) -> Path:
    """Write all CSVs, plots and summary.txt; returns the output folder."""
    import matplotlib
    matplotlib.use("Agg")  # draw to files only (safe next to the Pygame window)
    out = Path(out_dir) if out_dir else Path("output") / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(out / "trajectories.csv", list(analyze_traffic.TrajectorySample._fields),
               [(s.time_s, s.vehicle_id, s.lane_id, round(s.position_m, 3), round(s.speed_mps, 3),
                 round(s.acceleration_mps2, 3),
                 "" if math.isnan(s.gap_to_leader_m) else round(s.gap_to_leader_m, 3), s.veh_type)
                for s in rec.traj])
    _write_csv(out / "detector_crossings.csv", list(analyze_traffic.Crossing._fields),
               [(c.station, c.lane_id, round(c.time_s, 3), c.vehicle_id, round(c.speed_mps, 3))
                for c in rec.crossings])
    cells = analyze_traffic.edie_cells(rec, sim.length_m)
    _write_csv(out / "edie_cells.csv", list(cells[0].keys()) if cells else ["empty"],
               [_rounded_values(r, 3) for r in cells])
    vehicles = list(rec.vehicles.values())
    if vehicles:
        _write_csv(out / "vehicles.csv", list(vehicles[0].keys()), [_rounded_values(r, 4) for r in vehicles])
    obs = sim.observation_log
    if obs:
        _write_csv(out / "detector_observations.csv", list(obs[0].keys()), [list(r.values()) for r in obs])

    summary = plot.make_plots(sim, rec, cells, out)
    lines = [f"Traffic analysis summary - simulated time {sim.time_s:.0f} s "
             f"(first {analysis_constants.WARMUP_S:.0f} s excluded), dt {sim.dt_s} s", ""]
    lines += [f"{k}: {v}" for k, v in summary.items()]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    return out

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
    rec = analyze_traffic.TrafficRecorder()
    pending_disturbances = sorted(args.disturbance)
    n_steps = int(round(args.minutes * 60 / args.dt))
    for _ in range(n_steps):
        if pending_disturbances and sim.time_s >= pending_disturbances[0] - 1e-9:
            pending_disturbances.pop(0)
            analyze_traffic.trigger_disturbance(sim, rec)
        sim.step()
        rec.record(sim)
    out = export(sim, rec, args.out)
    print(f"Exported data and plots to {out}/")
    print((out / "summary.txt").read_text())


if __name__ == "__main__":
    main()
