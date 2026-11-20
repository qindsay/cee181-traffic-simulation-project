# cee181-traffic-simulation-project

To run this project, install all requirements.

```
pip3 install -r requirements.txt
```

Then, in your project directory terminal, run
```
python3 pygame_main.py
```

Keys: `SPACE` pause, `M` ramp meter on/off, `B` lead-vehicle braking disturbance, `C` enable/disable cav,
`UP/DOWN` speed, `R` reset, `P` export data + traffic-theory plots, click a car to inspect it. Note: pressing `C` also resets the entire simulation.

## Drivers and CAVs

- **Human drivers** use the IDM with Gaussian (z-score) parameters, including a
  **reaction time** (0.7 s on average): each driver sees the vehicle ahead as it
  was one reaction time ago. An emergency reflex (time-to-collision < 1.5 s)
  makes them react immediately when a crash is about to happen.
- **CAVs** (circles in the window; only passenger cars can be automated) use our
  own controller, updated every 0.1 s: constant 1.0 s time gap, cruise/follow law
  with string-stable gains, an RSS safety-distance check that overrides comfort,
  and comfort limits on acceleration (+1.5 / -3 m/s²) and jerk. All settings and
  equations are in the CAV block of `constants/vehicle_constants.py`.
- CAV share: `C` in the window uses `CAV_PENETRATION` (20%); the command line uses
  `--cav` (default 0 = no CAVs).

## Traffic-theory plots and data (`analysis/`)

Press `P` in the window (after at least 3 simulated minutes), or run without a window:

```
python3 -m analysis.traffic_analysis_main --minutes 20
python3 -m analysis.traffic_analysis_main --minutes 20 --cav 0.25 --seed 7 --disturbance 400 800 --out output/cav0.25_seed7
python3 -m analysis.traffic_analysis_main --minutes 20 --meter --disturbance 400 800
```

Options: `--cav` share of cars that are CAVs (0-1, default 0), `--seed` random
traffic (default 42), `--disturbance` times (s) when a lane-1 vehicle brakes hard,
`--meter` ramp meter on, `--dt` time step (keep 0.1 for all comparison runs),
`--out` output folder.

Everything goes to `output/run_<date_time>/`:

| File | What it shows |
|---|---|
| `1_fundamental_diagrams.png` | flow-density, speed-density, speed-flow with Greenshields, Greenberg and our own smooth curve through the data (its peak = capacity) |
| `2_time_space.png` | vehicle trajectories colored by speed; congestion wave speed, measured vs. Δq/Δk and IDM theory |
| `3_cumulative_counts.png` | cumulative vehicle counts upstream vs downstream (queueing view) |
| `4_headways.png` | time-headway distributions vs exponential (Poisson) and shifted exponential |
| `5_spot_speeds.png` | spot-speed study with normal fit and 15th / 50th / 85th percentiles |
| `6_micro_fundamental.png` | gap vs speed against the IDM equilibrium curve |
| `7_driver_parameters.png` | generated driver parameters vs their Gaussian (z-score) design |
| `summary.txt` | all fitted and measured numbers |
| `*.csv` | trajectories, detector crossings, Edie cells (q, k, u), vehicles, detector records |

Density, flow and space-mean speed come from Edie's generalized definitions on
100 m x 30 s space-time cells of the trajectories (point detectors cannot measure
density). Plot units follow the lectures: veh/h, veh/mi, mph.

## Reproducing the mid-report figures and tables

1. Run the 9 experiments (3 CAV shares x 3 seeds, 20 min each, braking events at 400 s and 800 s).
   Windows Command Prompt:

   ```
   for %c in (0 0.25 0.5) do for %s in (42 1 7) do py -3.12 -m analysis.traffic_analysis_main --minutes 20 --cav %c --seed %s --disturbance 400 800 --out output/cav%c_seed%s
   ```

   macOS / Linux:

   ```
   for c in 0 0.25 0.5; do for s in 42 1 7; do python3 -m analysis.traffic_analysis_main --minutes 20 --cav $c --seed $s --disturbance 400 800 --out output/cav${c}_seed${s}; done; done
   ```

2. Build the report figures and tables from those 9 folders (also runs the 10-follower string-stability test):

   ```
   python3 -m analysis.report_figures --runs output --out output/report
   ```

| Report item | File |
|---|---|
| Fig. 1 simulator screenshot | `output/report/fig1_simulator.png` (needs pygame) |
| Fig. 2 string-stability test | `output/report/fig6_string_stability.png` |
| Fig. 3 speed, time stopped, congested intervals | `output/report/fig2_headline.png` |
| Fig. 4 time-space diagrams | `output/report/fig3_time_space.png` |
| Fig. 5 fundamental diagrams with top-10 capacity | `output/report/fig4_fundamental.png` |
| Fig. 6 spot speeds | `output/report/fig5_spot_speeds.png` |
| Table 4 run results | `output/report/table_runs.csv` |
| Table 5 capacity estimates | `output/report/table_fd_fits.csv` |
| Appendix C graphs | `output/cav0_seed42/` and `output/cav0.5_seed42/` (made by each run) |

Capacity is measured directly from the data as the average flow of the 10
highest-flow Edie cells (`top_flow_capacity` in `analysis/analyze_traffic.py`).
A smooth trend line along the top of the cells (`fit_smooth_curve`: 90th-percentile
flow in 5 veh/mi density bins, smoothed) shows the shape of the diagram. A 10 s detector interval
counts as congested when its mean speed is below 15 m/s for at least 3 intervals in a row.
