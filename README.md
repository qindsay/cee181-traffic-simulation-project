# cee181-traffic-simulation-project

To run this project, install all requirements.

```
pip3 install -r requirements.txt
```

Then, in your project directory terminal, run
```
python3 main_pygame.py
```

Keys: `SPACE` pause, `M` ramp meter on/off, `B` lead-vehicle braking disturbance,
`UP/DOWN` speed, `R` reset, `P` export data + traffic-theory plots, click a car to inspect it.

## Traffic-theory plots and data (`traffic_analysis.py`)

Press `P` in the window (after at least 3 simulated minutes), or run without a window:

```
python3 traffic_analysis.py --minutes 20
python3 traffic_analysis.py --minutes 20 --meter --disturbance 400 800
```

Everything goes to `output/run_<date_time>/`:

| File | What it shows |
|---|---|
| `1_fundamental_diagrams.png` | flow-density, speed-density, speed-flow with Greenshields and Greenberg fits (capacity, critical and jam density) |
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

## Fixes made (October 2026)

- Ramp merges and lane changes check that the moving car can follow its new
  leader (before, cars merged a metre behind slower cars and drove through them:
  about 100 overlaps per 15 simulated minutes; now zero).
- Merges/lane changes are decided every 0.2 s / 1 s instead of every step, and
  arrivals are drawn as exponential headways in continuous time, so results do
  not change with the time step (use dt = 0.05, 0.1 or 0.2 s).
- Arrivals and vehicle parameters use separate random streams (they used to
  draw the same numbers).
- Ramp meter: a released car is no longer stopped and released a second time; a
  red meter now also stops a car whose leader already passed the line.
- The stored acceleration is the one actually applied (stopped cars used to
  report -9 m/s²); vehicle-state log rows use the time after the step.
- A braking disturbance can no longer override a harder brake that is needed.
- Ramp cars in the merge zone adjust to the mainline car beside them (at most
  comfortable braking), so fewer get stuck at the taper end.
- Exact normal CDF for the z-score percentile.

`v2_proposed/` is a larger reference version (ALINEA control loop, tests,
scenario files); it is not needed to run this project.
