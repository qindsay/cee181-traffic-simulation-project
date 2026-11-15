# cee181-traffic-simulation-project

To run this project, install all requirements.

```
pip3 install -r requirements.txt
```

Then, in your project directory terminal, run
```
python3 pygame_main.py
```

Keys: `SPACE` pause, `M` ramp meter on/off, `B` lead-vehicle braking disturbance,
`UP/DOWN` speed, `R` reset, `P` export data + traffic-theory plots, click a car to inspect it.

## Traffic-theory plots and data (`analysis/`)

Press `P` in the window (after at least 3 simulated minutes), or run without a window:

```
python3 -m analysis.traffic_analysis_main --minutes 20
python3 -m analysis.traffic_analysis_main --minutes 20 --meter --disturbance 400 800
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
