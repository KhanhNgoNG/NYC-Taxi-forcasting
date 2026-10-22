# NYC Taxi Demand Forecasting Demo

Adaptive Graph WaveNet V4 (holiday), 263 nodes, forecast horizon 30 minutes.

## Setup (Windows)

This repository contains source code and directory placeholders only.
Model weights, datasets, configuration, map geometry and the training notebook
are supplied separately. Git LFS is not required.

```powershell
git clone https://github.com/KhanhNgoNG/NYC-Taxi-forcasting.git
cd NYC-Taxi-forcasting
```

From the cloned repository root (the local development folder is named `demo`):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Local artifact placement

Copy the original exported V4 bundle files to these paths relative to the
repository root. These files are ignored by Git and must be obtained separately
from the project owner. Do not replace them with dummy data or another model.

```text
config/
  v4_config.json
model/
  gwn_v4_seed42.pt
  learned_adaptive_adj_v4.npy          # optional reference/debug artifact
graph/
  adjacency_corr_weighted.npy
  zone_metadata.csv
inference/
  x_test.npy
  cal_w_test.npy
  cal_t_test.npy
  lag_test.npy
  y_test.npy
  demand_current_test.npy
  target_indices_test.npy
  current_timestamps_test.npy
  forecast_timestamps_test.npy
map/
  taxi_zones.shp
  taxi_zones.shx
  taxi_zones.dbf
  taxi_zones.prj
  taxi_zones.cpg
reference/
  02-graph-wavenet-adaptive-graph.ipynb  # required only by verify_bundle.py
  README.txt                         # optional original bundle notes
  model_state_keys.txt               # optional checkpoint key reference
```

Use the original `v4_config.json`, including `cfg_full`, `train_mean`,
`train_std` and `lag_dim`. Keep `node_idx`, `LocationID`, `zone`, `borough`
in the metadata. Inputs must use the original exported node/channel order:
`x [samples,48,263]`, `cal_w [samples,48,11]`, `cal_t [samples,11]`,
`lag [samples,5,263]`. All arrays must share sample count; forecast timestamps
must be current timestamps + 30 minutes. Do not regenerate lag/calendar features.
All five shapefile components must share the `taxi_zones` basename.
Raw training/validation datasets and manifest.json are not needed by the app.

The app validates required artifacts and reports missing or incompatible files.
It cannot run inference until the runtime artifacts above have been copied.

## Run

```powershell
streamlit run app.py
```

Alternatively, without activation:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
.venv\Scripts\python.exe verify_bundle.py
```

## Demo logic

Select an existing current timestamp t. The backend reads one exported sample
from memory-mapped inputs, runs the checkpoint, inverse transforms with train
mean/std and clips predictions to zero exactly as notebook evaluation does.
Forecast is for t+30 minutes. Delta = forecast minus observed demand at t.
Default alert: abs(delta) >= 50 trips (red), for both increases and decreases.
Alerts are sorted by absolute change; Delta keeps its sign. Threshold is adjustable; alert red takes
precedence over other color bands. All metrics and tables follow Region.
Zone drill-down is available in the sidebar; polygons support hover and click popup.

Architecture is extracted verbatim from
`reference/02-graph-wavenet-adaptive-graph.ipynb` (class `AdaptiveGWNV3`).
Checkpoint loading is strict; adaptive adjacency comes from node embeddings.
No training, reconstructed features, or precomputed predictions are used.
CUDA is selected when available, otherwise CPU. Only one sample enters device memory.
CPU inference uses float32; CUDA AMP used by notebook evaluation may cause small
numerical differences. Dataset timestamps are naive; no timezone conversion is applied.

Map uses the supplied shapefile converted to EPSG:4326 and joined on LocationID.
Zones without geometry are reported and retained in tables/totals.
Basemap tiles and Folium JavaScript/CSS require an internet connection in the browser.
The notebook's Kaggle paths remain as unused CFG defaults only; runtime paths
resolve relative to this directory via pathlib.

`VALIDATION.md` records local verification results. `verify_bundle.py`
checks strict loading, CPU inference, timestamps, mapping, map colors and controls.
