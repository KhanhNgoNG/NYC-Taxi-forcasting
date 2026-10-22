# Local verification

Verified 2026-10-22:

- Runtime class/function ASTs match the supplied notebook exactly.
- Checkpoint strict=True: all keys matched successfully.
- Fixed supports rebuilt from graph agree with checkpoint buffer.
- Real CPU forward sample 0: [1,263], all finite; inverse transform and clipping pass.
- All 8,784 timestamp pairs differ by 30 minutes.
- Sample 0: current 2025-07-01 23:30 -> forecast 2025-07-02 00:00.
- Exported current demand agrees with inverse-normalized last input step.
- 263 metadata nodes and 263 geometry IDs, no missing model IDs.
- Shuffled metadata still maps correctly by node_idx and LocationID.
- Maps render for all boroughs and both modes, with tooltip fields in HTML.
- Alert red takes precedence, including thresholds below 25.
- Streamlit AppTest: no exceptions/errors at startup, Manhattan filter,
  zone 161 drill-down, Forecast Demand mode and no-alert threshold.
- Streamlit server starts; localhost health endpoint returns HTTP 200 ok.

Repeat backend/map checks: python verify_bundle.py

Tested CPU float32 with torch 2.14.1, Streamlit 1.66.1, GeoPandas 1.2.0,
Folium 0.20.0, streamlit-folium 0.27.4 in project .venv.
Browser hover/pan and remote tile loading were not manually exercised.
CUDA was not tested. Notebook AMP metrics may differ slightly from float32.
No full-test evaluation was run.
