import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from config import DEFAULT_THRESHOLD
from inference import InferenceBackend
from map_utils import load_zones, merge_zone_values, build_map, legend_html

st.set_page_config(page_title='NYC Taxi Demand Forecasting', layout='wide')


@st.cache_resource
def get_backend():
    return InferenceBackend()


@st.cache_data
def get_zones():
    return load_zones()


@st.cache_data(max_entries=64)
def predict(index):
    return get_backend().predict_sample(index)


def show_table(frame):
    columns = ['LocationID', 'zone', 'borough', 'demand_t', 'pred_t30', 'delta']
    st.dataframe(frame[columns].rename(columns={'zone': 'Zone', 'borough': 'Borough',
        'demand_t': 'Demand t', 'pred_t30': 'Forecast t+30', 'delta': 'Delta'}),
        hide_index=True, width='stretch', column_config={
            name: st.column_config.NumberColumn(format='%.1f') for name in ['Demand t', 'Forecast t+30', 'Delta']})


st.title('NYC Taxi Demand Forecasting')
st.caption('Adaptive Graph WaveNet V4 (holiday) · 263 zones · 30-minute forecast')
try:
    backend = get_backend()
    zones = get_zones()
    dates = np.datetime_as_string(backend.current, unit='D')
    with st.sidebar:
        st.header('Forecast controls')
        date = st.selectbox('Date', list(dict.fromkeys(dates)))
        candidates = np.flatnonzero(dates == date).tolist()
        index = st.selectbox('Time', candidates, format_func=lambda i: str(backend.current[i])[11:16])
        region = st.selectbox('Region', ['All NYC'] + sorted(backend.metadata.borough.dropna().unique().tolist()))
        threshold = st.number_input('Alert threshold (trips)', min_value=1, value=DEFAULT_THRESHOLD, step=1)
        mode = st.selectbox('Map mode', ['Demand Change', 'Forecast Demand'])
    result = predict(index)
    values = backend.zone_values(result, threshold)
    subset = values if region == 'All NYC' else values[values.borough == region]
    lookup = subset.set_index('LocationID')
    options = [None] + subset.LocationID.tolist()
    with st.sidebar:
        selected = st.selectbox('Taxi Zone', options,
            format_func=lambda v: 'All zones' if v is None else f'{v} - {lookup.loc[v, "zone"]}')
        st.caption(f'Device: {backend.device} · {len(backend.current):,} test samples')
    current_label = str(result['current_timestamp']).replace('T', ' ')
    forecast_label = str(result['forecast_timestamp']).replace('T', ' ')
    cols = st.columns(4)
    cols[0].metric('Current time', current_label)
    cols[1].metric('Forecast time', forecast_label)
    cols[2].metric('Total forecast', f'{subset.pred_t30.sum():,.1f}')
    cols[3].metric('Alert zones', int(subset.alert.sum()))
    st.caption(f'Metrics and tables: {region}. Timestamps use the original dataset local time.')
    frame = merge_zone_values(zones, values, result, region)
    absent = sorted(set(subset.LocationID) - set(frame.LocationID))
    if absent:
        st.info(f'{len(absent)} model zones have no matching polygon: {absent}. They remain in totals and tables.')
    st.subheader('Interactive NYC Taxi Zone Map')
    st_folium(build_map(frame, threshold, mode, selected), height=560, use_container_width=True,
              key=f'map-{region}-{mode}', returned_objects=[])
    st.markdown(legend_html(threshold, mode), unsafe_allow_html=True)
    st.subheader('Demand change alerts')
    alerts = subset[subset.alert].sort_values('delta', key=lambda s: s.abs(), ascending=False)
    if alerts.empty:
        st.info('No zones exceed the current threshold.')
    else:
        show_table(alerts)
    st.subheader('Selected Zone Details')
    if selected is None:
        st.caption('Select a Taxi Zone in the sidebar to inspect its forecast.')
    else:
        row = lookup.loc[selected]
        st.write(f'{selected} · {row.zone} · {row.borough}')
        a, b, c, d = st.columns(4)
        a.metric('Demand at t', f'{row.demand_t:,.1f}')
        b.metric('Forecast at t+30', f'{row.pred_t30:,.1f}')
        c.metric('Change', f'{row.delta:+,.1f}')
        d.metric('Percentage change', f'{row.delta / row.demand_t * 100:+.1f}%' if row.demand_t > 0 else 'N/A')
        if row.alert:
            st.error('HIGH DEMAND CHANGE — ' + ('INCREASE' if row.delta > 0 else 'DECREASE'))
        else:
            st.info('Below alert threshold')
    st.subheader('Top predicted demand zones')
    show_table(subset.nlargest(10, 'pred_t30'))
except Exception as exc:
    st.error(f'Unable to run the demo: {exc}')
    st.stop()
