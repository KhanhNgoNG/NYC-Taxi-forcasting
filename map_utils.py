"""Geometry joins, colors, tooltips and Folium rendering."""
import folium
import geopandas as gpd
import numpy as np
from branca.colormap import linear
from config import MAP_PATH


def load_zones():
    zones = gpd.read_file(MAP_PATH)
    if 'LocationID' not in zones or zones.crs is None:
        raise ValueError('Shapefile requires LocationID and a declared CRS')
    zones = zones[['LocationID', 'geometry']].to_crs('EPSG:4326')
    zones['LocationID'] = zones.LocationID.astype(int)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty]
    # Multipart records belonging to the same zone are one selectable feature.
    return zones.dissolve(by='LocationID', as_index=False)


def merge_zone_values(zones, values, result, region='All NYC'):
    merged = zones.merge(values, on='LocationID', how='left', validate='one_to_one')
    if merged.node_idx.isna().any():
        raise ValueError('Geometry LocationID missing from model metadata')
    if region != 'All NYC':
        merged = merged[merged.borough == region].copy()
    merged['Current time'] = str(result['current_timestamp']).replace('T', ' ')
    merged['Forecast time'] = str(result['forecast_timestamp']).replace('T', ' ')
    merged['Status'] = np.where(merged.alert, 'HIGH DEMAND CHANGE', 'Below alert threshold')
    return merged


def increase_color(delta, threshold):
    if abs(delta) >= threshold:
        return '#dc2626'
    if delta <= 0:
        return '#16a34a'
    return '#eab308' if delta < 25 else '#f97316'


def legend_html(threshold, mode):
    if mode != 'Demand Change':
        return ''
    bands = [('green', f'-{threshold:g} < change ≤ 0'), ('#eab308', f'0 < change < {min(25, threshold):g}'),
             ('#dc2626', f'|Change| ≥ {threshold:g} (increase or decrease alert)')]
    if threshold > 25:
        bands.insert(2, ('#f97316', f'25 ≤ change < {threshold:g}'))
    return ' &nbsp; '.join(f'<span style="color:{c}">●</span> {label}' for c, label in bands)


def build_map(frame, threshold=50, mode='Demand Change', selected_id=None):
    m = folium.Map(location=[40.72, -73.95], zoom_start=10, tiles='OpenStreetMap')
    if frame.empty:
        return m
    display = frame.copy()
    for col in ('demand_t', 'pred_t30', 'delta'):
        display[col] = display[col].round(1)
    vmax = max(float(frame.pred_t30.max()), 1.0)
    scale = linear.YlOrRd_09.scale(0, vmax)
    display['fill_color'] = [increase_color(d, threshold) for d in frame.delta] if mode == 'Demand Change' else [scale(v) for v in frame.pred_t30]
    fields = ['LocationID', 'zone', 'borough', 'Current time', 'Forecast time', 'demand_t', 'pred_t30', 'delta', 'Status']
    data = display[['geometry', *fields, 'fill_color']].to_json()
    folium.GeoJson(data, name='Taxi Zones',
        style_function=lambda feature: {'fillColor': feature['properties']['fill_color'],
            'color': '#111827', 'weight': 3 if feature['properties']['LocationID'] == selected_id else 0.6, 'fillOpacity': 0.7},
        highlight_function=lambda _: {'weight': 3, 'fillOpacity': 0.9},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=['Zone ID', 'Zone', 'Borough', 'Current time', 'Forecast time', 'Demand t', 'Forecast t+30', 'Change', 'Status'], sticky=False),
        popup=folium.GeoJsonPopup(fields=['LocationID', 'zone'])).add_to(m)
    west, south, east, north = frame.total_bounds
    m.fit_bounds([[south, west], [north, east]])
    if mode != 'Demand Change':
        scale.caption = 'Forecast demand at t+30 (trips)'
        scale.add_to(m)
    return m
