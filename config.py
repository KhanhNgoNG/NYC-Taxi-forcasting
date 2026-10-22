from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / 'model/gwn_v4_seed42.pt'
CONFIG_PATH = PROJECT_ROOT / 'config/v4_config.json'
GRAPH_PATH = PROJECT_ROOT / 'graph/adjacency_corr_weighted.npy'
METADATA_PATH = PROJECT_ROOT / 'graph/zone_metadata.csv'
INFERENCE_DIR = PROJECT_ROOT / 'inference'
MAP_PATH = PROJECT_ROOT / 'map/taxi_zones.shp'
DEFAULT_THRESHOLD = 50
