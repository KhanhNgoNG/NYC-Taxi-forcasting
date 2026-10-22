"""Validated, memory-mapped model-ready inputs and real checkpoint inference."""
import json
from threading import Lock
import numpy as np
import pandas as pd
import torch
from config import CONFIG_PATH, MODEL_PATH, GRAPH_PATH, METADATA_PATH, INFERENCE_DIR, MAP_PATH
from model import CFG, AdaptiveGWNV3, build_fixed_supports


class InferenceBackend:
    def __init__(self, device=None):
        names = ('x_test', 'cal_w_test', 'cal_t_test', 'lag_test', 'y_test',
                 'demand_current_test', 'target_indices_test',
                 'current_timestamps_test', 'forecast_timestamps_test')
        required = [CONFIG_PATH, MODEL_PATH, GRAPH_PATH, METADATA_PATH]
        required += [INFERENCE_DIR / f'{n}.npy' for n in names]
        required += [MAP_PATH.with_suffix(ext) for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg')]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError('Missing required artifacts:\n' + '\n'.join(missing))
        self.conf = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        self.arrays = {n: np.load(INFERENCE_DIR / f'{n}.npy', mmap_mode='r', allow_pickle=False) for n in names}
        n = len(self.arrays['x_test'])
        if n == 0:
            raise ValueError('Empty inference dataset')
        tails = {'x_test': (48, 263), 'cal_w_test': (48, 11), 'cal_t_test': (11,),
                 'lag_test': (5, 263), 'y_test': (263,), 'demand_current_test': (263,),
                 'target_indices_test': (), 'current_timestamps_test': (), 'forecast_timestamps_test': ()}
        for name, tail in tails.items():
            a = self.arrays[name]
            if a.shape != (n, *tail):
                raise ValueError(f'{name}: expected {(n, *tail)}, got {a.shape}')
            expected = 'M' if 'timestamps' in name else ('i' if name == 'target_indices_test' else 'f')
            if a.dtype.kind != expected:
                raise ValueError(f'{name}: invalid dtype {a.dtype}')
        self.current = self.arrays['current_timestamps_test']
        self.forecast = self.arrays['forecast_timestamps_test']
        if np.isnat(self.current).any() or np.isnat(self.forecast).any() or not np.all(self.forecast - self.current == np.timedelta64(30, 'm')):
            raise ValueError('Forecast timestamps must equal current timestamps + 30 minutes')
        if len(np.unique(self.current)) != n:
            raise ValueError('Duplicate current timestamps')
        self.metadata = pd.read_csv(METADATA_PATH)
        if not {'node_idx', 'LocationID', 'zone', 'borough'} <= set(self.metadata.columns):
            raise ValueError('Missing required metadata columns')
        if len(self.metadata) != 263 or set(self.metadata.node_idx) != set(range(263)) or not self.metadata.LocationID.is_unique:
            raise ValueError('Metadata must map 263 unique nodes to unique LocationIDs')
        self.metadata = self.metadata.sort_values('node_idx').reset_index(drop=True)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.lock = Lock()
        cfg = CFG(**self.conf['cfg_full'])
        adj = np.load(GRAPH_PATH, allow_pickle=False)
        if adj.shape != (263, 263) or not np.isfinite(adj).all():
            raise ValueError('Invalid fixed adjacency')
        self.model = AdaptiveGWNV3(263, build_fixed_supports(adj), cfg, self.conf['lag_dim'])
        state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        if not torch.allclose(state['fixed'], self.model.fixed, atol=1e-6):
            raise ValueError('Checkpoint fixed supports disagree with exported graph')
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.predict_sample(0)

    def predict_sample(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.current):
            raise IndexError('Sample index outside test dataset')
        tensors = []
        for name in ('x_test', 'cal_w_test', 'cal_t_test', 'lag_test'):
            a = np.array(self.arrays[name][index:index + 1], dtype=np.float32, copy=True)
            if not np.isfinite(a).all():
                raise ValueError(f'{name}[{index}] contains nonfinite values')
            tensors.append(torch.from_numpy(a).to(self.device))
        with self.lock, torch.no_grad():
            out = self.model(*tensors)
        if out.shape != (1, 263) or not torch.isfinite(out).all():
            raise ValueError('Model output must be finite [1,263]')
        pred = np.maximum(out.float().cpu().numpy()[0] * self.conf['train_std'] + self.conf['train_mean'], 0.0)
        current = np.array(self.arrays['demand_current_test'][index], copy=True)
        if not np.isfinite(current).all() or not np.isfinite(pred).all():
            raise ValueError('Nonfinite demand or prediction')
        return dict(current_timestamp=self.current[index], forecast_timestamp=self.forecast[index],
                    demand_current=current, prediction=pred, delta=pred-current)

    def predict_at_timestamp(self, timestamp):
        hits = np.flatnonzero(self.current == np.datetime64(timestamp))
        if len(hits) != 1:
            raise ValueError(f'No unique sample for timestamp {timestamp}')
        return self.predict_sample(int(hits[0]))

    def zone_values(self, result, threshold=50):
        frame = self.metadata.copy()
        nodes = frame.node_idx.to_numpy(dtype=int)
        for col, key in [('demand_t', 'demand_current'), ('pred_t30', 'prediction'), ('delta', 'delta')]:
            frame[col] = result[key][nodes]
        frame['alert'] = frame.delta.abs() >= threshold
        return frame
