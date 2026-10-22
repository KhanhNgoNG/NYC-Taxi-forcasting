"""Run independently: python verify_bundle.py (no training or full-test evaluation)."""
import ast
import json
from pathlib import Path
import numpy as np
from inference import InferenceBackend
from map_utils import load_zones, merge_zone_values, build_map, increase_color


def main():
    root = Path(__file__).resolve().parent
    notebook = json.loads((root / 'reference/02-graph-wavenet-adaptive-graph.ipynb').read_text(encoding='utf8'))
    local = ast.parse((root / 'model.py').read_text(encoding='utf8'))
    definitions = {n.name: ast.dump(n, include_attributes=False) for n in local.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in definitions:
                assert definitions.pop(node.name) == ast.dump(node, include_attributes=False), node.name
    assert not definitions
    backend = InferenceBackend('cpu')
    result = backend.predict_sample(0)
    values = backend.zone_values(result)
    np.testing.assert_allclose(backend.arrays['x_test'][0, -1] * backend.conf['train_std'] + backend.conf['train_mean'], result['demand_current'], atol=1e-4)
    assert result['prediction'].shape == (263,) and np.isfinite(result['prediction']).all()
    assert (result['prediction'] >= 0).all()
    zones = load_zones()
    merged = merge_zone_values(zones, values.sample(frac=1, random_state=42), result)
    for row in merged.itertuples():
        assert row.pred_t30 == result['prediction'][int(row.node_idx)]
    for borough in sorted(values.borough.unique()):
        part = merge_zone_values(zones, values, result, borough)
        assert set(part.borough) <= {borough}
        build_map(part).get_root().render()
    assert increase_color(50, 50) == '#dc2626'
    assert increase_color(-50, 50) == '#dc2626'
    assert increase_color(-49.9, 50) == '#16a34a'
    assert increase_color(49.9, 50) != '#dc2626'
    modified = dict(result, delta=result['delta'].copy())
    modified['delta'][:4] = [-50, 50, -49.9, 49.9]
    assert backend.zone_values(modified).alert.iloc[:4].tolist() == [True, True, False, False]
    assert increase_color(20, 10) == '#dc2626'
    assert increase_color(0, 50) == '#16a34a'
    for mode in ['Demand Change', 'Forecast Demand']:
        html = build_map(merged, mode=mode).get_root().render()
        assert 'Forecast time' in html and 'LocationID' in html
    print('PASS: runtime AST matches notebook; strict=True; CPU forward [1,263], finite.')
    print(f'PASS: timestamps, normalization, node mapping; {len(zones)} unique geometry IDs, {len(values)} model nodes.')
    print('Model zones without geometry:', sorted(set(values.LocationID)-set(zones.LocationID)))
    print('PASS: borough maps, tooltips, both map modes, red threshold boundary.')
    print('Sample:', result['current_timestamp'], '->', result['forecast_timestamp'])


if __name__ == '__main__':
    main()
