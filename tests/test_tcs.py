import json

import numpy as np
import pandas as pd
import synthetic

from retail_demand import config
from retail_demand.tcs import data, experiments, features


def _write(tmp_path, d):
    for k, v in d.items():
        v.to_csv(tmp_path / f"{k}.csv", index=False)


def _panel(tmp_path, **kw):
    d = synthetic.tcs(**kw)
    _write(tmp_path, d)
    raw = data.load(tmp_path)
    return d, raw, data.build_panel(raw)


def test_panel_alignment_prices_and_quality(tmp_path):
    d, raw, p = _panel(tmp_path)
    assert p.sales.shape == (12, len(p.dates)) and not np.isnan(p.sales).any()
    i = p.series.index[(p.series.outlet == 222) & (p.series.product_identifier == 423)][0]
    row = d["train_data"].query("outlet == 222 and product_identifier == 423").sales.to_numpy()
    np.testing.assert_array_equal(p.sales[i], row)
    wk = d["date_to_week_id_map"].week_id.to_numpy()
    pr = d["product_prices"].query("outlet == 222 and product_identifier == 423").set_index("week_id").sell_price
    np.testing.assert_allclose(p.price[i], pr.reindex(wk).to_numpy(), rtol=1e-6)    # weekly price spread over its days
    q = data.quality(raw, p)
    assert q["uniqueness"]["duplicate_product_outlet_date"] == 0
    assert q["completeness"]["rows"] == q["completeness"]["grid_cells"] and q["completeness"]["sales_rows_without_price"] == 0
    assert q["validity"]["negative_sales"] == 0 and sum(q["relevance"]["demand_classes"].values()) == 12


def test_demand_classes():
    smooth = np.full(200, 5.0)
    intermittent = np.tile([0, 0, 0, 4.0], 50)
    lumpy = np.tile([0, 0, 0, 1.0, 0, 0, 0, 20.0], 25)
    cls = data.classify(np.vstack([smooth, intermittent, lumpy]))["class"].tolist()
    assert cls == ["smooth", "intermittent", "lumpy"]


def test_croston_sba_tsb_converge_and_are_causal():
    y = np.tile([0.0, 4.0], 300)[None, :]                  # 4 units every second day: mean demand 2
    a = 0.1
    paths = features.intermittent_paths(y, alpha=a)
    assert abs(paths["croston"][0, -1] - 2.0) < 1e-6
    assert abs(paths["tsb"][0, -1] - 2.0) < 0.15
    np.testing.assert_allclose(paths["sba"], paths["croston"] * (1 - a / 2))
    y2 = y.copy()
    y2[0, 400:] = 50.0                                     # change the future only
    p2 = features.intermittent_paths(y2, alpha=a)
    for k in paths:
        np.testing.assert_array_equal(paths[k][0, :400], p2[k][0, :400])


def test_origin_features_use_nothing_after_the_origin_except_the_planned_price(tmp_path):
    _, _, p = _panel(tmp_path)
    o = 300
    f0 = features.origin_rows(p, o, features.intermittent_paths(p.sales))
    p.sales[:, o + 1:] = p.sales[:, o + 1:] * 3 + 2
    f1 = features.origin_rows(p, o, features.intermittent_paths(p.sales))
    pd.testing.assert_frame_equal(f0[features.FEATURES], f1[features.FEATURES])
    assert not np.allclose(f0.y.to_numpy(), f1.y.to_numpy())
    assert set(f0.h) == set(range(1, config.TCS_HORIZON + 1)) and (f0.target_idx == o + f0.h).all()


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    d = synthetic.tcs()
    _write(tmp_path, d)
    monkeypatch.setattr(config, "TCS_DIR", tmp_path)
    small = dict(max_iter=40, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)
    monkeypatch.setattr(experiments, "HGB", small)
    full = experiments.candidates
    keep = ("Ridge", "Random forest", "Gradient boosting (Poisson)")
    monkeypatch.setattr(experiments, "candidates", lambda seed=0: {k: v for k, v in full(seed).items() if k in keep})
    monkeypatch.setattr(experiments, "TUNE_GRIDS", {"Gradient boosting": [dict(learning_rate=0.2, max_iter=20)],
                                                   "Ridge": [{"ridge__alpha": 1.0}]})
    out = tmp_path / "out"
    experiments.run(experiments.STAGES, out)
    t = json.loads((out / "test.json").read_text())
    assert t["table"][t["model"]]["rmse"] < t["table"]["Zero (forecast nothing)"]["rmse"]
    v1 = json.loads((out / "v1.json").read_text())
    assert set(v1["categories"]) == {"others", "fast_moving_consumer_goods", "drinks_and_food"}
    ex = json.loads((out / "explain.json").read_text())
    assert ex["additivity_max_error"] < 1e-6
    dec = json.loads((out / "decision.json").read_text())
    assert all(0 < r["fill_rate"] <= 1 and r["scale_factor_from_calibration"] > 0 for r in dec["policies"].values())
    hyp = json.loads((out / "hypotheses.json").read_text())
    assert {h["id"] for h in hyp} >= {"H1", "H2", "H3", "H4", "H6"}
