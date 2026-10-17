import json

import numpy as np
import pandas as pd
import synthetic

from retail_demand import config
from retail_demand.favorita import data, experiments, features


def _write(tmp_path, d):
    for k, v in d.items():
        v.to_csv(tmp_path / f"{k}.csv", index=False)


def test_panel_alignment_and_quality(tmp_path):
    d = synthetic.favorita()
    _write(tmp_path, d)
    raw = data.load(tmp_path)
    p = data.build_panel(raw)
    assert p.sales.shape == (9, len(p.dates))
    i = p.series.index[(p.series.store_nbr == 2) & (p.series.family == "BEVERAGES")][0]
    row = d["train"][(d["train"].store_nbr == 2) & (d["train"].family == "BEVERAGES")].set_index("date").sales
    np.testing.assert_allclose(p.sales[i, p.dates.get_indexer(row.index)], row.to_numpy())
    assert np.isnan(p.sales[:, p.dates.get_loc(pd.Timestamp("2015-12-25"))]).all()      # closed day stays unobserved
    q = data.quality(raw, p)
    assert q["completeness"]["dates_absent_from_file"] == ["2015-12-25"]
    assert q["completeness"]["series_never_selling"] == 3                                 # BOOKS in every store
    assert q["uniqueness"]["duplicate_store_family_date"] == 0


def test_origin_features_use_nothing_after_the_origin_except_planned_inputs(tmp_path):
    d = synthetic.favorita()
    _write(tmp_path, d)
    p = data.build_panel(data.load(tmp_path))
    o = 300
    f0 = features.origin_rows(p, o)
    p.sales[:, o + 1:] = p.sales[:, o + 1:] * 5 + 3                                     # change the future
    p.transactions[:, o + 1:] = 1.0
    f1 = features.origin_rows(p, o)
    pd.testing.assert_frame_equal(f0[features.FEATURES], f1[features.FEATURES])
    assert not np.allclose(f0.y.to_numpy(), f1.y.to_numpy())                            # targets did change


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    d = synthetic.favorita(days=560)
    _write(tmp_path, d)
    monkeypatch.setattr(config, "FAVORITA_DIR", tmp_path)
    monkeypatch.setattr(config, "FAV_TEST_START", str((d["train"].date.max() - pd.Timedelta(days=15)).date()))
    monkeypatch.setattr(config, "FAV_TRAIN_FROM", "2015-03-01")
    monkeypatch.setattr(config, "FAV_N_VALID_WINDOWS", 2)
    full = experiments.candidates()
    keep = ("Ridge (log scale)", "Gradient boosting (log scale, squared error)", "Random forest (log scale)")
    monkeypatch.setattr(experiments, "candidates", lambda seed=0: {k: full[k] for k in keep})
    monkeypatch.setattr(experiments, "TUNE_GRIDS", {"Gradient boosting": [dict(learning_rate=0.1, max_iter=40, max_leaf_nodes=15,
                                                                                min_samples_leaf=20, l2_regularization=1.0)],
                                                   "Random forest": [{"min_samples_leaf": 20, "max_features": 0.5}]})
    out = tmp_path / "out"
    experiments.run(experiments.STAGES, out)
    t = json.loads((out / "test.json").read_text())
    assert t["table"][t["model"]]["wape"] < t["table"]["Mean of the last 14 days"]["wape"]
    ex = json.loads((out / "explain.json").read_text())
    assert ex["additivity_max_error"] < 1e-8 and ex["permutation_importance_wape_increase"]
    dec = json.loads((out / "decision.json").read_text())
    assert all(0 < r["cycle_service_level"] <= 1 for cyc in dec["cycles"].values() for r in cyc.values())
    assert {h["id"] for h in json.loads((out / "hypotheses.json").read_text())} >= {"H1", "H5"}
