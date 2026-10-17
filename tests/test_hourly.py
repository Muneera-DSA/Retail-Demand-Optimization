import json

import pandas as pd
import synthetic

from retail_demand import config
from retail_demand.hourly import data, experiments, features, models, quality


def test_grid_classifies_outage_and_gaps():
    raw = synthetic.hourly()
    g = data.to_grid(raw)
    assert len(g) == g.date.nunique() * 24 and g.ts.is_unique
    assert (g[g.date.between("2019-09-10", "2019-09-14")].status == "outage").all()
    gaps = g[g.status == "gap"]
    assert len(gaps) > 0 and set(gaps.hour) <= {3, 4}
    assert (g.status == "observed").sum() == len(raw)


def test_week_ahead_features_ignore_the_last_week():
    """Changing demand in the 167 hours before t (or later) must not change t's features."""
    g = data.to_grid(synthetic.hourly(gaps=False))
    f0 = features.build(g)
    t = 24 * 300
    g2 = g.copy()
    g2.loc[t - 167:, "demand"] = g2.loc[t - 167:, "demand"] * 10 + 7
    pd.testing.assert_series_equal(f0.loc[t, features.FEATURES], features.build(g2).loc[t, features.FEATURES])
    g3 = g.copy()
    g3.loc[t - 168, "demand"] += 1000                                  # exactly one week before: allowed, must matter
    assert not f0.loc[t, features.FEATURES].equals(features.build(g3).loc[t, features.FEATURES])


def test_quality_checks_report_known_problems():
    raw = synthetic.hourly()
    dup = pd.concat([raw, raw.iloc[[5]]])
    q = quality.run(dup, data.to_grid(raw), test_start="2019-11-01")
    assert q["uniqueness"]["duplicate_date_hour"] == 1
    assert q["validity"]["hour_outside_0_23"] == 0
    assert any(r["days"] == 5 for r in q["completeness"]["missing_day_runs"])


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    path = tmp_path / "hourly.csv"
    synthetic.hourly(days=800).to_csv(path, index=False)
    monkeypatch.setattr(config, "HOURLY_CSV", path)
    monkeypatch.setattr(config, "HOURLY_TEST_START", "2021-01-01")
    monkeypatch.setattr(config, "HOURLY_CV_FOLDS", 2)
    monkeypatch.setattr(config, "HOURLY_CV_FOLD_WEEKS", 4)
    full = models.candidates()
    keep = ("Ridge (hour-of-week dummies + history)", "Gradient boosting (Poisson)", "Extra trees")
    monkeypatch.setattr(models, "candidates", lambda seed=0: {k: full[k] for k in keep})
    monkeypatch.setattr(models, "TUNE_GRIDS", {"Ridge": [{"ridge__alpha": 10}], "Extra trees": [],
                                              "Gradient boosting": [dict(learning_rate=0.1, max_iter=30, max_leaf_nodes=7,
                                                                         min_samples_leaf=50, l2_regularization=1.0)]})
    out = tmp_path / "out"
    experiments.run(experiments.STAGES, out)
    t = json.loads((out / "test.json").read_text())
    # the synthetic demand is a fixed weekly profile plus noise: any model must beat last week's value
    assert t["table"][t["model"]]["mae"] < t["table"]["Seasonal naive (same hour last week)"]["mae"]
    assert json.loads((out / "explain.json").read_text())["additivity_max_error"] < 1e-8
    assert {h["id"] for h in json.loads((out / "hypotheses.json").read_text())} >= {"H1", "H5", "H6", "H7"}
    dec = json.loads((out / "decision.json").read_text())
    assert 0.5 < dec["plans"]["Model: 75th-percentile forecast"]["service_level"] <= 1
