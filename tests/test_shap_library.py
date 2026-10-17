"""Cross-check: our TreeSHAP against the `shap` package (runs in CI, where shap is installed)."""

import numpy as np
import pytest

shap = pytest.importorskip("shap")

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402

from retail_demand.common import treeshap  # noqa: E402


@pytest.mark.parametrize("model", [RandomForestRegressor(n_estimators=20, max_depth=6, random_state=0),
                                   HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=15, random_state=0)])
def test_matches_shap_tree_explainer(model):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))
    y = X[:, 0] * 3 + np.sin(X[:, 1]) + X[:, 2] * X[:, 3] + rng.normal(scale=0.2, size=500)
    model.fit(X, y)
    ours, base = treeshap.shap_values(model, X[:25])
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    theirs = explainer.shap_values(X[:25])
    np.testing.assert_allclose(ours, theirs, atol=1e-4, rtol=1e-4)
    assert base == pytest.approx(float(np.ravel(explainer.expected_value)[0]), abs=1e-4)
