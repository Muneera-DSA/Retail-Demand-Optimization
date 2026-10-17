"""Exact SHAP values for scikit-learn tree ensembles (path-dependent TreeSHAP).

Algorithm 2 of Lundberg, Erion & Lee (2018), "Consistent individualized feature attribution
for tree ensembles". The conditional expectation E[f(x) | x_S] follows each tree's training
cover at splits on features outside S. Values are in the model's raw output space and add
up exactly: base_value + sum(phi) == raw prediction (checked in the tests, and cross-checked
against the `shap` package in CI).

Supported: HistGradientBoostingRegressor, DecisionTreeRegressor, RandomForestRegressor,
ExtraTreesRegressor (numeric features; NaN routed as the model routes it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tree:
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    missing_left: np.ndarray
    value: np.ndarray        # leaf values (internal nodes ignored)
    cover: np.ndarray        # training samples reaching each node
    is_leaf: np.ndarray
    scale: float = 1.0       # weight of this tree in the ensemble

    def go_left(self, node: int, x: np.ndarray) -> bool:
        v = x[self.feature[node]]
        if np.isnan(v):
            return bool(self.missing_left[node])
        return bool(v <= self.threshold[node])


def trees_from_model(model) -> tuple[list[Tree], float]:
    """Return (trees, base offset) such that raw prediction = offset + sum(scale * tree(x))."""
    name = type(model).__name__
    if name == "HistGradientBoostingRegressor":
        trees = []
        for it in model._predictors:
            n = it[0].nodes
            trees.append(Tree(left=n["left"].astype(int), right=n["right"].astype(int), feature=n["feature_idx"].astype(int),
                              threshold=n["num_threshold"].astype(float), missing_left=n["missing_go_to_left"].astype(bool),
                              value=n["value"].astype(float), cover=n["count"].astype(float), is_leaf=n["is_leaf"].astype(bool)))
        return trees, float(np.ravel(model._baseline_prediction)[0])
    if name == "DecisionTreeRegressor":
        return [_sk_tree(model.tree_, 1.0)], 0.0
    if name in ("RandomForestRegressor", "ExtraTreesRegressor"):
        k = len(model.estimators_)
        return [_sk_tree(e.tree_, 1.0 / k) for e in model.estimators_], 0.0
    raise TypeError(f"TreeSHAP not implemented for {name}")


def _sk_tree(t, scale: float) -> Tree:
    miss = getattr(t, "missing_go_to_left", np.zeros(t.node_count, dtype=np.uint8))
    return Tree(left=t.children_left.astype(int), right=t.children_right.astype(int), feature=t.feature.astype(int),
                threshold=t.threshold.astype(float), missing_left=np.asarray(miss).astype(bool),
                value=t.value[:, 0, 0].astype(float), cover=t.weighted_n_node_samples.astype(float),
                is_leaf=t.children_left == -1, scale=scale)


# ------------------------------------------------------------------ path operations
# A path is four parallel lists: feature d, zero fraction z, one fraction o, weight w.

def _extend(d, z, o, w, pz, po, pi):
    d, z, o, w = d + [pi], z + [pz], o + [po], w + [1.0 if not w else 0.0]
    n = len(w) - 1
    for i in range(n - 1, -1, -1):
        w[i + 1] += po * w[i] * (i + 1) / (n + 1)
        w[i] = pz * w[i] * (n - i) / (n + 1)
    return d, z, o, w


def _unwind(d, z, o, w, k):
    n = len(w) - 1
    ok, zk = o[k], z[k]
    w = list(w)
    nxt = w[n]
    for i in range(n - 1, -1, -1):
        if ok != 0:
            tmp = w[i]
            w[i] = nxt * (n + 1) / ((i + 1) * ok)
            nxt = tmp - w[i] * zk * (n - i) / (n + 1)
        else:
            w[i] = w[i] * (n + 1) / (zk * (n - i))
    return d[:k] + d[k + 1:], z[:k] + z[k + 1:], o[:k] + o[k + 1:], w[:n]


def _unwound_sum(z, o, w, k):
    n = len(w) - 1
    ok, zk = o[k], z[k]
    nxt, total = w[n], 0.0
    for i in range(n - 1, -1, -1):
        if ok != 0:
            tmp = nxt * (n + 1) / ((i + 1) * ok)
            total += tmp
            nxt = w[i] - tmp * zk * (n - i) / (n + 1)
        else:
            total += (w[i] / zk) / ((n - i) / (n + 1))
    return total


def tree_shap_one(tree: Tree, x: np.ndarray, n_features: int) -> np.ndarray:
    phi = np.zeros(n_features)

    def recurse(node, d, z, o, w, pz, po, pi):
        d, z, o, w = _extend(d, z, o, w, pz, po, pi)
        if tree.is_leaf[node]:
            for i in range(1, len(w)):
                s = _unwound_sum(z, o, w, i)
                phi[d[i]] += s * (o[i] - z[i]) * tree.value[node]
            return
        hot, cold = (tree.left[node], tree.right[node]) if tree.go_left(node, x) else (tree.right[node], tree.left[node])
        f = tree.feature[node]
        iz = io = 1.0
        if f in d[1:]:
            k = d.index(f, 1)
            iz, io = z[k], o[k]
            d, z, o, w = _unwind(d, z, o, w, k)
        c = tree.cover[node]
        recurse(hot, d, z, o, w, iz * tree.cover[hot] / c, io, f)
        recurse(cold, d, z, o, w, iz * tree.cover[cold] / c, 0.0, f)

    recurse(0, [], [], [], [], 1.0, 1.0, -1)
    return phi * tree.scale


def expected_value(tree: Tree) -> float:
    """Cover-weighted mean leaf value: the tree's prediction with no feature known."""
    def ev(node):
        if tree.is_leaf[node]:
            return tree.value[node]
        lc, rc = tree.cover[tree.left[node]], tree.cover[tree.right[node]]
        return (lc * ev(tree.left[node]) + rc * ev(tree.right[node])) / (lc + rc)
    return ev(0) * tree.scale


def shap_values(model, X: np.ndarray, progress: int = 0) -> tuple[np.ndarray, float]:
    """(phi of shape (n, p), base value). Raw prediction = base + phi.sum(1)."""
    X = np.asarray(X, float)
    trees, offset = trees_from_model(model)
    phi = np.zeros(X.shape)
    for r in range(len(X)):
        for t in trees:
            phi[r] += tree_shap_one(t, X[r], X.shape[1])
        if progress and (r + 1) % progress == 0:
            print(f"[treeshap] {r + 1}/{len(X)} rows", flush=True)
    base = offset + sum(expected_value(t) for t in trees)
    return phi, float(base)


def raw_predict(model, X: np.ndarray) -> np.ndarray:
    """Raw (pre-link) ensemble output computed by walking the extracted trees."""
    X = np.asarray(X, float)
    trees, offset = trees_from_model(model)
    out = np.full(len(X), offset)
    for t in trees:
        for r in range(len(X)):
            node = 0
            while not t.is_leaf[node]:
                node = t.left[node] if t.go_left(node, X[r]) else t.right[node]
            out[r] += t.value[node] * t.scale
    return out


def brute_force_shap(tree: Tree, x: np.ndarray, n_features: int) -> np.ndarray:
    """Exact Shapley values of the path-dependent value function by enumerating subsets
    (exponential; tests only)."""
    from itertools import combinations
    from math import factorial

    def cond_exp(S, node=0):
        if tree.is_leaf[node]:
            return tree.value[node]
        if tree.feature[node] in S:
            return cond_exp(S, tree.left[node] if tree.go_left(node, x) else tree.right[node])
        lc, rc = tree.cover[tree.left[node]], tree.cover[tree.right[node]]
        return (lc * cond_exp(S, tree.left[node]) + rc * cond_exp(S, tree.right[node])) / (lc + rc)

    M = n_features
    phi = np.zeros(M)
    for i in range(M):
        others = [j for j in range(M) if j != i]
        for size in range(M):
            for S in combinations(others, size):
                wgt = factorial(size) * factorial(M - size - 1) / factorial(M)
                phi[i] += wgt * (cond_exp(set(S) | {i}) - cond_exp(set(S)))
    return phi * tree.scale
