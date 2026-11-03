"""Real-data screening helpers, checked on small synthetic tables (no network)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tabbias.realdata import CANDIDATES, TEST_SIZE, make_split, mean_ci, per_seed_gaps, summarize, to_xy


def table(rows=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 3)) * [1, 10, 100] + [0, 5, -50]
    y = (X[:, 0] > 0.3).astype(int)
    return X, y


def test_split_sizes_are_disjoint_stratified_and_standardized_on_train():
    X, y = table()
    X_tr, y_tr, X_te, y_te = make_split(X, y, 100, seed=0)
    assert (len(y_tr), len(y_te)) == (100, TEST_SIZE)
    assert abs(y_tr.mean() - y.mean()) < 0.02 and abs(y_te.mean() - y.mean()) < 0.02
    np.testing.assert_allclose(X_tr.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(X_tr.std(axis=0), 1, atol=1e-9)
    assert not {tuple(r) for r in X_tr} & {tuple(r) for r in X_te}


def test_split_refuses_a_table_too_small_for_train_plus_test():
    X, y = table(rows=350)
    with pytest.raises(ValueError):
        make_split(X, y, 100, seed=0)


def test_to_xy_keeps_numeric_features_and_maps_labels_in_sorted_order():
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "sex": ["m", "f", "m", "f"], "t": ["yes", "no", "yes", "no"]})
    X, y = to_xy(frame, "t")
    assert X.shape == (4, 1)
    np.testing.assert_array_equal(y, [1, 0, 1, 0])


def test_to_xy_uses_only_the_listed_features():
    """A nominal code stored as integers (an area code) must not count as a numeric feature."""
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "area_code": [415, 408, 415, 510], "t": ["1", "0", "1", "0"]})
    X, _ = to_xy(frame, "t", features=["a"])
    assert X.shape == (4, 1)


def test_to_xy_rejects_a_multiclass_target():
    frame = pd.DataFrame({"a": [1.0, 2, 3], "t": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        to_xy(frame, "t")


def runs_with(tree_acc, smooth_acc, tree_ll, smooth_ll, seeds=3, dataset="d", n=100):
    rows = []
    for seed in range(seeds):
        for model in ("rf", "gbdt"):
            rows.append(dict(dataset=dataset, n=n, seed=seed, model=model, accuracy=tree_acc[seed], log_loss=tree_ll))
        for model in ("svm", "mlp", "gp"):
            rows.append(dict(dataset=dataset, n=n, seed=seed, model=model, accuracy=smooth_acc, log_loss=smooth_ll))
        rows.append(dict(dataset=dataset, n=n, seed=seed, model="logreg", accuracy=0.0, log_loss=9.0))
    return pd.DataFrame(rows)


def test_gaps_are_positive_when_trees_win_and_ignore_other_models():
    gaps = per_seed_gaps(runs_with([0.9, 0.9, 0.9], 0.8, tree_ll=0.3, smooth_ll=0.5))
    np.testing.assert_allclose(gaps.delta_accuracy, 0.1)
    np.testing.assert_allclose(gaps.delta_log_loss, 0.2)


def test_mean_ci_matches_a_t_interval():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    mean, low, high = mean_ci(values)
    half = stats.t.ppf(0.975, 3) * values.std(ddof=1) / 2
    assert mean == pytest.approx(0.25)
    assert (low, high) == pytest.approx((0.25 - half, 0.25 + half))


def test_summary_flags_saturation_at_n100_only():
    saturated = runs_with([0.99] * 3, 0.99, 0.1, 0.1, dataset="easy", n=100)
    hard = runs_with([0.7] * 3, 0.7, 0.6, 0.6, dataset="hard", n=100)
    easy_500 = runs_with([0.99] * 3, 0.99, 0.1, 0.1, dataset="hard", n=500)
    table = summarize(pd.concat([saturated, hard, easy_500])).set_index(["dataset", "n"])
    assert table.loc[("easy", 100), "saturated_at_n100"]
    assert not table.loc[("hard", 100), "saturated_at_n100"]
    assert not table.loc[("hard", 500), "saturated_at_n100"]


def test_candidate_names_are_unique_ids():
    ids = [c.openml_id for c in CANDIDATES.values()]
    assert len(ids) == len(set(ids))
