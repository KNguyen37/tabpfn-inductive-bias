"""The contract, synthetic data and sweep loop, with fast models only."""
import numpy as np
import pandas as pd
import pytest

from tabbias.classic import CLASSIC_MODELS, make_classic
from tabbias.contract import METRIC_COLUMNS, SYNTH_DATASETS, SYNTH_N, SYNTH_VARIANTS, RunKey, load_grid, save_grid
from tabbias.sweep import aggregate, enumerate_keys, run_sweep
from tabbias.synth import make_problem


@pytest.mark.parametrize("name", SYNTH_DATASETS)
def test_problem_is_deterministic_and_balanced(name):
    a, b = make_problem(name, "clean", 3), make_problem(name, "clean", 3)
    np.testing.assert_array_equal(a.X_context, b.X_context)
    assert a.X_context.shape == (max(SYNTH_N), 2)
    for n in SYNTH_N:
        assert a.y_context[:n].sum() == n // 2  # every context size is balanced when clean


def test_seeds_differ():
    assert not np.array_equal(make_problem("moons", "clean", 0).X_context, make_problem("moons", "clean", 1).X_context)


def test_context_and_test_are_disjoint():
    problem = make_problem("xor", "clean", 0)
    context = {tuple(row) for row in problem.X_context}
    assert not context & {tuple(row) for row in problem.X_test}


def test_label_noise_touches_context_only():
    clean, noisy = make_problem("moons", "clean", 0), make_problem("moons", "noise20", 0)
    np.testing.assert_array_equal(clean.y_test, noisy.y_test)
    rate = (clean.y_context != noisy.y_context).mean()
    assert 0.12 < rate < 0.28


def test_rotation_rotates_the_same_draw():
    clean, rotated = make_problem("moons", "clean", 0), make_problem("moons", "rot45", 0)
    np.testing.assert_array_equal(clean.y_context, rotated.y_context)
    assert not np.allclose(clean.X_context, rotated.X_context)


def test_grid_is_square_and_covers_the_data():
    problem = make_problem("spirals_tight", "clean", 0)
    points = problem.grid_points()
    assert points.shape == (40000, 2)
    assert points[:, 0].min() == points[:, 1].min() == -problem.half_width
    assert np.abs(problem.X_context).max() < problem.half_width


def test_far_field_points_exclude_the_context_hull():
    problem = make_problem("moons", "clean", 0)
    assert 0 < len(problem.far_field_points(10)) < 14400
    assert (0, 0) not in {tuple(p) for p in problem.far_field_points(200)}


def test_run_key_rejects_paths_that_would_escape_the_tree():
    with pytest.raises(ValueError):
        RunKey("../evil", "clean", 10, 0, "rf")


def test_grid_round_trip(tmp_path):
    p = np.random.default_rng(0).random((20, 20))
    path = RunKey("moons", "clean", 10, 0, "rf").grid_path(tmp_path)
    save_grid(path, p, 2.5)
    loaded, half_width = load_grid(path)
    np.testing.assert_allclose(loaded, p, atol=1e-6)
    assert half_width == 2.5


def test_seed_reaches_stochastic_models():
    """The pilot fixed random_state at 0, so seed spread excluded model randomness."""
    assert make_classic("rf", 7).random_state == 7
    assert make_classic("mlp", 7).random_state == 7
    assert make_classic("gbdt", 7).random_state == 7


@pytest.mark.parametrize("model", CLASSIC_MODELS)
def test_every_classic_model_handles_the_smallest_context(tmp_path, model):
    keys = enumerate_keys(["circles"], ["clean"], [10], [0], [model])
    tally = run_sweep(keys, make_classic, tmp_path)
    assert tally["failed"] == 0, (tmp_path / "failures.log").read_text()


def test_sweep_resumes_and_aggregates(tmp_path):
    keys = enumerate_keys(["moons"], ["clean", "noise10"], [10, 30], [0], ["logreg", "knn"])
    assert run_sweep(keys, make_classic, tmp_path)["done"] == 8

    keys[0].run_path(tmp_path).unlink()
    tally = run_sweep(keys, make_classic, tmp_path)
    assert (tally["skipped"], tally["done"]) == (7, 1)

    table = tmp_path / "metrics.csv"
    assert aggregate(tmp_path, table) == 8
    frame = pd.read_csv(table)
    assert tuple(frame.columns) == METRIC_COLUMNS
    assert frame.drop(columns="device").notna().all().all()


def test_failed_run_is_logged_and_does_not_stop_the_sweep(tmp_path):
    def factory(name, seed):
        if name == "knn":
            raise RuntimeError("boom")
        return make_classic(name, seed)

    keys = enumerate_keys(["moons"], ["clean"], [10], [0], ["knn", "logreg"])
    tally = run_sweep(keys, factory, tmp_path)
    assert (tally["failed"], tally["done"]) == (1, 1)
    assert "boom" in (tmp_path / "failures.log").read_text()


def test_tabpfn_refuses_to_run_outside_colab(monkeypatch):
    from tabbias.tabpfn_models import make_tabpfn

    monkeypatch.delenv("TABBIAS_COMPUTE", raising=False)
    with pytest.raises(RuntimeError, match="Colab"):
        make_tabpfn("tabpfn_v2", 0)


def test_all_variants_are_constructible():
    for variant in SYNTH_VARIANTS:
        make_problem("circles", variant, 0)
