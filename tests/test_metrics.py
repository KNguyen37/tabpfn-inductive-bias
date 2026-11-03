"""The metrics must behave correctly on surfaces whose shape we know."""
import numpy as np
import pytest

from tabbias.metrics import (
    axis_alignment,
    expected_calibration_error,
    far_field_confidence,
    jaggedness,
    jump_share,
    transition_width,
)

HALF = 3.0


def ramp(k, angle_deg, res=200):
    axis = np.linspace(-HALF, HALF, res)
    x, y = np.meshgrid(axis, axis)
    angle = np.deg2rad(angle_deg)
    return 1 / (1 + np.exp(-np.clip(k * (x * np.cos(angle) + y * np.sin(angle)), -40, 40)))


def plateaus(angle_deg, res=200):
    axis = np.linspace(-HALF, HALF, res)
    x, y = np.meshgrid(axis, axis)
    angle = np.deg2rad(angle_deg)
    u = x * np.cos(angle) + y * np.sin(angle)
    return np.array([0.0, 0.3, 0.6, 1.0])[np.digitize(u, [-1, 0, 1])]


def test_axis_alignment_measures_direction_and_ignores_sharpness():
    """This is why S could not serve as a step score (feedback item 1)."""
    assert axis_alignment(ramp(1, 0)) == pytest.approx(1.0, abs=1e-3)
    assert axis_alignment(ramp(60, 0)) == pytest.approx(1.0, abs=1e-3)
    assert axis_alignment(ramp(1, 45)) == pytest.approx(1 / np.sqrt(2), abs=1e-3)
    assert axis_alignment(ramp(60, 45)) == pytest.approx(1 / np.sqrt(2), abs=0.01)


@pytest.mark.parametrize("angle", [0, 15, 22.5, 30, 45, 60, 67.5, 90])
def test_jump_share_separates_smooth_from_sharp_at_any_angle(angle):
    assert jump_share(ramp(1, angle)) < 0.05
    assert jump_share(ramp(60, angle)) > 0.75


@pytest.mark.parametrize("k", [1, 5, 60])
def test_jump_share_is_stable_under_rotation(k):
    scores = [jump_share(ramp(k, angle)) for angle in (0, 15, 22.5, 30, 45, 60, 67.5, 90)]
    assert max(scores) - min(scores) < 0.1


@pytest.mark.parametrize("k", [1, 60])
def test_jump_share_is_stable_across_grid_resolution(k):
    assert jump_share(ramp(k, 22.5, res=400)) == pytest.approx(jump_share(ramp(k, 22.5, res=200)), abs=0.06)


def test_jump_share_treats_tree_plateaus_as_steps():
    assert jump_share(plateaus(0)) > 0.95
    assert jump_share(plateaus(45)) > 0.95


def test_transition_width_recovers_the_logistic_width():
    # A logistic ramp of slope k crosses 0.1 to 0.9 over 2*ln(9)/k.
    assert transition_width(ramp(5, 0), HALF) == pytest.approx(2 * np.log(9) / 5, rel=0.05)


def test_transition_width_is_inflated_by_plateaus():
    """Documents why jump share, not width, is the primary sharpness metric."""
    assert transition_width(plateaus(0), HALF) > 10 * transition_width(ramp(60, 0), HALF)


def test_jaggedness_tracks_boundary_length_and_not_resolution():
    base = jaggedness(ramp(5, 0))
    # A straight edge at this angle crosses a chord 1/cos(angle) longer on a square grid.
    assert jaggedness(ramp(5, 33)) == pytest.approx(base / np.cos(np.deg2rad(33)), rel=0.02)
    assert jaggedness(ramp(5, 0, res=400)) == pytest.approx(base, rel=0.05)


def test_flat_surface_has_no_defined_shape():
    flat = np.full((50, 50), 0.5)
    assert np.isnan(jump_share(flat))
    assert np.isnan(axis_alignment(flat))
    assert transition_width(flat, HALF) == 0.0


def test_ece_is_zero_when_confidence_matches_accuracy():
    p = np.r_[np.full(100, 0.8), np.full(100, 0.2)]
    y = np.r_[np.ones(80), np.zeros(20), np.zeros(80), np.ones(20)].astype(int)
    assert expected_calibration_error(p, y) == pytest.approx(0.0, abs=1e-9)


def test_far_field_confidence_handles_empty_input():
    assert np.isnan(far_field_confidence(np.empty(0)))
    assert far_field_confidence(np.array([0.5, 0.9])) == pytest.approx(0.7)
