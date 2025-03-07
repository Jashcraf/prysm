"""Tests for basic geometry."""
import math

import pytest

import numpy as np

from prysm import geometry, coordinates


@pytest.mark.parametrize('sides, samples', [
    [5,  128],
    [10, 128],
    [25, 128],
    [5,  256],
    [25, 68]])
def test_regular_polygon(sides, samples):
    x, y = coordinates.make_xy_grid(samples, diameter=2)
    mask = geometry.regular_polygon(sides, 1, x, y)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (samples, samples)


@pytest.mark.parametrize('sigma, samples', [
    [0.5, 128],
    [5,   256]])
def test_gaussian(sigma, samples):
    x, y = coordinates.make_xy_grid(samples, diameter=2)
    assert type(geometry.gaussian(sigma, x, y)) is np.ndarray


def test_rotated_ellipse_fails_if_minor_is_bigger_than_major():
    minor = 1
    major = 0.5
    with pytest.raises(ValueError):
        geometry.rotated_ellipse(width_major=major, width_minor=minor, x=None, y=None)


@pytest.mark.parametrize('maj, min, majang', [
    [1, 0.5, 0],
    [1, 1, 5],
    [0.8, 0.1, 90]])
def test_rotated_ellipse(maj, min, majang):
    x, y = coordinates.make_xy_grid(32, diameter=2)
    assert type(geometry.rotated_ellipse(x=x, y=y,
                                         width_major=maj,
                                         width_minor=min,
                                         major_axis_angle=majang)) is np.ndarray


def test_circle_correct_area():
    x, y = coordinates.make_xy_grid(256, diameter=2)
    dx = x[0, 1] - x[0, 0]
    r_samples = 100
    r_circle = dx*r_samples
    r, _ = coordinates.cart_to_polar(x, y)
    mask = geometry.circle(r_circle, r)
    expected_area_of_circle = r_samples*r_samples * math.pi
    assert mask.sum() == pytest.approx(expected_area_of_circle, abs=3)


def test_truecircle_correct_area():
    x, y = coordinates.make_xy_grid(256, diameter=2)
    dx = x[0, 1] - x[0, 0]
    r_samples = 100
    r_circle = dx*r_samples
    r, _ = coordinates.cart_to_polar(x, y)
    mask = geometry.truecircle(r_circle, r)
    expected_area_of_circle = r_samples*r_samples * math.pi
    assert mask.sum() == pytest.approx(expected_area_of_circle, abs=1.5)


@pytest.mark.parametrize('vanes', [2, 3, 5, 6, 10])
def test_generate_spider_doesnt_error(vanes):
    x, y = coordinates.make_xy_grid(32, diameter=2)
    mask = geometry.spider(vanes, 1, x, y)
    assert isinstance(mask, np.ndarray)


def test_rectangle_correct_area():
    # really this test should be done for a rectangle that is less than the
    # entire array
    x, y = coordinates.make_xy_grid(256, diameter=2)
    mask = geometry.rectangle(1, x, y)
    expected = x.size
    assert mask.sum() == expected


def test_rectangle_doesnt_break_angle():
    x, y = coordinates.make_xy_grid(16, diameter=2)
    mask = geometry.rectangle(1, x, y, angle=45)
    assert mask.any()


def test_offset_circle():
    # [-16, 15] grid
    x, y = coordinates.make_xy_grid(32, dx=1)
    c = geometry.offset_circle(3, x, y, center=(2, 2))
    s = c.sum()
    assert s == 29  # 29 = roundup of 3^2 * pi


def test_annulus_functions():
    x, y = coordinates.make_xy_grid(32, dx=1)
    r = np.hypot(x, y)
    annul = geometry.annulus(5, 6, r)
    assert annul.any()


def test_rectangle_with_corner_fillets_function():
    x, y = coordinates.make_xy_grid(32, dx=1)
    mask = geometry.rectangle_with_corner_fillets(10, 10, 2, x, y)
    assert mask.any()
