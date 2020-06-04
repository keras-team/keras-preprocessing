import numpy as np
import pytest

from keras_preprocessing.image import affine_transformations


def test_random_transforms():
    x = np.random.random((2, 28, 28))
    assert affine_transformations.random_rotation(x, 45).shape == (2, 28, 28)
    assert affine_transformations.random_shift(x, 1, 1).shape == (2, 28, 28)
    assert affine_transformations.random_shear(x, 20).shape == (2, 28, 28)
    assert affine_transformations.random_channel_shift(x, 20).shape == (2, 28, 28)


def test_deterministic_transform():
    x = np.ones((3, 3, 3))
    x_rotated = np.array([[[0., 0., 0.],
                           [0., 0., 0.],
                           [1., 1., 1.]],
                          [[0., 0., 0.],
                           [1., 1., 1.],
                           [1., 1., 1.]],
                          [[0., 0., 0.],
                           [0., 0., 0.],
                           [1., 1., 1.]]])
    assert np.allclose(affine_transformations.apply_affine_transform(
        x, theta=45, channel_axis=2, fill_mode='constant'), x_rotated)


def test_random_zoom():
    x = np.random.random((2, 28, 28))
    assert affine_transformations.random_zoom(x, (5, 5)).shape == (2, 28, 28)
    assert np.allclose(x, affine_transformations.random_zoom(x, (1, 1)))


def test_random_zoom_error():
    with pytest.raises(ValueError):
        affine_transformations.random_zoom(0, zoom_range=[0])


def test_apply_brightness_shift_error(monkeypatch):
    monkeypatch.setattr(affine_transformations, 'ImageEnhance', None)
    with pytest.raises(ImportError):
        affine_transformations.apply_brightness_shift(0, [0])


def test_random_brightness(monkeypatch):
    monkeypatch.setattr(affine_transformations,
                        'apply_brightness_shift', lambda x, y: (x, y))
    assert (0, 3.) == affine_transformations.random_brightness(0, (3, 3))


def test_random_brightness_error():
    with pytest.raises(ValueError):
        affine_transformations.random_brightness(0, [0])


def test_apply_affine_transform_error(monkeypatch):
    monkeypatch.setattr(affine_transformations, 'scipy', None)
    with pytest.raises(ImportError):
        affine_transformations.apply_affine_transform(0)
