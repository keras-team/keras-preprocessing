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
                           [1., 1., 1.],
                           [0., 0., 0.]],
                          [[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]],
                          [[0., 0., 0.],
                           [1., 1., 1.],
                           [0., 0., 0.]]])
    assert np.allclose(
        affine_transformations.apply_affine_transform(x,
                                                      theta=45,
                                                      row_axis=0,
                                                      col_axis=1,
                                                      channel_axis=2,
                                                      fill_mode='constant'),
        x_rotated)


def test_matrix_center():
    x = np.expand_dims(np.array([
        [0, 1],
        [0, 0],
    ]), -1)
    x_rotated90 = np.expand_dims(np.array([
        [1, 0],
        [0, 0],
    ]), -1)

    assert np.allclose(
        affine_transformations.apply_affine_transform(x,
                                                      theta=90,
                                                      row_axis=0,
                                                      col_axis=1,
                                                      channel_axis=2),
        x_rotated90)


def test_translation():
    x = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ])
    x_up = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    x_dn = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    x_left = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    x_right = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ])

    # Channels first
    x_test = np.expand_dims(x, 0)

    # Horizontal translation
    assert np.alltrue(x_left == np.squeeze(
        affine_transformations.apply_affine_transform(x_test, tx=1)))
    assert np.alltrue(x_right == np.squeeze(
        affine_transformations.apply_affine_transform(x_test, tx=-1)))

    # change axes: x<->y
    assert np.alltrue(x_left == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, ty=1, row_axis=2, col_axis=1)))
    assert np.alltrue(x_right == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, ty=-1, row_axis=2, col_axis=1)))

    # Vertical translation
    assert np.alltrue(x_up == np.squeeze(
        affine_transformations.apply_affine_transform(x_test, ty=1)))
    assert np.alltrue(x_dn == np.squeeze(
        affine_transformations.apply_affine_transform(x_test, ty=-1)))

    # change axes: x<->y
    assert np.alltrue(x_up == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, tx=1, row_axis=2, col_axis=1)))
    assert np.alltrue(x_dn == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, tx=-1, row_axis=2, col_axis=1)))

    # Channels last
    x_test = np.expand_dims(x, -1)

    # Horizontal translation
    assert np.alltrue(x_left == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, tx=1, row_axis=0, col_axis=1, channel_axis=2)))
    assert np.alltrue(x_right == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, tx=-1, row_axis=0, col_axis=1, channel_axis=2)))

    # change axes: x<->y
    assert np.alltrue(x_left == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, ty=1, row_axis=1, col_axis=0, channel_axis=2)))
    assert np.alltrue(x_right == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, ty=-1, row_axis=1, col_axis=0, channel_axis=2)))

    # Vertical translation
    assert np.alltrue(x_up == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, ty=1, row_axis=0, col_axis=1, channel_axis=2)))
    assert np.alltrue(x_dn == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, ty=-1, row_axis=0, col_axis=1, channel_axis=2)))

    # change axes: x<->y
    assert np.alltrue(x_up == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, tx=1, row_axis=1, col_axis=0, channel_axis=2)))
    assert np.alltrue(x_dn == np.squeeze(
        affine_transformations.apply_affine_transform(
            x_test, tx=-1, row_axis=1, col_axis=0, channel_axis=2)))


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
                        'apply_brightness_shift', lambda x, y, z: (x, y))
    assert (0, 3.) == affine_transformations.random_brightness(0, (3, 3))


def test_random_brightness_error():
    with pytest.raises(ValueError):
        affine_transformations.random_brightness(0, [0])


def test_random_brightness_scale():
    img = np.ones((1, 1, 3)) * 128
    zeros = np.zeros((1, 1, 3))
    must_be_128 = affine_transformations.random_brightness(img, [1, 1], False)
    assert np.array_equal(img, must_be_128)
    must_be_0 = affine_transformations.random_brightness(img, [1, 1], True)
    assert np.array_equal(zeros, must_be_0)


def test_random_brightness_scale_outside_range_positive():
    img = np.ones((1, 1, 3)) * 1024
    zeros = np.zeros((1, 1, 3))
    must_be_1024 = affine_transformations.random_brightness(img, [1, 1], False)
    assert np.array_equal(img, must_be_1024)
    must_be_0 = affine_transformations.random_brightness(img, [1, 1], True)
    assert np.array_equal(zeros, must_be_0)


def test_random_brightness_scale_outside_range_negative():
    img = np.ones((1, 1, 3)) * -1024
    zeros = np.zeros((1, 1, 3))
    must_be_neg_1024 = affine_transformations.random_brightness(img, [1, 1], False)
    assert np.array_equal(img, must_be_neg_1024)
    must_be_0 = affine_transformations.random_brightness(img, [1, 1], True)
    assert np.array_equal(zeros, must_be_0)


def test_apply_affine_transform_error(monkeypatch):
    monkeypatch.setattr(affine_transformations, 'scipy', None)
    with pytest.raises(ImportError):
        affine_transformations.apply_affine_transform(0)
