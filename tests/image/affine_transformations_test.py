import numpy as np

from keras_preprocessing.image import affine_transformations


def test_random_transforms():
    x = np.random.random((2, 28, 28))
    assert affine_transformations.random_rotation(x, 45).shape == (2, 28, 28)
    assert affine_transformations.random_shift(x, 1, 1).shape == (2, 28, 28)
    assert affine_transformations.random_shear(x, 20).shape == (2, 28, 28)
    assert affine_transformations.random_zoom(x, (5, 5)).shape == (2, 28, 28)
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
