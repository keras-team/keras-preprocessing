import numpy as np
import pytest
from keras_preprocessing.image import secure_image


def test_transform_img():
    x = np.random.random((100, 100, 3))
    assert secure_image.transform_img(20, x, 100, 100).shape == (100, 100, 3)


def test_rot():
    x = np.random.random((100, 100, 3))
    assert secure_image.rot(x, 2, 0, 0).shape == (100, 100, 3)


def test_decrypt_img():
    path = "/some/path"
    with pytest.raises(Exception):
        secure_image.decrypt_img(path, "password", 100, 100)


def test_trandorm():
    path = "/path/to/src"
    dest = "path/to/dest"
    with pytest.raises(EnvironmentError):
        secure_image.transform(path, dest, count=1, block_size=39, image_x=100, image_y=100)


def test_encrypt_directory():
    path = "/path/to/src"
    dest = "path/to/dest"
    with pytest.raises(EnvironmentError):
        secure_image.encrypt_directory(path, dest, image_x=100, image_y=100, password="PASS")
