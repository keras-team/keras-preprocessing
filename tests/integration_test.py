import pytest
import sys

if sys.version_info > (3, 4):
    from importlib import reload


def test_dynamic_backend_setting():
    import keras_preprocessing
    reload(keras_preprocessing)
    assert keras_preprocessing._KERAS_MODULE is None
    from tensorflow import keras
    keras_preprocessing.set_keras_module(keras)
    from keras_preprocessing import image
    assert image.K == keras.backend

    import keras
    keras_preprocessing.set_keras_module(keras)
    reload(image)  # Note that this is required for now.
    assert image.K == keras.backend


if __name__ == '__main__':
    pytest.main([__file__])
