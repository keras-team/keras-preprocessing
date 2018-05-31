import pytest
import sys

if sys.version_info > (3, 4):
    from importlib import reload


def test_dynamic_backend_setting():
    import keras_preprocessing
    reload(keras_preprocessing)
    assert keras_preprocessing._KERAS_MODULE is None
    from tensorflow import keras as keras_ref
    keras_preprocessing.set_keras_module('tensorflow.keras')
    from keras_preprocessing import image
    assert image.keras_module() is keras_ref

    import keras as keras_ref
    keras_preprocessing.set_keras_module('keras')
    assert image.keras_module() is keras_ref
    reload(image)
    assert image.keras_module() is keras_ref


if __name__ == '__main__':
    pytest.main([__file__])
