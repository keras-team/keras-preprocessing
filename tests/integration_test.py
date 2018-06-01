import pytest
import sys

if sys.version_info > (3, 4):
    from importlib import reload


def test_that_internal_imports_are_not_overriden():
    # Test that changing the keras module after importing
    # Keras does not override keras.preprocessing's keras module
    import keras_preprocessing
    reload(keras_preprocessing)
    assert keras_preprocessing._KERAS_BACKEND is None

    import keras
    if not hasattr(keras.preprocessing.image, 'image'):
        return  # Old Keras, don't run.

    import tensorflow as tf
    keras_preprocessing.set_keras_submodules(backend=tf.keras.backend,
                                             utils=tf.keras.utils)
    assert keras.preprocessing.image.image.backend is keras.backend

    # Now test the reverse order
    del keras
    reload(keras_preprocessing)
    assert keras_preprocessing._KERAS_BACKEND is None

    keras_preprocessing.set_keras_submodules(backend=tf.keras.backend,
                                             utils=tf.keras.utils)
    import keras
    assert keras.preprocessing.image.image.backend is keras.backend


if __name__ == '__main__':
    pytest.main([__file__])
