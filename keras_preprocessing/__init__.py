"""Enables dynamic setting of underlying Keras module.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_KERAS_MODULE = None


def keras_module():
    global _KERAS_MODULE
    if _KERAS_MODULE is None:
        # Use `import keras` as default
        # then fallback to `tf.keras`
        try:
            import keras
        except ImportError:
            try:
                from tensorflow import keras
            except ImportError:
                raise ImportError(
                    'You must have Keras (or TensorFlow) '
                    'installed in order to use keras_preprocessing.')
        set_keras_module(keras)
    return _KERAS_MODULE


def set_keras_module(module):
    global _KERAS_MODULE
    _keras_module_validation(module)
    _KERAS_MODULE = module


def _keras_module_validation(module):
    # TODO: add check on version to make sure
    # this version of keras-preprocessing is
    # compatible with the provided Keras version
    # i.e. `if module.__version__ < ...`
    pass
