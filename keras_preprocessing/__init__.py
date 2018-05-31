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
        set_keras_module('keras')
    if _KERAS_MODULE == 'tensorflow.keras':
        # Due to TF namespace structure,
        # can't `__import__('tensorflow.keras')`.
        # Use workaround.
        tf = __import__('tensorflow')
        keras = tf.keras
    else:
        keras = __import__(_KERAS_MODULE, fromlist=['keras'])
    # TODO: check that the Keras version is compatible with
    # the current module.
    return keras


def set_keras_module(module):
    global _KERAS_MODULE
    _KERAS_MODULE = module
