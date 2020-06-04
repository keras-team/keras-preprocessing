import pytest

import keras_preprocessing


def test_api_modules():
    expected_exposed_modules = [
        'image',
        'sequence',
        'text'
    ]
    for _module in expected_exposed_modules:
        assert hasattr(keras_preprocessing, _module)


def test_get_keras_submodule(monkeypatch):
    monkeypatch.setattr(keras_preprocessing, '_KERAS_BACKEND', 'backend')
    assert 'backend' == keras_preprocessing.get_keras_submodule('backend')
    monkeypatch.setattr(keras_preprocessing, '_KERAS_UTILS', 'utils')
    assert 'utils' == keras_preprocessing.get_keras_submodule('utils')


def test_get_keras_submodule_errors(monkeypatch):
    with pytest.raises(ImportError):
        keras_preprocessing.get_keras_submodule('something')

    monkeypatch.setattr(keras_preprocessing, '_KERAS_BACKEND', None)
    with pytest.raises(ImportError):
        keras_preprocessing.get_keras_submodule('backend')

    with pytest.raises(ImportError):
        keras_preprocessing.get_keras_submodule('utils')
