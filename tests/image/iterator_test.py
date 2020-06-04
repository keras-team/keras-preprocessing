from keras_preprocessing.image import iterator


def test_iterator_empty_directory():
    # Testing with different batch sizes
    for batch_size in [0, 32]:
        data_iterator = iterator.Iterator(0, batch_size, False, 0)
        ret = next(data_iterator.index_generator)
        assert ret.size == 0
