import numpy as np
import pytest

from PIL import Image

from keras_preprocessing.image import numpy_array_iterator
from keras_preprocessing.image import utils
from keras_preprocessing.image.image_data_generator import ImageDataGenerator


@pytest.fixture(scope='module')
def all_test_images():
    img_w = img_h = 20
    rgb_images = []
    rgba_images = []
    gray_images = []
    for n in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        imarray = np.random.rand(img_w, img_h, 3) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        rgb_images.append(im)

        imarray = np.random.rand(img_w, img_h, 4) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        rgba_images.append(im)

        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = Image.fromarray(
            imarray.astype('uint8').squeeze()).convert('L')
        gray_images.append(im)

    return [rgb_images, rgba_images, gray_images]


@pytest.fixture(scope='module')
def image_data_generator():
    return ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=0.2,
        channel_shift_range=0.,
        brightness_range=(1, 5),
        fill_mode='nearest',
        cval=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        interpolation_order=1
    )


def test_numpy_array_iterator(image_data_generator, all_test_images, tmpdir):
    for test_images in all_test_images:
        img_list = []
        for im in test_images:
            img_list.append(utils.img_to_array(im)[None, ...])
        images = np.vstack(img_list)
        dsize = images.shape[0]

        iterator = numpy_array_iterator.NumpyArrayIterator(
            images,
            np.arange(images.shape[0]),
            image_data_generator,
            shuffle=False,
            save_to_dir=str(tmpdir),
            batch_size=3
        )
        x, y = next(iterator)
        assert x.shape == images[:3].shape
        assert list(y) == [0, 1, 2]

        # Test with sample weights
        iterator = numpy_array_iterator.NumpyArrayIterator(
            images,
            np.arange(images.shape[0]),
            image_data_generator,
            shuffle=False,
            sample_weight=np.arange(images.shape[0]) + 1,
            save_to_dir=str(tmpdir),
            batch_size=3
        )
        x, y, w = iterator.next()
        assert x.shape == images[:3].shape
        assert list(y) == [0, 1, 2]
        assert list(w) == [1, 2, 3]

        # Test with `shuffle=True`
        iterator = numpy_array_iterator.NumpyArrayIterator(
            images,
            np.arange(images.shape[0]),
            image_data_generator,
            shuffle=True,
            save_to_dir=str(tmpdir),
            batch_size=3,
            seed=42
        )
        x, y = iterator.next()
        assert x.shape == images[:3].shape
        # Check that the sequence is shuffled.
        assert list(y) != [0, 1, 2]

        # Test without y
        iterator = numpy_array_iterator.NumpyArrayIterator(
            images,
            None,
            image_data_generator,
            shuffle=True,
            save_to_dir=str(tmpdir),
            batch_size=3
        )
        x = iterator.next()
        assert type(x) is np.ndarray
        assert x.shape == images[:3].shape

        # Test with a single miscellaneous input data array
        x_misc1 = np.random.random(dsize)
        iterator = numpy_array_iterator.NumpyArrayIterator(
            (images, x_misc1),
            np.arange(dsize),
            image_data_generator,
            shuffle=False,
            batch_size=2
        )
        for i, (x, y) in enumerate(iterator):
            assert x[0].shape == images[:2].shape
            assert (x[1] == x_misc1[(i * 2):((i + 1) * 2)]).all()
            if i == 2:
                break

        # Test with two miscellaneous inputs
        x_misc2 = np.random.random((dsize, 3, 3))
        iterator = numpy_array_iterator.NumpyArrayIterator(
            (images, [x_misc1, x_misc2]),
            np.arange(dsize),
            image_data_generator,
            shuffle=False,
            batch_size=2
        )
        for i, (x, y) in enumerate(iterator):
            assert x[0].shape == images[:2].shape
            assert (x[1] == x_misc1[(i * 2):((i + 1) * 2)]).all()
            assert (x[2] == x_misc2[(i * 2):((i + 1) * 2)]).all()
            if i == 2:
                break

        # Test cases with `y = None`
        iterator = numpy_array_iterator.NumpyArrayIterator(
            images,
            None,
            image_data_generator,
            batch_size=3
        )
        x = iterator.next()
        assert type(x) is np.ndarray
        assert x.shape == images[:3].shape

        iterator = numpy_array_iterator.NumpyArrayIterator(
            (images, x_misc1),
            None,
            image_data_generator,
            batch_size=3,
            shuffle=False
        )
        x = iterator.next()
        assert type(x) is list
        assert x[0].shape == images[:3].shape
        assert (x[1] == x_misc1[:3]).all()

        iterator = numpy_array_iterator.NumpyArrayIterator(
            (images, [x_misc1, x_misc2]),
            None,
            image_data_generator,
            batch_size=3,
            shuffle=False
        )
        x = iterator.next()
        assert type(x) is list
        assert x[0].shape == images[:3].shape
        assert (x[1] == x_misc1[:3]).all()
        assert (x[2] == x_misc2[:3]).all()

        # Test with validation split
        generator = ImageDataGenerator(validation_split=0.2)
        iterator = numpy_array_iterator.NumpyArrayIterator(
            images,
            None,
            generator,
            batch_size=3
        )
        x = iterator.next()
        assert isinstance(x, np.ndarray)
        assert x.shape == images[:3].shape

        # Test some failure cases:
        x_misc_err = np.random.random((dsize + 1, 3, 3))

        with pytest.raises(ValueError) as e_info:
            numpy_array_iterator.NumpyArrayIterator(
                (images, x_misc_err),
                np.arange(dsize),
                generator,
                batch_size=3
            )
        assert str(e_info.value).find('All of the arrays in') != -1

        with pytest.raises(ValueError) as e_info:
            numpy_array_iterator.NumpyArrayIterator(
                (images, x_misc1),
                np.arange(dsize + 1),
                generator,
                batch_size=3
            )
        assert str(e_info.value).find('`x` (images tensor) and `y` (labels) ') != -1

        # Test `flow` behavior as Sequence
        seq = numpy_array_iterator.NumpyArrayIterator(
            images,
            np.arange(images.shape[0]),
            generator,
            shuffle=False, save_to_dir=str(tmpdir),
            batch_size=3
        )
        assert len(seq) == images.shape[0] // 3 + 1
        x, y = seq[0]
        assert x.shape == images[:3].shape
        assert list(y) == [0, 1, 2]

        # Test with `shuffle=True`
        seq = numpy_array_iterator.NumpyArrayIterator(
            images,
            np.arange(images.shape[0]),
            generator,
            shuffle=True,
            save_to_dir=str(tmpdir),
            batch_size=3,
            seed=123
        )
        x, y = seq[0]
        # Check that the sequence is shuffled.
        assert list(y) != [0, 1, 2]
        # `on_epoch_end` should reshuffle the sequence.
        seq.on_epoch_end()
        x2, y2 = seq[0]
        assert list(y) != list(y2)

    # test order_interpolation
    labels = np.array([[2, 2, 0, 2, 2],
                       [1, 3, 2, 3, 1],
                       [2, 1, 0, 1, 2],
                       [3, 1, 0, 2, 0],
                       [3, 1, 3, 2, 1]])
    label_generator = ImageDataGenerator(
        rotation_range=90.,
        interpolation_order=0
    )
    labels_gen = numpy_array_iterator.NumpyArrayIterator(
        labels[np.newaxis, ..., np.newaxis],
        None,
        label_generator,
        seed=123
    )
    assert (np.unique(labels) == np.unique(next(labels_gen))).all()
