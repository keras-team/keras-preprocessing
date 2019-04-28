import numpy as np
import pytest

from PIL import Image

from keras_preprocessing.image import image_data_generator
from keras_preprocessing.image import utils


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


def test_image_data_generator(all_test_images):
    for test_images in all_test_images:
        img_list = []
        for im in test_images:
            img_list.append(utils.img_to_array(im)[None, ...])

        image_data_generator.ImageDataGenerator(
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


def test_image_data_generator_with_validation_split(all_test_images):
    for test_images in all_test_images:
        img_list = []
        for im in test_images:
            img_list.append(utils.img_to_array(im)[None, ...])

        images = np.vstack(img_list)
        labels = np.concatenate([
            np.zeros((int(len(images) / 2),)),
            np.ones((int(len(images) / 2),))])
        generator = image_data_generator.ImageDataGenerator(validation_split=0.5)

        # training and validation sets would have different
        # number of classes, because labels are sorted
        with pytest.raises(ValueError,
                           match='Training and validation subsets '
                                 'have different number of classes after '
                                 'the split.*'):
            generator.flow(images, labels,
                           shuffle=False, batch_size=10,
                           subset='validation')

        labels = np.concatenate([
            np.zeros((int(len(images) / 4),)),
            np.ones((int(len(images) / 4),)),
            np.zeros((int(len(images) / 4),)),
            np.ones((int(len(images) / 4),))
        ])

        seq = generator.flow(images, labels,
                             shuffle=False, batch_size=10,
                             subset='validation')

        x, y = seq[0]
        assert 2 == len(np.unique(y))

        seq = generator.flow(images, labels,
                             shuffle=False, batch_size=10,
                             subset='training')
        x2, y2 = seq[0]
        assert 2 == len(np.unique(y2))

        with pytest.raises(ValueError):
            generator.flow(images, np.arange(images.shape[0]),
                           shuffle=False, batch_size=3,
                           subset='foo')


def test_image_data_generator_with_split_value_error():
    with pytest.raises(ValueError):
        image_data_generator.ImageDataGenerator(validation_split=5)


def test_image_data_generator_invalid_data():
    generator = image_data_generator.ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        data_format='channels_last')
    # Test fit with invalid data
    with pytest.raises(ValueError):
        x = np.random.random((3, 10, 10))
        generator.fit(x)

    # Test flow with invalid data
    with pytest.raises(ValueError):
        x = np.random.random((32, 10, 10))
        generator.flow(np.arange(x.shape[0]))


def test_image_data_generator_fit():
    generator = image_data_generator.ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=(0.2, 0.2),
        channel_shift_range=0.,
        brightness_range=(1, 5),
        fill_mode='nearest',
        cval=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        interpolation_order=1,
        data_format='channels_last'
    )
    x = np.random.random((32, 10, 10, 3))
    generator.fit(x, augment=True)
    # Test grayscale
    x = np.random.random((32, 10, 10, 1))
    generator.fit(x)
    # Test RBG
    x = np.random.random((32, 10, 10, 3))
    generator.fit(x)
    # Test more samples than dims
    x = np.random.random((32, 4, 4, 1))
    generator.fit(x)
    generator = image_data_generator.ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=True,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=(0.2, 0.2),
        channel_shift_range=0.,
        brightness_range=(1, 5),
        fill_mode='nearest',
        cval=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        interpolation_order=1,
        data_format='channels_first'
    )
    x = np.random.random((32, 10, 10, 3))
    generator.fit(x, augment=True)
    # Test grayscale
    x = np.random.random((32, 1, 10, 10))
    generator.fit(x)
    # Test RBG
    x = np.random.random((32, 3, 10, 10))
    generator.fit(x)
    # Test more samples than dims
    x = np.random.random((32, 1, 4, 4))
    generator.fit(x)


def test_image_data_generator_flow(all_test_images, tmpdir):
    for test_images in all_test_images:
        img_list = []
        for im in test_images:
            img_list.append(utils.img_to_array(im)[None, ...])

        images = np.vstack(img_list)
        dsize = images.shape[0]
        generator = image_data_generator.ImageDataGenerator(
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

        generator.flow(
            images,
            np.arange(images.shape[0]),
            shuffle=False,
            save_to_dir=str(tmpdir),
            batch_size=3
        )

        generator.flow(
            images,
            np.arange(images.shape[0]),
            shuffle=False,
            sample_weight=np.arange(images.shape[0]) + 1,
            save_to_dir=str(tmpdir),
            batch_size=3
        )

        # Test with `shuffle=True`
        generator.flow(
            images, np.arange(images.shape[0]),
            shuffle=True,
            save_to_dir=str(tmpdir),
            batch_size=3,
            seed=42
        )

        # Test without y
        generator.flow(
            images,
            None,
            shuffle=True,
            save_to_dir=str(tmpdir),
            batch_size=3
        )

        # Test with a single miscellaneous input data array
        x_misc1 = np.random.random(dsize)
        generator.flow(
            (images, x_misc1),
            np.arange(dsize),
            shuffle=False,
            batch_size=2
        )

        # Test with two miscellaneous inputs
        x_misc2 = np.random.random((dsize, 3, 3))
        generator.flow(
            (images, [x_misc1, x_misc2]),
            np.arange(dsize),
            shuffle=False,
            batch_size=2
        )

        # Test cases with `y = None`
        generator.flow(images, None, batch_size=3)
        generator.flow((images, x_misc1), None, batch_size=3, shuffle=False)
        generator.flow(
            (images, [x_misc1, x_misc2]),
            None,
            batch_size=3,
            shuffle=False
        )
        generator = image_data_generator.ImageDataGenerator(validation_split=0.2)
        generator.flow(images, batch_size=3)

        # Test some failure cases:
        x_misc_err = np.random.random((dsize + 1, 3, 3))
        with pytest.raises(ValueError) as e_info:
            generator.flow((images, x_misc_err), np.arange(dsize), batch_size=3)
        assert str(e_info.value).find('All of the arrays in') != -1

        with pytest.raises(ValueError) as e_info:
            generator.flow((images, x_misc1), np.arange(dsize + 1), batch_size=3)
        assert str(e_info.value).find('`x` (images tensor) and `y` (labels) ') != -1

        # Test `flow` behavior as Sequence
        generator.flow(
            images,
            np.arange(images.shape[0]),
            shuffle=False,
            save_to_dir=str(tmpdir),
            batch_size=3
        )

        # Test with `shuffle=True`
        generator.flow(
            images,
            np.arange(images.shape[0]),
            shuffle=True, save_to_dir=str(tmpdir),
            batch_size=3, seed=123
        )

    # test order_interpolation
    labels = np.array([[2, 2, 0, 2, 2],
                       [1, 3, 2, 3, 1],
                       [2, 1, 0, 1, 2],
                       [3, 1, 0, 2, 0],
                       [3, 1, 3, 2, 1]])

    label_generator = image_data_generator.ImageDataGenerator(
        rotation_range=90.,
        interpolation_order=0
    )
    label_generator.flow(
        x=labels[np.newaxis, ..., np.newaxis],
        seed=123
    )


def test_valid_args():
    with pytest.raises(ValueError):
        image_data_generator.ImageDataGenerator(brightness_range=0.1)


def test_batch_standardize(all_test_images):
    # ImageDataGenerator.standardize should work on batches
    for test_images in all_test_images:
        img_list = []
        for im in test_images:
            img_list.append(utils.img_to_array(im)[None, ...])

        images = np.vstack(img_list)
        generator = image_data_generator.ImageDataGenerator(
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
            vertical_flip=True)
        generator.fit(images, augment=True)

        transformed = np.copy(images)
        for i, im in enumerate(transformed):
            transformed[i] = generator.random_transform(im)
        transformed = generator.standardize(transformed)


def test_deterministic_transform():
    x = np.ones((32, 32, 3))
    generator = image_data_generator.ImageDataGenerator(
        rotation_range=90,
        fill_mode='constant')
    x = np.random.random((32, 32, 3))
    assert np.allclose(generator.apply_transform(x, {'flip_vertical': True}),
                       x[::-1, :, :])
    assert np.allclose(generator.apply_transform(x, {'flip_horizontal': True}),
                       x[:, ::-1, :])
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
    assert np.allclose(generator.apply_transform(x, {'theta': 45}),
                       x_rotated)


def test_random_transforms():
    x = np.random.random((2, 28, 28))
    # Test get_random_transform with predefined seed
    seed = 1
    generator = image_data_generator.ImageDataGenerator(
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=0.2,
        channel_shift_range=0.1,
        brightness_range=(1, 5),
        horizontal_flip=True,
        vertical_flip=True)
    transform_dict = generator.get_random_transform(x.shape, seed)
    transform_dict2 = generator.get_random_transform(x.shape, seed * 2)
    assert transform_dict['theta'] != 0
    assert transform_dict['theta'] != transform_dict2['theta']
    assert transform_dict['tx'] != 0
    assert transform_dict['tx'] != transform_dict2['tx']
    assert transform_dict['ty'] != 0
    assert transform_dict['ty'] != transform_dict2['ty']
    assert transform_dict['shear'] != 0
    assert transform_dict['shear'] != transform_dict2['shear']
    assert transform_dict['zx'] != 0
    assert transform_dict['zx'] != transform_dict2['zx']
    assert transform_dict['zy'] != 0
    assert transform_dict['zy'] != transform_dict2['zy']
    assert transform_dict['channel_shift_intensity'] != 0
    assert (transform_dict['channel_shift_intensity'] !=
            transform_dict2['channel_shift_intensity'])
    assert transform_dict['brightness'] != 0
    assert transform_dict['brightness'] != transform_dict2['brightness']

    # Test get_random_transform without any randomness
    generator = image_data_generator.ImageDataGenerator()
    transform_dict = generator.get_random_transform(x.shape, seed)
    assert transform_dict['theta'] == 0
    assert transform_dict['tx'] == 0
    assert transform_dict['ty'] == 0
    assert transform_dict['shear'] == 0
    assert transform_dict['zx'] == 1
    assert transform_dict['zy'] == 1
    assert transform_dict['channel_shift_intensity'] is None
    assert transform_dict['brightness'] is None


if __name__ == '__main__':
    pytest.main([__file__])
