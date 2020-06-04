import os
import shutil
import tempfile

import numpy as np
import pytest

from PIL import Image

from keras_preprocessing.image import image_data_generator


@pytest.fixture(scope='module')
def all_test_images():
    img_w = img_h = 20
    rgb_images = []
    rgba_images = []
    gray_images = []
    gray_images_16bit = []
    gray_images_32bit = []
    for n in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        # RGB
        imarray = np.random.rand(img_w, img_h, 3) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        rgb_images.append(im)
        # RGBA
        imarray = np.random.rand(img_w, img_h, 4) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        rgba_images.append(im)
        # 8-bit grayscale
        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = Image.fromarray(imarray.astype('uint8').squeeze()).convert('L')
        gray_images.append(im)
        # 16-bit grayscale
        imarray = np.array(
            np.random.randint(-2147483648, 2147483647, (img_w, img_h))
        )
        im = Image.fromarray(imarray.astype('uint16'))
        gray_images_16bit.append(im)
        # 32-bit grayscale
        im = Image.fromarray(imarray.astype('uint32'))
        gray_images_32bit.append(im)

    return [rgb_images, rgba_images,
            gray_images, gray_images_16bit, gray_images_32bit]


def test_directory_iterator(all_test_images, tmpdir):
    num_classes = 2

    # create folders and subfolders
    paths = []
    for cl in range(num_classes):
        class_directory = 'class-{}'.format(cl)
        classpaths = [
            class_directory,
            os.path.join(class_directory, 'subfolder-1'),
            os.path.join(class_directory, 'subfolder-2'),
            os.path.join(class_directory, 'subfolder-1', 'sub-subfolder')
        ]
        for path in classpaths:
            tmpdir.join(path).mkdir()
        paths.append(classpaths)

    # save the images in the paths
    count = 0
    filenames = []
    for test_images in all_test_images:
        for im in test_images:
            # rotate image class
            im_class = count % num_classes
            # rotate subfolders
            classpaths = paths[im_class]
            filename = os.path.join(
                classpaths[count % len(classpaths)],
                'image-{}.png'.format(count))
            filenames.append(filename)
            im.save(str(tmpdir / filename))
            count += 1

    # create iterator
    generator = image_data_generator.ImageDataGenerator()
    dir_iterator = generator.flow_from_directory(str(tmpdir))

    # check number of classes and images
    assert len(dir_iterator.class_indices) == num_classes
    assert len(dir_iterator.classes) == count
    assert set(dir_iterator.filenames) == set(filenames)

    # Test invalid use cases
    with pytest.raises(ValueError):
        generator.flow_from_directory(str(tmpdir), color_mode='cmyk')
    with pytest.raises(ValueError):
        generator.flow_from_directory(str(tmpdir), class_mode='output')

    def preprocessing_function(x):
        """This will fail if not provided by a Numpy array.
        Note: This is made to enforce backward compatibility.
        """

        assert x.shape == (26, 26, 3)
        assert type(x) is np.ndarray

        return np.zeros_like(x)

    # Test usage as Sequence
    generator = image_data_generator.ImageDataGenerator(
        preprocessing_function=preprocessing_function)
    dir_seq = generator.flow_from_directory(str(tmpdir),
                                            target_size=(26, 26),
                                            color_mode='rgb',
                                            batch_size=3,
                                            class_mode='categorical')
    assert len(dir_seq) == np.ceil(count / 3.)
    x1, y1 = dir_seq[1]
    assert x1.shape == (3, 26, 26, 3)
    assert y1.shape == (3, num_classes)
    x1, y1 = dir_seq[5]
    assert (x1 == 0).all()

    with pytest.raises(ValueError):
        x1, y1 = dir_seq[14]  # there are 40 images and batch size is 3


def test_directory_iterator_class_mode_input(all_test_images, tmpdir):
    tmpdir.join('class-1').mkdir()

    # save the images in the paths
    count = 0
    for test_images in all_test_images:
        for im in test_images:
            filename = str(
                tmpdir / 'class-1' / 'image-{}.png'.format(count))
            im.save(filename)
            count += 1

    # create iterator
    generator = image_data_generator.ImageDataGenerator()
    dir_iterator = generator.flow_from_directory(str(tmpdir),
                                                 class_mode='input')
    batch = next(dir_iterator)

    # check if input and output have the same shape
    assert(batch[0].shape == batch[1].shape)
    # check if the input and output images are not the same numpy array
    input_img = batch[0][0]
    output_img = batch[1][0]
    output_img[0][0][0] += 1
    assert(input_img[0][0][0] != output_img[0][0][0])


@pytest.mark.parametrize('validation_split,num_training', [
    (0.25, 30),
    (0.50, 20),
    (0.75, 10),
])
def test_directory_iterator_with_validation_split(all_test_images,
                                                  validation_split,
                                                  num_training):
    num_classes = 2
    tmp_folder = tempfile.mkdtemp(prefix='test_images')

    # create folders and subfolders
    paths = []
    for cl in range(num_classes):
        class_directory = 'class-{}'.format(cl)
        classpaths = [
            class_directory,
            os.path.join(class_directory, 'subfolder-1'),
            os.path.join(class_directory, 'subfolder-2'),
            os.path.join(class_directory, 'subfolder-1', 'sub-subfolder')
        ]
        for path in classpaths:
            os.mkdir(os.path.join(tmp_folder, path))
        paths.append(classpaths)

    # save the images in the paths
    count = 0
    filenames = []
    for test_images in all_test_images:
        for im in test_images:
            # rotate image class
            im_class = count % num_classes
            # rotate subfolders
            classpaths = paths[im_class]
            filename = os.path.join(
                classpaths[count % len(classpaths)],
                'image-{}.png'.format(count))
            filenames.append(filename)
            im.save(os.path.join(tmp_folder, filename))
            count += 1

    # create iterator
    generator = image_data_generator.ImageDataGenerator(
        validation_split=validation_split
    )

    with pytest.raises(ValueError):
        generator.flow_from_directory(tmp_folder, subset='foo')

    train_iterator = generator.flow_from_directory(tmp_folder,
                                                   subset='training')
    assert train_iterator.samples == num_training

    valid_iterator = generator.flow_from_directory(tmp_folder,
                                                   subset='validation')
    assert valid_iterator.samples == count - num_training

    # check number of classes and images
    assert len(train_iterator.class_indices) == num_classes
    assert len(train_iterator.classes) == num_training
    assert len(set(train_iterator.filenames) &
               set(filenames)) == num_training

    shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    pytest.main([__file__])
