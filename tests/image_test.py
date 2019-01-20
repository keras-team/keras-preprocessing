import pytest
from PIL import Image
import numpy as np
import os
import tempfile
import shutil
import keras
import pandas as pd
import random

# TODO: remove the 3 lines below once the Keras release
# is configured to use keras_preprocessing
import keras_preprocessing
keras_preprocessing.set_keras_submodules(
    backend=keras.backend, utils=keras.utils)

# This enables this import
from keras_preprocessing import image


class TestImage(object):

    def setup_class(cls):
        cls.img_w = cls.img_h = 20
        rgb_images = []
        rgba_images = []
        gray_images = []
        for n in range(8):
            bias = np.random.rand(cls.img_w, cls.img_h, 1) * 64
            variance = np.random.rand(cls.img_w, cls.img_h, 1) * (255 - 64)
            imarray = np.random.rand(cls.img_w, cls.img_h, 3) * variance + bias
            im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
            rgb_images.append(im)

            imarray = np.random.rand(cls.img_w, cls.img_h, 4) * variance + bias
            im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
            rgba_images.append(im)

            imarray = np.random.rand(cls.img_w, cls.img_h, 1) * variance + bias
            im = Image.fromarray(
                imarray.astype('uint8').squeeze()).convert('L')
            gray_images.append(im)

        cls.all_test_images = [rgb_images, rgba_images, gray_images]

    def teardown_class(cls):
        del cls.all_test_images

    def test_image_data_generator(self, tmpdir):
        for test_images in self.all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image.ImageDataGenerator(
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
                interpolation_order=1)
            generator.fit(images, augment=True)

            for x, y in generator.flow(images, np.arange(images.shape[0]),
                                       shuffle=False,
                                       save_to_dir=str(tmpdir),
                                       batch_size=3):
                assert x.shape == images[:3].shape
                assert list(y) == [0, 1, 2]
                break

            # Test with sample weights
            for x, y, w in generator.flow(
                    images, np.arange(images.shape[0]),
                    shuffle=False,
                    sample_weight=np.arange(images.shape[0]) + 1,
                    save_to_dir=str(tmpdir),
                    batch_size=3):
                assert x.shape == images[:3].shape
                assert list(y) == [0, 1, 2]
                assert list(w) == [1, 2, 3]
                break

            # Test with `shuffle=True`
            for x, y in generator.flow(images, np.arange(images.shape[0]),
                                       shuffle=True,
                                       save_to_dir=str(tmpdir),
                                       batch_size=3,
                                       seed=42):
                assert x.shape == images[:3].shape
                # Check that the sequence is shuffled.
                assert list(y) != [0, 1, 2]
                break

            # Test without y
            for x in generator.flow(images, None,
                                    shuffle=True,
                                    save_to_dir=str(tmpdir),
                                    batch_size=3):
                assert type(x) is np.ndarray
                assert x.shape == images[:3].shape
                # Check that the sequence is shuffled.
                break

            # Test with a single miscellaneous input data array
            dsize = images.shape[0]
            x_misc1 = np.random.random(dsize)

            for i, (x, y) in enumerate(generator.flow(
                    (images, x_misc1),
                    np.arange(dsize),
                    shuffle=False,
                    batch_size=2)):
                assert x[0].shape == images[:2].shape
                assert (x[1] == x_misc1[(i * 2):((i + 1) * 2)]).all()
                if i == 2:
                    break

            # Test with two miscellaneous inputs
            x_misc2 = np.random.random((dsize, 3, 3))

            for i, (x, y) in enumerate(generator.flow(
                    (images, [x_misc1, x_misc2]),
                    np.arange(dsize),
                    shuffle=False,
                    batch_size=2)):
                assert x[0].shape == images[:2].shape
                assert (x[1] == x_misc1[(i * 2):((i + 1) * 2)]).all()
                assert (x[2] == x_misc2[(i * 2):((i + 1) * 2)]).all()
                if i == 2:
                    break

            # Test cases with `y = None`
            x = generator.flow(images, None, batch_size=3).next()
            assert type(x) is np.ndarray
            assert x.shape == images[:3].shape
            x = generator.flow((images, x_misc1), None,
                               batch_size=3, shuffle=False).next()
            assert type(x) is list
            assert x[0].shape == images[:3].shape
            assert (x[1] == x_misc1[:3]).all()
            x = generator.flow((images, [x_misc1, x_misc2]), None,
                               batch_size=3, shuffle=False).next()
            assert type(x) is list
            assert x[0].shape == images[:3].shape
            assert (x[1] == x_misc1[:3]).all()
            assert (x[2] == x_misc2[:3]).all()

            generator = image.ImageDataGenerator(validation_split=0.2)
            x = generator.flow(images, batch_size=3).next()
            assert isinstance(x, np.ndarray)
            assert x.shape == images[:3].shape

            # Test some failure cases:
            x_misc_err = np.random.random((dsize + 1, 3, 3))

            with pytest.raises(ValueError) as e_info:
                generator.flow((images, x_misc_err), np.arange(dsize),
                               batch_size=3)
            assert str(e_info.value).find(
                'All of the arrays in') != -1

            with pytest.raises(ValueError) as e_info:
                generator.flow((images, x_misc1), np.arange(dsize + 1),
                               batch_size=3)
            assert str(e_info.value).find(
                '`x` (images tensor) and `y` (labels) ') != -1

            # Test `flow` behavior as Sequence
            seq = generator.flow(images, np.arange(images.shape[0]),
                                 shuffle=False, save_to_dir=str(tmpdir),
                                 batch_size=3)
            assert len(seq) == images.shape[0] // 3 + 1
            x, y = seq[0]
            assert x.shape == images[:3].shape
            assert list(y) == [0, 1, 2]

            # Test with `shuffle=True`
            seq = generator.flow(images, np.arange(images.shape[0]),
                                 shuffle=True, save_to_dir=str(tmpdir),
                                 batch_size=3, seed=123)
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

        label_generator = image.ImageDataGenerator(rotation_range=90.,
                                                   interpolation_order=0)
        labels_gen = label_generator.flow(x=labels[np.newaxis, ..., np.newaxis],
                                          seed=123)
        assert (np.unique(labels) == np.unique(next(labels_gen))).all()

    def test_image_data_generator_with_validation_split(self):
        for test_images in self.all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            labels = np.concatenate([
                np.zeros((int(len(images) / 2),)),
                np.ones((int(len(images) / 2),))])
            generator = image.ImageDataGenerator(validation_split=0.5)

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

    def test_image_data_generator_with_split_value_error(self):
        with pytest.raises(ValueError):
            generator = image.ImageDataGenerator(validation_split=5)

    def test_image_data_generator_invalid_data(self):
        generator = image.ImageDataGenerator(
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

    def test_image_data_generator_fit(self):
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            zoom_range=(0.2, 0.2),
            data_format='channels_last')
        # Test grayscale
        x = np.random.random((32, 10, 10, 1))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 10, 10, 3))
        generator.fit(x)
        # Test more samples than dims
        x = np.random.random((32, 4, 4, 1))
        generator.fit(x)
        generator = image.ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_first')
        # Test grayscale
        x = np.random.random((32, 1, 10, 10))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 3, 10, 10))
        generator.fit(x)
        # Test more samples than dims
        x = np.random.random((32, 1, 4, 4))
        generator.fit(x)

    def test_directory_iterator(self, tmpdir):
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
        for test_images in self.all_test_images:
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
        generator = image.ImageDataGenerator()
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
        generator = image.ImageDataGenerator(
            preprocessing_function=preprocessing_function)
        dir_seq = generator.flow_from_directory(str(tmpdir),
                                                target_size=(26, 26),
                                                color_mode='rgb',
                                                batch_size=3,
                                                class_mode='categorical')
        assert len(dir_seq) == np.ceil(count / 3)
        x1, y1 = dir_seq[1]
        assert x1.shape == (3, 26, 26, 3)
        assert y1.shape == (3, num_classes)
        x1, y1 = dir_seq[5]
        assert (x1 == 0).all()

        with pytest.raises(ValueError):
            x1, y1 = dir_seq[9]

    def test_directory_iterator_class_mode_input(self, tmpdir):
        tmpdir.join('class-1').mkdir()

        # save the images in the paths
        count = 0
        for test_images in self.all_test_images:
            for im in test_images:
                filename = str(
                    tmpdir / 'class-1' / 'image-{}.png'.format(count))
                im.save(filename)
                count += 1

        # create iterator
        generator = image.ImageDataGenerator()
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
        (0.25, 18),
        (0.50, 12),
        (0.75, 6),
    ])
    def test_directory_iterator_with_validation_split(self,
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
        for test_images in self.all_test_images:
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
        generator = image.ImageDataGenerator(validation_split=validation_split)

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

    def test_dataframe_iterator(self, tmpdir):
        num_classes = 2

        # save the images in the tmpdir
        count = 0
        filenames = []
        filepaths = []
        filenames_without = []
        for test_images in self.all_test_images:
            for im in test_images:
                filename = "image-{}.png".format(count)
                filename_without = "image-{}".format(count)
                filenames.append(filename)
                filepaths.append(os.path.join(str(tmpdir), filename))
                filenames_without.append(filename_without)
                im.save(str(tmpdir / filename))
                count += 1

        df = pd.DataFrame({
            "filename": filenames,
            "class": [str(random.randint(0, 1)) for _ in filenames],
            "filepaths": filepaths
        })

        # create iterator
        iterator = image.DataFrameIterator(df, str(tmpdir))
        batch = next(iterator)
        assert len(batch) == 2
        assert isinstance(batch[0], np.ndarray)
        assert isinstance(batch[1], np.ndarray)
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(df, x_col='filepaths')
        df_iterator_dir = generator.flow_from_dataframe(df, str(tmpdir))
        df_sparse_iterator = generator.flow_from_dataframe(df, str(tmpdir),
                                                           class_mode="sparse")
        if np.isnan(df_sparse_iterator.classes).any():
            raise ValueError('Invalid values.')

        df_regression = pd.DataFrame({"filename": filenames,
                                      "col1": [random.randrange(0, 1)
                                               for _ in filenames],
                                      "col2": [random.randrange(0, 1)
                                               for _ in filenames]})
        df_multiple_y_iterator = generator.flow_from_dataframe(
            df_regression, str(tmpdir), y_col=["col1", "col2"], class_mode="other")
        df_regression = pd.DataFrame({"filename": filenames,
                                      "col1": [random.randrange(0, 1)
                                               for _ in filenames],
                                      "col2": [random.randrange(0, 1)
                                               for _ in filenames]},
                                     dtype=str)
        batch_x, batch_y = next(df_multiple_y_iterator)
        with pytest.raises(TypeError):
            df_multiple_y_iterator = generator.flow_from_dataframe(
                df_regression, str(tmpdir), y_col=["col1", "col2"],
                class_mode="other")
        with pytest.raises(TypeError):
            df_single_y_iterator = generator.flow_from_dataframe(
                df_regression, str(tmpdir), y_col="col1", class_mode="other")
        # check number of classes and images
        assert len(df_iterator.class_indices) == num_classes
        assert len(df_iterator.classes) == count
        assert set(df_iterator.filenames) == set(filepaths)
        assert len(df_iterator_dir.class_indices) == num_classes
        assert len(df_iterator_dir.classes) == count
        assert set(df_iterator_dir.filenames) == set(filenames)
        assert batch_y.shape[1] == 2
        # test shuffle=False
        _, batch_y = next(generator.flow_from_dataframe(df, str(tmpdir),
                                                        shuffle=False,
                                                        class_mode="sparse"))
        assert (batch_y == df['class'].astype('float')[:len(batch_y)]).all()
        # Test invalid use cases
        with pytest.raises(ValueError):
            generator.flow_from_dataframe(df, str(tmpdir), color_mode='cmyk')
        with pytest.raises(ValueError):
            generator.flow_from_dataframe(df, str(tmpdir), class_mode='output')
        with pytest.warns(DeprecationWarning):
            generator.flow_from_dataframe(df, str(tmpdir), has_ext=True)
        with pytest.warns(DeprecationWarning):
            generator.flow_from_dataframe(df, str(tmpdir), has_ext=False)

        def preprocessing_function(x):
            """This will fail if not provided by a Numpy array.
            Note: This is made to enforce backward compatibility.
            """

            assert x.shape == (26, 26, 3)
            assert type(x) is np.ndarray

            return np.zeros_like(x)

        # Test usage as Sequence
        generator = image.ImageDataGenerator(
            preprocessing_function=preprocessing_function)
        dir_seq = generator.flow_from_dataframe(df, str(tmpdir),
                                                target_size=(26, 26),
                                                color_mode='rgb',
                                                batch_size=3,
                                                class_mode='categorical')
        assert len(dir_seq) == np.ceil(count / 3)
        x1, y1 = dir_seq[1]
        assert x1.shape == (3, 26, 26, 3)
        assert y1.shape == (3, num_classes)
        x1, y1 = dir_seq[5]
        assert (x1 == 0).all()

        with pytest.raises(ValueError):
            x1, y1 = dir_seq[9]

    def test_valid_args(self):
        with pytest.raises(ValueError):
            image.ImageDataGenerator(brightness_range=0.1)

    def test_dataframe_iterator_class_mode_input(self, tmpdir):
        # save the images in the paths
        count = 0
        filenames = []
        for test_images in self.all_test_images:
            for im in test_images:
                filename = 'image-{}.png'.format(count)
                im.save(str(tmpdir / filename))
                filenames.append(filename)
                count += 1
        df = pd.DataFrame({"filename": filenames})
        generator = image.ImageDataGenerator()
        df_autoencoder_iterator = generator.flow_from_dataframe(df, str(tmpdir),
                                                                x_col="filename",
                                                                y_col=None,
                                                                class_mode="input")

        batch = next(df_autoencoder_iterator)

        # check if input and output have the same shape and they're the same
        assert(batch[0].all() == batch[1].all())
        # check if the input and output images are not the same numpy array
        input_img = batch[0][0]
        output_img = batch[1][0]
        output_img[0][0][0] += 1
        assert(input_img[0][0][0] != output_img[0][0][0])

        df_autoencoder_iterator = generator.flow_from_dataframe(df, str(tmpdir),
                                                                x_col="filename",
                                                                y_col="class",
                                                                class_mode="input")

        batch = next(df_autoencoder_iterator)

        # check if input and output have the same shape and they're the same
        assert(batch[0].all() == batch[1].all())
        # check if the input and output images are not the same numpy array
        input_img = batch[0][0]
        output_img = batch[1][0]
        output_img[0][0][0] += 1
        assert(input_img[0][0][0] != output_img[0][0][0])

    def test_dataframe_iterator_class_mode_categorical_multi_label(self, tmpdir):
        # save the images in the paths
        filenames = []
        count = 0
        for test_images in self.all_test_images:
            for im in test_images:
                filename = 'image-{}.png'.format(count)
                im.save(str(tmpdir / filename))
                filenames.append(filename)
                count += 1
        label_opt = ['a', 'b', ['a'], ['b'], ['a', 'b'], ['b', 'a']]
        df = pd.DataFrame({
            "filename": filenames,
            "class": [random.choice(label_opt) for _ in filenames[:-2]] + ['b', 'a']
        })
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(df, str(tmpdir))
        batch_x, batch_y = next(df_iterator)
        assert isinstance(batch_x, np.ndarray)
        assert len(batch_x.shape) == 4
        assert isinstance(batch_y, np.ndarray)
        assert batch_y.shape == (len(batch_x), 2)
        for labels in batch_y:
            assert all(l in {0, 1} for l in labels)

        # use OrderedDict to mantain order in python 2.7 and allow for checks
        # on first 3 batches
        df = pd.DataFrame({
            "filename": filenames,
            "class": [['b', 'a']] + ['b'] + [['c']] + [random.choice(label_opt)
                                                       for _ in filenames[:-3]]
        })
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(df, str(tmpdir), shuffle=False)
        batch_x, batch_y = next(df_iterator)
        assert isinstance(batch_x, np.ndarray)
        assert len(batch_x.shape) == 4
        assert isinstance(batch_y, np.ndarray)
        assert batch_y.shape == (len(batch_x), 3)
        for labels in batch_y:
            assert all(l in {0, 1} for l in labels)
        assert (batch_y[0] == np.array([1, 1, 0])).all()
        assert (batch_y[1] == np.array([0, 1, 0])).all()
        assert (batch_y[2] == np.array([0, 0, 1])).all()

    @pytest.mark.parametrize('validation_split,num_training', [
        (0.25, 18),
        (0.50, 12),
        (0.75, 6),
    ])
    def test_dataframe_iterator_with_validation_split(self,
                                                      validation_split,
                                                      num_training, tmpdir):
        num_classes = 2

        # save the images in the tmpdir
        count = 0
        filenames = []
        filenames_without = []
        for test_images in self.all_test_images:
            for im in test_images:
                filename = "image-{}.png".format(count)
                filename_without = "image-{}".format(count)
                filenames.append(filename)
                filenames_without.append(filename_without)
                im.save(str(tmpdir / filename))
                count += 1

        df = pd.DataFrame({"filename": filenames,
                           "class": [str(random.randint(0, 1)) for _ in filenames]})
        # create iterator
        generator = image.ImageDataGenerator(validation_split=validation_split)
        df_sparse_iterator = generator.flow_from_dataframe(df,
                                                           str(tmpdir),
                                                           class_mode="sparse")
        if np.isnan(next(df_sparse_iterator)[:][1]).any():
            raise ValueError('Invalid values.')

        with pytest.raises(ValueError):
            generator.flow_from_dataframe(
                df, tmpdir, subset='foo')

        train_iterator = generator.flow_from_dataframe(df, str(tmpdir),
                                                       subset='training')
        assert train_iterator.samples == num_training

        valid_iterator = generator.flow_from_dataframe(df, str(tmpdir),
                                                       subset='validation')
        assert valid_iterator.samples == count - num_training

        # check number of classes and images
        assert len(train_iterator.class_indices) == num_classes
        assert len(train_iterator.classes) == num_training
        assert len(set(train_iterator.filenames) &
                   set(filenames)) == num_training

    def test_dataframe_iterator_with_custom_indexed_dataframe(self, tmpdir):
        num_classes = 2

        # save the images in the tmpdir
        count = 0
        filenames = []
        for test_images in self.all_test_images:
            for im in test_images:
                filename = "image-{}.png".format(count)
                filenames.append(filename)
                im.save(str(tmpdir / filename))
                count += 1

        # create dataframes
        classes = np.random.randint(num_classes, size=len(filenames))
        classes = [str(c) for c in classes]
        df = pd.DataFrame({"filename": filenames,
                           "class": classes})
        df2 = pd.DataFrame({"filename": filenames,
                            "class": classes},
                           index=np.arange(1, len(filenames) + 1))
        df3 = pd.DataFrame({"filename": filenames,
                            "class": classes},
                           index=filenames)

        # create iterators
        seed = 1
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, str(tmpdir), seed=seed)
        df2_iterator = generator.flow_from_dataframe(
            df2, str(tmpdir), seed=seed)
        df3_iterator = generator.flow_from_dataframe(
            df3, str(tmpdir), seed=seed)

        # Test all iterators return same pairs of arrays
        for _ in range(len(filenames)):
            a1, c1 = next(df_iterator)
            a2, c2 = next(df2_iterator)
            a3, c3 = next(df3_iterator)
            assert np.array_equal(a1, a2)
            assert np.array_equal(a1, a3)
            assert np.array_equal(c1, c2)
            assert np.array_equal(c1, c3)

    def test_dataframe_iterator_n(self, tmpdir):

        # save the images in the tmpdir
        count = 0
        filenames = []
        for test_images in self.all_test_images:
            for im in test_images:
                filename = "image-{}.png".format(count)
                filenames.append(filename)
                im.save(str(tmpdir / filename))
                count += 1

        # exclude first two items
        n_files = len(filenames)
        input_filenames = filenames[2:]

        # create dataframes
        classes = np.random.randint(2, size=len(input_filenames))
        classes = [str(c) for c in classes]
        df = pd.DataFrame({"filename": input_filenames})
        df2 = pd.DataFrame({"filename": input_filenames,
                            "class": classes})

        # create iterators
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, str(tmpdir), class_mode=None)
        df2_iterator = generator.flow_from_dataframe(
            df2, str(tmpdir), class_mode='binary')

        # Test the number of items in iterators
        assert df_iterator.n == n_files - 2
        assert df2_iterator.n == n_files - 2

    def test_dataframe_iterator_absolute_path(self, tmpdir):

        # save the images in the tmpdir
        count = 0
        file_paths = []
        for test_images in self.all_test_images:
            for im in test_images:
                filename = "image-{:0>5}.png".format(count)
                file_path = str(tmpdir / filename)
                file_paths.append(file_path)
                im.save(file_path)
                count += 1

        # prepare an image with a forbidden extension.
        file_path_fbd = str(tmpdir / 'image-forbid.fbd')
        shutil.copy(file_path, file_path_fbd)

        # create dataframes
        classes = np.random.randint(2, size=len(file_paths))
        classes = [str(c) for c in classes]
        df = pd.DataFrame({"filename": file_paths})
        df2 = pd.DataFrame({"filename": file_paths,
                            "class": classes})
        df3 = pd.DataFrame({"filename": ['image-not-exist.png'] + file_paths})
        df4 = pd.DataFrame({"filename": file_paths + [file_path_fbd]})

        # create iterators
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, None, class_mode=None,
            shuffle=False, batch_size=1)
        df2_iterator = generator.flow_from_dataframe(
            df2, None, class_mode='binary',
            shuffle=False, batch_size=1)
        df3_iterator = generator.flow_from_dataframe(
            df3, None, class_mode=None,
            shuffle=False, batch_size=1)
        df4_iterator = generator.flow_from_dataframe(
            df4, None, class_mode=None,
            shuffle=False, batch_size=1)

        validation_split = 0.2
        generator_split = image.ImageDataGenerator(validation_split=validation_split)
        df_train_iterator = generator_split.flow_from_dataframe(
            df, None, class_mode=None,
            shuffle=False, subset='training', batch_size=1)
        df_val_iterator = generator_split.flow_from_dataframe(
            df, None, class_mode=None,
            shuffle=False, subset='validation', batch_size=1)

        # Test the number of items in iterators
        assert df_iterator.n == len(file_paths)
        assert df2_iterator.n == len(file_paths)
        assert df3_iterator.n == len(file_paths)
        assert df4_iterator.n == len(file_paths)
        assert df_val_iterator.n == int(validation_split * len(file_paths))
        assert df_train_iterator.n == len(file_paths) - df_val_iterator.n

        # Test flow_from_dataframe
        for i in range(len(file_paths)):
            a1 = next(df_iterator)
            a2, _ = next(df2_iterator)
            a3 = next(df3_iterator)
            a4 = next(df4_iterator)

            if i < df_val_iterator.n:
                a5 = next(df_val_iterator)
            else:
                a5 = next(df_train_iterator)

            assert np.array_equal(a1, a2)
            assert np.array_equal(a1, a3)
            assert np.array_equal(a1, a4)
            assert np.array_equal(a1, a5)

    def test_dataframe_iterator_with_drop_duplicates(self, tmpdir):

        # save the images in the tmpdir
        count = 0
        filenames = []
        for test_images in self.all_test_images:
            for im in test_images:
                filename = "image-{:0>5}.png".format(count)
                filenames.append(filename)
                im.save(str(tmpdir / filename))
                count += 1

        # prepare input_filenames
        n_files = len(filenames)
        idx_rand, idx_rand2 = np.random.randint(1, n_files, size=2)
        input_filenames = filenames[::-1]  # reversed
        input_filenames2 = filenames[:idx_rand] + filenames[:idx_rand2]

        # create dataframes
        df = pd.DataFrame({"filename": input_filenames})
        df2 = pd.DataFrame({"filename": input_filenames2})

        # create iterators
        generator = image.ImageDataGenerator()
        df_drop_iterator = generator.flow_from_dataframe(
            df2, str(tmpdir), class_mode=None, drop_duplicates=True)
        df_no_drop_iterator = generator.flow_from_dataframe(
            df2, str(tmpdir), class_mode=None, drop_duplicates=False)

        # Test drop_duplicates
        assert df_drop_iterator.n == len(set(input_filenames2))
        assert df_no_drop_iterator.n == len(input_filenames2)

    def test_dataframe_iterator_with_subdirs(self, tmpdir):
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
        for test_images in self.all_test_images:
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

        # create dataframe
        classes = np.random.randint(num_classes, size=len(filenames))
        classes = [str(c) for c in classes]
        df = pd.DataFrame({"filename": filenames,
                           "class": classes})

        # create iterator
        generator = image.ImageDataGenerator()
        df_iterator = generator.flow_from_dataframe(
            df, str(tmpdir), class_mode='binary')

        # Test the number of items in iterator
        assert df_iterator.n == len(filenames)
        assert set(df_iterator.filenames) == set(filenames)

    def test_img_utils(self):
        height, width = 10, 8

        # Test th data format
        # Test RGB 3D
        x = np.random.random((3, height, width))
        img = image.array_to_img(x, data_format='channels_first')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_first')
        assert x.shape == (3, height, width)
        # Test RGBA 3D
        x = np.random.random((4, height, width))
        img = image.array_to_img(x, data_format='channels_first')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_first')
        assert x.shape == (4, height, width)
        # Test 2D
        x = np.random.random((1, height, width))
        img = image.array_to_img(x, data_format='channels_first')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_first')
        assert x.shape == (1, height, width)

        # Test tf data format
        # Test RGB 3D
        x = np.random.random((height, width, 3))
        img = image.array_to_img(x, data_format='channels_last')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_last')
        assert x.shape == (height, width, 3)
        # Test RGBA 3D
        x = np.random.random((height, width, 4))
        img = image.array_to_img(x, data_format='channels_last')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_last')
        assert x.shape == (height, width, 4)
        # Test 2D
        x = np.random.random((height, width, 1))
        img = image.array_to_img(x, data_format='channels_last')
        assert img.size == (width, height)
        x = image.img_to_array(img, data_format='channels_last')
        assert x.shape == (height, width, 1)

        # Test invalid use case
        with pytest.raises(ValueError):
            x = np.random.random((height, width))  # not 3D
            img = image.array_to_img(x, data_format='channels_first')
        with pytest.raises(ValueError):
            x = np.random.random((height, width, 3))
            # unknown data_format
            img = image.array_to_img(x, data_format='channels')
        with pytest.raises(ValueError):
            # neither RGB, RGBA, or gray-scale
            x = np.random.random((height, width, 5))
            img = image.array_to_img(x, data_format='channels_last')
        with pytest.raises(ValueError):
            x = np.random.random((height, width, 3))
            # unknown data_format
            img = image.img_to_array(x, data_format='channels')
        with pytest.raises(ValueError):
            # neither RGB, RGBA, or gray-scale
            x = np.random.random((height, width, 5, 3))
            img = image.img_to_array(x, data_format='channels_last')

    def test_random_transforms(self):
        x = np.random.random((2, 28, 28))
        assert image.random_rotation(x, 45).shape == (2, 28, 28)
        assert image.random_shift(x, 1, 1).shape == (2, 28, 28)
        assert image.random_shear(x, 20).shape == (2, 28, 28)
        assert image.random_zoom(x, (5, 5)).shape == (2, 28, 28)
        assert image.random_channel_shift(x, 20).shape == (2, 28, 28)

        # Test get_random_transform with predefined seed
        seed = 1
        generator = image.ImageDataGenerator(
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
        generator = image.ImageDataGenerator()
        transform_dict = generator.get_random_transform(x.shape, seed)
        assert transform_dict['theta'] == 0
        assert transform_dict['tx'] == 0
        assert transform_dict['ty'] == 0
        assert transform_dict['shear'] == 0
        assert transform_dict['zx'] == 1
        assert transform_dict['zy'] == 1
        assert transform_dict['channel_shift_intensity'] is None
        assert transform_dict['brightness'] is None

    def test_deterministic_transform(self):
        x = np.ones((32, 32, 3))
        generator = image.ImageDataGenerator(
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
        assert np.allclose(image.apply_affine_transform(
            x, theta=45, channel_axis=2, fill_mode='constant'), x_rotated)

    def test_batch_standardize(self):
        # ImageDataGenerator.standardize should work on batches
        for test_images in self.all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(image.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image.ImageDataGenerator(
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

    def test_load_img(self, tmpdir):
        filename_rgb = str(tmpdir / 'rgb_image.png')
        filename_rgba = str(tmpdir / 'rgba_image.png')

        original_rgb_array = np.array(255 * np.random.rand(100, 100, 3),
                                      dtype=np.uint8)
        original_rgb = image.array_to_img(original_rgb_array, scale=False)
        original_rgb.save(filename_rgb)

        original_rgba_array = np.array(255 * np.random.rand(100, 100, 4),
                                       dtype=np.uint8)
        original_rgba = image.array_to_img(original_rgba_array, scale=False)
        original_rgba.save(filename_rgba)

        # Test that loaded image is exactly equal to original.

        loaded_im = image.load_img(filename_rgb)
        loaded_im_array = image.img_to_array(loaded_im)
        assert loaded_im_array.shape == original_rgb_array.shape
        assert np.all(loaded_im_array == original_rgb_array)

        loaded_im = image.load_img(filename_rgba, color_mode='rgba')
        loaded_im_array = image.img_to_array(loaded_im)
        assert loaded_im_array.shape == original_rgba_array.shape
        assert np.all(loaded_im_array == original_rgba_array)

        loaded_im = image.load_img(filename_rgb, color_mode='grayscale')
        loaded_im_array = image.img_to_array(loaded_im)
        assert loaded_im_array.shape == (original_rgb_array.shape[0],
                                         original_rgb_array.shape[1], 1)

        # Test that nothing is changed when target size is equal to original.

        loaded_im = image.load_img(filename_rgb, target_size=(100, 100))
        loaded_im_array = image.img_to_array(loaded_im)
        assert loaded_im_array.shape == original_rgb_array.shape
        assert np.all(loaded_im_array == original_rgb_array)

        loaded_im = image.load_img(filename_rgba, color_mode='rgba',
                                   target_size=(100, 100))
        loaded_im_array = image.img_to_array(loaded_im)
        assert loaded_im_array.shape == original_rgba_array.shape
        assert np.all(loaded_im_array == original_rgba_array)

        loaded_im = image.load_img(filename_rgb, color_mode='grayscale',
                                   target_size=(100, 100))
        loaded_im_array = image.img_to_array(loaded_im)
        assert loaded_im_array.shape == (original_rgba_array.shape[0],
                                         original_rgba_array.shape[1], 1)

        # Test down-sampling with bilinear interpolation.

        loaded_im = image.load_img(filename_rgb, target_size=(25, 25))
        loaded_im_array = image.img_to_array(loaded_im)
        assert loaded_im_array.shape == (25, 25, 3)

        loaded_im = image.load_img(filename_rgba, color_mode='rgba',
                                   target_size=(25, 25))
        loaded_im_array = image.img_to_array(loaded_im)
        assert loaded_im_array.shape == (25, 25, 4)

        loaded_im = image.load_img(filename_rgb, color_mode='grayscale',
                                   target_size=(25, 25))
        loaded_im_array = image.img_to_array(loaded_im)
        assert loaded_im_array.shape == (25, 25, 1)

        # Test down-sampling with nearest neighbor interpolation.

        loaded_im_nearest = image.load_img(filename_rgb, target_size=(25, 25),
                                           interpolation="nearest")
        loaded_im_array_nearest = image.img_to_array(loaded_im_nearest)
        assert loaded_im_array_nearest.shape == (25, 25, 3)
        assert np.any(loaded_im_array_nearest != loaded_im_array)

        loaded_im_nearest = image.load_img(filename_rgba, color_mode='rgba',
                                           target_size=(25, 25),
                                           interpolation="nearest")
        loaded_im_array_nearest = image.img_to_array(loaded_im_nearest)
        assert loaded_im_array_nearest.shape == (25, 25, 4)
        assert np.any(loaded_im_array_nearest != loaded_im_array)

        # Check that exception is raised if interpolation not supported.

        loaded_im = image.load_img(filename_rgb, interpolation="unsupported")
        with pytest.raises(ValueError):
            loaded_im = image.load_img(filename_rgb, target_size=(25, 25),
                                       interpolation="unsupported")

if __name__ == '__main__':
    pytest.main([__file__])
