import os
import random
import shutil

import numpy as np
import pandas as pd
import pytest

from PIL import Image

from keras_preprocessing.image import dataframe_iterator
from keras_preprocessing.image import image_data_generator


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


def test_dataframe_iterator(all_test_images, tmpdir):
    num_classes = 2

    # save the images in the tmpdir
    count = 0
    filenames = []
    filepaths = []
    filenames_without = []
    for test_images in all_test_images:
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
    iterator = dataframe_iterator.DataFrameIterator(df, str(tmpdir))
    batch = next(iterator)
    assert len(batch) == 2
    assert isinstance(batch[0], np.ndarray)
    assert isinstance(batch[1], np.ndarray)
    generator = image_data_generator.ImageDataGenerator()
    df_iterator = generator.flow_from_dataframe(df, x_col='filepaths')
    df_iterator_dir = generator.flow_from_dataframe(df, str(tmpdir))
    df_sparse_iterator = generator.flow_from_dataframe(df, str(tmpdir),
                                                       class_mode="sparse")
    assert not np.isnan(df_sparse_iterator.classes).any()
    # check number of classes and images
    assert len(df_iterator.class_indices) == num_classes
    assert len(df_iterator.classes) == count
    assert set(df_iterator.filenames) == set(filepaths)
    assert len(df_iterator_dir.class_indices) == num_classes
    assert len(df_iterator_dir.classes) == count
    assert set(df_iterator_dir.filenames) == set(filenames)
    # test without shuffle
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
    generator = image_data_generator.ImageDataGenerator(
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


def test_dataframe_iterator_validate_filenames(all_test_images, tmpdir):
    # save the images in the paths
    count = 0
    filenames = []
    for test_images in all_test_images:
        for im in test_images:
            filename = 'image-{}.png'.format(count)
            im.save(str(tmpdir / filename))
            filenames.append(filename)
            count += 1
    df = pd.DataFrame({"filename": filenames + ['test.jpp', 'test.jpg']})
    generator = image_data_generator.ImageDataGenerator()
    df_iterator = generator.flow_from_dataframe(df,
                                                str(tmpdir),
                                                class_mode="input")
    assert len(df_iterator.filenames) == len(df['filename']) - 2
    df_iterator = generator.flow_from_dataframe(df,
                                                str(tmpdir),
                                                class_mode="input",
                                                validate_filenames=False)
    assert len(df_iterator.filenames) == len(df['filename'])


def test_dataframe_iterator_sample_weights(all_test_images, tmpdir):
    # save the images in the paths
    count = 0
    filenames = []
    for test_images in all_test_images:
        for im in test_images:
            filename = 'image-{}.png'.format(count)
            im.save(str(tmpdir / filename))
            filenames.append(filename)
            count += 1
    df = pd.DataFrame({"filename": filenames})
    df['weight'] = ([2, 5] * len(df))[:len(df)]
    generator = image_data_generator.ImageDataGenerator()
    df_iterator = generator.flow_from_dataframe(df, str(tmpdir),
                                                x_col="filename",
                                                y_col=None,
                                                shuffle=False,
                                                batch_size=5,
                                                weight_col='weight',
                                                class_mode="input")

    batch = next(df_iterator)
    assert len(batch) == 3  # (x, y, weights)
    # check if input and output have the same shape and they're the same
    assert(batch[0].all() == batch[1].all())
    # check if the input and output images are not the same numpy array
    input_img = batch[0][0]
    output_img = batch[1][0]
    output_img[0][0][0] += 1
    assert input_img[0][0][0] != output_img[0][0][0]
    assert np.array_equal(np.array([2, 5, 2, 5, 2]), batch[2])

    # fail
    df['weight'] = (['2', '5'] * len(df))[:len(df)]
    with pytest.raises(TypeError):
        image_data_generator.ImageDataGenerator().flow_from_dataframe(
            df,
            weight_col='weight',
            class_mode="input"
        )


def test_dataframe_iterator_class_mode_input(all_test_images, tmpdir):
    # save the images in the paths
    count = 0
    filenames = []
    for test_images in all_test_images:
        for im in test_images:
            filename = 'image-{}.png'.format(count)
            im.save(str(tmpdir / filename))
            filenames.append(filename)
            count += 1
    df = pd.DataFrame({"filename": filenames})
    generator = image_data_generator.ImageDataGenerator()
    df_autoencoder_iterator = generator.flow_from_dataframe(df, str(tmpdir),
                                                            x_col="filename",
                                                            y_col=None,
                                                            class_mode="input")

    batch = next(df_autoencoder_iterator)

    # check if input and output have the same shape and they're the same
    assert np.allclose(batch[0], batch[1])
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


def test_dataframe_iterator_class_mode_categorical_multi_label(all_test_images,
                                                               tmpdir):
    # save the images in the paths
    filenames = []
    count = 0
    for test_images in all_test_images:
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
    generator = image_data_generator.ImageDataGenerator()
    df_iterator = generator.flow_from_dataframe(df, str(tmpdir))
    batch_x, batch_y = next(df_iterator)
    assert isinstance(batch_x, np.ndarray)
    assert len(batch_x.shape) == 4
    assert isinstance(batch_y, np.ndarray)
    assert batch_y.shape == (len(batch_x), 2)
    for labels in batch_y:
        assert all(l in {0, 1} for l in labels)

    # on first 3 batches
    df = pd.DataFrame({
        "filename": filenames,
        "class": [['b', 'a']] + ['b'] + [['c']] + [random.choice(label_opt)
                                                   for _ in filenames[:-3]]
    })
    generator = image_data_generator.ImageDataGenerator()
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


def test_dataframe_iterator_class_mode_multi_output(all_test_images, tmpdir):
    # save the images in the paths
    filenames = []
    count = 0
    for test_images in all_test_images:
        for im in test_images:
            filename = 'image-{}.png'.format(count)
            im.save(str(tmpdir / filename))
            filenames.append(filename)
            count += 1
    # fit both outputs are a single number
    df = pd.DataFrame({"filename": filenames}).assign(
        output_0=np.random.uniform(size=len(filenames)),
        output_1=np.random.uniform(size=len(filenames))
    )
    df_iterator = image_data_generator.ImageDataGenerator().flow_from_dataframe(
        df, y_col=['output_0', 'output_1'], directory=str(tmpdir),
        batch_size=3, shuffle=False, class_mode='multi_output'
    )
    batch_x, batch_y = next(df_iterator)
    assert isinstance(batch_x, np.ndarray)
    assert len(batch_x.shape) == 4
    assert isinstance(batch_y, list)
    assert len(batch_y) == 2
    assert np.array_equal(batch_y[0],
                          np.array(df['output_0'].tolist()[:3]))
    assert np.array_equal(batch_y[1],
                          np.array(df['output_1'].tolist()[:3]))
    # if one of the outputs is a 1D array
    df['output_1'] = [np.random.uniform(size=(2, 2, 1)).flatten()
                      for _ in range(len(df))]
    df_iterator = image_data_generator.ImageDataGenerator().flow_from_dataframe(
        df, y_col=['output_0', 'output_1'], directory=str(tmpdir),
        batch_size=3, shuffle=False, class_mode='multi_output'
    )
    batch_x, batch_y = next(df_iterator)
    assert isinstance(batch_x, np.ndarray)
    assert len(batch_x.shape) == 4
    assert isinstance(batch_y, list)
    assert len(batch_y) == 2
    assert np.array_equal(batch_y[0],
                          np.array(df['output_0'].tolist()[:3]))
    assert np.array_equal(batch_y[1],
                          np.array(df['output_1'].tolist()[:3]))
    # if one of the outputs is a 2D array
    df['output_1'] = [np.random.uniform(size=(2, 2, 1))
                      for _ in range(len(df))]
    df_iterator = image_data_generator.ImageDataGenerator().flow_from_dataframe(
        df, y_col=['output_0', 'output_1'], directory=str(tmpdir),
        batch_size=3, shuffle=False, class_mode='multi_output'
    )
    batch_x, batch_y = next(df_iterator)
    assert isinstance(batch_x, np.ndarray)
    assert len(batch_x.shape) == 4
    assert isinstance(batch_y, list)
    assert len(batch_y) == 2
    assert np.array_equal(batch_y[0],
                          np.array(df['output_0'].tolist()[:3]))
    assert np.array_equal(batch_y[1],
                          np.array(df['output_1'].tolist()[:3]))
    # fail if single column
    with pytest.raises(TypeError):
        image_data_generator.ImageDataGenerator().flow_from_dataframe(
            df, y_col='output_0',
            directory=str(tmpdir),
            class_mode='multi_output'
        )


def test_dataframe_iterator_class_mode_raw(all_test_images, tmpdir):
    # save the images in the paths
    filenames = []
    count = 0
    for test_images in all_test_images:
        for im in test_images:
            filename = 'image-{}.png'.format(count)
            im.save(str(tmpdir / filename))
            filenames.append(filename)
            count += 1
    # case for 1D output
    df = pd.DataFrame({"filename": filenames}).assign(
        output_0=np.random.uniform(size=len(filenames)),
        output_1=np.random.uniform(size=len(filenames))
    )
    df_iterator = image_data_generator.ImageDataGenerator().flow_from_dataframe(
        df, y_col='output_0', directory=str(tmpdir),
        batch_size=3, shuffle=False, class_mode='raw'
    )
    batch_x, batch_y = next(df_iterator)
    assert isinstance(batch_x, np.ndarray)
    assert len(batch_x.shape) == 4
    assert isinstance(batch_y, np.ndarray)
    assert batch_y.shape == (3,)
    assert np.array_equal(batch_y, df['output_0'].values[:3])
    # case with a 2D output
    df_iterator = image_data_generator.ImageDataGenerator().flow_from_dataframe(
        df, y_col=['output_0', 'output_1'], directory=str(tmpdir),
        batch_size=3, shuffle=False, class_mode='raw'
    )
    batch_x, batch_y = next(df_iterator)
    assert isinstance(batch_x, np.ndarray)
    assert len(batch_x.shape) == 4
    assert isinstance(batch_y, np.ndarray)
    assert batch_y.shape == (3, 2)
    assert np.array_equal(batch_y,
                          df[['output_0', 'output_1']].values[:3])


@pytest.mark.parametrize('validation_split,num_training', [
    (0.25, 18),
    (0.50, 12),
    (0.75, 6),
])
def test_dataframe_iterator_with_validation_split(all_test_images, validation_split,
                                                  num_training, tmpdir):
    num_classes = 2

    # save the images in the tmpdir
    count = 0
    filenames = []
    filenames_without = []
    for test_images in all_test_images:
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
    generator = image_data_generator.ImageDataGenerator(
        validation_split=validation_split
    )
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


def test_dataframe_iterator_with_custom_indexed_dataframe(all_test_images, tmpdir):
    num_classes = 2

    # save the images in the tmpdir
    count = 0
    filenames = []
    for test_images in all_test_images:
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
    generator = image_data_generator.ImageDataGenerator()
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


def test_dataframe_iterator_n(all_test_images, tmpdir):

    # save the images in the tmpdir
    count = 0
    filenames = []
    for test_images in all_test_images:
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
    generator = image_data_generator.ImageDataGenerator()
    df_iterator = generator.flow_from_dataframe(
        df, str(tmpdir), class_mode=None)
    df2_iterator = generator.flow_from_dataframe(
        df2, str(tmpdir), class_mode='binary')

    # Test the number of items in iterators
    assert df_iterator.n == n_files - 2
    assert df2_iterator.n == n_files - 2


def test_dataframe_iterator_absolute_path(all_test_images, tmpdir):

    # save the images in the tmpdir
    count = 0
    file_paths = []
    for test_images in all_test_images:
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
    generator = image_data_generator.ImageDataGenerator()
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
    generator_split = image_data_generator.ImageDataGenerator(
        validation_split=validation_split
    )
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


def test_dataframe_iterator_with_subdirs(all_test_images, tmpdir):
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

    # create dataframe
    classes = np.random.randint(num_classes, size=len(filenames))
    classes = [str(c) for c in classes]
    df = pd.DataFrame({"filename": filenames,
                       "class": classes})

    # create iterator
    generator = image_data_generator.ImageDataGenerator()
    df_iterator = generator.flow_from_dataframe(
        df, str(tmpdir), class_mode='binary')

    # Test the number of items in iterator
    assert df_iterator.n == len(filenames)
    assert set(df_iterator.filenames) == set(filenames)


if __name__ == '__main__':
    pytest.main([__file__])
