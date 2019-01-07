"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from .iterator import Iterator
from .utils import (array_to_img,
                    get_extension,
                    img_to_array,
                    _list_valid_filenames_in_directory,
                    load_img)


class DataFrameIterator(Iterator):
    """Iterator capable of reading images from a directory on disk
        through a dataframe.

    # Arguments
        dataframe: Pandas dataframe containing the filepaths relative to
            `directory` or absolute paths if `directory` is None of the images
            in a column and classes in another column/s that can be fed as raw
            target data.
        directory: string, path to the directory to read images from. Directory to
            under which all the images are present. If None, data in x_col column
            should be absolute paths.
        image_data_generator: Instance of `ImageDataGenerator` to use for
            random transformations and normalization. If None, no transformations
            and normalizations are made.
        x_col: Column in dataframe that contains all the filenames (or absolute
            paths, if directory is set to None).
        y_col: Column/s in dataframe that has the target data.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, names of
            each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `"other"`: targets are the data(numpy array) of y_col data
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        drop_duplicates: Boolean, whether to drop duplicate rows based on filename.
    """
    allowed_class_modes = {
        'categorical', 'binary', 'sparse', 'input', 'other', None
    }

    def __init__(self,
                 dataframe,
                 directory=None,
                 image_data_generator=None,
                 x_col="filename",
                 y_col="class",
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 drop_duplicates=True):
        super(DataFrameIterator, self).common_init(image_data_generator,
                                                   target_size,
                                                   color_mode,
                                                   data_format,
                                                   save_to_dir,
                                                   save_prefix,
                                                   save_format,
                                                   subset,
                                                   interpolation)
        df = dataframe.copy()
        if drop_duplicates:
            df.drop_duplicates(x_col, inplace=True)
        self.x_col = x_col
        self.directory = directory
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype

        if not classes:
            classes = []
            if class_mode not in ["other", "input", None]:
                classes = list(np.sort(df[y_col].unique()))
        else:
            if class_mode in ["other", "input", None]:
                raise ValueError('classes cannot be set if class_mode'
                                 ' is either "other" or "input" or None.')
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        df = self._filter_valid_filepaths(df)
        if self.split:
            num_files = len(df)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            df = df.iloc[start: stop, :]
        self.filenames = df[x_col].tolist()

        if class_mode not in ["other", "input", None]:
            classes = df[y_col].values
            self.classes = np.array([self.class_indices[cls] for cls in classes])
        elif class_mode == "other":
            self.data = df[y_col].values
            if type(y_col) == str:
                y_col = [y_col]
            if "object" in list(df[y_col].dtypes):
                raise TypeError("y_col column/s must be numeric datatypes.")
        self.samples = len(self.filenames)
        if self.num_classes > 0:
            print('Found %d images belonging to %d classes.' %
                  (self.samples, self.num_classes))
        else:
            print('Found %d images.' % self.samples)

        super(DataFrameIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=self.dtype)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            if self.directory is not None:
                img_path = os.path.join(self.directory, fname)
            else:
                img_path = fname
            img = load_img(img_path,
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=self.dtype)
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'other':
            batch_y = self.data[index_array]
        else:
            return batch_x
        return batch_x, batch_y

    def _filter_valid_filepaths(self, df):
        """Keep only dataframe rows with valid filenames

        # Arguments
            df: Pandas dataframe containing filenames in a column

        # Returns
            absolute paths to image files
        """
        filepaths = df[self.x_col].map(
            lambda fname: os.path.join(self.directory or '', fname)
        )
        format_check = filepaths.map(get_extension).isin(self.white_list_formats)
        existence_check = filepaths.map(os.path.isfile)
        return df[format_check & existence_check]
