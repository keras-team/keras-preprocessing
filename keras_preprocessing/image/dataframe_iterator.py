"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from pandas.api.types import is_numeric_dtype

from .iterator import BatchFromFilesMixin, Iterator
from .utils import (array_to_img,
                    get_extension,
                    img_to_array,
                    load_img)


class DataFrameIterator(BatchFromFilesMixin, Iterator):
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

        super(DataFrameIterator, self).set_processing_attrs(image_data_generator,
                                                            target_size,
                                                            color_mode,
                                                            data_format,
                                                            save_to_dir,
                                                            save_prefix,
                                                            save_format,
                                                            subset,
                                                            interpolation)
        df = dataframe.copy()
        self.directory = directory
        self.class_mode = class_mode
        self.dtype = dtype
        self._check_params(df, x_col, y_col, classes)
        if drop_duplicates:
            df.drop_duplicates(x_col, inplace=True)
        # check which image files are valid and keep them
        df = self._filter_valid_filepaths(df, x_col)
        classes = classes or []
        if class_mode not in ["other", "input", None]:
            df, classes = self._filter_classes(df, y_col, classes)
            num_classes = len(classes)
            # build an index of all the unique classes
            self.class_indices = dict(zip(classes, range(len(classes))))
        if self.split:
            num_files = len(df)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            df = df.iloc[start: stop, :]
        if class_mode not in ["other", "input", None]:
            self.classes = self.get_classes(df, y_col)
        self.filenames = df[x_col].tolist()
        if class_mode == "other":
            self.data = df[y_col].values
            if isinstance(y_col, str):
                y_col = [y_col]
            if "object" in set(df[y_col].dtypes):
                raise TypeError("y_col column/s must be numeric datatypes.")
        self.samples = len(self.filenames)
        if class_mode in ["other", "input", None]:
            print('Found {} images.'.format(self.samples))
        else:
            print('Found {} images belonging to {} classes.'
                  .format(self.samples, num_classes))
        super(DataFrameIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)

    def _check_params(self, df, x_col, y_col, classes):
        classes = set(classes)
        # check class mode is one of the currently supported
        if self.class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(self.class_mode, self.allowed_class_modes))
        # check that filenames/filepaths column values are all strings
        if not all(df[x_col].apply(lambda x: isinstance(x, str))):
            raise ValueError('All values in column x_col={} must be strings.'
                             .format(x_col))
        # check that labels are string or numeric if binary and sparse output
        # Note: sparse class_mode supports numeric, string, list and tuple
        is_numeric_or_str = (is_numeric_dtype(df[y_col]) or
                             all(df[y_col].apply(lambda x: isinstance(x, str))))
        if self.class_mode in {'binary', 'sparse'} and not is_numeric_or_str:
            raise TypeError('If class_mode="{}", y_col="{}" column '
                            'values must be numeric or string. '
                            .format(self.class_mode, y_col))
        # check that if binary there are only 2 different classes
        if self.class_mode == 'binary':
            if classes:
                if len(classes) != 2:
                    raise ValueError('If class_mode="binary" there must be 2 '
                                     'classes. {} class(es) were given.'
                                     .format(len(classes)))
            elif df[y_col].nunique() != 2:
                raise ValueError('If class_mode="binary" there must be 2 classes. '
                                 'Found {} classes.'.format(df[y_col].nunique()))
        # check that no classes are given if class_mode other or input
        if classes and self.class_mode in ["other", "input", None]:
            raise ValueError('classes cannot be set if class_mode'
                             ' is either "other" or "input" or None.')

    def get_classes(self, df, y_col):
        labels = []
        for label in df[y_col]:
            if isinstance(label, (list, tuple)):
                labels.append([self.class_indices[lbl] for lbl in label])
            else:
                labels.append(self.class_indices[label])
        return labels

    @staticmethod
    def _filter_classes(df, y_col, classes):
        df = df.copy()

        def remove_classes(labels, classes):
            if isinstance(labels, (list, tuple)):
                labels = [cls for cls in labels if cls in classes]
                return labels or None
            elif isinstance(labels, str):
                return labels if labels in classes else None
            else:
                raise TypeError(
                    "Expect string, list or tuple but found {} in {} column "
                    .format(type(x), y_col)
                )

        if classes:
            classes = set(classes)  # sort and prepare for membership lookup
            df[y_col] = df[y_col].apply(lambda x: remove_classes(x, classes))
        else:
            classes = set()
            for v in df[y_col]:
                if isinstance(v, (list, tuple)):
                    classes.update(v)
                else:
                    classes.add(v)
        return df.dropna(subset=[y_col]), sorted(classes)

    def _filter_valid_filepaths(self, df, x_col):
        """Keep only dataframe rows with valid filenames

        # Arguments
            df: Pandas dataframe containing filenames in a column

        # Returns
            absolute paths to image files
        """
        filepaths = df[x_col].map(
            lambda fname: os.path.join(self.directory or '', fname)
        )
        format_check = filepaths.map(get_extension).isin(self.white_list_formats)
        existence_check = filepaths.map(os.path.isfile)
        return df[format_check & existence_check]

    @property
    def filepaths(self):
        root = self.directory or ''
        return [os.path.join(root, fname) for fname in self.filenames]

    @property
    def labels(self):
        return self.classes

    @property
    def data(self):
        return self._data
