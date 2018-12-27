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
        dataframe: Pandas dataframe containing the filenames
            (or paths relative to `directory`) of the images in a column and
            classes in another column/s that can be fed as raw target data.
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
            if used with dataframe,this will be the directory to under which
            all the images are present.
            You could also set it to None if data in x_col column are
            absolute paths.
        image_data_generator: Instance of `ImageDataGenerator` to use for
            random transformations and normalization. If None, no transformations
            and normalizations are made.
        x_col: Column in dataframe that contains all the filenames (or absolute
            paths, if directory is set to None).
        y_col: Column/s in dataframe that has the target data.
        has_ext: bool, Whether the filenames in x_col has extensions or not.
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
        sort: Boolean, whether to sort dataframe by filename (before shuffle).
        drop_duplicates: Boolean, whether to drop duplicate rows based on filename.
    """
    allowed_class_modes = {
        'categorical', 'binary', 'sparse', 'input', 'other', None
    }

    def __init__(self, dataframe,
                 directory,
                 image_data_generator=None,
                 x_col="filename",
                 y_col="class",
                 has_ext=True,
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
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 sort=True,
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
        self.df = dataframe.copy()
        if drop_duplicates:
            self.df.drop_duplicates(x_col, inplace=True)
        self.x_col = x_col
        self.df[x_col] = self.df[x_col].astype(str)
        self.directory = directory
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            if class_mode not in ["other", "input", None]:
                classes = list(np.sort(self.df[y_col].unique()))
        else:
            if class_mode in ["other", "input", None]:
                raise ValueError('classes cannot be set if class_mode'
                                 ' is either "other" or "input" or None.')
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        # Second, build an index of the images.
        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')

        if self.directory is not None:
            filenames = _list_valid_filenames_in_directory(
                directory,
                self.white_list_formats,
                None,
                class_indices=self.class_indices,
                follow_links=follow_links,
                df=True)
        else:
            if not has_ext:
                raise ValueError('has_ext cannot be set to False'
                                 ' if directory is None.')
            filenames = self._list_valid_filepaths(self.white_list_formats)

        if has_ext:
            ext_exist = False
            if get_extension(self.df[x_col].values[0]) in self.white_list_formats:
                ext_exist = True
            if not ext_exist:
                raise ValueError('has_ext is set to True but'
                                 ' extension not found in x_col')
            self.df = self.df[self.df[x_col].isin(filenames)]
            if sort:
                self.df.sort_values(by=x_col, inplace=True)
            self.filenames = list(self.df[x_col])
        else:
            without_ext_with = {f[:-1 * (len(f.split(".")[-1]) + 1)]: f
                                for f in filenames}
            filenames_without_ext = [f[:-1 * (len(f.split(".")[-1]) + 1)]
                                     for f in filenames]
            self.df = self.df[self.df[x_col].isin(filenames_without_ext)]
            if sort:
                self.df.sort_values(by=x_col, inplace=True)
            self.filenames = [without_ext_with[f] for f in list(self.df[x_col])]

        if self.split:
            num_files = len(self.filenames)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            self.df = self.df.iloc[start: stop, :]
            self.filenames = self.filenames[start: stop]

        if class_mode not in ["other", "input", None]:
            classes = self.df[y_col].values
            self.classes = np.array([self.class_indices[cls] for cls in classes])
        elif class_mode == "other":
            self.data = self.df[y_col].values
            if type(y_col) == str:
                y_col = [y_col]
            if "object" in list(self.df[y_col].dtypes):
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

    def _list_valid_filepaths(self, white_list_formats):
        df_paths = self.df[self.x_col]
        format_check = df_paths.map(get_extension).isin(white_list_formats)
        existence_check = df_paths.map(os.path.isfile)
        valid_filepaths = list(df_paths[np.logical_and(format_check,
                                                       existence_check)])
        return valid_filepaths
