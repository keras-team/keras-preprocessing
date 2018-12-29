from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from .iterator import Iterator
from .utils import (array_to_img,
                    get_extension,
                    img_to_array,
                    load_img)


class DictionaryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk through a dictionary.

    # Arguments
        dictionary: Filenames or filepaths as keys and class labels as values.
        image_data_generator: Instance of `ImageDataGenerator` to use for
            random transformations and normalization.
        directory: Filenames in directory will be looked in this directory
            if provided. Only PNG, JPG, JPEG, BMP, PPM, TIF or TIFF extensions
            will be considered.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Use only these classes (e.g. `["dogs", "cats"]`).
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample images
            (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if
            the target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also supported.
            If PIL version 3.4.0 or newer is installed, "box" and "hamming"
                are also supported.
            By default, "nearest" is used.
        dtype: Dtype to use for generated arrays.
    """
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self, dictionary, image_data_generator,
                 directory=None,
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
                 dtype='float32'):
        super(DictionaryIterator, self).common_init(image_data_generator,
                                                    target_size,
                                                    color_mode,
                                                    data_format,
                                                    save_to_dir,
                                                    save_prefix,
                                                    save_format,
                                                    subset,
                                                    interpolation)
        dictionary, classes = self._filter_classes(dictionary, classes)
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        # build an index of all the unique classes.
        self.class_indices = dict(zip(classes, range(len(classes))))
        # check which image files are valid
        self.filepaths, self.labels = self._list_filepaths_and_labels(dictionary,
                                                                      directory)
        if self.split:  # create training or validation split
            num_files = len(self.filepaths)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            self.filepaths = self.filepaths[start: stop]
            self.labels = self.labels[start: stop]

        print('Found %d images belonging to %d classes.' %
              (len(self.filepaths), len(self.class_indices)))
        super(DictionaryIterator, self).__init__(len(self.filepaths),
                                                 batch_size,
                                                 shuffle,
                                                 seed)

    def _filter_classes(self, dictionary, classes):
        if classes:
            classes = set(classes)
            dictionary = {k: v for k, v in dictionary.items() if v in classes}
        else:
            classes = set()
            for v in dictionary.values():
                if isinstance(v, (list, tuple)):
                    classes.union(v)
                else:
                    classes.add(v)
        return dictionary, classes

    def _list_filepaths_and_labels(self, dictionary, directory):
        """List valid filepaths and their respective class labels

        If directory is None the filenames in the directory keys will
        be taken as absolute paths to the image files. If directory is not
        None it will be taken as the root directory of the filenames in the
        dictionary keys.

        # Arguments
            dictionary: Filenames as keys and class labels as values.
            directory: Path to the directory to read images from.

        # Returns
            filepaths, labels: filepaths with an allowed extension and the
                corresponding class labels
        """
        filepaths, labels, root = [], [], directory or ''
        for filename, label in dictionary.items():
            filepath = os.path.join(root, filename)
            if (get_extension(filename) in self.white_list_formats and
                    os.path.isfile(filepath)):
                filepaths.append(filepath)
                if isinstance(label, (list, tuple)):
                    labels.extend(self.class_indices[lbl] for lbl in label)
                else:
                    labels.append(self.class_indices[label])
        return filepaths, labels

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        for i, j in enumerate(index_array):
            img = load_img(self.filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`, but not PIL images.
            if hasattr(img, 'close'):
                img.close()
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
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.labels[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.labels[n_observation]] = 1.
        else:
            return batch_x
        return batch_x, batch_y
