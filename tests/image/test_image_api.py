from keras_preprocessing import image


def test_api_classes():
    expected_exposed_classes = [
        'DataFrameIterator',
        'DirectoryIterator',
        'ImageDataGenerator',
        'Iterator',
        'NumpyArrayIterator',
    ]
    for _class in expected_exposed_classes:
        assert hasattr(image, _class)


def test_api_functions():
    expected_exposed_functions = [
        'flip_axis',
        'random_rotation',
        'random_shift',
        'random_shear',
        'random_zoom',
        'apply_channel_shift',
        'random_channel_shift',
        'apply_brightness_shift',
        'random_brightness',
        'transform_matrix_offset_center',
        'apply_affine_transform',
        'validate_filename',
        'save_img',
        'load_img',
        'list_pictures',
        'array_to_img',
        'img_to_array'
    ]
    for function in expected_exposed_functions:
        assert hasattr(image, function)
