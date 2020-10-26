import sys

from setuptools import setup
from setuptools import find_packages

long_description = '''
Keras Preprocessing is the data preprocessing
and data augmentation module of the Keras deep learning library.
It provides utilities for working with image data, text data,
and sequence data.

Read the documentation at: https://keras.io/

Keras Preprocessing may be imported directly
from an up-to-date installation of Keras:

```
from keras import preprocessing
```

Keras Preprocessing is compatible with Python 2.7-3.6
and is distributed under the MIT license.
'''

setup(name='Keras_Preprocessing',
      version='1.1.2',
      description='Easy data preprocessing and data augmentation '
                  'for deep learning models',
      long_description=long_description,
      author='Keras Team',
      url='https://github.com/keras-team/keras-preprocessing',
      download_url='https://github.com/keras-team/'
                   'keras-preprocessing/tarball/1.1.2',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'six>=1.9.0'],
      extras_require={
          'tests': ['pandas',
                    'Pillow' if sys.version_info >= (3, 0) else 'pillow',
                    'tensorflow',  # CPU version
                    'keras',
                    'pytest',
                    'pytest-xdist',
                    'pytest-cov'],
          'pep8': ['flake8'],
          'image': ['scipy>=0.14',
                    'Pillow>=5.2.0'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
