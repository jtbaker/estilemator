from __future__ import (absolute_import, division, print_function)

import os
import sys
from setuptools import setup

setup(
    name='estilemator',
    version='0.1.0',
    author='Jason Baker',
    author_email='jason.th.baker@gmail.com',
    packages=['estilemator'],
    url='http://github.com/jtbaker/estilemator',
    license='LICENSE.txt',
    description='Generate slippy map tiles from your 2D estimator model.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "mercantile",
        "rasterio",
        "matplotlib",
        "pytest",
        "joblib",
        "affine"
    ],
)
