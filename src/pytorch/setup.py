from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

# trying to import the required torch package
try:
    import torch
except ImportError:
    raise Exception('SPE requires PyTorch to be installed. aborting')

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Proceed to setup
setup(
    name='spe',
    version='0.1',
    description='stochastic positional encoding for PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Antoine Liutkus',
    author_email='antoine.liutkus@inria.fr',
    packages=['spe'],
    keywords='pytorch',
    install_requires=[
        'torch>=1.7',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)
