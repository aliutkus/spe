from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Proceed to setup
setup(
    name='jax-spe',
    version='0.1',
    description='stochastic positional encoding for PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ondřej Cífka',
    author_email='cifkao@gmail.com',
    packages=['jax_spe'],
    install_requires=[
        'jax>=0.2.6',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)
