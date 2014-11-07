from setuptools import setup, find_packages
import codecs
import os

import verhulst

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='verhulst',

    version=verhulst.__version__,

    description=('Statistical and plotting routines for evaluating binary '
                 'logistic regressions.'),
    long_description=long_description,

    url='https://github.com/rpetchler/verhulst',

    author='Ross Petchler',
    author_email='ross.petchler@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='data analysis logitistic regression statistics plotting',

    packages=find_packages(exclude=["contrib", "docs", "tests*"]),

    install_requires = ['numpy', 'scipy', 'matplotlib']
)
