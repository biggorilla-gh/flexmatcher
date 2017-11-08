#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'pandas',
    'scipy',
    'scikit-learn>=0.18',
    'numpy',
    'python-Levenshtein==0.12.0'
]

test_requirements = [
    'pandas',
    'scipy',
    'scikit-learn>=0.18',
    'numpy'
]

setup(
    name='flexmatcher',
    version='1.0.2',
    description="FlexMatcher is a schema matching package in Python which handles the problem of matching multiple schemas to a single mediated schema.",
    long_description=readme + '\n\n' + history,
    author="BigGorilla Team",
    author_email='thebiggorilla.team@gmail.com',
    url='https://github.com/biggorilla-gh/flexmatcher',
    packages=[
        'flexmatcher',
        'flexmatcher.utils',
        'flexmatcher.classify'
    ],
    package_dir={'flexmatcher': 'flexmatcher',
                 'flexmatcher.utils': 'flexmatcher/utils',
                 'flexmatcher.classify': 'flexmatcher/classify'},
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='flexmatcher',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
