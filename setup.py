# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

setup(
    name='k_means_clustering',
    version='0.1.0',
    description='k-means clustering',
    long_description=readme,
    author='Dixhom',
    author_email='kuborisho@gmail.com',
    url='',
    license='MIT',
	install_requires=['numpy'],
    packages=find_packages(exclude=('tests', 'docs'))
)

