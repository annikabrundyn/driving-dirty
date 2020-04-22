#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='driving-dirty',
      version='0.0.1',
      description='DL 2020 Project',
      author='',
      author_email='',
      url='https://github.com/annikabrundyn/driving-dirty',
      install_requires=[
            'pytorch-lightning'
      ],
      packages=find_packages()
      )
