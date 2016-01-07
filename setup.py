#!/usr/bin/env python
""" Setup and install noweats.  """

from distutils.core import setup
from setuptools import find_packages

setup(name='NowEats',
      version='1.0',
      description='NowEats Twitter feed',
      author='Brandon Reiss',
      author_email='blr246@nyu.edu',
      url='https://github.com/blr246/noweats',
      packages=find_packages(),
      scripts=[
          'bin/collect_nyc',
          'bin/process_file',
          'bin/process_new',
          'bin/compress_data',
          'bin/link_numpy',
      ],
      )
