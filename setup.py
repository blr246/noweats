#!/usr/bin/env python

#from distutils.core import setup
from setuptools import find_packages, setup

setup(name='NowEats',
      version='1.0',
      description='NowEats Twitter feed',
      author='Brandon Reiss',
      author_email='blr246@nyu.edu',
      url='https://github.com/blr246/noweats',
      packages=find_packages(),
      install_requires=['tweepy'],
      #scripts=[
      #    'bin/collect_nyc',
      #],
      )
