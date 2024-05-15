# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Richard_Cui on 2024/5/15 10:56.
from setuptools import setup, find_packages
setup(
          name='sft-data-generator',
          version='0.0.1',
          packages=find_packages(where='src'),
          package_dir={'': 'src'}
      )
