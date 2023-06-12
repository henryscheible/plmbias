from distutils.core import setup
from setuptools import find_packages

setup(name="plmbias",
      version="0.0.2",
      description="Tools for NLP research",
      license="MIT",
      author="Henry Scheible",
      author_email="henry.scheible@gmail.com",
      url="https://github.com/henryscheible/plmbias",
      install_requires=[
          'torch',
          'transformers',
          'datasets',
          'evaluate',
          'sklearn',
          'captum',
          'retry'
      ],
      packages=find_packages()
)