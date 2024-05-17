from setuptools import setup, find_packages

setup(name='sba_nn',
      packages=find_packages(),
      install_required = ['numpy', 'pandas', 'matplotlib'],
      version ='1.0')