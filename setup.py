from setuptools import setup, find_packages
import numpy
from Cython.Build import cythonize

setup(name='dea_tsmask',
      description='DEA Timeseries cloud and cloud shadow classifier',
      version='0.0.1',
      author='DEA team',
      author_email='earth.observation@ga.gov.au',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.5',
      ext_modules=cythonize('dea_tsmask/impl.pyx'),
      include_dirs=[numpy.get_include()])
