from setuptools import setup

from my_pip_package import __version__

setup(
    name='neuralnet-fs',
    version=__version__,

    url='https://github.com/aureliusnoble/neuralnet-fs/new/main',
    author='Aurelius Noble',
    author_email='a.j.noble@lse.com',

    py_modules=['nnfs'],
    
    install_requires=[
    'numpy',
],
)
