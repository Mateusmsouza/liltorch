from setuptools import setup, find_packages

setup(
    name='liltorch',
    version='0.0.1',
    description='Small neural network library made only with raw python',
    author='Mateus Souza',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
