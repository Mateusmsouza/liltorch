from setuptools import setup, find_packages

setup(
    name='liltorch',
    version='0.0.2',
    description='Small neural network library made with numpy and raw python',
    author='Mateus Souza',
    packages=find_packages(),
    install_requires=['numpy'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
