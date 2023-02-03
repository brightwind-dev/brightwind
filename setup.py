from setuptools import setup, find_packages
import codecs
import os.path


with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='brightwind',
    # Update version number here, below and in _init_.py:
    version=get_version("brightwind/__init__.py"),
    packages=['brightwind', 'brightwind.load', 'brightwind.utils', 'brightwind.export', 'brightwind.analyse',
              'brightwind.transform', 'brightwind.demo_datasets'],
    package_data={
        # If any package contains *.mplstyle or *.txt files, include them:
        '': ['*.mplstyle', 'demo_datasets/*.csv', 'demo_datasets/*.txt', 'demo_datasets/*.json'],
    },
    url='https://github.com/brightwind-dev/brightwind.git',
    # UPDATE VERSION NUMBER HERE:
    download_url='https://github.com/brightwind-dev/brightwind/archive/v2.0.0.tar.gz',
    license='MIT',
    author='Stephen Holleran of BrightWind Ltd',
    author_email='stephen@brightwindanalysis.com',
    description='Scripts for wind resource data processing.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=['BRIGHT', 'WIND', 'RESOURCE', 'DATA', 'ANALYSTS', 'PROCESSING', 'WASP', 'ROSE', 'WINDFARMER', 'OPENWIND',
              'WIND PRO', 'WINDOGRAPHER'],
    install_requires=[
        'pandas>=0.24.0, <=1.5.3',
        'numpy>=1.16.4',
        'scikit-learn>=0.19.1',
        'matplotlib>=3.0.3',
        'requests>=2.20.0',
        'scipy>=0.19.1',
        'pytest>= 4.1.0',
        'six>= 1.12.0',
        'python-dateutil>=2.8.0',
        'ipywidgets>=7.4.2',
        'ipython>=7.4.0',
        'gmaps>=0.9.0',
        'boto3>=1.9.66',
        'colormap>=1.0.1',
        'easydev>=0.10.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)
