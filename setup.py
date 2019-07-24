from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='brightwind',
    # Update version number here:
    version='0.2.6',
    packages=['brightwind', 'brightwind.load', 'brightwind.utils', 'brightwind.export', 'brightwind.analyse',
              'brightwind.transform', 'brightwind.datasets'],
    package_data={
        # If any package contains *.mplstyle or *.txt files, include them:
        '': ['*.mplstyle', 'datasets/demo/*.csv', 'datasets/demo/*.txt'],
    },
    url='https://github.com/brightwind-dev/brightwind.git',
    # Update version number here:
    download_url = 'https://github.com/brightwind-dev/brightwind/archive/v0.2.6.tar.gz',
    license='GNU Lesser General Public License v3 or later (LGPLv3+)',
    author='Stephen Holleran and Inder Preet of BrightWind Ltd',
    author_email='stephen@brightwindanalysis.com',
    description='Scripts for wind resource data processing.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=['BRIGHT', 'WIND', 'RESOURCE', 'DATA', 'ANALYSTS', 'PROCESSING', 'WASP', 'ROSE', 'WINDFARMER', 'OPENWIND',
              'WIND PRO', 'WINDOGRAPHER'],
    install_requires=[
        'pandas>=0.24.0',
        'numpy>=1.14.6',
        'scikit-learn>=0.19.1',
        'matplotlib>=3.0.3',
        'requests>=2.20.0',
        'scipy>=0.19.1',
        'pytest>= 4.1.0',
        'six>= 1.12.0',
        'dateutil>=2.8.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
    ],
)
