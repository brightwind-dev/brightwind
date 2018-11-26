from setuptools import setup

setup(
    name='brightwind',
    version='0.0.1',
    packages=['brightwind', 'brightwind.load', 'brightwind.utils', 'brightwind.export', 'brightwind.analyse',
              'brightwind.transform'],
    url='https://github.com/brightwindanalysis/brightwind.git',
    license='MIT',
    author='BrightWind Ltd',
    author_email='inder@brightwindanalysis',
    description='Scripts for wind-analysts'
)
