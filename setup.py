from setuptools import setup

setup(
    name='brightwind',
    version='0.1.0',
    packages=['brightwind', 'brightwind.load', 'brightwind.utils', 'brightwind.export', 'brightwind.analyse',
              'brightwind.transform'],
    url='https://github.com/brightwind-dev/brightwind.git',
    license='LGPL-3.0-or-later',
    author='Stephen Holleran and Inder Preet of BrightWind Ltd',
    author_email='stephen@brightwindanalysis',
    description='Scripts for wind-analysts'
)
