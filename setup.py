from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='brightwind',
    version='0.2.0',
    packages=['brightwind', 'brightwind.load', 'brightwind.utils', 'brightwind.export', 'brightwind.analyse',
              'brightwind.transform'],
    url='https://github.com/brightwind-dev/brightwind.git',
	download_url = 'https://github.com/brightwind-dev/brightwind/archive/v0.2.0.tar.gz',
    license='LGPL-3.0-or-later',
    author='Stephen Holleran and Inder Preet of BrightWind Ltd',
    author_email='stephen@brightwindanalysis.com',
    description='Scripts for wind resource data processing.',
    long_description=long_description,
    long_description_content_type="text/markdown",
	keywords = ['WIND', 'RESOURCE', 'DATA', 'ANALYSTS', 'PROCESSING', 'WASP', 'ROSE', 'WINDFARMER', 'OPENWIND', 'WIND PRO', 'WINDOGRAPHER'],
	classifiers=[
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"License :: OSI Approved :: LGPL-3.0-or-later",
		"Operating System :: OS Independent",
	],
)
