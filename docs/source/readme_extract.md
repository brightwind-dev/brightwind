# brightwind
------
The brightwind python library aims to empower wind resource analysts and
establish a common industry standard toolset.
<br>

---------
##### Features
The library provides wind analysts with easy to use tools for working with
meteorological data. It supports loading of meteorological data, averaging,
filtering, plotting, correlations, shear analysis, long term adjustments, etc.
The library can export a resulting long term adjusted tab file to be used in
other software.

----------------
##### Benefits

The key benefits to an open-sourced, scriptable library is that it provides traceability
and validation by the industry. An open-source toolset provides complete transparency
of the calculations that underpin a wind energy assessment leading to a standardisation
of these calculations by the industry thus removing any confusion as to how the data is
manipulated.

As it is scriptable it can be clearly seen exactly how the data was processed making it easier
for reviewing and checking. It can also be used as the foundation building block
in the automation of wind analysis tasks and software development.

--------
##### License
The library is licensed under the GNU Lesser General Public License v3.0. It is the
intention to allow commercial products or services to be built on top of this
library but any modifications to the library itself should be kept open-sourced.
Keeping this open-sourced will ensure that the library becomes the de facto
library used by wind analysts, establishing a common, standard industry wide
toolset for wind data processing .

---------

Project website : https://github.com/brightwind-dev/brightwind

------------
##### Installation
> Note: This is an initial beta release of the library. Anything may change at any
> time. The public API must not be considered stable.

First clone or download the github repository to your machine. From the command prompt, navigate to the root folder of the brightwind repository where you can find the setup.py file and run:
```
C:\...\brightwind> pip install -e .
```
Don't forget the dot at the end. This command will install the package using pip.

Requires Python version 3.6 or later. If you work on Windows, the [Anaconda](https://www.anaconda.com/download/) Python
distribution will install everything you need including [pip](https://www.w3schools.com/python/python_pip.asp) for installing packages and
the [Jupyter Notebook](https://jupyter.org/) for using the library.

---