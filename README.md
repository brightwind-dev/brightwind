# brightwind
--------------

The brightwind python library aims to empower wind resource analysts and
establish a common industry standard toolset.
<br>

###### Features
The library provides wind analysts with easy to use tools for working with
meteorological data. It supports loading of meteorological data, averaging,
filtering, plotting, correlations, shear analysis, long term adjustments, etc.
The library can export a resulting long term adjusted tab file to be used in
other software.

###### Benefits
The key benefits to an open-sourced, scriptable library is that it provides traceability
and validation by the industry. An open-source toolset provides complete transparency
of the calculations that underpin a wind energy assessment leading to a standardisation
of these calculations by the industry thus removing any confusion as to how the data is
manipulated.

As it is scriptable it can be clearly seen exactly how the data was processed making it easier
for reviewing and checking. It can also be used as the foundation building block
in the automation of wind analysis tasks and software development.

###### License
The library is licensed under the GNU Lesser General Public License v3.0. It is the
intention to allow commercial products or services to be built on top of this
library but any modifications to the library itself should be kept open-sourced.
Keeping this open-sourced will ensure that the library becomes the de facto
library used by wind analysts, establishing a common, standard industry wide
toolset for wind data processing .

<br>

Documentation website (work in progress): https://brightwind-dev.github.io/brightwind-docs/

<br>

---
### Installation
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

<br>

---
### Usage
```python
import brightwind as bw

data = bw.load_csv(r'C:\...\file_with_data.csv')
print(data)
```
```python
bw.basic_stats(data)
```
<br>

---
### Test datasets
A test dataset is included in this repository and is used to test functions in the code. The source of the dataset is:

<br>

| Dataset            | Source           | Notes  |
|:------------------ |:-------------|:-----|
| Offshore-CREYAP-2  | [Offshore-CREYAP-2-Data-pack](http://www.ewea.org/events/workshops/past-workshops/resource-assessment-2015/offshore-creyap-part-ii/) | Two offshore met masts with MERRA data. |
| CREYAP Pt II       | [CREYAP-Pt-2](http://www.ewea.org/events/workshops/past-workshops/resource-assessment-2015/offshore-creyap-part-ii/)      | Onshore 50m met mast from the CREYAP Pt II along with additional MERRA-2 reference data  |
| Campbell Scientific | Anonymous | A modified 2 year met mast dataset in the format of a Campbell Scientific CR1000 logger along with associated 18-yr MERRA-2 data. |

<br>

---
### Contributing
If you wish to be involved or find out more please contact stephen@brightwindanalysis.com.

Most contributors use the PyCharm IDE and follow the built in PyCharm code style.

<br>

---
### Sphinx-docs
###### setting up Sphinx-docs

Currently we used Sphinx to automate documentation. Download sphinx-bootstrap-theme using the following command:

```
pip install sphinx-bootstrap-theme
```
You may need the following to work with Jupyter Notebooks
```
pip install nbsphinx --user
```
If a module file has not yet been created this will need to be done e.g.
```
sphinx-doc -f -o source/ ../brightwind/load
```
###### running
If already set up then simply run
```
make html
```

The documentation is hosted using github pages and there is a separate repo brightwind-docs to host files for the webpage. To know how to set up the repos to contribute to the documentations, see README for brightwind-docs.
<br>

---
