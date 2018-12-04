# brightwind
--------------

<p align="center">
  <img src="https://user-images.githubusercontent.com/23189856/45693728-a171bc80-bb55-11e8-90a1-e5257b07efc0.jpg" height="600" width="800">
</p>

<br>

The brightwind library aims to empower wind resource analysts and establish a common industry standard toolset.

Brightwind’s open source python library provides wind analysts with easy to use tools for working with
meteorological data. It supports loading of meteorological data, averaging, filtering, plotting, correlations, shear analysis,
long term adjustments, etc. The library can export a resulting long term adjusted tab file to be used in
other software.

The library is licensed under the GNU Lesser General Public License v3.0. It is the intention to allow commercial
products or services to be built on top of this library but any modifications to the library itself should be kept
open-sourced. Keeping this open-sourced will ensure that the library becomes the de facto library used by wind analysts,
establishing a common, standard industry wide toolset for wind data processing .

<br>

---
### Install
From the command prompt, navigate to the root folder of the brightwind repository where you can find the setup.py file and run:
```
C:\...\brightwind> pip install -e .
```
Don't forget the dot at the end. This command will install the package using pip.

Requires Python version 3.6.
<br>

---
### Usage
```python
import brightwind as bw

data = bw.load_csv('C:\...\file_with_data.csv')
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




<br>

---
### Sphinx-docs
##### setting up

From Anaconda prompt navigate to brightwind/sphinx-docs

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
##### running
If already set up then simply run
```
make html
```
<br>

---