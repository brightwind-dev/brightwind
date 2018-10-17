# brightwind
--------------

<p align="center">
  <img src="https://user-images.githubusercontent.com/23189856/45693728-a171bc80-bb55-11e8-90a1-e5257b07efc0.jpg" height="600" width="800">
</p>



Brightwind’s open source python library provides wind analysts with easy to use methods for working with
meteorological data. It supports loading of meteorological data, analysing it in various ways including
correlations, frequency tables etc., also tools for transforming the data including averaging, filtering,
etc. are available. The library also exports files which can be used in other softwares like .tab files

---
### Install
From the command prompt, navigate to the root folder of the brightwind repository where you can find the setup.py file and run:
```
C:\...\brightwind> pip install -e .
```
Don't forget the dot at the end. This command will install the package using pip.

---
### Usage
```
import brightwind as bw

data = bw.load_timeseries('C:\...\file_with_data.csv')
bw.basic_stats(data)
```



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
