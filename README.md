--------------
```
     __         _       __    __           _           __
    / /_  _____(_)___  / /_  / /__      __(_)___  ___ / /
   / __ \/ ___/ / __ \/ __ \/ __/ | /| / / / __ \/ __  /
  / /_/ / /  / / /_/ / / / / /_ | |/ |/ / / / / / /_/ /
 /_.___/_/  /_/\__, /_/ /_/\__/ |__/|__/_/_/ /_/\__,_/
              /____/
 ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**A Python library primarily for wind resource assessments.**

--------------

<br>

Brightwind is a Python library specifically built for wind analysis. It can load in wind speed, wind direction and 
other metrological timeseries data. There are various plots you can use to understand this data and to find any 
potential issues. You can perform many common functions to the data such as shear and long-term adjustments. The 
resulting adjusted data is then outputted as a frequency distribution tab file which can be used in wind analysis 
software such as WAsP.

This library can also be used for solar resource analysis.

<br>

---
### Installation

You can use pip from the command line to install the library.

```
C:\Users\Stephen> pip install brightwind
```
It is advisable to use a separate environment to avoid any dependency clashes with other libraries such as Pandas, Numpy 
or Matplotlib you may already have installed.

<br>

For those that do not have Python installed and are just getting started, we recommend installing Anaconda. Anaconda is 
a Python distribution for scientific computing and so provides everything you need, Python, pip and Jupyter Notebook 
along with libraries such as Pandas, Numpy and Matplotlib. Datacamp provide a good tutorial for [installing 
Anaconda on Windows](https://www.datacamp.com/tutorial/installing-anaconda-windows) to get started.

Once Anaconda is installed, you can use the **Anaconda Prompt** to run the above command line `pip install brightwind`. 
Or first use **Anaconda Navigator** to create an environment.

---
### Documentation

Documentation on how to get setup and use the library can be found at https://brightwind-dev.github.io/brightwind-docs/

<br>

Example usage of the brightwind library is shown below using Jupyter Notebook. Jupyter Notebook is a powerful way to 
immediately see the results of code you have written.
<br>

<p>

![demo_image_1](read_me_1.png)
![demo_image_2](read_me_2.png)
</p>




<br>

##### Features
The library provides wind analysts with easy to use tools for working with
meteorological data. It supports loading of meteorological data, averaging,
filtering, plotting, correlations, shear analysis, long term adjustments, etc.
The library can then export a resulting long term adjusted tab file to be used in
other wind analysis software.

<br>

##### Benefits
The key benefits to an open-source library is that it provides complete transparency
and traceability. Anyone in the industry can review any part of the code and suggest changes,
thus creating a standardised, validated toolkit for the industry.

By default, during an assessment every manipulation or adjustment made to the wind data is
contained in a single file. This can easily be reviewed and checked by internal reviewers or,
as the underlying code is open-sourced, there is no reason why this file cannot be sent to
3rd parties for review thus increasing the effectiveness of a banks due diligence.

<br>

##### License
The library is licensed under the MIT license.

<br>

---
### Test datasets
A test dataset is included in this repository and is used to demonstrate function and test functions in the code. 
Other files and datasets are also included to complement this demo dataset. These are outlined below:

<br>

| Dataset               | Source           | Notes  |
|:--------------------- |:-------------|:-----|
| demo_data.csv         | BrightWind | A modified 2 year met mast dataset in csv and Campbell Scientific format. |
| MERRA-2_XX_2000-01-01_2017-06-30.csv | NASA [GES DISC](https://disc.gsfc.nasa.gov/) | 4 x MERRA-2 18-yr datasets to complement the demo data for long term analyses. |
| demo_cleaning_file.csv | BrightWind | A file containing information on what periods to clean out from the demo data. |
| windographer_flagging_log.txt | BrightWind | The same cleaning info as found in 'demo_cleaning_file.csv' formatted as a Windographer flagging file. |
| demo_data_iea43_wra_data_model.json | BrightWind | A JSON file formatted according to the IEA Wind Task 43 WRA Data Model standard which describes the mast configuration for the demo data. |

<br>

---
### Contributing
If you wish to be involved or find out more please contact stephen@brightwindanalysis.com.

More information can be found in the [contributing.md](https://github.com/brightwind-dev/brightwind/blob/master/contributing.md) section of the website.

<br>
