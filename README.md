--------------
```
     __         _       __    __           _           __
    / /_  _____(_)___  / /_  / /__      __(_)___  ___ / /
   / __ \/ ___/ / __ \/ __ \/ __/ | /| / / / __ \/ __  /
  / /_/ / /  / / /_/ / / / / /_ | |/ |/ / / / / / /_/ /
 /_.___/_/  /_/\__, /_/ /_/\__/ |__/|__/_/_/ /_/\__,_/
              /____/
 ```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**A Python library for analysing wind resource timeseries data.**

--------------

<br>

Brightwind can load in wind speed, wind direction and other metrological timeseries data. There are various plots you can use to understand this data and find any potential issues to filter out. You can perform many common functions to the data such as shear and long-term adjustments. The resulting adjusted data is then outputted as a frequency distribution tab file.

This library isn't only for wind data, it is also used for solar resource data.

<br>

---
### Installation

The library can be installed by using pip from the command line (for those that have pip installed).

<br>

```
C:\Users\Stephen> pip install brightwind
```

<br>

For those that do not already have Python or pip, please follow this tutorial,
[getting started on Windows](https://brightwind-dev.github.io/brightwind-docs/tutorials/getting_started_on_windows.html),
to get set up.

---
### Documentation

Documentation on how to get setup and use the library can be found at https://brightwind-dev.github.io/brightwind-docs/

<br>

Example usage is shown below via a Jupyter Notebook.
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
The library can export a resulting long term adjusted tab file to be used in
other software.

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
A test dataset is included in this repository and is used to test functions in the code. The source of the dataset is:

<br>

| Dataset            | Source           | Notes  |
|:------------------ |:-------------|:-----|
| Demo data          | Anonymous | A modified 2 year met mast dataset in various logger formats along with associated 18-yr MERRA-2 data. |

<br>

---
### Contributing
If you wish to be involved or find out more please contact stephen@brightwindanalysis.com.

More information can be found in the [contributing.md](https://github.com/brightwind-dev/brightwind/blob/master/contributing.md) section of the website.

<br>
