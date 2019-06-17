
# Basic Analysis
***
## Using the brightwind library in Jupyter Notebook

* With brightwind installed and opened in the Jupyter notebook, you can begin to analyse data using a wide range of functions. First, import the brightwind folder into the Jupyter notebook.
```
Import brightind as bw
```
* Now, to display the available functions type
```
bw.
```
and press 'Tab'.
<br>
<br>
* Once you have selected which function from the list to use, open parenthesis, i.e:
```
bw.monthly_means() 
```
* To view the list of necessary inputs for this function, press ‘Shift’ and ‘Tab’ simultaneously.
<br>

![Bw_function.png](attachment:Bw_function.png)

***

## Importing Data
* Most data analysis will involve the importing of data from excel spreadsheets or .csv files. To import data into the brightwind workspace from a <em>CSV</em> file, we can use the load_csv function.
<br>

* For example, to import a demo <em>.csv</em> file such as the  <em>MERRA-2_SE_2000-01-01_2017-06-30.csv</em> distributed with brightwind, type:

```
mydata = bw.load_csv(r’C:\Users\myuser\brightwind\datasets\demo\MERRA-2_SE_2000-01-01_2017-06-30.csv’) 

```
and press ‘Shift’ and ‘Enter’.

* This will load the data form the excel spreadsheet into a pandas DataFrame, <em>mydata</em>, shown below:

 ![DataFrame_Example.png](attachment:DataFrame_Example.png)

<br>

* Once this data is loaded into the brightwind environment, it can be used within different functions. For example, to calculate the monthly mean wind speeds from the first column of mydata, i.e the <em>WS50m_m/s</em>  column, for the first six months of the year, type:
```
bw.monthly_means(mydata.loc[:'2000-06-30 23:00:00','WS50m_m/s'])
```

* This should produce the following graph: 

![Graph.png](attachment:Graph.png)
***


```python

```
