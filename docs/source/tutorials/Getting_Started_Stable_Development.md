
# Getting Started 
***
This guide will lead you through all the steps and installations required to start using the brightwind library, assuming a basic level of computer knowledge. These steps will include:

- Installing Anaconda Python Distribution
- Installing the brightwind library
- Opening the brightwind library in Jupyter notebooks

If you are familiar with Python environments, Git and would like to use the development verison of the brightwind library, you may prefer to follow the instructions for downloading the development (unstable) code.
***

## Setup
***
### Step 1:
- First, we need to download the Anaconda Python distribution, found here: https://www.anaconda.com/distribution/. Please ensure you download the current version of the installer for your machine, as if not this will cause problems.

- Once the installation has completed, open 'Search' and search 'Anaconda'. You should see two applications appear, the Anaconda Prompt and the Anaconda Navigator, as shown in the image below:

![Getting_Started_Img_4.png](attachment:Getting_Started_Img_4.png)
<br>
- If you cannot see both applications, it is likely that Anaconda did not install properly, as this has been known to occur. Please uninstall and reinstall the program and try again. Instructions on how to install
Anaconda can be found here: https://docs.anaconda.com/anaconda/install/
***


### Step 2
- Click on Anaconda  prompt in your computer’s search bar and open it. Navigate to the root folder of your folder of choice, i.e. the folder <em>brightwind_demo</em> in this case,  and type:

```
C:\...\brightwind_demo> pip install brightwind 
```

![Pip_install_bw.png](attachment:Pip_install_bw.png)
<br>


- If you receive errors stating that the command <em> pip </em> does not exist, it is likely that Anaconda did not install properly, as this has been known to occur. Please uninstall and reinstall the program and try again. 
<br> <br>
***


### Step 3:

* To begin using brightwind for data analysis, first open Anaconda Navigator, which you should have downloaded and installed in Steps 1 and 2.
<br>
<br>
* On the Anaconda Navigator start-up page, open up the Jupyter Notebook (not Jupyter Lab). This will open a window in your default web browser. This web browser window should display a directory of all files on your computer. Navigate to the folder in which you installed brightwind in Step 2 above, or whichever other folder you would like to work in. Jupyter will be able to find brightwind no matter which folder you work in.

![Getting_Started_Img_1.png](attachment:Getting_Started_Img_1.png)

<br>

* Now, in the top right-hand corner of the screen, click ‘New’ and then ‘Python 3’ to open a new Python kernel. 


![Getting_Started_Img_2.png](attachment:Getting_Started_Img_2.png)

<br>

* To import the brightwind code and start using the functions, type:
```
import brightwind as bw
```
into the cell and press ‘Shift’ and ‘Enter’ to import the code.

![Getting_Started_Img_3.png](attachment:Getting_Started_Img_3.png)
 
 ***



```python

```
