# Xpipe3
# A X-ray Chandra Code for Galaxy Clusters

The code has three operational steps: pre-process, imaging, analysis. The two last modes are depedent from the previous ones.

### Prerequisites

* [CIAO](http://cxc.harvard.edu/ciao/download/) - is the software package developed by the Chandra X-Ray Center for analysing data from the Chandra X-ray Telescope.

Before anything, you must always initialize the ciao environment. 
```
ciao
```
Further, the CIAO package comes from with its own python dependencies and directories. So, at the first time CIAO initialization you should install the libraries bellow although we have already in your computer.


* [Astropy](https://www.astropy.org) - It's used to manage the fits tables
* [ConfigParser] 

### Setup the code

The script is config file base. In order to run the code, setup the `./Xpipe_Config.ini` file. 

Section `[paths]`: there are two paths, one to storage the X-ray Chandra data files and the other is for the outputs. 
Section `[Files]`: the input catalog
Section `[Columns]`: set the columns names.

Section `[Mode]`: choose the steps to run the code. It is bolean `True` or `False`.

If `[parallel]`, you might choose the number of cores and the number of jobs per time. The `batchStart` and `batchMax` parameters are integers values that are used as index in the input catalog.

### Run the code
After all the steps above, you just run:

```
python main.py
```

## Authors

* **Johnny H. Esteves**[PurpleBooth](https://github.com/estevesjh)

