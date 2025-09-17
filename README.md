# Tauc Plot Analyser

This Python application was created to help calculate bandgaps easily from UV-Vis absorbance spectra. Currently only works with data exported from UV-3600 Shimadzu (.csv or .txt files). 

## Features
- Sliders to choose selection areas
- Downloading graph
- Choosing bandgap transition type and calculation of bandgap

## Installation
Download the python scripts and requirements.txt. To create the environment of the app from the .txt file run
```
conda env create --name tauc_plot_analyser --file requirements.txt 
``` 
on a python terminal. To activate the environment run 
```
conda activate tauc_plot_analyser
``` 
You can then run
```
python tauc_plot_analyser_app.py
``` 
from the terminal.

## Create desktop shortcut
You can create a desktop shortcut by running this command in the terminal
```
pyshortcut -n TPA -i App_icon.ico tauc_plot_analyser_app.py
```
More information about pyshortcut can be found [here](https://newville.github.io/pyshortcuts/#)

## Known issues and bugs
- None for now

