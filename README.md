# Spectroscopic Transit Search

These are Python tools for a Spectroscopic Transit Search, as detailed in [this paper](https://arxiv.org/abs/1903.08186). This github repository contains the following repositories:
* [figures](https://github.com/lennartvansluijs/spectroscopic-transit-search/tree/master/figures) contains the final Figures
* [data](https://github.com/lennartvansluijs/spectroscopic-transit-search/tree/master/data) contains the spectral data of Beta Pictoris
* [scripts](https://github.com/lennartvansluijs/spectroscopic-transit-search/tree/master/scripts) contains the Python scripts used to create the plots

## Dependencies
You should have `git`, `python`, `astropy`, `lmfit` (only Figure 7) installed in order to run all Python scripts. All scripts were written using Python 3. Some C++ code is used as well and should be compiled using a C++ compiler as `g++`.

## Quick start
1. Clone the git directory to your local storage area.
```python
git clone https://github.com/lennartvansluijs/spectroscopic-transit-search <local folder>
```
2. In the cloned directory, go to the scripts folder and compile the C++ code using your favourite compiler. When using `g++`
```bash
cd scripts
g++ -Wall -fPIC -O3 -march=native -fopenmp -c utils.cpp
g++ -shared -o libutils.so utils.o
```
3. Now you should be ready to run all scripts generating Figures 1-8 in Jupyter Notebook by running
```bash
jupyter notebook plot_fig1.ipynb
jupyter notebook plot_fig2.ipynb
...
```

## License

The code is released under a BSD 2-clause license.
