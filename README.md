# Spectroscopic Transit Search

These are Python tools for a Spectroscopic Transit Search, as detailed in van Sluijs et al. (2019). A copy of the paper, figures and final results are stored in the *paper* folder.

## Dependencies
You should have `git`, `python`, `astropy`, `lmfit` installed in order to run all Python scripts. Some C++ code is used as well and should be compiled using a C++ compiler as `g++`.

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

## Citing this code

The code is registered at the [ASCL](http://ascl.net/) at
[...](...) and should be cited in papers
as: ...

## License

The code is released under a BSD 2-clause license.
