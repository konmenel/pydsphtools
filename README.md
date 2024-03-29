# PyDSPHtools
Some functions I often re-use for DualSPHysics cases. I am slowly adding functionality as I need it.

# Dependencies
- Python 3.8
- Numpy
- Pandas
- Scipy
- lxml

# Installation
## Download the package
If you have Git install you can just clone the package by running:
```bash
git clone https://github.com/konmenel/pydsphtools.git
```

Else you can simple download the zip from https://github.com/konmenel/pydsphtools

## (Optional) Create a virtual environment
### With Conda
Create a new conda environment:
```bash
conda env create -f environment.yml
```

Active the environment:
```bash
conda activate pydsphtools
```
### With venv
Create a new virtual environment
```bash
python -m venv ./venv
```

Active the environment:
```bash
venv\Scripts\Activate
```

Install the dependencies
```bash
python -m pip install -r requirements.txt 
```

## Install the package
To install the package locally simply run:
```bash
pip install -e .
```

# Documentation
The documentation generated with [pdoc](https://pdoc3.github.io/pdoc/) of the package can be found in the `doc/` directory. Open the file `doc/pydsphtools/index.html` in the browser to read it.

To generate the documenation install pdoc using pip
```bash
pip install pdoc3
```

and then run from the root directory of package
```bash
pdoc --html -c latex_math=True -o doc pydsphtool
```
