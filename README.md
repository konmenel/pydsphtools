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

## (Optional) Create a conda environment for development
Create a new conda environment:
```bash
conda env create -f environment.yml
```

Active the environment:
```bash
conda activate pdsh-dev
```

## Install the package
To install the package locally simply run:
```bash
pip install .
```

To install the development package run:
```bash
pip install -e ".[dev]"
```

# Documentation
The documentation generated with [pdoc](https://pdoc3.github.io/pdoc/) of the package can be found in the `doc/` directory. Open the file `doc/pydsphtools/index.html` in the browser to read it.

To generate the documenation run
```bash
python generate_doc.py
```
