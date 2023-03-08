# PyDSPHtools
Some functions I often re-use for DualSPHysics cases. I am slowly adding functionality as I need it.

# Dependencies
- Python 3.8
- Numpy
- Pandas
- Scipy

# Installation
## Download the package
If you have Git install you can just clone the package by running:
```console
$ git clone https://github.com/konmenel/pydsphtools.git
```

Else you can simple download the zip from https://github.com/konmenel/pydsphtools

## (Optional) Create a virtual environment
### With Conda
Create a new conda environment:
```console
$ conda env create -f environment.yml
```

Active the environment:
```console
$ conda activate pydsphtools
```
### With venv
Create a new virtual environment
```console
$ python -m venv ./venv
```

Active the environment:
```console
$ venv\Scripts\Activate
```

Install the dependencies
```console
$ python -m pip install -r requirements.txt 
```

## Install the package
To install the package locally simply run:
```console
$ pip install -e .
```
