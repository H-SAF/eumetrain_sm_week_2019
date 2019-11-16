#!/bin/bash

unset PYTHON_DIR
unset PYTHON_VERSION
unset PYTHON_LIB
unset PYTHONHOME
unset PYTHONPATH
unset PYTHON_INCLUDE
#Script to create a conda environment sm_env and install the necessary required packages

curl https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-MacOSX-x86_64.sh -o miniconda.sh

bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda create -n sm_env
conda install -n sm_env-c conda-forge numpyscipypandas matplotlib rasterio geopandas netCDF4 pyflakes statsmodels cartopy basemap basemap-data-hires cython h5py jupyter gdal python=3.6 metview pybufr-ecmwf pykdtree pygrib pyresample
source activate sm_env
conda install -c conda-forge cdo

pip install --upgrade pip
pip install ascat pytesmo metview
